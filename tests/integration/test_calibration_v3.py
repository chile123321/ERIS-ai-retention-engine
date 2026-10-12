"""Small synthetic fixtures only; no production training or locked holdout access."""

import copy
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import sklearn.calibration as sklearn_calibration
from sklearn.impute import SimpleImputer

from eris_ml.evaluation.calibration_reporting import reliability_bins
from eris_ml.evaluation.calibration_v3_reporting import (
    analyze,
    append_summary,
    plot_bins,
    policy_metrics,
    select_method,
)
from eris_ml.evaluation.feature_ablation import make_shared_folds
from eris_ml.evaluation.threshold_policy_v2 import top_k
from eris_ml.features.schema_v3 import V3InputValidator, load_protocol
from eris_ml.models import baseline_v3 as baseline
from eris_ml.models import calibration_v3 as calibration

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def setup():
    config = load_protocol(ROOT / "configs/features/feature_set_v3_final_15.yaml")
    policy = calibration.load_policy(ROOT)
    rng = np.random.default_rng(72)
    columns = {
        f["name"]: rng.choice(f.get("vocabulary", f.get("values", list(range(1, 20)))), 80)
        for f in config["features"]
    }
    frame = pd.DataFrame(columns)
    frame["Attrition"] = [1, 0, 0, 0] * 20
    frame["Age"] = np.where(frame.Attrition == 1, 20, 48) + rng.integers(0, 4, 80)
    for feature in ("TrainingTimesLastYear", "YearsWithCurrManager"):
        frame[feature] = frame[feature].astype(float)
        frame.loc[::7, feature] = np.nan
    frame["source_row"] = np.arange(100, 180)
    frame["EmployeeNumber"] = np.arange(1000, 1080)
    assignments = frame[baseline.IDS].copy()
    assignments["fold"] = make_shared_folds(frame.Attrition.to_numpy())
    parameters = {"model__n_estimators": 9, "model__max_depth": 3, "model__n_jobs": 1}
    return config, policy, frame, assignments, parameters


def reference_for(setup, model, fold):
    config, _, frame, assignments, parameters = setup
    pipeline = calibration.pipeline_for(config, model, parameters if model == "rf_tuned" else {})
    names = [f["name"] for f in config["features"]]
    train, valid = assignments.fold != fold, assignments.fold == fold
    pipeline.fit(frame.loc[train, names], frame.loc[train, "Attrition"])
    probability = pipeline.predict_proba(frame.loc[valid, names])[:, 1]
    reference = assignments.loc[valid].rename(columns={"Attrition": "y_true"}).copy()
    reference["model"] = f"{model}_raw"
    reference["probability"] = probability
    reference["feature_set"] = "v3_final_15"
    return reference


@pytest.mark.parametrize("model", calibration.MODELS)
def test_five_fits_only_and_calibrator_preprocessing_isolation(setup, monkeypatch, model):
    config, policy, frame, assignments, parameters = setup
    reference = reference_for(setup, model, 1).sample(frac=1, random_state=9)
    before = frame.copy(deep=True)
    train = np.flatnonzero(assignments.fold.to_numpy() != 1)
    splits = calibration.tuning.inner_splits(frame.iloc[train].Attrition.to_numpy())
    expected = [set(frame.iloc[train[it]].index) for it, _ in splits] + [
        set(frame.iloc[train].index)
    ]
    fit_sets, imputer_sets, calibrator_rows = [], [], []
    validate_fit, impute_fit = V3InputValidator.fit, SimpleImputer.fit
    sigmoid_fit = sklearn_calibration._SigmoidCalibration.fit

    def validate_spy(self, X, y=None):
        assert list(X.columns) == [f["name"] for f in config["features"]]
        fit_sets.append(set(X.index))
        return validate_fit(self, X, y)

    def impute_spy(self, X, y=None):
        imputer_sets.append(set(X.index))
        result = impute_fit(self, X, y)
        np.testing.assert_allclose(result.statistics_, X.median().to_numpy())
        return result

    def sigmoid_spy(self, X, y, sample_weight=None):
        assert len(y) == len(train)
        np.testing.assert_array_equal(y, frame.iloc[train].Attrition.to_numpy())
        calibrator_rows.append(len(y))
        return sigmoid_fit(self, X, y, sample_weight=sample_weight)

    monkeypatch.setattr(V3InputValidator, "fit", validate_spy)
    monkeypatch.setattr(SimpleImputer, "fit", impute_spy)
    monkeypatch.setattr(sklearn_calibration._SigmoidCalibration, "fit", sigmoid_spy)
    result = calibration.fit_outer(
        frame,
        assignments,
        config,
        policy,
        model,
        1,
        parameters if model == "rf_tuned" else {},
        reference,
    )
    assert fit_sets == expected and imputer_sets == expected
    assert calibrator_rows == [len(train)]
    assert result["base_estimator_fits"] == 5
    assert result["raw_max_absolute_difference"] < 1e-9
    pd.testing.assert_frame_equal(frame, before)
    val = set(assignments.loc[assignments.fold == 1, "source_row"])
    assert not val & set(result["outer_training_source_rows"])
    assert all(not val & set(rows) for rows in result["calibration_validation_source_rows"])
    predictions = pd.DataFrame(result["predictions"])
    assert predictions.groupby(["model", "scenario"]).size().eq(len(val)).all()
    assert np.isfinite(predictions.probability).all()
    assert predictions.probability.between(0, 1).all()
    # All four validation scenarios reused the same five fits, no stress refitting.
    assert len(fit_sets) == 5 and len(calibrator_rows) == 1


def test_small_end_to_end_oof_metrics_plot_and_append(setup, tmp_path):
    config, policy, frame, assignments, parameters = setup
    checkpoints, references = [], []
    for model in calibration.MODELS:
        for fold in range(1, 6):
            reference = reference_for(setup, model, fold)
            references.append(reference)
            checkpoints.append(
                calibration.fit_outer(
                    frame,
                    assignments,
                    config,
                    policy,
                    model,
                    fold,
                    parameters if model == "rf_tuned" else {},
                    reference,
                )
            )
    oof = pd.DataFrame([r for c in checkpoints for r in c["predictions"]])
    oof = calibration.validate_results(oof.sample(frac=1, random_state=4), assignments)
    assert len(oof) == 80 * 4 * 4
    reference = pd.concat(references, ignore_index=True)
    reference = pd.concat(
        [
            reference,
            reference.loc[reference.model == "rf_tuned_raw"].assign(model=calibration.REFERENCE),
        ],
        ignore_index=True,
    )
    analysis, bins = analyze(oof, reference, policy)
    assert bins.groupby("model")["count"].sum().eq(80).all()
    assert bins.groupby("model")["positives"].sum().eq(20).all()
    for candidate in calibration.CANDIDATES:
        for percent, expected in ((5, 0), (10, 5), (15, 10)):
            row = next(r for r in analysis["top_k"][candidate] if r["capacity_percent"] == percent)
            assert row["pooled"]["alerts"] == expected  # floor on each 16-row fold, not global
        original = analysis["optional_missing"][candidate]["original"]["pooled"]
        assert original["mean_absolute_probability_change"] == 0
        assert original["top_10_overlap_fraction"] == 1
    plot = tmp_path / "reliability.png"
    plot_bins(bins, plot)
    assert plot.read_bytes().startswith(b"\x89PNG")
    with pytest.raises(FileExistsError):
        plot_bins(bins, plot)
    summary = tmp_path / "summary.md"
    original = b"baseline\r\ntuning\n"
    summary.write_bytes(original)
    preflight = {
        "summary_before": summary.read_text(encoding="utf-8"),
        "summary_before_bytes": len(original),
        "summary_before_sha256": hashlib.sha256(original).hexdigest(),
    }
    append_summary(summary, preflight, "c9", analysis, checkpoints)
    saved = summary.read_bytes()
    append_summary(summary, preflight, "c9", analysis, checkpoints)
    assert saved == summary.read_bytes() and saved.startswith(original)
    assert "ensemble=False" in summary.read_text(encoding="utf-8")


def test_raw_regression_is_by_id_and_rejects_changed_probability(setup):
    _, policy, _, assignments, _ = setup
    raw = reference_for(setup, "logistic_baseline", 1)
    assert (
        calibration.check_raw(
            raw.sample(frac=1),
            raw,
            assignments.loc[assignments.fold == 1],
            policy["raw_regression"],
        )
        == 0
    )
    changed = raw.copy()
    changed["probability"] = 0.5
    with pytest.raises(ValueError, match="Raw regression failed"):
        calibration.check_raw(
            changed, raw, assignments.loc[assignments.fold == 1], policy["raw_regression"]
        )


def test_top_k_ties_and_batch_budget_do_not_use_labels():
    frame = pd.DataFrame(
        {
            "source_row": np.arange(23)[::-1],
            "probability": [0.5] * 23,
            "fold": [1] * 11 + [2] * 12,
            "y_true": [0, 1] * 11 + [0],
        }
    )
    selected = top_k(frame.probability.to_numpy(), frame.source_row.to_numpy(), 10)
    assert sorted(frame.loc[selected == 1, "source_row"].tolist()) == [0, 1]
    result = policy_metrics(frame, 10)
    assert result["pooled"]["alerts"] == 2
    assert [r["k"] for r in result["folds"]] == [1, 1]
    shuffled = frame.sample(frac=1, random_state=2)
    assert policy_metrics(shuffled, 10) == result


def test_rank_reversal_is_not_silently_accepted():
    raw = np.linspace(0.01, 0.99, 100)
    ids = np.arange(100)
    passed = calibration.rank_diagnostics(raw, raw**2, ids, -2.0, 1e-12)
    assert passed["material_pairwise_inversions"] == 0
    with pytest.raises(ValueError, match="Material|not increasing"):
        calibration.rank_diagnostics(raw, 1 - raw, ids, 2.0, 1e-12)


def test_recommendation_guards_not_forced_win(setup):
    _, policy, _, _, _ = setup
    base = {
        "pooled_oof": {
            "brier_score": 0.12,
            "log_loss": 0.4,
            "average_precision": 0.5,
            "roc_auc": 0.8,
        },
        "folds": [
            {
                "fold": k,
                "brier_score": 0.12,
                "log_loss": 0.4,
                "average_precision": 0.5,
                "roc_auc": 0.8,
            }
            for k in range(1, 6)
        ],
    }
    assert select_method(base, base, policy["recommendation"])["recommended_method"] == "raw"
    improved = copy.deepcopy(base)
    for item in [improved["pooled_oof"], *improved["folds"]]:
        item["brier_score"] -= 0.0005
        item["log_loss"] -= 0.01
    chosen = select_method(base, improved, policy["recommendation"])
    assert chosen["recommended_method"] == "sigmoid" and chosen["small_positive_brier_improvement"]
    for key in ("average_precision", "roc_auc"):
        bad = copy.deepcopy(improved)
        bad["pooled_oof"][key] -= 0.006
        assert select_method(base, bad, policy["recommendation"])["recommended_method"] == "raw"
    bad = copy.deepcopy(improved)
    for fold in bad["folds"][2:]:
        fold["brier_score"] = 0.13
    assert select_method(base, bad, policy["recommendation"])["recommended_method"] == "raw"


def test_bins_duplicate_edges_and_probability_validation():
    rows = reliability_bins([0, 1, 0, 1], [0.5] * 4, count=5)
    assert len(rows) == 1 and rows[0]["count"] == 4
    for value in (np.nan, np.inf, -0.1, 1.1):
        with pytest.raises(ValueError):
            reliability_bins([0, 1], [0.2, value], count=5)


def test_alignment_and_final_test_guard_stop_before_fit(setup, monkeypatch, tmp_path):
    config, policy, frame, assignments, _ = setup
    invalid = assignments.copy()
    invalid.loc[0, "EmployeeNumber"] = -1

    def forbidden(*args, **kwargs):
        pytest.fail("No pipeline fit for invalid alignment")

    monkeypatch.setattr(calibration, "pipeline_for", forbidden)
    with pytest.raises(ValueError, match="mismatch before"):
        calibration.fit_outer(
            frame, invalid, config, policy, "logistic_baseline", 1, {}, pd.DataFrame()
        )
    with pytest.raises(ValueError):
        baseline.safe_path(tmp_path, "data/final_test_raw_v1.csv")
    with pytest.raises(ValueError):
        calibration.run(tmp_path, "r1")


def test_baseline_logistic_cannot_receive_tuned_params(setup):
    config, _, _, _, _ = setup
    with pytest.raises(ValueError, match="cannot receive"):
        calibration.pipeline_for(config, "logistic_baseline", {"model__C": 0.01})
