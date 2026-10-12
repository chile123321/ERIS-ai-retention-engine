"""Synthetic remediation isolation/alignment/guardrail tests; no consumed data access."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import sklearn.calibration as sklearn_calibration
from sklearn.impute import SimpleImputer

from eris_ml.evaluation.feature_ablation import make_shared_folds
from eris_ml.evaluation.remediation_v3_reporting import analyze, guardrails, report_results
from eris_ml.features.schema_v3 import V3InputValidator, load_protocol
from eris_ml.models import baseline_v3 as baseline
from eris_ml.models import remediation_v3 as study

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def setup():
    config = load_protocol(ROOT / "configs/features/feature_set_v3_final_15.yaml")
    rng = np.random.default_rng(91)
    frame = pd.DataFrame(
        {
            f["name"]: rng.choice(f.get("vocabulary", f.get("values", [1, 2, 3])), 80)
            for f in config["features"]
        }
    )
    frame["Attrition"] = [0, 0, 0, 1] * 20
    frame["Age"] = np.where(frame.Attrition.eq(1), 23, 49) + rng.integers(0, 3, 80)
    frame["TotalWorkingYears"] = np.where(frame.Attrition.eq(1), 1, 20)
    frame["source_row"] = np.arange(100, 180)
    frame["EmployeeNumber"] = np.arange(1000, 1080)
    assignments = frame[baseline.IDS].copy()
    assignments["fold"] = make_shared_folds(frame.Attrition.to_numpy())
    params = {
        "n_estimators": 7,
        "max_depth": 3,
        "min_samples_leaf": 2,
        "min_samples_split": 2,
        "class_weight": "balanced",
        "random_state": 42,
        "n_jobs": 1,
        "max_features": "sqrt",
    }
    return config, frame, assignments, params


def test_outer_inner_isolation_and_optional_copies(setup, monkeypatch):
    config, frame, assignment, params = setup
    before = frame.copy(deep=True)
    train = np.flatnonzero(assignment.fold.ne(1).to_numpy())
    inner = study.tuning.inner_splits(frame.iloc[train].Attrition.to_numpy())
    expected = (
        [set(frame.iloc[train].index)]
        + [set(frame.iloc[train[t]].index) for t, _ in inner]
        + [set(frame.iloc[train].index)]
    )
    seen, imputers, calibrator = [], [], []
    validate_fit, impute_fit = V3InputValidator.fit, SimpleImputer.fit
    sigmoid_fit = sklearn_calibration._SigmoidCalibration.fit

    def validate(self, X, y=None):
        assert list(X.columns) == [f["name"] for f in config["features"]]
        seen.append(set(X.index))
        return validate_fit(self, X, y)

    def impute(self, X, y=None):
        imputers.append(set(X.index))
        result = impute_fit(self, X, y)
        np.testing.assert_array_equal(result.statistics_, X.median())
        return result

    def sigmoid(self, X, y, sample_weight=None):
        np.testing.assert_array_equal(y, frame.iloc[train].Attrition.to_numpy())
        calibrator.append(len(y))
        return sigmoid_fit(self, X, y, sample_weight=sample_weight)

    monkeypatch.setattr(V3InputValidator, "fit", validate)
    monkeypatch.setattr(SimpleImputer, "fit", impute)
    monkeypatch.setattr(sklearn_calibration._SigmoidCalibration, "fit", sigmoid)
    result = study.fit_fold(frame, assignment, config, params, "rf_balanced", 1)
    assert seen == imputers == expected and calibrator == [len(train)]
    assert result["base_fits"] == 6 and result["calibrator_fits"] == 1
    assert set(result["train_source_rows"]) == set(frame.iloc[train].source_row)
    assert len(result["predictions"]) == 2 * 4 * len(frame[assignment.fold.eq(1)])
    pd.testing.assert_frame_equal(frame, before)


def test_no_age_really_removed_and_seed_deterministic(setup, monkeypatch):
    config, frame, assignment, params = setup
    template = study.pipeline_for(config, params, "rf_unweighted_no_age")
    assert template.named_steps["model"].get_params()["class_weight"] is None
    used = [
        c for _, _, columns in template.named_steps["preprocessor"].transformers for c in columns
    ]
    assert len(used) == 14 and "Age" not in used
    original = V3InputValidator.fit

    def check(self, X, y=None):
        assert len(X.columns) == 14 and "Age" not in X
        assert not {"Gender", "MaritalStatus", "source_row", "EmployeeNumber", "Attrition"} & set(X)
        return original(self, X, y)

    monkeypatch.setattr(V3InputValidator, "fit", check)
    first = study.fit_fold(frame, assignment, config, params, "rf_unweighted_no_age", 1)
    second = study.fit_fold(frame, assignment, config, params, "rf_unweighted_no_age", 1)
    assert first["predictions"] == second["predictions"]
    assert first["feature_order"] == [f["name"] for f in config["features"] if f["name"] != "Age"]


@pytest.mark.parametrize(
    "path",
    [
        "data/processed/final_test_raw_v1.csv",
        "artifacts/predictions/v3_final_15/f1/holdout_predictions.csv",
        "artifacts/reports/v3_final_15/f1/final_metrics.json",
        "artifacts/reports/v3_final_15/final_evaluation_consumed.json",
    ],
)
def test_final_paths_rejected_before_access(tmp_path, path):
    with pytest.raises(ValueError):
        study.study_path(tmp_path, path)


def test_guardrails_specific_failures():
    def row():
        return {
            "pooled_oof": {
                "average_precision": 0.5,
                "roc_auc": 0.8,
                "brier_score": 0.1,
                "log_loss": 0.3,
            },
            "mean_bias": 0.01,
            "folds": [{"fold": i, "brier_score": 0.1, "log_loss": 0.3} for i in range(1, 6)],
        }

    scores = {name: row() for name in study.PRIMARY}
    dummy = {"pooled_oof": {"brier_score": 0.14, "log_loss": 0.44}}
    assert all(r["eligible"] for r in guardrails(scores, dummy).values())
    scores["rf_balanced_raw"]["mean_bias"] = 0.10
    assert not guardrails(scores, dummy)["rf_balanced_raw"]["eligible"]
    sig = scores["rf_unweighted_sigmoid"]
    sig["pooled_oof"]["average_precision"] = 0.47
    for fold in sig["folds"][:3]:
        fold["brier_score"] = 0.11
    outcome = guardrails(scores, dummy)["rf_unweighted_sigmoid"]
    assert not outcome["checks"]["ap_drop_le_0_02"]
    assert not outcome["checks"]["brier_worse_folds_le_2"]


def test_end_to_end_all_candidates_alignment_reports_and_no_final_io(setup, tmp_path, monkeypatch):
    config, frame, assignment, params = setup
    original_open = Path.open

    def safe_open(path, *args, **kwargs):
        normalized = path.as_posix().lower()
        assert "final_test" not in normalized and "/f1/" not in normalized
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", safe_open)
    records = [
        study.fit_fold(frame, assignment, config, params, family, fold)
        for family in study.FAMILIES
        for fold in range(1, 6)
    ]
    assert sum(r["base_fits"] for r in records) == 70
    oof = study.validate_oof(
        pd.DataFrame([p for r in records for p in r["predictions"]]), assignment
    )
    assert len(oof) == 80 * 6 * 4
    assert oof.groupby(["model", "scenario"]).size().eq(80).all()
    assert np.isfinite(oof.probability).all() and oof.probability.between(0, 1).all()
    with pytest.raises(ValueError):
        study.validate_oof(oof.iloc[1:], assignment)
    wrong = assignment.copy()
    wrong.loc[0, "Attrition"] = 1 - wrong.loc[0, "Attrition"]
    with pytest.raises(ValueError):
        study.validate_oof(oof, wrong)
    dummy = oof.loc[(oof.model == "logistic_baseline_raw") & (oof.scenario == "original")].copy()
    dummy["model"] = "dummy_prior"
    dummy["probability"] = 0.25
    analysis, bins, groups = analyze(oof, dummy, frame)
    assert set(analysis["guardrails"]) == set(study.PRIMARY)
    assert analysis["recommendation"]["best_ranking_candidate"] != study.SENSITIVITY
    assert bins.groupby("model")["count"].sum().eq(80).all()
    assert groups.loc[groups.fold.eq(0)].groupby("model").n.sum().eq(80).all()
    report = tmp_path / "devfix_test"
    report.mkdir()
    report_results(report, analysis, bins, groups)
    assert (report / "reliability.png").exists()
    text = (report / "recommendation.md").read_text("utf-8")
    assert all(name in text for name in study.PRIMARY)
    assert not list(report.rglob("*.joblib"))
    hashes = {p: study.sha256_file(p) for p in report.iterdir()}
    report_results(report, analysis, bins, groups)
    assert hashes == {p: study.sha256_file(p) for p in hashes}


def test_existing_output_refused_without_training(tmp_path, monkeypatch):
    (tmp_path / "artifacts/reports/v3_final_15/devfix1").mkdir(parents=True)

    def forbidden(*args, **kwargs):
        pytest.fail("Existing run must be rejected before preflight or fitting")

    monkeypatch.setattr(study, "prepare", forbidden)
    with pytest.raises(FileExistsError):
        study.run(tmp_path, "devfix1")
