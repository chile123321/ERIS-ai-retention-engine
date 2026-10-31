"""Small synthetic fits only: isolation, frozen-schema reuse and honest OOF comparisons."""

import copy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import sklearn.calibration as sklearn_calibration
from sklearn.impute import SimpleImputer

from eris_ml.evaluation.feature_ablation import make_shared_folds
from eris_ml.features.schema_v3 import V3InputValidator, load_protocol
from eris_ml.models import adaboost_v3 as study

ROOT = Path(__file__).resolve().parents[2]
SMALL_GRID = {
    "model__estimator__max_depth": [1, 2],
    "model__n_estimators": [3],
    "model__learning_rate": [0.5],
}


@pytest.fixture
def fixture():
    config = load_protocol(ROOT / "configs/features/feature_set_v3_final_15.yaml")
    rng = np.random.default_rng(31)
    frame = pd.DataFrame(
        {
            f["name"]: rng.choice(f.get("vocabulary", f.get("values", [1, 2, 3, 4])), 80)
            for f in config["features"]
        }
    )
    frame["Attrition"] = [0, 0, 0, 1] * 20
    frame["source_row"] = np.arange(100, 180)
    frame["EmployeeNumber"] = np.arange(1000, 1080)
    frame.loc[[1, 5, 20], "TrainingTimesLastYear"] = np.nan
    assignments = frame[study.baseline.IDS].copy()
    assignments["fold"] = make_shared_folds(frame.Attrition.to_numpy())
    return config, frame, assignments


def test_pipeline_reuses_schema_and_rejects_invalid(fixture):
    config, frame, _ = fixture
    pipeline = study.pipeline_for(config)
    model = pipeline.named_steps["model"]
    assert model.n_estimators == 50 and model.learning_rate == 1.0
    assert model.random_state == model.estimator.random_state == 42
    assert model.estimator.max_depth == 1 and model.estimator.class_weight is None
    used = [
        c for _, _, columns in pipeline.named_steps["preprocessor"].transformers for c in columns
    ]
    assert len(used) == 15 and set(used) == {f["name"] for f in config["features"]}
    X = frame[used].copy()
    X.loc[0, "Department"] = "unknown employer category"
    with pytest.raises(ValueError, match="vocabulary"):
        pipeline.fit(X, frame.Attrition)
    with pytest.raises(ValueError, match="Unexpected"):
        pipeline.fit(frame, frame.Attrition)


def test_fit_and_calibrator_only_see_outer_training(fixture, monkeypatch):
    config, frame, assignments = fixture
    original_frame = frame.copy(deep=True)
    outer = np.flatnonzero(assignments.fold.to_numpy() != 1)
    splits = study.inner_splits(frame.iloc[outer].Attrition.to_numpy())
    allowed = [set(frame.iloc[outer].index)] + [set(frame.iloc[outer[t]].index) for t, _ in splits]
    seen, imputed, calibrated = [], [], []
    original_validate = V3InputValidator.fit
    original_impute = SimpleImputer.fit
    original_sigmoid = sklearn_calibration._SigmoidCalibration.fit

    def validate(self, X, y=None):
        assert set(X.index) in allowed
        assert list(X.columns) == [f["name"] for f in config["features"]]
        seen.append(set(X.index))
        return original_validate(self, X, y)

    def impute(self, X, y=None):
        assert set(X.index) in allowed
        imputed.append(set(X.index))
        fitted = original_impute(self, X, y)
        np.testing.assert_allclose(fitted.statistics_, X.median())
        return fitted

    def sigmoid(self, X, y, sample_weight=None):
        np.testing.assert_array_equal(y, frame.iloc[outer].Attrition.to_numpy())
        calibrated.append(len(y))
        return original_sigmoid(self, X, y, sample_weight=sample_weight)

    monkeypatch.setattr(V3InputValidator, "fit", validate)
    monkeypatch.setattr(SimpleImputer, "fit", impute)
    monkeypatch.setattr(sklearn_calibration._SigmoidCalibration, "fit", sigmoid)
    oof, trace = study.fit_fold(frame, assignments, config, 1, SMALL_GRID)
    assert seen == imputed and len(calibrated) == 2
    # 2 grid candidates x 4 inner fits + refit + baseline + 2*(4+1) calibration fits.
    assert len(seen) == 20
    assert set(oof.model) == set(study.MODELS)
    assert len(oof) == 4 * int(assignments.fold.eq(1).sum())
    assert trace["diagnostics"]["tuned"]["raw_refit_max_difference"] < 1e-12
    pd.testing.assert_frame_equal(frame, original_frame)


def test_validation_labels_cannot_change_search_or_predictions(fixture):
    config, frame, assignments = fixture
    first, trace = study.fit_fold(frame, assignments, config, 1, SMALL_GRID)
    changed, changed_assignment = frame.copy(), assignments.copy()
    valid = assignments.fold.eq(1)
    changed.loc[valid, "Attrition"] = 1 - changed.loc[valid, "Attrition"]
    changed_assignment.loc[valid, "Attrition"] = changed.loc[valid, "Attrition"]
    second, trace2 = study.fit_fold(changed, changed_assignment, config, 1, SMALL_GRID)
    np.testing.assert_allclose(first.probability, second.probability, rtol=1e-12, atol=1e-15)
    assert trace["best_parameters"] == trace2["best_parameters"]
    assert trace["search_results"] == trace2["search_results"]


def test_complete_oof_alignment_top_k_report_and_repeatability(fixture, monkeypatch):
    config, frame, assignments = fixture
    original_open = Path.open

    def guarded_open(path, *args, **kwargs):
        assert "final_test" not in path.as_posix() and "/f1/" not in path.as_posix()
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    blocks = [study.fit_fold(frame, assignments, config, f, SMALL_GRID)[0] for f in range(1, 6)]
    repeat, _ = study.fit_fold(frame, assignments, config, 1, SMALL_GRID)
    pd.testing.assert_frame_equal(blocks[0], repeat)
    oof = pd.concat(blocks, ignore_index=True)
    reference = oof[oof.model.eq("adaboost_baseline_raw")].copy()
    reference["model"] = study.RF  # Synthetic comparator; never load production data in tests.
    combined = pd.concat([oof, reference], ignore_index=True)
    result = study.analyze(combined.sample(frac=1, random_state=2), assignments)
    for record in result["models"].values():
        assert record["rows"] == 80
        assert len(record["folds"]) == 5
        assert [r["pooled"]["alerts"] for r in record["top_k"]] == [0, 5, 10]
    report = study.report_text(result)
    assert all(m in report for m in study.MODELS + [study.RF])
    assert "EXPLORATORY" in report and "EmployeeNumber" not in report
    broken = combined.copy()
    broken.loc[0, "fold"] = 99
    with pytest.raises(ValueError, match="fold mismatch"):
        study.analyze(broken, assignments)
    for invalid in (float("nan"), float("inf"), -0.1, 1.1):
        broken = combined.copy()
        broken.loc[0, "probability"] = invalid
        with pytest.raises(ValueError):
            study.analyze(broken, assignments)
    with pytest.raises(ValueError, match="duplicated"):
        study.analyze(pd.concat([combined, combined.iloc[:1]]), assignments)


def test_tie_break_and_calibration_guardrails():
    params = [
        {"model__estimator__max_depth": d, "model__n_estimators": n, "model__learning_rate": lr}
        for d, n, lr in [(2, 3, 1), (1, 5, 1), (1, 3, 0.5)]
    ]
    assert study.choose_best({"mean_test_score": [0.5, 0.5, 0.5], "params": params}) == 2
    with pytest.raises(ValueError, match="Nonfinite"):
        study.choose_best({"mean_test_score": [np.nan], "params": params[:1]})
    raw = {
        "pooled": {"average_precision": 0.5, "roc_auc": 0.8, "brier_score": 0.12, "log_loss": 0.4},
        "folds": [{"brier_score": 0.12}] * 5,
    }
    sigmoid = copy.deepcopy(raw)
    sigmoid["pooled"]["brier_score"] = 0.11
    sigmoid["folds"] = [{"brier_score": 0.11}] * 5
    assert study.calibration_proposal(raw, sigmoid)["proposed_method"] == "sigmoid"
    for metric in ("average_precision", "roc_auc"):
        bad = copy.deepcopy(sigmoid)
        bad["pooled"][metric] -= 0.0051
        assert study.calibration_proposal(raw, bad)["proposed_method"] == "raw"
    assert not study.calibration_proposal(raw, sigmoid)["independently_evaluated_selection_policy"]
    flags = study.top_k(np.ones(20) * 0.5, np.arange(20)[::-1], 10)
    assert np.flatnonzero(flags).tolist() == [18, 19]


@pytest.mark.parametrize(
    "relative",
    [
        "data/processed/final_test_raw_v1.csv",
        "artifacts/predictions/v3_final_15/f1/predictions.csv",
    ],
)
def test_final_guard_before_io(tmp_path, relative):
    with pytest.raises(ValueError):
        study.study_path(tmp_path, relative)


def test_existing_output_never_overwritten(tmp_path):
    folder = tmp_path / "artifacts/reports/adaboost_v3/a1"
    folder.mkdir(parents=True)
    sentinel = folder / "original.txt"
    sentinel.write_text("preserve")
    with pytest.raises(ValueError, match="already exist"):
        study.run(tmp_path, "a1")
    assert sentinel.read_text() == "preserve"
