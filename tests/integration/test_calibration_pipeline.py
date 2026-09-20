"""Small synthetic end-to-end calibration checks; no real data is loaded."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from sklearn.model_selection import StratifiedKFold

from eris_ml.evaluation.calibration_reporting import (
    candidate_configuration,
    plot_reliability,
    summarize_calibration,
    write_calibration_report,
)
from eris_ml.features.definitions import load_feature_definition
from eris_ml.models.calibration import OOF_COLUMNS, run_cross_fitted_calibration
from eris_ml.models.factory import _load_xgb_classifier, create_estimator
from eris_ml.models.training import build_model_pipeline
from eris_ml.models.tuning import XGBOOST_FIXED, load_yaml_config

ROOT = Path(__file__).resolve().parents[2]


def _frame() -> tuple[pd.DataFrame, object]:
    generator = np.random.default_rng(12)
    definition = load_feature_definition(ROOT / "configs/features/feature_set_v1_full.yaml")
    rows = 48
    target = np.array([0] * 36 + [1] * 12)
    generator.shuffle(target)
    frame = pd.DataFrame(index=range(rows))
    for column in definition.nominal:
        frame[column] = generator.choice(["A", "B"], rows)
    for column in definition.ordinal:
        frame[column] = generator.integers(1, 5, rows)
    for column in definition.numeric:
        frame[column] = generator.normal(size=rows)
    frame["Age"] += target
    frame["source_row"] = np.arange(rows)
    frame["EmployeeNumber"] = 1000 + np.arange(rows)
    frame["Attrition"] = target
    return frame, definition


def _setup(frame, definition):
    protocol = load_yaml_config(ROOT / "configs/calibration/calibration_protocol_v1.yaml")
    protocol["outer_cv"]["n_splits"] = 2
    protocol["calibration_cv"]["n_splits"] = 2
    logistic = load_yaml_config(ROOT / "configs/models/logistic_regression.yaml")
    xgboost = load_yaml_config(ROOT / "configs/models/xgboost_tuned_v1.yaml")
    fold_parameters = {
        fold: {
            "n_estimators": 5, "max_depth": 1, "learning_rate": 0.1,
            "min_child_weight": 1, "subsample": 0.8, "colsample_bytree": 0.8,
            "gamma": 0.0, "reg_alpha": 0.1, "reg_lambda": 3.0,
        }
        for fold in (1, 2)
    }
    labels = frame["Attrition"].to_numpy()
    features = frame.loc[:, definition.all_features]
    prior = frame.loc[:, ["source_row", "EmployeeNumber", "Attrition"]].copy()
    prior["outer_fold"] = 0
    prior["logistic_baseline_probability"] = np.nan
    prior["xgboost_tuned_probability"] = np.nan
    splitter = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
    for fold, (training, validation) in enumerate(splitter.split(features, labels), start=1):
        prior.loc[validation, "outer_fold"] = fold
        for model, estimator in (
            ("logistic_baseline", create_estimator(logistic)),
            ("xgboost_tuned", _load_xgb_classifier()(**XGBOOST_FIXED, **fold_parameters[fold])),
        ):
            pipeline = build_model_pipeline(definition, estimator)
            pipeline.fit(features.iloc[training], labels[training])
            prior.loc[validation, f"{model}_probability"] = pipeline.predict_proba(
                features.iloc[validation]
            )[:, 1]
    return protocol, logistic, xgboost, fold_parameters, prior


def test_cross_fitted_calibration_is_deterministic_and_reports_all_candidates(
    tmp_path, monkeypatch
) -> None:
    frame, definition = _frame()
    protocol, logistic, xgboost, fold_parameters, prior = _setup(frame, definition)
    import eris_ml.models.calibration as calibration

    original_fit = calibration.CalibratedClassifierCV.fit
    fitted_indices = []

    def spy_fit(self, x, y, **kwargs):
        fitted_indices.append(set(x.index.tolist()))
        assert len(x) == 24  # outer-training only
        return original_fit(self, x, y, **kwargs)

    monkeypatch.setattr(calibration.CalibratedClassifierCV, "fit", spy_fit)
    first = run_cross_fitted_calibration(
        frame, definition, logistic, xgboost, fold_parameters, prior, protocol
    )
    second = run_cross_fitted_calibration(
        frame, definition, logistic, xgboost, fold_parameters, prior, protocol
    )
    assert len(fitted_indices) == 8 * 2  # 2 models x 2 methods x 2 folds x 2 runs
    validation_sets = {
        fold: set(prior.index[prior["outer_fold"] == fold].tolist()) for fold in (1, 2)
    }
    for indices in fitted_indices:
        assert indices.isdisjoint(validation_sets[1]) or indices.isdisjoint(validation_sets[2])
    assert len(first) == len(frame) == len(second)
    assert first["source_row"].is_unique and first["EmployeeNumber"].is_unique
    assert set(first["outer_fold"]) == {1, 2}
    for column in OOF_COLUMNS:
        assert first[column].notna().all()
        assert np.isfinite(first[column]).all()
        assert first[column].between(0, 1).all()
        np.testing.assert_allclose(first[column], second[column], rtol=1e-7, atol=1e-9)
    np.testing.assert_allclose(
        first["logistic_raw_probability"], prior["logistic_baseline_probability"]
    )
    np.testing.assert_allclose(
        first["xgboost_raw_probability"], prior["xgboost_tuned_probability"]
    )
    summary = summarize_calibration(first)
    plot = tmp_path / "calibration_reliability_v1.png"
    report = tmp_path / "calibration_v1.md"
    plot_reliability(summary, plot)
    metadata = {
        "development_rows": 48, "negative_count": 36, "positive_count": 12,
        "feature_set_version": "v1-full", "feature_count": 25,
        "development_sha256": "synthetic", "outer_splits": 2, "outer_seed": 42,
        "step10a_oof_sha256": "synthetic", "step10a_search_sha256": "synthetic",
        "logistic_config_sha256": "synthetic", "xgboost_config_sha256": "synthetic",
        "xgboost_fold_parameters": "synthetic",
        "calibration_splits": 2, "calibration_seed": 43, "bins": 10,
        "python_version": "test", "pandas_version": "test", "sklearn_version": "test",
        "xgboost_version": "test", "matplotlib_version": "test",
    }
    write_calibration_report(summary, metadata, report, plot)
    text = report.read_text(encoding="utf-8")
    for model in ("Logistic baseline", "XGBoost tuned"):
        for method in ("raw", "sigmoid", "isotonic"):
            assert f"| {model} | {method} |" in text
    assert "Locked final test accessed: No" in text
    assert plot.is_file() and plot.stat().st_size > 0
    for model in ("logistic", "xgboost"):
        candidate = candidate_configuration(model, summary["decisions"][model]["method"])
        path = tmp_path / f"{model}.yaml"
        path.write_text(yaml.safe_dump(candidate), encoding="utf-8")
        saved = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert saved["calibration"]["method"] in {"none", "sigmoid", "isotonic"}
        assert saved["selection"] == {
            "development_only": True, "final_test_evaluated": False,
            "production_approved": False,
        }
    assert not list(tmp_path.glob("*.joblib"))


def test_prior_oof_regression_failure_stops_before_result() -> None:
    frame, definition = _frame()
    protocol, logistic, xgboost, fold_parameters, prior = _setup(frame, definition)
    prior.loc[0, "logistic_baseline_probability"] = 0.999
    with pytest.raises(RuntimeError, match="raw regression check failed"):
        run_cross_fitted_calibration(
            frame, definition, logistic, xgboost, fold_parameters, prior, protocol
        )


def test_prior_oof_fold_alignment_failure_stops_before_fit(monkeypatch) -> None:
    frame, definition = _frame()
    protocol, logistic, xgboost, fold_parameters, prior = _setup(frame, definition)
    prior.loc[0, "outer_fold"] = 99
    import eris_ml.models.calibration as calibration

    def forbidden_fit(self, x, y, **kwargs):
        raise AssertionError("Fit should not be reached")

    monkeypatch.setattr(calibration.CalibratedClassifierCV, "fit", forbidden_fit)
    with pytest.raises(ValueError, match="outer-fold assignment"):
        run_cross_fitted_calibration(
            frame, definition, logistic, xgboost, fold_parameters, prior, protocol
        )
