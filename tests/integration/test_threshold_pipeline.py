"""Synthetic nested threshold-policy checks; never load real employee data."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from sklearn.model_selection import StratifiedKFold

from eris_ml.evaluation.threshold_reporting import (
    candidate_configuration,
    plot_tradeoff,
    recommend_candidates,
    write_threshold_report,
)
from eris_ml.evaluation.thresholding import run_cross_fitted_policies
from eris_ml.features.definitions import load_feature_definition
from eris_ml.models.factory import _load_xgb_classifier, create_estimator
from eris_ml.models.training import build_model_pipeline
from eris_ml.models.tuning import XGBOOST_FIXED, load_yaml_config

ROOT = Path(__file__).resolve().parents[2]


def _fixture():
    rng = np.random.default_rng(19)
    definition = load_feature_definition(ROOT / "configs/features/feature_set_v1_full.yaml")
    rows = 48
    labels = np.array([0] * 36 + [1] * 12)
    rng.shuffle(labels)
    frame = pd.DataFrame(index=range(rows))
    for name in definition.nominal:
        frame[name] = rng.choice(["A", "B"], rows)
    for name in definition.ordinal:
        frame[name] = rng.integers(1, 5, rows)
    for name in definition.numeric:
        frame[name] = rng.normal(size=rows)
    frame["Age"] += labels
    frame["source_row"] = np.arange(rows)
    frame["EmployeeNumber"] = 3000 + np.arange(rows)
    frame["Attrition"] = labels
    logistic = load_yaml_config(ROOT / "configs/models/logistic_regression.yaml")
    xgboost = load_yaml_config(ROOT / "configs/models/xgboost_tuned_v1.yaml")
    params = {
        fold: {
            "n_estimators": 5, "max_depth": 1, "learning_rate": 0.1,
            "min_child_weight": 1, "subsample": 0.8, "colsample_bytree": 0.8,
            "gamma": 0.0, "reg_alpha": 0.1, "reg_lambda": 3.0,
        }
        for fold in (1, 2)
    }
    previous = frame.loc[:, ["source_row", "EmployeeNumber", "Attrition"]].copy()
    previous["outer_fold"] = 0
    previous["logistic_raw_probability"] = np.nan
    previous["xgboost_raw_probability"] = np.nan
    features = frame.loc[:, definition.all_features]
    outer = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
    for fold, (training, validation) in enumerate(outer.split(features, labels), start=1):
        previous.loc[validation, "outer_fold"] = fold
        for model, estimator in (
            ("logistic", create_estimator(logistic)),
            ("xgboost", _load_xgb_classifier()(**XGBOOST_FIXED, **params[fold])),
        ):
            fitted = build_model_pipeline(definition, estimator).fit(
                features.iloc[training], labels[training]
            )
            previous.loc[validation, f"{model}_raw_probability"] = (
                fitted.predict_proba(features.iloc[validation])[:, 1]
            )
    policy = load_yaml_config(ROOT / "configs/threshold/threshold_policy_v1.yaml")
    return frame, definition, logistic, xgboost, params, previous, policy


def test_nested_policies_are_deterministic_and_training_only(tmp_path, monkeypatch) -> None:
    frame, definition, logistic, xgboost, params, previous, policy = _fixture()
    import sklearn.pipeline

    original_fit = sklearn.pipeline.Pipeline.fit
    fitted_indices = []

    def spy_fit(self, features, labels, **kwargs):
        fitted_indices.append(set(features.index.tolist()))
        return original_fit(self, features, labels, **kwargs)

    monkeypatch.setattr(sklearn.pipeline.Pipeline, "fit", spy_fit)
    first = run_cross_fitted_policies(
        frame, definition, logistic, xgboost, params, previous, policy,
        outer_splits=2, inner_splits=2,
    )
    second = run_cross_fitted_policies(
        frame, definition, logistic, xgboost, params, previous, policy,
        outer_splits=2, inner_splits=2,
    )
    outer_validation = {
        fold: set(previous.index[previous["outer_fold"] == fold].tolist())
        for fold in (1, 2)
    }
    assert fitted_indices
    assert all(
        indices.isdisjoint(outer_validation[1]) or indices.isdisjoint(outer_validation[2])
        for indices in fitted_indices
    )
    output = first["oof"]
    assert len(output) == 48 and output["source_row"].is_unique
    assert set(output["outer_fold"]) == {1, 2}
    for model in ("xgboost", "logistic"):
        np.testing.assert_allclose(
            output[f"{model}_raw_probability"], previous[f"{model}_raw_probability"]
        )
    for column in output:
        if column.endswith("_prediction"):
            assert output[column].notna().all()
            assert output[column].isin([0, 1]).all()
            np.testing.assert_array_equal(output[column], second["oof"][column])
    assert set(first["summary"]["kind"]) == {"capacity", "recall", "cost"}
    assert len(first["fold_records"]) == 2 * 2 * 9
    assert first["summary"]["feasible_folds"].eq(2).all()
    decision = recommend_candidates(first)
    assert len(decision["candidates"]) <= 3
    candidate = candidate_configuration(first, decision)
    path = tmp_path / "candidates.yaml"
    path.write_text(yaml.safe_dump(candidate), encoding="utf-8")
    saved = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert saved["business_approved"] is False
    assert saved["production_approved"] is False
    assert saved["final_test_evaluated"] is False
    assert "production_threshold" not in saved
    plot = tmp_path / "threshold_tradeoff_v1.png"
    report = tmp_path / "threshold_selection_v1.md"
    metadata = {
        "negative_count": 36, "positive_count": 12, "feature_version": "v1-full",
        "feature_count": 25, "development_sha256": "synthetic",
        "calibration_oof_sha256": "synthetic", "xgboost_search_sha256": "synthetic",
        "xgboost_config_sha256": "synthetic", "logistic_config_sha256": "synthetic",
        "policy_sha256": "synthetic", "fold_parameters": "synthetic",
        "python_version": "test", "pandas_version": "test", "sklearn_version": "test",
        "xgboost_version": "test", "matplotlib_version": "test",
    }
    plot_tradeoff(first, plot)
    write_threshold_report(first, decision, metadata, report, plot)
    text = report.read_text(encoding="utf-8")
    for scenario in (
        "capacity_10_percent", "capacity_15_percent", "capacity_20_percent",
        "capacity_25_percent", "minimum_recall_60_percent",
        "minimum_recall_70_percent", "cost_fn_2x", "cost_fn_5x", "cost_fn_10x",
    ):
        assert scenario in text
    assert plot.is_file() and plot.stat().st_size > 0
    assert not list(tmp_path.glob("*.joblib"))


def test_raw_regression_check_stops_on_changed_outer_probability() -> None:
    frame, definition, logistic, xgboost, params, previous, policy = _fixture()
    previous.loc[0, "xgboost_raw_probability"] = 0.999
    with pytest.raises(RuntimeError, match="raw regression check failed"):
        run_cross_fitted_policies(
            frame, definition, logistic, xgboost, params, previous, policy,
            outer_splits=2, inner_splits=2,
        )


def test_fold_alignment_rejected_before_fitting(monkeypatch) -> None:
    frame, definition, logistic, xgboost, params, previous, policy = _fixture()
    previous.loc[0, "outer_fold"] = 99
    import sklearn.pipeline

    def forbidden_fit(self, features, labels, **kwargs):
        raise AssertionError("Should reject before fitting")

    monkeypatch.setattr(sklearn.pipeline.Pipeline, "fit", forbidden_fit)
    with pytest.raises(ValueError, match="outer folds"):
        run_cross_fitted_policies(
            frame, definition, logistic, xgboost, params, previous, policy,
            outer_splits=2, inner_splits=2,
        )
