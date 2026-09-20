"""Integration tests for leakage-safe baseline cross-validation."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer

from eris_ml.evaluation.reporting import (
    write_class_weight_report,
    write_tree_baseline_report,
)
from eris_ml.features.definitions import load_feature_definition
from eris_ml.models.factory import create_estimator
from eris_ml.models.training import (
    build_model_pipeline,
    evaluate_baselines,
    evaluate_candidates,
)

ROOT = Path(__file__).resolve().parents[2]
FEATURE_CONFIG = ROOT / "configs" / "features" / "feature_set_v1_full.yaml"
MODEL_CONFIG = {
    "model": "logistic_regression",
    "role": "baseline",
    "random_seed": 42,
    "parameters": {
        "solver": "liblinear",
        "C": 1.0,
        "max_iter": 2000,
        "class_weight": None,
    },
}
BALANCED_MODEL_CONFIG = {
    **MODEL_CONFIG,
    "role": "imbalance_candidate",
    "candidate_name": "logistic_regression_balanced",
    "parameters": {**MODEL_CONFIG["parameters"], "class_weight": "balanced"},
}
RANDOM_FOREST_CONFIG = {
    "model": "random_forest",
    "role": "tree_baseline",
    "random_seed": 42,
    "parameters": {
        "n_estimators": 500,
        "max_depth": None,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "max_features": "sqrt",
        "class_weight": None,
        "n_jobs": -1,
    },
}
XGBOOST_CONFIG = {
    "model": "xgboost",
    "role": "tree_baseline",
    "random_seed": 42,
    "parameters": {
        "n_estimators": 300,
        "max_depth": 3,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 1,
        "reg_lambda": 1.0,
        "scale_pos_weight": 1.0,
        "eval_metric": "logloss",
        "tree_method": "hist",
        "n_jobs": -1,
    },
}
EVALUATION_CONFIG = {
    "cross_validation": {"n_splits": 5, "shuffle": True, "random_seed": 42},
    "classification_threshold": 0.5,
}


def _synthetic_development() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    rows = 150
    target = np.zeros(rows, dtype=int)
    target[rng.choice(rows, size=30, replace=False)] = 1
    definition = load_feature_definition(FEATURE_CONFIG)
    frame = pd.DataFrame(index=np.arange(rows))

    for name in definition.nominal:
        frame[name] = rng.choice(["category_a", "category_b"], size=rows)
    for name in definition.ordinal:
        frame[name] = rng.integers(1, 5, size=rows)
    for index, name in enumerate(definition.numeric):
        frame[name] = 20 + index + rng.normal(0, 4, rows)
    frame["Age"] = 35 + target * 5 + rng.normal(0, 4, rows)

    frame["Attrition"] = target
    frame["EmployeeNumber"] = np.arange(1000, 1000 + rows)
    frame["source_row"] = np.arange(1, rows + 1)
    return frame


def test_baseline_pipeline_produces_complete_deterministic_oof_predictions() -> None:
    frame = _synthetic_development()
    definition = load_feature_definition(FEATURE_CONFIG)

    first = evaluate_baselines(frame, definition, MODEL_CONFIG, EVALUATION_CONFIG)
    second = evaluate_baselines(frame, definition, MODEL_CONFIG, EVALUATION_CONFIG)

    assert set(first) == {"dummy", "logistic_regression"}
    for model_name in first:
        result = first[model_name]
        assert len(result["fold_metrics"]) == 5
        assert len(result["oof_probabilities"]) == len(frame)
        assert np.isfinite(result["oof_probabilities"]).all()
        assert (
            (result["oof_probabilities"] >= 0)
            & (result["oof_probabilities"] <= 1)
        ).all()
        assert set(result["fold_assignments"]) == {1, 2, 3, 4, 5}
        assert np.array_equal(
            result["fold_assignments"], second[model_name]["fold_assignments"]
        )
        assert np.allclose(
            result["oof_probabilities"], second[model_name]["oof_probabilities"]
        )

    assert np.array_equal(
        first["dummy"]["fold_assignments"],
        first["logistic_regression"]["fold_assignments"],
    )
    assert first["dummy"]["oof_metrics"]["recall"] == 0.0
    assert (
        first["logistic_regression"]["oof_metrics"]["average_precision"]
        > first["dummy"]["oof_metrics"]["average_precision"]
    )


def test_complete_pipeline_excludes_target_identifiers_and_provenance() -> None:
    definition = load_feature_definition(FEATURE_CONFIG)
    pipeline = build_model_pipeline(definition, create_estimator(MODEL_CONFIG))

    preprocessor = pipeline.named_steps["preprocessor"]
    assert isinstance(preprocessor, ColumnTransformer)
    assert "model" in pipeline.named_steps
    transformed_columns = {
        column for _, _, columns in preprocessor.transformers for column in columns
    }
    assert transformed_columns == set(definition.all_features)
    assert {"Attrition", "EmployeeNumber", "source_row"}.isdisjoint(transformed_columns)


def test_three_candidates_share_folds_and_balanced_improves_minority_detection() -> None:
    frame = _synthetic_development()
    definition = load_feature_definition(FEATURE_CONFIG)
    features = frame.loc[:, definition.all_features]
    candidates = {
        "dummy_prior": {"model": "dummy", "parameters": {"strategy": "prior"}},
        "logistic_regression": MODEL_CONFIG,
        "logistic_regression_balanced": BALANCED_MODEL_CONFIG,
    }

    first = evaluate_candidates(
        candidates,
        definition,
        features,
        frame["Attrition"],
        EVALUATION_CONFIG["cross_validation"],
        EVALUATION_CONFIG["classification_threshold"],
    )
    second = evaluate_candidates(
        candidates,
        definition,
        features,
        frame["Attrition"],
        EVALUATION_CONFIG["cross_validation"],
        EVALUATION_CONFIG["classification_threshold"],
    )

    assert set(first) == set(candidates)
    reference_folds = first["dummy_prior"]["fold_assignments"]
    for candidate_name, result in first.items():
        assert len(result["oof_probabilities"]) == len(frame)
        assert np.isfinite(result["oof_probabilities"]).all()
        assert ((result["oof_probabilities"] >= 0) & (result["oof_probabilities"] <= 1)).all()
        assert np.array_equal(result["fold_assignments"], reference_folds)
        assert np.array_equal(
            result["fold_assignments"], second[candidate_name]["fold_assignments"]
        )
        assert np.allclose(
            result["oof_probabilities"], second[candidate_name]["oof_probabilities"]
        )

    unweighted = first["logistic_regression"]["oof_metrics"]
    balanced = first["logistic_regression_balanced"]["oof_metrics"]
    assert (
        balanced["recall"] > unweighted["recall"]
        or balanced["false_negative_count"] < unweighted["false_negative_count"]
    )


def test_class_weight_report_contains_model_and_operational_tables(tmp_path) -> None:
    frame = _synthetic_development()
    definition = load_feature_definition(FEATURE_CONFIG)
    candidates = {
        "dummy_prior": {"model": "dummy", "parameters": {"strategy": "prior"}},
        "logistic_regression": MODEL_CONFIG,
        "logistic_regression_balanced": BALANCED_MODEL_CONFIG,
    }
    results = evaluate_candidates(
        candidates,
        definition,
        frame.loc[:, definition.all_features],
        frame["Attrition"],
        EVALUATION_CONFIG["cross_validation"],
        EVALUATION_CONFIG["classification_threshold"],
    )
    destination = tmp_path / "class_weight_report.md"
    metadata = {
        "development_rows": len(frame),
        "positive_count": int(frame["Attrition"].sum()),
        "positive_rate": float(frame["Attrition"].mean()),
        "development_sha256": "synthetic",
        "feature_set_version": definition.version,
        "feature_count": len(definition.all_features),
        "n_splits": 5,
        "shuffle": True,
        "random_seed": 42,
        "threshold": 0.5,
        "python_version": "test",
        "pandas_version": "test",
        "sklearn_version": "test",
    }

    write_class_weight_report(results, destination, metadata)
    report = destination.read_text(encoding="utf-8")

    assert "Accuracy" in report
    assert "PR-AUC" in report
    assert "Operational impact" in report
    assert "Predicted positive" in report


def test_tree_candidates_share_folds_record_diagnostics_and_report(tmp_path) -> None:
    frame = _synthetic_development()
    definition = load_feature_definition(FEATURE_CONFIG)
    candidates = {
        "dummy_prior": {"model": "dummy", "parameters": {"strategy": "prior"}},
        "logistic_regression": MODEL_CONFIG,
        "logistic_regression_balanced": BALANCED_MODEL_CONFIG,
        "random_forest": RANDOM_FOREST_CONFIG,
        "xgboost": XGBOOST_CONFIG,
    }
    arguments = (
        candidates,
        definition,
        frame.loc[:, definition.all_features],
        frame["Attrition"],
        EVALUATION_CONFIG["cross_validation"],
        EVALUATION_CONFIG["classification_threshold"],
    )

    first = evaluate_candidates(*arguments)
    second = evaluate_candidates(*arguments)

    assert set(first) == set(candidates)
    reference_folds = first["dummy_prior"]["fold_assignments"]
    for candidate_name, result in first.items():
        assert len(result["oof_probabilities"]) == len(frame)
        assert np.isfinite(result["oof_probabilities"]).all()
        assert ((result["oof_probabilities"] >= 0) & (result["oof_probabilities"] <= 1)).all()
        assert set(result["fold_assignments"]) == {1, 2, 3, 4, 5}
        assert np.array_equal(result["fold_assignments"], reference_folds)
        assert np.allclose(
            result["oof_probabilities"], second[candidate_name]["oof_probabilities"]
        )
        for fold_metrics in result["fold_metrics"]:
            assert 0 <= fold_metrics["train_average_precision"] <= 1
            assert 0 <= fold_metrics["validation_average_precision"] <= 1
            assert fold_metrics["overfitting_gap"] == pytest.approx(
                fold_metrics["train_average_precision"]
                - fold_metrics["validation_average_precision"]
            )
            assert fold_metrics["fit_time_seconds"] >= 0
            assert fold_metrics["prediction_time_seconds"] >= 0

    metadata = {
        "development_rows": len(frame),
        "negative_count": int((frame["Attrition"] == 0).sum()),
        "positive_count": int(frame["Attrition"].sum()),
        "positive_rate": float(frame["Attrition"].mean()),
        "development_sha256": "synthetic",
        "feature_set_version": definition.version,
        "feature_count": len(definition.all_features),
        "n_splits": 5,
        "shuffle": True,
        "random_seed": 42,
        "threshold": 0.5,
        "python_version": "test",
        "pandas_version": "test",
        "sklearn_version": "test",
        "xgboost_version": "test",
    }
    destination = tmp_path / "tree_report.md"
    write_tree_baseline_report(first, destination, metadata, candidates)
    report = destination.read_text(encoding="utf-8")

    for label in ("Dummy", "Logistic unweighted", "Logistic balanced", "Random Forest", "XGBoost"):
        assert label in report
    assert "Locked final test accessed: No" in report
    assert "Mean train PR-AUC" in report
    assert "Mean fit time" in report
