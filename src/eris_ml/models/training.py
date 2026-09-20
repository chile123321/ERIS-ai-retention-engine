"""Model-training orchestration interfaces."""

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.pipeline import Pipeline

from eris_ml.evaluation.cross_validation import run_cross_validation
from eris_ml.features.definitions import FeatureDefinition, validate_feature_columns
from eris_ml.features.preprocessing import build_preprocessor
from eris_ml.models.factory import create_estimator


def normalize_binary_target(target: pd.Series) -> pd.Series:
    """Normalize the approved Attrition encodings to integer zero/one values."""
    mapped = target.replace({"No": 0, "Yes": 1})
    numeric = pd.to_numeric(mapped, errors="coerce")
    if numeric.isna().any() or set(numeric.unique().tolist()) != {0, 1}:
        raise ValueError("Attrition must contain both classes encoded as No/Yes or 0/1.")
    return numeric.astype(int)


def build_model_pipeline(
    feature_definition: FeatureDefinition, estimator: BaseEstimator
) -> Pipeline:
    """Build an unfitted, leakage-safe preprocessing and classifier pipeline."""
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(feature_definition)),
            ("model", estimator),
        ]
    )


def train_model(pipeline: Pipeline, features: pd.DataFrame, target: pd.Series) -> Pipeline:
    """Fit and return a cloned pipeline without mutating the supplied estimator."""
    fitted = clone(pipeline)
    fitted.fit(features, target)
    return fitted


def evaluate_candidates(
    candidate_configs: Mapping[str, dict[str, Any]],
    feature_definition: FeatureDefinition,
    features: pd.DataFrame,
    target: pd.Series,
    cv_config: Mapping[str, Any],
    threshold: float,
) -> dict[str, dict[str, Any]]:
    """Evaluate configured candidates on identical deterministic OOF folds."""
    if not candidate_configs:
        raise ValueError("At least one candidate configuration is required.")
    validate_feature_columns(features.columns, feature_definition)
    normalized_target = normalize_binary_target(target)

    results: dict[str, dict[str, Any]] = {}
    reference_assignments: np.ndarray | None = None
    for candidate_name, candidate_config in candidate_configs.items():
        pipeline = build_model_pipeline(
            feature_definition, create_estimator(candidate_config)
        )
        result = run_cross_validation(
            pipeline,
            features,
            normalized_target,
            cv_config,
            threshold=threshold,
        )
        assignments = np.asarray(result["fold_assignments"])
        if reference_assignments is None:
            reference_assignments = assignments
        elif not np.array_equal(reference_assignments, assignments):
            raise RuntimeError("Candidate models did not use identical fold assignments.")
        results[candidate_name] = result
    return results


def evaluate_baselines(
    development_data: pd.DataFrame,
    feature_definition: FeatureDefinition,
    logistic_config: dict[str, Any],
    evaluation_config: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    """Generate OOF results for Dummy and Logistic Regression baselines."""
    required = set(feature_definition.all_features) | {"Attrition"}
    missing = sorted(required - set(development_data.columns))
    if missing:
        raise ValueError(f"Development data is missing required columns: {missing}.")

    features = development_data.loc[:, feature_definition.all_features].copy()
    validate_feature_columns(features.columns, feature_definition)
    cv_config = evaluation_config.get("cross_validation")
    if not isinstance(cv_config, Mapping):
        raise ValueError("Evaluation configuration must define cross_validation.")
    threshold = evaluation_config.get("classification_threshold", 0.5)
    if not isinstance(threshold, (int, float)) or not np.isfinite(threshold):
        raise ValueError("classification_threshold must be a finite number.")

    candidate_configs = {
        "dummy_prior": {"model": "dummy", "parameters": {"strategy": "prior"}},
        "logistic_regression": logistic_config,
    }
    candidate_results = evaluate_candidates(
        candidate_configs,
        feature_definition,
        features,
        development_data["Attrition"],
        cv_config,
        float(threshold),
    )
    return {
        "dummy": candidate_results["dummy_prior"],
        "logistic_regression": candidate_results["logistic_regression"],
    }
