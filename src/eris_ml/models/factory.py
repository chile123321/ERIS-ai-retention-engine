"""Estimator factory interfaces."""

from typing import Any

from sklearn.base import BaseEstimator
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


class EstimatorConfigError(ValueError):
    """Raised when a model configuration is unsupported or unsafe."""


def _parameters(config: dict[str, Any], allowed: set[str]) -> dict[str, Any]:
    parameters = config.get("parameters", {})
    if not isinstance(parameters, dict):
        raise EstimatorConfigError("'parameters' must be a mapping.")
    unexpected = sorted(set(parameters) - allowed)
    if unexpected:
        raise EstimatorConfigError(f"Unapproved estimator parameters: {unexpected}.")
    return parameters


def _require_seed(config: dict[str, Any]) -> None:
    if config.get("random_seed", 42) != 42:
        raise EstimatorConfigError("Approved model random_seed must be 42.")


def _fixed_parameters(
    config: dict[str, Any], approved: dict[str, Any], model_name: str
) -> dict[str, Any]:
    parameters = _parameters(config, set(approved))
    resolved = {**approved, **parameters}
    if resolved != approved:
        raise EstimatorConfigError(
            f"{model_name} parameters must match the approved fixed baseline."
        )
    return resolved


def _load_xgb_classifier() -> type[BaseEstimator]:
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise EstimatorConfigError(
            "XGBoost is unavailable; install the declared 'explain' optional dependency."
        ) from exc
    return XGBClassifier


def create_estimator(config: dict[str, Any]) -> BaseEstimator:
    """Create an unfitted estimator from the approved baseline configuration."""
    if not isinstance(config, dict):
        raise EstimatorConfigError("Estimator configuration must be a mapping.")

    model = config.get("model")
    if model == "dummy":
        parameters = _parameters(config, {"strategy"})
        strategy = parameters.get("strategy", "prior")
        if strategy != "prior":
            raise EstimatorConfigError("Dummy baseline only permits strategy='prior'.")
        return DummyClassifier(strategy="prior")

    if model == "logistic_regression":
        parameters = _parameters(config, {"solver", "C", "max_iter", "class_weight"})
        defaults: dict[str, Any] = {
            "solver": "liblinear",
            "C": 1.0,
            "max_iter": 2000,
            "class_weight": None,
        }
        resolved = {**defaults, **parameters}
        if (
            resolved["solver"] != "liblinear"
            or resolved["C"] != 1.0
            or resolved["max_iter"] != 2000
            or resolved["class_weight"] not in (None, "balanced")
        ):
            raise EstimatorConfigError(
                "Logistic Regression parameters must match an approved class-weight candidate."
            )
        _require_seed(config)
        return LogisticRegression(random_state=42, **resolved)

    if model == "random_forest":
        random_forest_approved: dict[str, Any] = {
            "n_estimators": 500,
            "max_depth": None,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "max_features": "sqrt",
            "class_weight": None,
            "n_jobs": -1,
        }
        resolved = _fixed_parameters(config, random_forest_approved, "Random Forest")
        _require_seed(config)
        return RandomForestClassifier(random_state=42, **resolved)

    if model == "xgboost":
        xgboost_approved: dict[str, Any] = {
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
        }
        resolved = _fixed_parameters(config, xgboost_approved, "XGBoost")
        _require_seed(config)
        classifier = _load_xgb_classifier()
        return classifier(random_state=42, **resolved)

    raise EstimatorConfigError(f"Unsupported model type: {model!r}.")
