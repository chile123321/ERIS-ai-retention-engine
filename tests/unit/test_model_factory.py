"""Unit tests for approved estimator configurations."""

import builtins

import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from eris_ml.models.factory import EstimatorConfigError, create_estimator


def _logistic_config(class_weight):
    return {
        "model": "logistic_regression",
        "random_seed": 42,
        "parameters": {
            "solver": "liblinear",
            "C": 1.0,
            "max_iter": 2000,
            "class_weight": class_weight,
        },
    }


@pytest.mark.parametrize("class_weight", [None, "balanced"])
def test_factory_accepts_approved_class_weights(class_weight) -> None:
    estimator = create_estimator(_logistic_config(class_weight))

    assert isinstance(estimator, LogisticRegression)
    assert estimator.class_weight == class_weight
    assert estimator.random_state == 42


def test_factory_rejects_unapproved_class_weight() -> None:
    with pytest.raises(EstimatorConfigError, match="approved class-weight"):
        create_estimator(_logistic_config("invalid"))


RANDOM_FOREST_CONFIG = {
    "model": "random_forest",
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


def test_factory_creates_fixed_random_forest() -> None:
    estimator = create_estimator(RANDOM_FOREST_CONFIG)

    assert isinstance(estimator, RandomForestClassifier)
    assert estimator.random_state == 42
    assert estimator.n_estimators == 500
    assert estimator.class_weight is None


def test_factory_creates_fixed_xgboost() -> None:
    estimator = create_estimator(XGBOOST_CONFIG)

    assert isinstance(estimator, XGBClassifier)
    assert estimator.random_state == 42
    assert estimator.scale_pos_weight == 1.0
    assert estimator.eval_metric == "logloss"


@pytest.mark.parametrize("config", [RANDOM_FOREST_CONFIG, XGBOOST_CONFIG])
def test_tree_factory_rejects_unapproved_parameter(config) -> None:
    invalid = {**config, "parameters": {**config["parameters"], "unexpected": 1}}

    with pytest.raises(EstimatorConfigError, match="Unapproved estimator parameters"):
        create_estimator(invalid)


def test_factory_rejects_unknown_model() -> None:
    with pytest.raises(EstimatorConfigError, match="Unsupported model type"):
        create_estimator({"model": "not_a_model"})


def test_xgboost_dependency_error_is_actionable(monkeypatch) -> None:
    original_import = builtins.__import__

    def fail_xgboost_import(name, *args, **kwargs):
        if name == "xgboost":
            raise ImportError("simulated missing dependency")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fail_xgboost_import)
    with pytest.raises(EstimatorConfigError, match="install.*explain"):
        create_estimator(XGBOOST_CONFIG)
