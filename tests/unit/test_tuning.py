"""Unit tests for strict development-only tuning configuration."""

from copy import deepcopy
from pathlib import Path

import pytest
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from eris_ml.data.loaders import load_development_data
from eris_ml.features.definitions import load_feature_definition
from eris_ml.models.factory import EstimatorConfigError
from eris_ml.models.tuning import (
    build_search,
    load_yaml_config,
    validate_search_config,
    validate_tuning_protocol,
)

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = ROOT / "configs/tuning/tuning_protocol_v1.yaml"
LOGISTIC_PATH = ROOT / "configs/tuning/logistic_regression_search_v1.yaml"
XGBOOST_PATH = ROOT / "configs/tuning/xgboost_search_v1.yaml"
FEATURE_PATH = ROOT / "configs/features/feature_set_v1_full.yaml"


def test_production_protocol_and_search_spaces_are_approved() -> None:
    protocol = validate_tuning_protocol(load_yaml_config(PROTOCOL_PATH))
    logistic = validate_search_config(
        load_yaml_config(LOGISTIC_PATH), "logistic_regression"
    )
    xgboost = validate_search_config(load_yaml_config(XGBOOST_PATH), "xgboost")

    assert protocol["outer_cv"]["n_splits"] == 5
    assert protocol["inner_cv"]["n_splits"] == 4
    assert protocol["outer_cv"]["random_state"] == 42
    assert protocol["inner_cv"]["random_state"] == 43
    assert len(logistic["search_space"]["C"]) * len(logistic["search_space"]["penalty"]) == 20
    assert xgboost["n_iter"] == 40
    assert xgboost["fixed_parameters"]["scale_pos_weight"] == 1.0


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        (None, "primary_metric", "accuracy"),
        (None, "threshold", 0.4),
        ("outer_cv", "type", "KFold"),
        ("inner_cv", "n_splits", 1),
    ],
)
def test_invalid_protocol_is_rejected(section, key, value) -> None:
    config = load_yaml_config(PROTOCOL_PATH)
    target = config if section is None else config[section]
    target[key] = value

    with pytest.raises(ValueError):
        validate_tuning_protocol(config)


def test_search_rejects_weighting_threshold_and_unknown_parameters() -> None:
    logistic = load_yaml_config(LOGISTIC_PATH)
    logistic["fixed_parameters"]["class_weight"] = "balanced"
    with pytest.raises(ValueError, match="unweighted"):
        validate_search_config(logistic, "logistic_regression")

    xgboost = load_yaml_config(XGBOOST_PATH)
    xgboost["search_space"]["scale_pos_weight"] = [1, 2]
    with pytest.raises(ValueError, match="exactly"):
        validate_search_config(xgboost, "xgboost")
    xgboost = load_yaml_config(XGBOOST_PATH)
    xgboost["search_space"]["threshold"] = [0.5]
    with pytest.raises(ValueError, match="exactly"):
        validate_search_config(xgboost, "xgboost")


def test_search_builders_prefix_model_parameters_and_set_search_methods() -> None:
    definition = load_feature_definition(FEATURE_PATH)
    protocol = validate_tuning_protocol(load_yaml_config(PROTOCOL_PATH))
    logistic = build_search(definition, load_yaml_config(LOGISTIC_PATH), protocol)
    xgboost = build_search(definition, load_yaml_config(XGBOOST_PATH), protocol)

    assert isinstance(logistic, GridSearchCV)
    assert set(logistic.param_grid) == {"model__C", "model__penalty"}
    assert logistic.scoring == "average_precision"
    assert logistic.cv.n_splits == 4
    assert logistic.estimator.named_steps["preprocessor"] is not None
    assert isinstance(xgboost, RandomizedSearchCV)
    assert xgboost.n_iter == 40
    assert xgboost.random_state == 42
    assert xgboost.estimator.named_steps["model"].n_jobs == 1


def test_xgboost_missing_dependency_has_clear_error(monkeypatch) -> None:
    import eris_ml.models.tuning as tuning

    def missing_dependency():
        raise EstimatorConfigError("XGBoost unavailable; install explain extra")

    monkeypatch.setattr(tuning, "_load_xgb_classifier", missing_dependency)
    protocol = load_yaml_config(PROTOCOL_PATH)
    config = load_yaml_config(XGBOOST_PATH)
    with pytest.raises(EstimatorConfigError, match="install explain"):
        build_search(load_feature_definition(FEATURE_PATH), config, protocol)


def test_final_test_path_is_rejected_before_opening() -> None:
    with pytest.raises(ValueError, match="prohibited"):
        load_development_data(Path("data/processed/final_test_raw_v1.csv"))


def test_search_config_does_not_mutate_input() -> None:
    config = load_yaml_config(LOGISTIC_PATH)
    original = deepcopy(config)
    validate_search_config(config, "logistic_regression")
    assert config == original
