"""Leakage-safe nested cross-validation for the two approved tuning families."""

import json
from collections import Counter
from collections.abc import Callable, Mapping
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.base import BaseEstimator, clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import (
    GridSearchCV,
    ParameterGrid,
    RandomizedSearchCV,
    StratifiedKFold,
)
from sklearn.pipeline import Pipeline

from eris_ml.evaluation.metrics import calculate_metrics
from eris_ml.features.definitions import FeatureDefinition, validate_feature_columns
from eris_ml.models.factory import _load_xgb_classifier, create_estimator
from eris_ml.models.training import build_model_pipeline, normalize_binary_target

FAMILIES = ("logistic_regression", "xgboost")
RESULT_NAMES = (
    "logistic_baseline",
    "logistic_tuned",
    "xgboost_baseline",
    "xgboost_tuned",
)
LOGISTIC_SEARCH_KEYS = {"penalty", "C"}
XGBOOST_SEARCH_KEYS = {
    "n_estimators",
    "max_depth",
    "learning_rate",
    "min_child_weight",
    "subsample",
    "colsample_bytree",
    "gamma",
    "reg_alpha",
    "reg_lambda",
}
LOGISTIC_FIXED = {
    "solver": "liblinear",
    "class_weight": None,
    "max_iter": 3000,
    "random_state": 42,
}
XGBOOST_FIXED = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "tree_method": "hist",
    "scale_pos_weight": 1.0,
    "n_jobs": 1,
    "random_state": 42,
}


def load_yaml_config(path: Path) -> dict[str, Any]:
    """Read one explicitly supplied YAML configuration."""
    try:
        value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"Could not read tuning configuration: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Tuning configuration must be a mapping: {path}")
    return value


def _keys(value: Any, expected: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError(f"{label} must contain exactly {sorted(expected)}.")
    return value


def _cv_settings(value: Any, label: str) -> dict[str, Any]:
    config = _keys(value, {"type", "n_splits", "shuffle", "random_state"}, label)
    if config["type"] != "StratifiedKFold" or config["shuffle"] is not True:
        raise ValueError(f"{label} must use shuffled StratifiedKFold.")
    if type(config["n_splits"]) is not int or config["n_splits"] < 2:
        raise ValueError(f"{label}.n_splits must be an integer >= 2.")
    if type(config["random_state"]) is not int:
        raise ValueError(f"{label}.random_state must be an integer.")
    return config


def validate_tuning_protocol(config: dict[str, Any]) -> dict[str, Any]:
    """Validate the fixed no-leakage tuning protocol; accept smaller test folds."""
    _keys(
        config,
        {
            "protocol_version",
            "primary_metric",
            "threshold",
            "outer_cv",
            "inner_cv",
            "search",
            "execution",
        },
        "tuning protocol",
    )
    if config["protocol_version"] != "tuning-protocol-v1":
        raise ValueError("Unsupported tuning protocol version.")
    if config["primary_metric"] != "average_precision":
        raise ValueError("Tuning primary metric must be average_precision.")
    if config["threshold"] != 0.5:
        raise ValueError("Threshold is fixed at 0.5; threshold tuning is prohibited.")
    _cv_settings(config["outer_cv"], "outer_cv")
    _cv_settings(config["inner_cv"], "inner_cv")
    search = _keys(config["search"], {"error_score", "return_train_score"}, "search")
    if search != {"error_score": "raise", "return_train_score": True}:
        raise ValueError("Search must raise errors and return train scores.")
    execution = _keys(
        config["execution"],
        {"outer_parallelism", "search_n_jobs", "estimator_n_jobs"},
        "execution",
    )
    if execution["outer_parallelism"] != 1 or execution["estimator_n_jobs"] != 1:
        raise ValueError("Outer CV and search estimators must each use one worker.")
    if type(execution["search_n_jobs"]) is not int or execution["search_n_jobs"] == 0:
        raise ValueError("search_n_jobs must be a nonzero integer.")
    return config


def validate_search_config(config: dict[str, Any], family: str) -> dict[str, Any]:
    """Reject unapproved search dimensions, weighting, scoring, or fixed values."""
    if family not in FAMILIES:
        raise ValueError(f"Unsupported tuning family: {family}.")
    root_keys = {"model", "search_type", "scoring", "fixed_parameters", "search_space"}
    if family == "xgboost":
        root_keys |= {"n_iter", "random_state"}
    _keys(config, root_keys, f"{family} search config")
    if config["model"] != family or config["scoring"] != "average_precision":
        raise ValueError(f"{family} search must score average_precision.")
    expected_type = "grid" if family == "logistic_regression" else "randomized"
    if config["search_type"] != expected_type:
        raise ValueError(f"{family} must use {expected_type} search.")
    fixed = LOGISTIC_FIXED if family == "logistic_regression" else XGBOOST_FIXED
    if config["fixed_parameters"] != fixed:
        raise ValueError(f"{family} fixed parameters must match the approved unweighted values.")
    expected_space = (
        LOGISTIC_SEARCH_KEYS if family == "logistic_regression" else XGBOOST_SEARCH_KEYS
    )
    space = _keys(config["search_space"], expected_space, f"{family} search_space")
    for name, values in space.items():
        if not isinstance(values, list) or not values:
            raise ValueError(f"{family}.{name} must be a non-empty list.")
        if name == "penalty":
            if set(values) - {"l1", "l2"}:
                raise ValueError("Logistic penalty must be l1 or l2.")
        elif any(type(value) not in (int, float) for value in values):
            raise ValueError(f"{family}.{name} must contain numeric values.")
        elif name in {"subsample", "colsample_bytree"}:
            if any(not 0 < value <= 1 for value in values):
                raise ValueError(f"{family}.{name} must be in (0, 1].")
        elif name in {"gamma", "reg_alpha"}:
            if any(value < 0 for value in values):
                raise ValueError(f"{family}.{name} must be nonnegative.")
        elif any(value <= 0 for value in values):
            raise ValueError(f"{family}.{name} must be positive.")
        if name in {"n_estimators", "max_depth", "min_child_weight"}:
            if any(type(value) is not int for value in values):
                raise ValueError(f"{family}.{name} must contain integers.")
    if family == "xgboost":
        if type(config["n_iter"]) is not int or config["n_iter"] < 1:
            raise ValueError("XGBoost n_iter must be a positive integer.")
        if config["random_state"] != 42:
            raise ValueError("XGBoost search random_state must be 42.")
        if config["n_iter"] > len(ParameterGrid(space)):
            raise ValueError("XGBoost n_iter exceeds the number of parameter combinations.")
    return config


def _splitter(settings: Mapping[str, Any]) -> StratifiedKFold:
    return StratifiedKFold(
        n_splits=settings["n_splits"],
        shuffle=settings["shuffle"],
        random_state=settings["random_state"],
    )


def _tuning_estimator(config: dict[str, Any], family: str) -> BaseEstimator:
    fixed = config["fixed_parameters"]
    if family == "logistic_regression":
        return LogisticRegression(**fixed)
    classifier = _load_xgb_classifier()
    return classifier(**fixed)


def build_search(
    feature_definition: FeatureDefinition,
    search_config: dict[str, Any],
    protocol: dict[str, Any],
    *,
    full_development: bool = False,
) -> GridSearchCV | RandomizedSearchCV:
    """Build an unfitted search over a complete preprocessing/model pipeline."""
    validate_tuning_protocol(protocol)
    family = search_config.get("model")
    if not isinstance(family, str):
        raise ValueError("Search configuration must name a model family.")
    validate_search_config(search_config, family)
    pipeline = build_model_pipeline(feature_definition, _tuning_estimator(search_config, family))
    parameter_space = {
        f"model__{name}": values for name, values in search_config["search_space"].items()
    }
    cv_settings = protocol["outer_cv"] if full_development else protocol["inner_cv"]
    common: dict[str, Any] = {
        "estimator": pipeline,
        "scoring": "average_precision",
        "cv": _splitter(cv_settings),
        "n_jobs": protocol["execution"]["search_n_jobs"],
        "error_score": "raise",
        "return_train_score": True,
        "refit": True,
    }
    if family == "logistic_regression":
        return GridSearchCV(param_grid=parameter_space, **common)
    return RandomizedSearchCV(
        param_distributions=parameter_space,
        n_iter=search_config["n_iter"],
        random_state=search_config["random_state"],
        **common,
    )


def _positive_probabilities(pipeline: Pipeline, features: pd.DataFrame) -> np.ndarray:
    classes = np.asarray(pipeline.classes_)
    positive_index = np.flatnonzero(classes == 1)
    if positive_index.size != 1:
        raise ValueError("Fitted model has no positive-class probability column.")
    return np.asarray(pipeline.predict_proba(features)[:, positive_index[0]], dtype=float)


def _fold_record(
    pipeline: Pipeline,
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_validation: pd.DataFrame,
    y_validation: np.ndarray,
    fold: int,
    fit_time: float,
    search_time: float,
    refit_time: float,
    inner_best_score: float | None,
    best_parameters: dict[str, Any] | None,
    threshold: float,
) -> tuple[dict[str, Any], np.ndarray]:
    train_probabilities = _positive_probabilities(pipeline, x_train)
    started = perf_counter()
    validation_probabilities = _positive_probabilities(pipeline, x_validation)
    prediction_time = perf_counter() - started
    validation_metrics = calculate_metrics(y_validation, validation_probabilities, threshold)
    train_ap = float(average_precision_score(y_train, train_probabilities))
    validation_ap = validation_metrics["average_precision"]
    return (
        {
            "outer_fold": fold,
            "train_rows": len(x_train),
            "validation_rows": len(x_validation),
            "train_average_precision": train_ap,
            "validation_average_precision": validation_ap,
            "overfitting_gap": train_ap - validation_ap,
            "inner_best_average_precision": inner_best_score,
            "fit_time_seconds": fit_time,
            "search_time_seconds": search_time,
            "refit_time_seconds": refit_time,
            "prediction_time_seconds": prediction_time,
            "best_parameters": best_parameters,
            **validation_metrics,
        },
        validation_probabilities,
    )


def _search_rows(
    search: GridSearchCV | RandomizedSearchCV, fold: int | str
) -> list[dict[str, Any]]:
    results = search.cv_results_
    rows = []
    for index, params in enumerate(results["params"]):
        rows.append(
            {
                "outer_fold": fold,
                "parameters": json.dumps(
                    {key.removeprefix("model__"): value for key, value in params.items()},
                    sort_keys=True,
                ),
                "mean_inner_validation_pr_auc": float(results["mean_test_score"][index]),
                "std_inner_validation_pr_auc": float(results["std_test_score"][index]),
                "mean_train_pr_auc": float(results["mean_train_score"][index]),
                "rank": int(results["rank_test_score"][index]),
                "mean_fit_time_seconds": float(results["mean_fit_time"][index]),
            }
        )
    return rows


def run_nested_cv(
    features: pd.DataFrame,
    target: pd.Series,
    feature_definition: FeatureDefinition,
    baseline_configs: Mapping[str, dict[str, Any]],
    search_configs: Mapping[str, dict[str, Any]],
    protocol: dict[str, Any],
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Evaluate two fixed baselines and two tuned candidates on shared outer folds."""
    validate_tuning_protocol(protocol)
    if set(baseline_configs) != set(FAMILIES) or set(search_configs) != set(FAMILIES):
        raise ValueError("Nested tuning requires Logistic and XGBoost baseline/search configs.")
    for family in FAMILIES:
        validate_search_config(search_configs[family], family)
    validate_feature_columns(features.columns, feature_definition)
    normalized = normalize_binary_target(target)
    labels = normalized.to_numpy()
    if len(features) != len(labels):
        raise ValueError("Features and target must have the same number of rows.")
    outer = _splitter(protocol["outer_cv"])
    minimum = protocol["outer_cv"]["n_splits"] * protocol["inner_cv"]["n_splits"]
    if normalized.value_counts().min() < minimum:
        raise ValueError("Each class needs enough rows for outer and inner stratification.")
    threshold = float(protocol["threshold"])
    oof = {name: np.full(len(features), np.nan) for name in RESULT_NAMES}
    records: dict[str, list[dict[str, Any]]] = {name: [] for name in RESULT_NAMES}
    search_results: dict[str, list[dict[str, Any]]] = {family: [] for family in FAMILIES}
    assignments = np.zeros(len(features), dtype=int)

    for fold, (training_indices, validation_indices) in enumerate(
        outer.split(features, labels), start=1
    ):
        assignments[validation_indices] = fold
        x_train = features.iloc[training_indices]
        x_validation = features.iloc[validation_indices]
        y_train = labels[training_indices]
        y_validation = labels[validation_indices]
        for family in FAMILIES:
            if progress is not None:
                progress(f"Outer fold {fold}/{outer.n_splits}: {family} baseline")
            baseline = build_model_pipeline(
                feature_definition, create_estimator(baseline_configs[family])
            )
            started = perf_counter()
            baseline = clone(baseline).fit(x_train, y_train)
            fit_time = perf_counter() - started
            name = "logistic_baseline" if family == "logistic_regression" else "xgboost_baseline"
            record, probabilities = _fold_record(
                baseline, x_train, y_train, x_validation, y_validation, fold,
                fit_time, 0.0, fit_time, None, None, threshold,
            )
            oof[name][validation_indices] = probabilities
            records[name].append(record)
            del baseline

            if progress is not None:
                progress(f"Outer fold {fold}/{outer.n_splits}: {family} inner search")
            search = build_search(feature_definition, search_configs[family], protocol)
            started = perf_counter()
            search.fit(x_train, y_train)
            search_time = perf_counter() - started
            best_parameters = {
                key.removeprefix("model__"): value for key, value in search.best_params_.items()
            }
            name = "logistic_tuned" if family == "logistic_regression" else "xgboost_tuned"
            record, probabilities = _fold_record(
                search.best_estimator_, x_train, y_train, x_validation, y_validation,
                fold, 0.0, search_time, float(search.refit_time_),
                float(search.best_score_), best_parameters, threshold,
            )
            oof[name][validation_indices] = probabilities
            records[name].append(record)
            search_results[family].extend(_search_rows(search, fold))
            del search

    if (assignments == 0).any():
        raise RuntimeError("Every row must receive one outer-fold assignment.")
    for name, probabilities in oof.items():
        invalid_bounds = ((probabilities < 0) | (probabilities > 1)).any()
        if not np.isfinite(probabilities).all() or invalid_bounds:
            raise RuntimeError(f"Incomplete or invalid nested OOF probabilities: {name}.")
    summary = {
        name: {
            key: {
                "mean": float(np.mean([row[key] for row in rows])),
                "standard_deviation": float(np.std([row[key] for row in rows], ddof=1)),
            }
            for key in (
                "train_average_precision", "validation_average_precision", "overfitting_gap",
                "fit_time_seconds", "search_time_seconds", "refit_time_seconds",
                "prediction_time_seconds",
            )
        }
        for name, rows in records.items()
    }
    return {
        "oof_probabilities": oof,
        "outer_fold": assignments,
        "oof_metrics": {
            name: calculate_metrics(labels, probabilities, threshold)
            for name, probabilities in oof.items()
        },
        "fold_records": records,
        "fold_summary": summary,
        "search_results": search_results,
    }


def run_full_development_search(
    features: pd.DataFrame,
    target: pd.Series,
    feature_definition: FeatureDefinition,
    search_configs: Mapping[str, dict[str, Any]],
    protocol: dict[str, Any],
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Search all development rows only after nested evaluation has finished."""
    normalized = normalize_binary_target(target)
    outputs: dict[str, Any] = {}
    for family in FAMILIES:
        if progress is not None:
            progress(f"Full-development search: {family}")
        search = build_search(
            feature_definition, search_configs[family], protocol, full_development=True
        )
        started = perf_counter()
        search.fit(features, normalized)
        outputs[family] = {
            "best_parameters": {
                key.removeprefix("model__"): value
                for key, value in search.best_params_.items()
            },
            "best_cv_score": float(search.best_score_),
            "search_time_seconds": perf_counter() - started,
            "search_results": _search_rows(search, "full"),
        }
        del search
    return outputs


def candidate_configuration(
    family: str, search_config: dict[str, Any], best_parameters: dict[str, Any]
) -> dict[str, Any]:
    """Produce an unfitted next-step candidate YAML payload."""
    if family not in FAMILIES:
        raise ValueError(f"Unsupported tuning family: {family}.")
    validate_search_config(search_config, family)
    if set(best_parameters) != set(search_config["search_space"]):
        raise ValueError("Best parameters must contain each approved search dimension.")
    parameters = {**search_config["fixed_parameters"], **best_parameters}
    random_seed = parameters.pop("random_state")
    return {
        "model": family,
        "role": "tuned_candidate",
        "candidate_name": f"{family}_tuned_v1",
        "random_seed": random_seed,
        "parameters": parameters,
    }


def parameter_frequencies(records: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    """Count selected values for each search parameter across outer folds."""
    values: dict[str, Counter[str]] = {}
    for row in records:
        for key, value in (row["best_parameters"] or {}).items():
            values.setdefault(key, Counter())[str(value)] += 1
    return {key: dict(counter) for key, counter in values.items()}
