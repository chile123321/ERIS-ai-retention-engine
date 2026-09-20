"""Development-only outer cross-fitted probability calibration."""

import json
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold

from eris_ml.features.definitions import FeatureDefinition, validate_feature_columns
from eris_ml.models.factory import _load_xgb_classifier, create_estimator
from eris_ml.models.training import build_model_pipeline, normalize_binary_target
from eris_ml.models.tuning import XGBOOST_FIXED, XGBOOST_SEARCH_KEYS

METHODS = ("raw", "sigmoid", "isotonic")
MODELS = ("logistic", "xgboost")
OOF_COLUMNS = tuple(f"{model}_{method}_probability" for model in MODELS for method in METHODS)
PRIOR_COLUMNS = (
    "source_row", "EmployeeNumber", "Attrition", "outer_fold",
    "logistic_baseline_probability", "xgboost_tuned_probability",
)


def _mapping(value: Any, keys: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError(f"{label} must contain exactly {sorted(keys)}.")
    return value


def _cv(value: Any, label: str, seed: int) -> dict[str, Any]:
    config = _mapping(value, {"type", "n_splits", "shuffle", "random_state"}, label)
    if (
        config["type"] != "StratifiedKFold"
        or type(config["n_splits"]) is not int
        or config["n_splits"] < 2
        or config["shuffle"] is not True
        or config["random_state"] != seed
    ):
        raise ValueError(f"{label} must be shuffled StratifiedKFold with seed {seed}.")
    return config


def validate_calibration_protocol(value: dict[str, Any]) -> dict[str, Any]:
    """Enforce the approved protocol; small synthetic tests may use fewer folds."""
    config = _mapping(
        value,
        {
            "protocol_version", "outer_cv", "calibration_cv", "methods",
            "primary_metric", "secondary_metrics", "calibration_bins",
            "operational_threshold", "execution",
        },
        "calibration protocol",
    )
    if config["protocol_version"] != "calibration-protocol-v1":
        raise ValueError("Unsupported calibration protocol version.")
    _cv(config["outer_cv"], "outer_cv", 42)
    _cv(config["calibration_cv"], "calibration_cv", 43)
    if config["methods"] != list(METHODS):
        raise ValueError("Calibration methods must be raw, sigmoid, isotonic exactly once.")
    if config["primary_metric"] != "brier_score":
        raise ValueError("Primary metric must be brier_score.")
    if config["secondary_metrics"] != [
        "log_loss", "expected_calibration_error", "pr_auc", "roc_auc"
    ]:
        raise ValueError("Secondary calibration metrics are not approved.")
    if config["calibration_bins"] != {"count": 10, "strategy": "quantile"}:
        raise ValueError("Calibration must use 10 quantile bins.")
    if type(config["operational_threshold"]) not in (int, float) or (
        config["operational_threshold"] != 0.5
    ):
        raise ValueError("Operational threshold must remain 0.5.")
    if config["execution"] != {"outer_parallelism": 1, "estimator_n_jobs": 1}:
        raise ValueError("Calibration must use one outer and estimator worker.")
    return config


def validate_xgboost_parameters(parameters: Mapping[str, Any]) -> dict[str, Any]:
    """Validate one fold's search result against the approved XGBoost dimensions."""
    if not isinstance(parameters, dict) or set(parameters) != XGBOOST_SEARCH_KEYS:
        raise ValueError("XGBoost fold parameters do not match the approved allowlist.")
    checked = dict(parameters)
    for key in ("n_estimators", "max_depth", "min_child_weight"):
        if type(checked[key]) is not int or checked[key] < 1:
            raise ValueError(f"XGBoost {key} must be a positive integer.")
    if checked["max_depth"] != 1:
        raise ValueError("XGBoost selected max_depth must be 1.")
    for key in XGBOOST_SEARCH_KEYS - {"n_estimators", "max_depth", "min_child_weight"}:
        number = checked[key]
        if type(number) not in (int, float) or not np.isfinite(number):
            raise ValueError(f"XGBoost {key} must be finite and numeric.")
        if key in {"subsample", "colsample_bytree"} and not 0 < number <= 1:
            raise ValueError(f"XGBoost {key} must be in (0, 1].")
        if key in {"gamma", "reg_alpha"} and number < 0:
            raise ValueError(f"XGBoost {key} must be nonnegative.")
        if key not in {"gamma", "reg_alpha"} and number <= 0:
            raise ValueError(f"XGBoost {key} must be positive.")
    return checked


def load_outer_best_parameters(
    search_results: pd.DataFrame, folds: int
) -> dict[int, dict[str, Any]]:
    """Read each unique rank-one outer XGBoost result; ignore full-development search."""
    if any("final_test" in str(column).lower() for column in search_results.columns):
        raise ValueError("XGBoost search artifact must not reference final test.")
    if search_results.astype(str).apply(
        lambda column: column.str.contains("final_test|locked_test|test_locked", case=False)
    ).to_numpy().any():
        raise ValueError("XGBoost search artifact must not reference final test.")
    required = {"outer_fold", "parameters", "rank"}
    if not required.issubset(search_results.columns):
        raise ValueError("XGBoost search artifact lacks outer_fold, parameters, or rank.")
    if "model" in search_results and not search_results["model"].eq("xgboost").all():
        raise ValueError("Search artifact contains a non-XGBoost model.")
    if search_results.empty:
        raise ValueError("XGBoost search artifact is empty.")
    outer_ids = search_results["outer_fold"].astype(str)
    expected = {str(fold) for fold in range(1, folds + 1)}
    if set(outer_ids) != expected | {"full"} and set(outer_ids) != expected:
        raise ValueError("XGBoost search artifact has missing or unexpected fold IDs.")
    results: dict[int, dict[str, Any]] = {}
    for fold in range(1, folds + 1):
        rows = search_results.loc[(outer_ids == str(fold)) & (search_results["rank"] == 1)]
        if len(rows) != 1:
            raise ValueError(f"XGBoost outer fold {fold} needs exactly one rank-one record.")
        try:
            parsed = json.loads(rows.iloc[0]["parameters"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid XGBoost parameters in outer fold {fold}.") from exc
        results[fold] = validate_xgboost_parameters(parsed)
    return results


def validate_prior_oof(
    frame: pd.DataFrame, target: pd.Series, previous: pd.DataFrame,
    protocol: dict[str, Any],
) -> np.ndarray:
    """Verify Step 10A row identity and exact deterministic outer-fold assignments."""
    missing = set(PRIOR_COLUMNS) - set(previous.columns)
    if missing or len(previous) != len(frame):
        raise ValueError(f"Prior OOF shape/columns mismatch; missing={sorted(missing)}.")
    if not previous["source_row"].is_unique or not previous["EmployeeNumber"].is_unique:
        raise ValueError("Prior OOF identifiers must be unique.")
    for column in ("source_row", "EmployeeNumber"):
        if not np.array_equal(frame[column].to_numpy(), previous[column].to_numpy()):
            raise ValueError(f"Prior OOF {column} row alignment failed.")
    labels = normalize_binary_target(target).to_numpy()
    if not np.array_equal(labels, previous["Attrition"].to_numpy()):
        raise ValueError("Prior OOF target alignment failed.")
    settings = protocol["outer_cv"]
    splitter = StratifiedKFold(
        n_splits=settings["n_splits"], shuffle=True, random_state=settings["random_state"]
    )
    assignments = np.zeros(len(frame), dtype=int)
    for fold, (_, validation) in enumerate(splitter.split(frame, labels), start=1):
        assignments[validation] = fold
    if not np.array_equal(assignments, previous["outer_fold"].to_numpy()):
        raise ValueError("Prior OOF outer-fold assignment differs from Step 10A protocol.")
    for column in ("logistic_baseline_probability", "xgboost_tuned_probability"):
        scores = previous[column].to_numpy(dtype=float)
        if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
            raise ValueError(f"Prior OOF {column} has invalid probabilities.")
    return assignments


def _positive_probability(estimator: BaseEstimator, features: pd.DataFrame) -> np.ndarray:
    classes = np.asarray(estimator.classes_)
    positive = np.flatnonzero(classes == 1)
    if len(positive) != 1:
        raise ValueError("Estimator has no positive-class probability column.")
    return np.asarray(estimator.predict_proba(features)[:, positive[0]], dtype=float)


def _raw_regression(previous: np.ndarray, current: np.ndarray, model: str, fold: int) -> None:
    if not np.allclose(previous, current, rtol=1e-7, atol=1e-9):
        difference = float(np.max(np.abs(previous - current)))
        raise RuntimeError(
            f"Step 10A raw regression check failed for {model}, outer fold {fold}; "
            f"max absolute difference={difference:.12g}."
        )


def run_cross_fitted_calibration(
    frame: pd.DataFrame,
    feature_definition: FeatureDefinition,
    logistic_config: dict[str, Any],
    xgboost_config: dict[str, Any],
    fold_parameters: Mapping[int, dict[str, Any]],
    previous_oof: pd.DataFrame,
    protocol: dict[str, Any],
    progress: Callable[[str], None] | None = None,
) -> pd.DataFrame:
    """Fit each raw/calibrated model only on outer-training, with Step 10A checks."""
    validate_calibration_protocol(protocol)
    if xgboost_config.get("model") != "xgboost" or (
        xgboost_config.get("parameters", {}).get("scale_pos_weight") != 1.0
    ):
        raise ValueError("XGBoost config must be the unweighted tuned candidate.")
    fixed = {
        key: xgboost_config["parameters"].get(key)
        for key in XGBOOST_FIXED if key != "random_state"
    }
    if fixed != {key: value for key, value in XGBOOST_FIXED.items() if key != "random_state"}:
        raise ValueError("XGBoost fixed parameters differ from Step 10A.")
    if xgboost_config.get("random_seed") != XGBOOST_FIXED["random_state"]:
        raise ValueError("XGBoost random seed differs from Step 10A.")
    if logistic_config.get("model") != "logistic_regression" or (
        logistic_config.get("parameters", {}).get("class_weight") is not None
    ):
        raise ValueError("Logistic baseline must have class_weight=null.")
    logistic_estimator = create_estimator(logistic_config)
    if set(fold_parameters) != set(range(1, protocol["outer_cv"]["n_splits"] + 1)):
        raise ValueError("XGBoost best parameters must cover every outer fold exactly.")
    for parameters in fold_parameters.values():
        validate_xgboost_parameters(parameters)
    required = set(feature_definition.all_features) | {
        "source_row", "EmployeeNumber", "Attrition"
    }
    if set(frame.columns) != required:
        raise ValueError("Development frame columns differ from the approved feature set.")
    features = frame.loc[:, feature_definition.all_features].copy()
    validate_feature_columns(features.columns, feature_definition)
    target = normalize_binary_target(frame["Attrition"])
    labels = target.to_numpy()
    assignments = validate_prior_oof(frame, target, previous_oof, protocol)
    minimum = protocol["outer_cv"]["n_splits"] * protocol["calibration_cv"]["n_splits"]
    if target.value_counts().min() < minimum:
        raise ValueError("Insufficient positives for outer and calibration folds.")
    predictions = {column: np.full(len(frame), np.nan) for column in OOF_COLUMNS}
    splitter = StratifiedKFold(
        n_splits=protocol["outer_cv"]["n_splits"], shuffle=True,
        random_state=protocol["outer_cv"]["random_state"],
    )
    classifier = _load_xgb_classifier()
    for fold, (training, validation) in enumerate(splitter.split(features, labels), start=1):
        if not np.all(assignments[validation] == fold):
            raise RuntimeError("Unexpected outer-fold assignment.")
        x_train, x_validation = features.iloc[training], features.iloc[validation]
        y_train = labels[training]
        estimators = {
            "logistic": logistic_estimator,
            "xgboost": classifier(**XGBOOST_FIXED, **fold_parameters[fold]),
        }
        for model, estimator in estimators.items():
            if progress is not None:
                progress(f"Outer fold {fold}/{splitter.n_splits}: {model} raw/sigmoid/isotonic")
            base = build_model_pipeline(feature_definition, estimator)
            raw = _positive_probability(clone(base).fit(x_train, y_train), x_validation)
            reference_column = (
                "logistic_baseline_probability" if model == "logistic"
                else "xgboost_tuned_probability"
            )
            _raw_regression(
                previous_oof[reference_column].to_numpy(dtype=float)[validation],
                raw, model, fold,
            )
            predictions[f"{model}_raw_probability"][validation] = raw
            for method in ("sigmoid", "isotonic"):
                calibration_cv = StratifiedKFold(
                    n_splits=protocol["calibration_cv"]["n_splits"], shuffle=True,
                    random_state=protocol["calibration_cv"]["random_state"],
                )
                calibrated = CalibratedClassifierCV(
                    estimator=clone(base), method=method, cv=calibration_cv,
                    ensemble=True, n_jobs=protocol["execution"]["estimator_n_jobs"],
                )
                calibrated.fit(x_train, y_train)
                predictions[f"{model}_{method}_probability"][validation] = (
                    _positive_probability(calibrated, x_validation)
                )
    output = pd.DataFrame(
        {
            "source_row": frame["source_row"].to_numpy(),
            "EmployeeNumber": frame["EmployeeNumber"].to_numpy(),
            "Attrition": labels,
            "outer_fold": assignments,
            **predictions,
        }
    )
    for column in OOF_COLUMNS:
        scores = output[column].to_numpy(dtype=float)
        if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
            raise RuntimeError(f"Incomplete or invalid calibrated OOF probabilities: {column}.")
    return output
