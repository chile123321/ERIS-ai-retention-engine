"""Leakage-safe, business-scenario threshold selection on development data only."""

from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import StratifiedKFold

from eris_ml.features.definitions import FeatureDefinition, validate_feature_columns
from eris_ml.models.calibration import validate_xgboost_parameters
from eris_ml.models.factory import _load_xgb_classifier, create_estimator
from eris_ml.models.training import build_model_pipeline, normalize_binary_target
from eris_ml.models.tuning import XGBOOST_FIXED

MODELS = ("xgboost", "logistic")
OUTER_SEED = 42
INNER_SEED = 43
CAPACITIES = (0.10, 0.15, 0.20, 0.25)
MINIMUM_RECALLS = (0.60, 0.70)
COST_RATIOS = (2.0, 5.0, 10.0)


def validate_policy(policy: dict[str, Any]) -> dict[str, Any]:
    """Reject unapproved scenarios, costs, or implied business approval."""
    required = {
        "policy_version", "model", "business_approved", "status", "selection_data",
        "capacity_scenarios", "recall_scenarios", "cost_sensitivity",
        "default_threshold_reference",
    }
    if not isinstance(policy, dict) or set(policy) != required:
        raise ValueError("Threshold policy root keys do not match v1.")
    if (
        policy["policy_version"] != "threshold-policy-v1"
        or policy["model"] != "xgboost_tuned_raw"
        or policy["business_approved"] is not False
        or policy["status"] != "development_scenario_analysis"
    ):
        raise ValueError("Threshold policy must remain an unapproved development analysis.")
    if (
        policy["selection_data"] != {
            "type": "cross_fitted_development", "final_test_used": False
        }
        or policy["selection_data"].get("final_test_used") is not False
    ):
        raise ValueError("Threshold selection must be cross-fitted development only.")
    capacities = policy["capacity_scenarios"]
    recalls = policy["recall_scenarios"]
    costs = policy["cost_sensitivity"]
    if not isinstance(capacities, list) or len(capacities) != len(CAPACITIES):
        raise ValueError("Exactly four capacity scenarios are required.")
    if not isinstance(recalls, list) or len(recalls) != len(MINIMUM_RECALLS):
        raise ValueError("Exactly two recall scenarios are required.")
    for fraction, scenario in zip(CAPACITIES, capacities, strict=True):
        if not isinstance(scenario, dict) or scenario != {
            "name": f"capacity_{round(fraction * 100)}_percent",
            "max_alert_rate": fraction,
            "objective": "maximize_recall",
            "tie_breaker": "maximize_precision",
        }:
            raise ValueError("Capacity scenario differs from the approved sensitivity grid.")
    for fraction, scenario in zip(MINIMUM_RECALLS, recalls, strict=True):
        if not isinstance(scenario, dict) or scenario != {
            "name": f"minimum_recall_{round(fraction * 100)}_percent",
            "min_recall": fraction,
            "objective": "minimize_alert_rate",
            "tie_breaker": "maximize_precision",
        }:
            raise ValueError("Recall scenario differs from the approved sensitivity grid.")
    if costs != {
        "false_positive_cost": 1.0,
        "false_negative_cost_ratios": list(COST_RATIOS),
    }:
        raise ValueError("Cost scenarios must be FP=1 and FN=2, 5, 10.")
    if type(policy["default_threshold_reference"]) not in (int, float) or (
        policy["default_threshold_reference"] != 0.5
    ):
        raise ValueError("Default threshold reference must be 0.5, not a selection rule.")
    return policy


def scenarios(policy: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the nine named development-only scenario specifications."""
    validate_policy(policy)
    values = [
        {"name": row["name"], "kind": "capacity", "value": row["max_alert_rate"]}
        for row in policy["capacity_scenarios"]
    ]
    values += [
        {"name": row["name"], "kind": "recall", "value": row["min_recall"]}
        for row in policy["recall_scenarios"]
    ]
    values += [
        {"name": f"cost_fn_{int(ratio)}x", "kind": "cost", "value": ratio}
        for ratio in COST_RATIOS
    ]
    return values


def validate_probability_vector(target: Any, probabilities: Any) -> tuple[np.ndarray, np.ndarray]:
    """Accept binary labels and finite raw probabilities, including 0 and 1."""
    labels = np.asarray(target)
    scores = np.asarray(probabilities, dtype=float)
    if labels.ndim != 1 or scores.ndim != 1 or not len(labels) or len(labels) != len(scores):
        raise ValueError("Target and probabilities must be aligned nonempty vectors.")
    if set(np.unique(labels).tolist()) - {0, 1}:
        raise ValueError("Target must contain only 0 and 1.")
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("Raw probabilities must be finite and within [0, 1].")
    return labels.astype(int), scores


def _safe_ratio(numerator: int | float, denominator: int | float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def confusion_metrics(tn: int, fp: int, fn: int, tp: int) -> dict[str, Any]:
    """Operational metrics with safe zero-denominator behavior."""
    total = tn + fp + fn + tp
    if min(tn, fp, fn, tp) < 0 or total == 0:
        raise ValueError("Confusion counts must be nonnegative and nonempty.")
    precision = _safe_ratio(tp, tp + fp)
    recall = _safe_ratio(tp, tp + fn)
    specificity = _safe_ratio(tn, tn + fp)
    return {
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "accuracy": _safe_ratio(tn + tp, total),
        "balanced_accuracy": (recall + specificity) / 2,
        "specificity": specificity,
        "precision": precision,
        "recall": recall,
        "f1": _safe_ratio(2 * tp, 2 * tp + fp + fn),
        "f2": _safe_ratio(5 * tp, 5 * tp + 4 * fn + fp),
        "alerts": tp + fp,
        "alert_rate": _safe_ratio(tp + fp, total),
        "false_negative_rate": _safe_ratio(fn, tp + fn),
        "false_positive_rate": _safe_ratio(fp, tn + fp),
        "fp_per_tp": None if tp == 0 else float(fp / tp),
        "fn_per_tp": None if tp == 0 else float(fn / tp),
    }


def metrics_at_threshold(target: Any, probabilities: Any, threshold: float) -> dict[str, Any]:
    labels, scores = validate_probability_vector(target, probabilities)
    if isinstance(threshold, (bool, np.bool_)) or not isinstance(
        threshold, (int, float, np.integer, np.floating)
    ):
        raise ValueError("Threshold must be finite and in [0, 1].")
    value = float(threshold)
    if not np.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("Threshold must be finite and in [0, 1].")
    prediction = scores >= value
    tp = int(np.sum(prediction & (labels == 1)))
    fp = int(np.sum(prediction & (labels == 0)))
    fn = int(np.sum(~prediction & (labels == 1)))
    tn = int(np.sum(~prediction & (labels == 0)))
    return {"threshold": value, **confusion_metrics(tn, fp, fn, tp)}


def threshold_curve(target: Any, probabilities: Any) -> pd.DataFrame:
    """Evaluate every unique probability boundary, plus 0 and 1, in O(n log n)."""
    labels, scores = validate_probability_vector(target, probabilities)
    thresholds = np.unique(np.concatenate(([0.0, 1.0], scores)))
    order = np.argsort(scores, kind="stable")
    sorted_scores = scores[order]
    prefix_positives = np.concatenate(([0], np.cumsum(labels[order], dtype=int)))
    first_alert = np.searchsorted(sorted_scores, thresholds, side="left")
    positives = int(np.sum(labels))
    negatives = len(labels) - positives
    rows = []
    for threshold, first, prefix in zip(
        thresholds, first_alert, prefix_positives[first_alert], strict=True
    ):
        tp = positives - int(prefix)
        fp = len(labels) - int(first) - tp
        fn = positives - tp
        tn = negatives - fp
        rows.append({"threshold": float(threshold), **confusion_metrics(tn, fp, fn, tp)})
    return pd.DataFrame(rows)


def select_threshold(curve: pd.DataFrame, scenario: Mapping[str, Any]) -> dict[str, Any] | None:
    """Apply one deterministic constraint/cost rule to threshold-training rows only."""
    if curve.empty or not {"threshold", "precision", "recall", "alert_rate", "fp", "fn"}.issubset(
        curve.columns
    ):
        raise ValueError("Threshold curve is empty or missing metrics.")
    kind, raw_value = scenario.get("kind"), scenario.get("value")
    if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
        raise ValueError("Scenario constraint must be finite and numeric.")
    value = float(raw_value)
    if not np.isfinite(value):
        raise ValueError("Scenario constraint must be finite and numeric.")
    if kind == "capacity":
        if not 0 <= value <= 1:
            raise ValueError("Capacity must be in [0, 1].")
        feasible = curve.loc[curve["alert_rate"] <= value]
        if feasible.empty:
            return None
        ordered = feasible.sort_values(
            ["recall", "precision", "threshold"], ascending=[False, False, False]
        )
    elif kind == "recall":
        if not 0 <= value <= 1:
            raise ValueError("Minimum recall must be in [0, 1].")
        feasible = curve.loc[curve["recall"] >= value]
        if feasible.empty:
            return None
        ordered = feasible.sort_values(
            ["alert_rate", "precision", "threshold"], ascending=[True, False, False]
        )
    elif kind == "cost":
        if value <= 0:
            raise ValueError("False-negative cost ratio must be positive.")
        if "tn" not in curve or "tp" not in curve:
            raise ValueError("Cost selection requires all confusion counts.")
        total = curve[["tn", "fp", "fn", "tp"]].sum(axis=1)
        if (total <= 0).any():
            raise ValueError("Cost selection requires nonempty rows.")
        feasible = curve.assign(
            total_cost=curve["fp"] + value * curve["fn"],
            normalized_cost=(curve["fp"] + value * curve["fn"]) / total,
        )
        ordered = feasible.sort_values(
            ["normalized_cost", "alert_rate", "precision", "threshold"],
            ascending=[True, True, False, False],
        )
    else:
        raise ValueError(f"Unknown threshold scenario kind: {kind!r}.")
    return dict(ordered.iloc[0].to_dict())


def threshold_stability(values: list[float]) -> dict[str, Any]:
    """Descriptive threshold dispersion; >0.10 std or >0.20 range is a warning."""
    if not values or not np.isfinite(values).all():
        raise ValueError("Threshold stability needs finite selected thresholds.")
    sample = np.asarray(values, dtype=float)
    deviation = float(np.std(sample, ddof=1)) if len(sample) > 1 else 0.0
    low, high = float(np.min(sample)), float(np.max(sample))
    return {
        "mean": float(np.mean(sample)), "median": float(np.median(sample)),
        "minimum": low, "maximum": high, "std": deviation,
        "unstable": bool(deviation > 0.10 or high - low > 0.20),
    }


def per_thousand(metrics: Mapping[str, Any]) -> dict[str, float]:
    total = sum(int(metrics[key]) for key in ("tn", "fp", "fn", "tp"))
    if total <= 0:
        raise ValueError("Per-1,000 conversion requires evaluated rows.")
    return {
        "alerts": 1000 * (metrics["tp"] + metrics["fp"]) / total,
        "true_positives": 1000 * metrics["tp"] / total,
        "false_positives": 1000 * metrics["fp"] / total,
        "missed_attrition": 1000 * metrics["fn"] / total,
    }


def _positive_probability(estimator: BaseEstimator, features: pd.DataFrame) -> np.ndarray:
    classes = np.asarray(estimator.classes_)
    positive = np.flatnonzero(classes == 1)
    if len(positive) != 1:
        raise ValueError("Fitted estimator has no positive class.")
    return np.asarray(estimator.predict_proba(features)[:, positive[0]], dtype=float)


def validate_calibration_oof(
    frame: pd.DataFrame, oof: pd.DataFrame, outer_splits: int = 5
) -> np.ndarray:
    """Verify exact 10B row identity, target, folds, and only the two raw vectors."""
    required = {
        "source_row", "EmployeeNumber", "Attrition", "outer_fold",
        "xgboost_raw_probability", "logistic_raw_probability",
    }
    if len(frame) != len(oof) or not required.issubset(oof.columns):
        raise ValueError("Calibration OOF row count or required columns differ.")
    if not oof["source_row"].is_unique or not oof["EmployeeNumber"].is_unique:
        raise ValueError("Calibration OOF identifiers must be unique.")
    for column in ("source_row", "EmployeeNumber"):
        if oof[column].isna().any() or not np.array_equal(
            frame[column].to_numpy(), oof[column].to_numpy()
        ):
            raise ValueError(f"Calibration OOF {column} alignment failed.")
    labels = normalize_binary_target(frame["Attrition"]).to_numpy()
    if not np.array_equal(labels, oof["Attrition"].to_numpy()):
        raise ValueError("Calibration OOF target alignment failed.")
    assignments = np.zeros(len(frame), dtype=int)
    splitter = StratifiedKFold(n_splits=outer_splits, shuffle=True, random_state=OUTER_SEED)
    for fold, (_, validation) in enumerate(splitter.split(frame, labels), start=1):
        assignments[validation] = fold
    if not np.array_equal(assignments, oof["outer_fold"].to_numpy()):
        raise ValueError("Calibration OOF outer folds differ from approved CV.")
    for model in MODELS:
        validate_probability_vector(labels, oof[f"{model}_raw_probability"].to_numpy())
    return assignments


def _inner_oof(
    features: pd.DataFrame, labels: np.ndarray, base: BaseEstimator, splits: int
) -> np.ndarray:
    """Fit four cloned full pipelines on inner-training only."""
    predictions = np.full(len(features), np.nan)
    splitter = StratifiedKFold(n_splits=splits, shuffle=True, random_state=INNER_SEED)
    for training, validation in splitter.split(features, labels):
        fitted = clone(base).fit(features.iloc[training], labels[training])
        predictions[validation] = _positive_probability(fitted, features.iloc[validation])
    validate_probability_vector(labels, predictions)
    return predictions


def run_cross_fitted_policies(
    frame: pd.DataFrame,
    definition: FeatureDefinition,
    logistic_config: dict[str, Any],
    xgboost_config: dict[str, Any],
    fold_parameters: Mapping[int, dict[str, Any]],
    calibration_oof: pd.DataFrame,
    policy: dict[str, Any],
    *,
    outer_splits: int = 5,
    inner_splits: int = 4,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Choose policies on inner OOF and evaluate only on untouched outer-validation."""
    specifications = scenarios(policy)
    if outer_splits < 2 or inner_splits < 2:
        raise ValueError("Outer and inner CV require at least two folds.")
    if set(fold_parameters) != set(range(1, outer_splits + 1)):
        raise ValueError("XGBoost fold parameters must cover every outer fold.")
    for parameters in fold_parameters.values():
        validate_xgboost_parameters(parameters)
    if xgboost_config.get("model") != "xgboost" or (
        xgboost_config.get("random_seed") != OUTER_SEED
    ):
        raise ValueError("Approved XGBoost tuned raw model config is required.")
    fixed = {key: xgboost_config.get("parameters", {}).get(key) for key in XGBOOST_FIXED
             if key != "random_state"}
    if fixed != {key: value for key, value in XGBOOST_FIXED.items()
                 if key != "random_state"}:
        raise ValueError("XGBoost fixed parameters differ from Step 10A.")
    if logistic_config.get("model") != "logistic_regression" or (
        logistic_config.get("parameters", {}).get("class_weight") is not None
    ):
        raise ValueError("Unweighted Logistic baseline is required.")
    logistic_estimator = create_estimator(logistic_config)
    expected_columns = set(definition.all_features) | {
        "source_row", "EmployeeNumber", "Attrition"
    }
    if set(frame.columns) != expected_columns:
        raise ValueError("Development frame has unexpected or missing columns.")
    features = frame.loc[:, definition.all_features].copy()
    validate_feature_columns(features.columns, definition)
    labels = normalize_binary_target(frame["Attrition"]).to_numpy()
    if np.bincount(labels).min() < outer_splits * inner_splits:
        raise ValueError("Too few members of a class for nested stratification.")
    assignments = validate_calibration_oof(frame, calibration_oof, outer_splits)
    output = frame.loc[:, ["source_row", "EmployeeNumber"]].copy()
    output["Attrition"] = labels
    output["outer_fold"] = assignments
    fold_records: list[dict[str, Any]] = []
    for model in MODELS:
        output[f"{model}_raw_probability"] = np.nan
        for spec in specifications:
            prefix = f"{model}_{spec['name']}"
            output[f"{prefix}_threshold"] = np.nan
            output[f"{prefix}_prediction"] = np.nan
    outer = StratifiedKFold(n_splits=outer_splits, shuffle=True, random_state=OUTER_SEED)
    classifier = _load_xgb_classifier()
    for fold, (training, validation) in enumerate(outer.split(features, labels), start=1):
        if not np.all(assignments[validation] == fold):
            raise RuntimeError("Outer-fold assignment changed.")
        x_train, x_validation = features.iloc[training], features.iloc[validation]
        y_train, y_validation = labels[training], labels[validation]
        estimators = {
            "xgboost": classifier(**XGBOOST_FIXED, **fold_parameters[fold]),
            "logistic": logistic_estimator,
        }
        for model, estimator in estimators.items():
            if progress is not None:
                progress(f"Outer fold {fold}/{outer_splits}: {model} inner OOF + raw validation")
            base = build_model_pipeline(definition, estimator)
            training_scores = _inner_oof(x_train, y_train, base, inner_splits)
            training_curve = threshold_curve(y_train, training_scores)
            fitted = clone(base).fit(x_train, y_train)
            validation_scores = _positive_probability(fitted, x_validation)
            previous = calibration_oof[f"{model}_raw_probability"].to_numpy()[validation]
            if not np.allclose(previous, validation_scores, rtol=1e-7, atol=1e-9):
                difference = float(np.max(np.abs(previous - validation_scores)))
                raise RuntimeError(
                    f"Step 10B raw regression check failed for {model} fold {fold}; "
                    f"max absolute difference={difference:.12g}."
                )
            output.loc[output.index[validation], f"{model}_raw_probability"] = validation_scores
            for spec in specifications:
                selected = select_threshold(training_curve, spec)
                if selected is None:
                    fold_records.append({
                        "model": model, "scenario": spec["name"], "kind": spec["kind"],
                        "outer_fold": fold, "feasible": False,
                    })
                    continue
                threshold = float(selected["threshold"])
                prediction = (validation_scores >= threshold).astype(int)
                prefix = f"{model}_{spec['name']}"
                output.loc[output.index[validation], f"{prefix}_threshold"] = threshold
                output.loc[output.index[validation], f"{prefix}_prediction"] = prediction
                observed = metrics_at_threshold(y_validation, validation_scores, threshold)
                fold_records.append({
                    "model": model, "scenario": spec["name"], "kind": spec["kind"],
                    "outer_fold": fold, "feasible": True,
                    "selected_threshold": threshold,
                    "training_alert_rate": selected["alert_rate"],
                    "training_recall": selected["recall"],
                    "validation_alert_rate": observed["alert_rate"],
                    "validation_precision": observed["precision"],
                    "validation_recall": observed["recall"],
                    "validation_f1": observed["f1"],
                    "validation_f2": observed["f2"],
                    **{key: observed[key] for key in ("tn", "fp", "fn", "tp")},
                })
    for model in MODELS:
        validate_probability_vector(labels, output[f"{model}_raw_probability"].to_numpy())
    fold_frame = pd.DataFrame(fold_records)
    summaries: list[dict[str, Any]] = []
    for model in MODELS:
        for spec in specifications:
            subset = fold_frame.loc[
                (fold_frame["model"] == model) & (fold_frame["scenario"] == spec["name"])
            ]
            feasible = int(subset["feasible"].sum())
            row: dict[str, Any] = {
                "model": model, "scenario": spec["name"], "kind": spec["kind"],
                "feasible_folds": feasible, "outer_folds": outer_splits,
            }
            if feasible == outer_splits:
                predictions = output[f"{model}_{spec['name']}_prediction"].to_numpy(dtype=int)
                tp = int(np.sum((predictions == 1) & (labels == 1)))
                fp = int(np.sum((predictions == 1) & (labels == 0)))
                fn = int(np.sum((predictions == 0) & (labels == 1)))
                tn = int(np.sum((predictions == 0) & (labels == 0)))
                stability = threshold_stability(subset["selected_threshold"].tolist())
                row.update(confusion_metrics(tn, fp, fn, tp))
                row.update({f"threshold_{key}": value for key, value in stability.items()})
                if spec["kind"] == "capacity":
                    row["validation_constraint_met"] = bool(row["alert_rate"] <= spec["value"])
                elif spec["kind"] == "recall":
                    row["validation_constraint_met"] = bool(row["recall"] >= spec["value"])
                if spec["kind"] == "cost":
                    row["total_cost"] = fp + spec["value"] * fn
                    row["normalized_cost"] = row["total_cost"] / len(labels)
            summaries.append(row)
    full_curves = []
    full_selected = []
    for model in MODELS:
        curve = threshold_curve(labels, calibration_oof[f"{model}_raw_probability"].to_numpy())
        full_curves.append(curve.assign(model=model))
        for spec in specifications:
            selected = select_threshold(curve, spec)
            full_selected.append({
                "model": model, "scenario": spec["name"], "kind": spec["kind"],
                "feasible": selected is not None,
                **({} if selected is None else selected),
            })
    return {
        "oof": output, "fold_records": fold_frame,
        "summary": pd.DataFrame(summaries),
        "full_curve": pd.concat(full_curves, ignore_index=True),
        "full_selected": pd.DataFrame(full_selected),
    }
