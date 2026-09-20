"""Development-only subgroup audit of frozen cross-fitted prediction policies."""

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from eris_ml.data.fairness_recovery import validate_development_fairness_frame
from eris_ml.evaluation.calibration_reporting import reliability_bins

ATTRIBUTES = ("Gender", "AgeGroup", "MaritalStatus", "Gender_AgeGroup")
CI_METRICS = ("alert_rate", "tpr", "fpr", "precision", "brier")
GAP_METRICS = ("selection_rate_difference", "tpr_difference", "fpr_difference",
               "equalized_odds_gap")
KEYS = ["source_row", "EmployeeNumber"]
EXPECTED_COLUMNS = {
    "calibration": {"source_row", "EmployeeNumber", "Attrition", "outer_fold",
                    "xgboost_raw_probability", "logistic_raw_probability"},
    "policy": {"source_row", "EmployeeNumber", "Attrition", "outer_fold",
               "xgboost_raw_probability", "logistic_raw_probability",
               "xgboost_capacity_15_percent_prediction",
               "logistic_capacity_15_percent_prediction",
               "xgboost_capacity_15_percent_threshold",
               "logistic_capacity_15_percent_threshold"},
}


def validate_protocol(protocol: dict[str, Any]) -> dict[str, Any]:
    """Enforce the frozen v1 audit scope and explicit holdout incident status."""
    fixed = {
        "protocol_version": "fairness-protocol-v1",
        "model": "xgboost_tuned_raw",
        "primary_policy": "capacity_15_percent",
        "working_threshold": 0.345651,
        "business_approved": False,
        "production_approved": False,
        "final_test_used": False,
        "holdout_status": "retained_with_protocol_deviation",
        "final_test_file_opened": False,
        "mixed_fairness_file_loaded_during_schema_inspection": True,
        "final_metrics_computed": False,
        "final_rows_used_for_model_decision": False,
        "audit_attributes": {"primary": ["Gender", "AgeGroup"],
                             "exploratory": ["MaritalStatus", "Gender_AgeGroup"]},
        "age_groups": [
            {"name": "age_18_29", "minimum": 18, "maximum": 29},
            {"name": "age_30_39", "minimum": 30, "maximum": 39},
            {"name": "age_40_49", "minimum": 40, "maximum": 49},
            {"name": "age_50_plus", "minimum": 50, "maximum": None},
        ],
        "minimum_support": {"total_rows": 50, "positive_rows": 10, "negative_rows": 10},
        "screening_thresholds": {"tpr_difference": 0.10, "fpr_difference": 0.10,
                                 "selection_rate_ratio": 0.80,
                                 "calibration_gap_absolute": 0.05},
        "sensitivity_thresholds": [0.5, 0.445293, 0.345651, 0.265658],
    }
    if not isinstance(protocol, dict) or set(protocol) != set(fixed) | {"bootstrap"}:
        raise ValueError("Fairness protocol keys differ from frozen v1 scope.")
    for key, value in fixed.items():
        if protocol[key] != value or (isinstance(value, bool) and protocol[key] is not value):
            raise ValueError(f"Fairness protocol has invalid {key}.")
    bootstrap = protocol["bootstrap"]
    if bootstrap != {"enabled": True, "iterations": 2000,
                     "confidence_level": 0.95, "random_state": 42}:
        raise ValueError("Bootstrap must be 2000 iterations, 95% CI, seed 42.")
    return protocol


def add_audit_groups(frame: pd.DataFrame) -> pd.DataFrame:
    """Assign fixed, predeclared age bands; reject out-of-domain ages."""
    result = frame.copy()
    age = pd.to_numeric(result["Age"], errors="raise")
    if age.isna().any() or age.lt(18).any() or age.mod(1).ne(0).any():
        raise ValueError("Age outside the declared adult integer domain.")
    result["AgeGroup"] = pd.cut(
        age, bins=[17, 29, 39, 49, np.inf],
        labels=["18-29", "30-39", "40-49", "50+"],
    ).astype(str)
    result["Gender_AgeGroup"] = result["Gender"].astype(str) + " × " + result["AgeGroup"]
    return result


def _validate_oof(frame: pd.DataFrame, kind: str, expected_rows: int) -> None:
    if not EXPECTED_COLUMNS[kind].issubset(frame.columns):
        raise ValueError(f"{kind} OOF is missing required columns.")
    if len(frame) != expected_rows or frame[KEYS].isna().any().any():
        raise ValueError(f"{kind} OOF row count or IDs are invalid.")
    if frame["source_row"].duplicated().any() or frame["EmployeeNumber"].duplicated().any():
        raise ValueError(f"{kind} OOF has duplicate IDs.")
    folds = frame["outer_fold"].value_counts().sort_index()
    if expected_rows == 1176 and set(folds.index) != {1, 2, 3, 4, 5}:
        raise ValueError(f"{kind} OOF outer folds do not match 1–5.")
    if folds.empty or (folds <= 0).any():
        raise ValueError(f"{kind} OOF has invalid outer folds.")
    if not frame["Attrition"].isin([0, 1]).all():
        raise ValueError(f"{kind} OOF target is not binary.")
    for column in ("xgboost_raw_probability", "logistic_raw_probability"):
        values = pd.to_numeric(frame[column], errors="raise").to_numpy(dtype=float)
        if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
            raise ValueError(f"{kind} OOF has invalid raw probabilities.")


def join_audit_inputs(
    development: pd.DataFrame, fairness: pd.DataFrame, calibration: pd.DataFrame,
    policy: pd.DataFrame, *, expected_rows: int = 1176,
    expected_target_counts: dict[int, int] | None = None,
    forbidden_source_rows: set[int] | None = None,
    forbidden_employee_numbers: set[int] | None = None,
) -> pd.DataFrame:
    """Require exact one-to-one ID/target/fold/probability agreement before any metric."""
    validate_development_fairness_frame(
        fairness, development, expected_rows=expected_rows,
        expected_target_counts=expected_target_counts,
        forbidden_source_rows=forbidden_source_rows,
        forbidden_employee_numbers=forbidden_employee_numbers,
    )
    for kind, frame in (("calibration", calibration), ("policy", policy)):
        _validate_oof(frame, kind, expected_rows)
        if set(map(tuple, frame[KEYS].to_numpy())) != set(map(tuple, development[KEYS].to_numpy())):
            raise ValueError(f"{kind} OOF IDs do not match development exactly.")
    joined = fairness.merge(calibration[list(EXPECTED_COLUMNS["calibration"])], on=KEYS,
                            how="left", validate="one_to_one", suffixes=("", "_cal"))
    joined = joined.merge(policy[list(EXPECTED_COLUMNS["policy"])], on=KEYS,
                          how="left", validate="one_to_one", suffixes=("", "_policy"))
    if len(joined) != expected_rows or joined.isna().any().any():
        raise ValueError("Audit join lost rows or contains missing values.")
    if not (joined["Attrition"] == joined["Attrition_cal"]).all() or not (
        joined["Attrition"] == joined["Attrition_policy"]
    ).all():
        raise ValueError("OOF targets do not align with development fairness target.")
    if not (joined["outer_fold"] == joined["outer_fold_policy"]).all():
        raise ValueError("Calibration and policy outer folds differ.")
    for model in ("xgboost", "logistic"):
        left = joined[f"{model}_raw_probability"].to_numpy(dtype=float)
        right = joined[f"{model}_raw_probability_policy"].to_numpy(dtype=float)
        if not np.allclose(left, right, rtol=1e-7, atol=1e-9):
            raise ValueError(f"{model} raw probabilities differ from calibration OOF.")
        prediction = joined[f"{model}_capacity_15_percent_prediction"].to_numpy()
        threshold = joined[f"{model}_capacity_15_percent_threshold"].to_numpy(dtype=float)
        if not np.isfinite(threshold).all() or ((threshold < 0) | (threshold > 1)).any():
            raise ValueError(f"{model} policy thresholds are invalid.")
        if not np.isin(prediction, [0, 1]).all() or not np.array_equal(
            prediction.astype(int), (left >= threshold).astype(int)
        ):
            raise ValueError(f"{model} policy predictions do not match fold thresholds.")
    return add_audit_groups(joined)


def _ratio(numerator: int | float, denominator: int | float) -> float | None:
    return float(numerator / denominator) if denominator else None


def group_metrics(
    target: Any, prediction: Any, probability: Any, *,
    support: Mapping[str, int], bins: int = 10, compute_ece: bool = True,
) -> dict[str, Any]:
    """Calculate descriptive operational and probability metrics for one subgroup."""
    y = np.asarray(target, dtype=int)
    predicted = np.asarray(prediction, dtype=int)
    scores = np.asarray(probability, dtype=float)
    if not len(y) or len(y) != len(predicted) or len(y) != len(scores):
        raise ValueError("Group vectors must have equal nonzero length.")
    if not np.isin(y, [0, 1]).all() or not np.isin(predicted, [0, 1]).all():
        raise ValueError("Group target and prediction must be binary.")
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("Group raw probability must be finite within [0, 1].")
    tn = int(np.sum((y == 0) & (predicted == 0)))
    fp = int(np.sum((y == 0) & (predicted == 1)))
    fn = int(np.sum((y == 1) & (predicted == 0)))
    tp = int(np.sum((y == 1) & (predicted == 1)))
    n = len(y)
    positives, negatives = tp + fn, tn + fp
    precision = _ratio(tp, tp + fp)
    recall = _ratio(tp, positives)
    fpr = _ratio(fp, negatives)
    tnr = _ratio(tn, negatives)
    fnr = _ratio(fn, positives)
    npv = _ratio(tn, tn + fn)
    supported = (n >= support["total_rows"] and positives >= support["positive_rows"]
                 and negatives >= support["negative_rows"])
    ece = None
    if supported and compute_ece:
        ece = float(sum(row["count"] / n * row["absolute_gap"]
                        for row in reliability_bins(y, scores, bins)))
    return {
        "n": n, "positive_count": positives, "negative_count": negatives,
        "base_rate": positives / n,
        "support_status": "supported" if supported else "insufficient support",
        "alerts": tp + fp, "alert_rate": (tp + fp) / n,
        "tpr": recall, "fnr": fnr, "fpr": fpr, "tnr": tnr,
        "precision": precision, "npv": npv, "accuracy": (tp + tn) / n,
        "balanced_accuracy": None if recall is None or tnr is None else (recall + tnr) / 2,
        "f1": _ratio(2 * tp, 2 * tp + fp + fn),
        "f2": _ratio(5 * tp, 5 * tp + 4 * fn + fp),
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "mean_probability": float(np.mean(scores)),
        "brier": float(np.mean((scores - y) ** 2)), "ece": ece,
        "calibration_gap": float(np.mean(scores) - np.mean(y)),
    }


def _gap(values: list[float | None]) -> float | None:
    valid = [value for value in values if value is not None]
    return float(max(valid) - min(valid)) if len(valid) >= 2 else None


def _min_max_ratio(values: list[float | None]) -> float | None:
    valid = [value for value in values if value is not None]
    return float(min(valid) / max(valid)) if len(valid) >= 2 and max(valid) > 0 else None


def disparity_metrics(groups: list[dict[str, Any]]) -> dict[str, Any]:
    """Max–min disparities without presuming a privileged reference group."""
    measures = {
        "selection_rate": "alert_rate", "tpr": "tpr", "fpr": "fpr",
        "precision": "precision", "fnr": "fnr", "brier": "brier",
        "calibration_gap": "calibration_gap",
    }
    result: dict[str, Any] = {"group_count": len(groups),
                              "supported_groups": sum(row["support_status"] == "supported"
                                                      for row in groups)}
    for name, column in measures.items():
        values = [row[column] for row in groups]
        result[f"{name}_difference"] = _gap(values)
        if name in ("selection_rate", "tpr", "fpr", "precision"):
            result[f"{name}_ratio"] = _min_max_ratio(values)
    tpr_gap, fpr_gap = result["tpr_difference"], result["fpr_difference"]
    result["equalized_odds_gap"] = (max(tpr_gap, fpr_gap)
                                    if tpr_gap is not None and fpr_gap is not None else None)
    return result


def _percentile_ci(values: list[float], iterations: int, level: float) -> tuple[float | None,
                                                                                 float | None, int]:
    if len(values) < int(0.8 * iterations):
        return None, None, len(values)
    alpha = (1 - level) / 2
    low, high = np.quantile(values, [alpha, 1 - alpha])
    return float(low), float(high), len(values)


def bootstrap_attribute(
    frame: pd.DataFrame, attribute: str, prediction_column: str,
    probability_column: str, support: Mapping[str, int], *,
    iterations: int = 2000, confidence_level: float = 0.95, seed: int = 42,
) -> tuple[dict[str, dict[str, tuple[float | None, float | None, int]]],
           dict[str, tuple[float | None, float | None, int]]]:
    """Paired group×target stratified bootstrap for subgroup and disparity CIs."""
    if attribute not in ATTRIBUTES or iterations < 1 or not 0 < confidence_level < 1:
        raise ValueError("Invalid bootstrap attribute or configuration.")
    rng = np.random.default_rng(seed)
    labels = frame[attribute].astype(str).to_numpy()
    target = frame["Attrition"].to_numpy(dtype=int)
    prediction = frame[prediction_column].to_numpy(dtype=int)
    probability = frame[probability_column].to_numpy(dtype=float)
    group_names = sorted(np.unique(labels).tolist())
    supported_names = {
        name for name in group_names
        if group_metrics(target[labels == name], prediction[labels == name],
                         probability[labels == name], support=support,
                         compute_ece=False)["support_status"] == "supported"
    }
    strata = [np.flatnonzero((labels == name) & (target == value))
              for name in group_names for value in (0, 1)]
    strata = [indices for indices in strata if len(indices)]
    samples: dict[str, dict[str, list[float]]] = {
        name: {metric: [] for metric in CI_METRICS} for name in group_names
    }
    gap_samples: dict[str, list[float]] = {metric: [] for metric in GAP_METRICS}
    for _ in range(iterations):
        drawn = np.concatenate([rng.choice(indices, size=len(indices), replace=True)
                                for indices in strata])
        drawn_labels = labels[drawn]
        rows = []
        for name in group_names:
            mask = drawn_labels == name
            metric = group_metrics(target[drawn][mask], prediction[drawn][mask],
                                   probability[drawn][mask], support=support,
                                   compute_ece=False)
            rows.append(metric)
            if name in supported_names:
                for column in CI_METRICS:
                    value = metric[column]
                    if value is not None:
                        samples[name][column].append(float(value))
        if len(supported_names) == len(group_names):
            gaps = disparity_metrics(rows)
            for column in GAP_METRICS:
                value = gaps[column]
                if value is not None:
                    gap_samples[column].append(float(value))
    group_ci = {
        name: {
            column: _percentile_ci(values, iterations, confidence_level)
            for column, values in columns.items()
        } for name, columns in samples.items()
    }
    gap_ci = {column: _percentile_ci(values, iterations, confidence_level)
              for column, values in gap_samples.items()}
    return group_ci, gap_ci


def screen_disparity(
    groups: list[dict[str, Any]], disparity: dict[str, Any],
    gap_ci: dict[str, tuple[float | None, float | None, int]],
    thresholds: Mapping[str, float],
) -> str:
    """Governance triage only, never a legal discrimination determination."""
    if disparity["supported_groups"] != disparity["group_count"] or len(groups) < 2:
        return "INSUFFICIENT_EVIDENCE"
    if any(disparity[column] is None for column in ("tpr_difference", "fpr_difference",
                                                      "selection_rate_ratio")):
        return "INSUFFICIENT_EVIDENCE"
    if (disparity["tpr_difference"] > thresholds["tpr_difference"]
        or disparity["fpr_difference"] > thresholds["fpr_difference"]
        or disparity["selection_rate_ratio"] < thresholds["selection_rate_ratio"]
        or any(abs(row["calibration_gap"]) > thresholds["calibration_gap_absolute"]
               for row in groups)):
        return "REVIEW"
    for column in ("tpr_difference", "fpr_difference", "selection_rate_difference"):
        if gap_ci[column][1] is None:
            return "INSUFFICIENT_EVIDENCE"
    upper_tpr = gap_ci["tpr_difference"][1]
    upper_fpr = gap_ci["fpr_difference"][1]
    if upper_tpr is None or upper_fpr is None:
        return "INSUFFICIENT_EVIDENCE"
    if (upper_tpr > thresholds["tpr_difference"]
        or upper_fpr > thresholds["fpr_difference"]):
        return "INSUFFICIENT_EVIDENCE"
    return "PASS"


def audit_attribute(
    frame: pd.DataFrame, attribute: str, prediction_column: str,
    probability_column: str, protocol: dict[str, Any], *,
    bootstrap_iterations: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Calculate group metrics, paired bootstrap intervals and triage for one attribute."""
    if attribute not in ATTRIBUTES or attribute not in frame:
        raise ValueError("Audit attribute is invalid or absent.")
    support = protocol["minimum_support"]
    rows = []
    for name, subset in frame.groupby(attribute, sort=True, observed=True):
        measured = group_metrics(subset["Attrition"], subset[prediction_column],
                                 subset[probability_column], support=support)
        rows.append({"attribute": attribute, "group": str(name), **measured})
    gap = disparity_metrics(rows)
    config = protocol["bootstrap"]
    group_ci, gap_ci = bootstrap_attribute(
        frame, attribute, prediction_column, probability_column, support=support,
        iterations=bootstrap_iterations or config["iterations"],
        confidence_level=config["confidence_level"], seed=config["random_state"],
    )
    for row in rows:
        for metric_name, (low, high, valid) in group_ci[row["group"]].items():
            row[f"{metric_name}_ci_low"] = low
            row[f"{metric_name}_ci_high"] = high
            row[f"{metric_name}_bootstrap_valid"] = valid
    gap.update({f"{metric}_ci_low": interval[0] for metric, interval in gap_ci.items()})
    gap.update({f"{metric}_ci_high": interval[1] for metric, interval in gap_ci.items()})
    gap.update({f"{metric}_bootstrap_valid": interval[2]
                for metric, interval in gap_ci.items()})
    gap["attribute"] = attribute
    gap["status"] = screen_disparity(rows, gap, gap_ci, protocol["screening_thresholds"])
    return rows, gap


def threshold_sensitivity(frame: pd.DataFrame, protocol: dict[str, Any]) -> list[dict[str, Any]]:
    """Descriptive only: apply four frozen thresholds to XGBoost raw OOF scores."""
    rows = []
    for threshold in protocol["sensitivity_thresholds"]:
        copy = frame.copy()
        copy["fixed_prediction"] = (
            copy["xgboost_raw_probability"].to_numpy(dtype=float) >= threshold
        ).astype(int)
        for attribute in ("Gender", "AgeGroup"):
            groups = [group_metrics(part["Attrition"], part["fixed_prediction"],
                                    part["xgboost_raw_probability"],
                                    support=protocol["minimum_support"])
                      for _, part in copy.groupby(attribute, sort=True, observed=True)]
            rows.append({"threshold": threshold, "attribute": attribute,
                         **disparity_metrics(groups)})
    return rows


def run_fairness_audit(
    frame: pd.DataFrame, protocol: dict[str, Any], *, bootstrap_iterations: int | None = None,
) -> dict[str, Any]:
    """Audit frozen capacity-15 predictions for both models without retraining."""
    validate_protocol(protocol)
    comparisons: dict[str, Any] = {}
    metric_rows: list[dict[str, Any]] = []
    disparity_rows: list[dict[str, Any]] = []
    for model in ("xgboost", "logistic"):
        prediction = f"{model}_capacity_15_percent_prediction"
        probability = f"{model}_raw_probability"
        overall = group_metrics(frame["Attrition"], frame[prediction], frame[probability],
                                support=protocol["minimum_support"])
        comparisons[model] = {"overall": overall, "disparities": {}}
        for attribute in ATTRIBUTES:
            groups, disparity = audit_attribute(
                frame, attribute, prediction, probability, protocol,
                bootstrap_iterations=bootstrap_iterations,
            )
            for row in groups:
                metric_rows.append({"model": model, "policy": "capacity_15_percent", **row})
            disparity_rows.append({"model": model, **disparity})
            comparisons[model]["disparities"][attribute] = disparity
    return {"metrics": pd.DataFrame(metric_rows), "disparities": pd.DataFrame(disparity_rows),
            "sensitivity": pd.DataFrame(threshold_sensitivity(frame, protocol)),
            "comparison": comparisons}
