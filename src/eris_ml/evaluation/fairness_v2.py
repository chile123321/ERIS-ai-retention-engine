"""Aggregate-only V2 audit of saved raw probabilities and saved batch decisions."""

from collections.abc import Iterator
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from eris_ml.data.fairness_recovery import validate_development_fairness_frame
from eris_ml.evaluation.fairness import add_audit_groups

FEATURES = ("v2_general_16", "v2_no_department_17")
CAPACITIES = (5, 10, 15)
IDS = ["source_row", "EmployeeNumber", "Attrition", "fold"]
GROUPS = {
    "Overall": ("all",),
    "AgeGroup": ("18-29", "30-39", "40-49", "50+"),
    "Gender": ("Female", "Male"),
    "MaritalStatus": ("Divorced", "Married", "Single"),
}
RATES = ("alert_rate", "tpr", "fpr", "precision")
PROB_METRICS = ("average_precision", "roc_auc", "brier", "mean_probability")


def validate_protocol(protocol: dict[str, Any]) -> None:
    """Reject changes to the preregistered scope and interpretation."""
    fixed = {
        "protocol_version": "fairness-protocol-v2", "model": "logistic_baseline",
        "probability_type": "raw", "feature_sets": list(FEATURES), "policy": "saved_top_k",
        "capacity_percent": list(CAPACITIES), "development_only": True,
        "business_approved": False, "production_approved": False, "final_test_used": False,
        "minimum_support": {"total_rows": 50, "positive_rows": 10, "negative_rows": 10},
        "precision_minimum_alerts": 10,
        "unsupported_metrics": "undefined_performance_metrics_keep_counts_and_score_distribution",
        "missing_unknown": "reject_before_metrics_no_imputation_or_row_dropping",
        "empty_fold_groups": "emit_zero_counts_undefined_rates_insufficient_evidence",
        "auc_one_class": "undefined",
        "audit_attributes": {"primary": ["AgeGroup", "Gender"], "exploratory": ["MaritalStatus"]},
        "domains": {"Gender": ["Female", "Male"],
                    "MaritalStatus": ["Divorced", "Married", "Single"]},
        "age_groups": [
            {"name": "18-29", "minimum": 18, "maximum": 29},
            {"name": "30-39", "minimum": 30, "maximum": 39},
            {"name": "40-49", "minimum": 40, "maximum": 49},
            {"name": "50+", "minimum": 50, "maximum": None},
        ],
        "screening_thresholds": {"tpr_difference": .10, "fpr_difference": .10,
                                 "selection_rate_ratio": .80, "calibration_gap_absolute": .05},
        "uncertainty": {
            "method": "wilson_binomial_descriptive_conditional_on_saved_predictions",
            "confidence_level": .95, "z": 1.959963984540054,
            "fold_summary": "mean_sample_sd_min_max_and_valid_fold_count",
            "significance_claim": False,
        },
        "top_k_verification": {
            "batch": "saved_outer_validation_fold", "k": "floor_rows_times_percent_divided_by_100",
            "tie_order": ["probability_descending", "source_row_ascending"],
            "no_subgroup_reranking": True,
        },
        "comparison_direction": "no_department_17_minus_general_16",
        "undefined_metrics": "null_with_explicit_denominators",
        "scopes": ["pooled", "fold_1", "fold_2", "fold_3", "fold_4", "fold_5"],
        "screening_interpretation": "descriptive_human_review_only_not_fairness_or_legal_approval",
        "metrics": {
            "probability": ["mean_probability", "probability_std", "p0", "p10", "p25", "p50",
                            "p75", "p90", "p100", "average_precision", "roc_auc", "brier"],
            "operational": ["n", "positives", "negatives", "alerts", "tp", "fp", "fn", "tn",
                            "alert_rate", "tpr", "fpr", "precision"],
        },
    }
    if protocol != fixed:
        raise ValueError("Fairness V2 protocol differs from the locked scope.")


def _aligned(frame: pd.DataFrame, expected: pd.DataFrame) -> pd.DataFrame:
    if (frame.columns.has_duplicates or not set(IDS).issubset(frame)
        or frame[IDS].isna().any().any() or not frame.source_row.is_unique
        or not frame.EmployeeNumber.is_unique):
        raise ValueError("Missing/duplicate OOF identities.")
    result = frame.sort_values("source_row").reset_index(drop=True)
    if not result[IDS].equals(expected):
        raise ValueError("OOF ID/target/fold alignment mismatch.")
    scores = result.probability.to_numpy(dtype=float)
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("Invalid probability.")
    return result


def validate_inputs(
    development: pd.DataFrame, assignment: pd.DataFrame, audit: pd.DataFrame,
    baseline: pd.DataFrame, calibration: pd.DataFrame, policy: pd.DataFrame, *,
    expected_rows: int = 1176, expected_counts: dict[int, int] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Validate before metrics; rank only to verify saved decisions, never replace them."""
    counts = {0: 986, 1: 190} if expected_counts is None else expected_counts
    if (len(development) != expected_rows
        or development.Attrition.value_counts().to_dict() != counts
        or not development.source_row.is_unique or not development.EmployeeNumber.is_unique
        or development[IDS[:3]].isna().any().any()):
        raise ValueError("Development contract mismatch.")
    if (len(assignment) != len(development)
        or not assignment[IDS[:3]].reset_index(drop=True).equals(
            development[IDS[:3]].reset_index(drop=True))
        or not np.array_equal(assignment.development_position, np.arange(len(development)))
        or set(assignment.fold) != {1, 2, 3, 4, 5}
        or not np.equal(assignment.fold, assignment.fold.astype(int)).all()):
        raise ValueError("Saved assignment mismatch.")
    validate_development_fairness_frame(
        audit, development, expected_rows=expected_rows, expected_target_counts=counts,
    )
    for attr in ("Gender", "MaritalStatus"):
        if not audit[attr].isin(GROUPS[attr]).all():
            raise ValueError(f"Missing/unknown audit group: {attr}.")
    grouped = add_audit_groups(audit)
    grouped["Overall"] = "all"
    expected = assignment[IDS].sort_values("source_row").reset_index(drop=True)
    raw = baseline.loc[
        baseline.feature_set.isin(FEATURES) & baseline.model.eq("logistic_regression")
    ]
    cal = calibration.loc[calibration.feature_set.isin(FEATURES) & calibration.method.eq("raw")]
    selected = policy.loc[policy.policy.eq("top_k")]
    if set(selected.feature_set) != set(FEATURES) or set(selected.capacity_percent) != set(
        CAPACITIES
    ):
        raise ValueError("Saved top-k candidate coverage mismatch.")
    if selected.columns.has_duplicates or selected.isna().any().any():
        raise ValueError("Missing policy values.")
    raw_parts, policy_parts = [], []
    for feature in FEATURES:
        reference = _aligned(raw.loc[raw.feature_set == feature], expected)
        calibrated_raw = _aligned(cal.loc[cal.feature_set == feature], expected)
        if not np.allclose(reference.probability, calibrated_raw.probability,
                           rtol=1e-7, atol=1e-9):
            raise ValueError("Baseline/calibration raw probability mismatch.")
        raw_parts.append(reference)
        for capacity in CAPACITIES:
            part = _aligned(selected.loc[
                (selected.feature_set == feature) & (selected.capacity_percent == capacity)
            ], expected)
            if not np.allclose(reference.probability, part.probability, rtol=1e-7, atol=1e-9):
                raise ValueError("Policy raw probability mismatch.")
            if not part.prediction.isin([0, 1]).all():
                raise ValueError("Saved prediction must be binary.")
            for _, batch in part.groupby("fold"):
                k = len(batch) * capacity // 100
                ranked = batch.sort_values(
                    ["probability", "source_row"], ascending=[False, True], kind="stable"
                )
                chosen = set(ranked.head(k).source_row)
                if not np.array_equal(batch.prediction, batch.source_row.isin(chosen).astype(int)):
                    raise ValueError("Saved top-k batch/tie/budget mismatch.")
            policy_parts.append(part)
    attributes = grouped[["source_row", "EmployeeNumber", *GROUPS]]
    joined = [pd.concat(parts, ignore_index=True).merge(
        attributes, on=["source_row", "EmployeeNumber"], validate="many_to_one"
    ) for parts in (raw_parts, policy_parts)]
    return joined[0], joined[1]


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _wilson(successes: int, count: int, z: float) -> tuple[float | None, float | None]:
    if not count:
        return None, None
    p = successes / count
    denominator = 1 + z * z / count
    center = (p + z * z / (2 * count)) / denominator
    half = z * np.sqrt(p * (1 - p) / count + z * z / (4 * count * count)) / denominator
    return max(0., float(center - half)), min(1., float(center + half))


def measure(
    labels: np.ndarray, scores: np.ndarray, protocol: dict[str, Any],
    predictions: np.ndarray | None = None,
) -> dict[str, Any]:
    """Keep counts/distributions; unsupported performance estimates remain undefined."""
    if (labels.ndim != 1 or labels.shape != scores.shape or not np.isin(labels, [0, 1]).all()
        or not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any()):
        raise ValueError("Invalid metric vectors.")
    n, positives = len(labels), int(np.sum(labels == 1))
    negatives = n - positives
    minimum = protocol["minimum_support"]
    supported = (n >= minimum["total_rows"] and positives >= minimum["positive_rows"]
                 and negatives >= minimum["negative_rows"])
    result: dict[str, Any] = {
        "n": n, "positives": positives, "negatives": negatives,
        "support": "supported" if supported else "insufficient evidence",
        "base_rate": _ratio(positives, n),
    }
    if predictions is None:
        result.update({
            "mean_probability": float(np.mean(scores)) if n else None,
            "probability_std": float(np.std(scores, ddof=1)) if n > 1 else None,
            "brier": float(np.mean((scores - labels) ** 2)) if supported else None,
            "average_precision": float(average_precision_score(labels, scores))
            if supported else None,
            "roc_auc": float(roc_auc_score(labels, scores)) if supported else None,
            "auc_status": ("defined" if supported else "undefined_missing_class"
                           if not positives or not negatives else "undefined_insufficient_support"),
        })
        for q in (0, 10, 25, 50, 75, 90, 100):
            result[f"p{q}"] = float(np.quantile(scores, q / 100)) if n else None
        result["calibration_gap"] = float(np.mean(scores - labels)) if n else None
        return result
    if predictions.shape != labels.shape or not np.isin(predictions, [0, 1]).all():
        raise ValueError("Invalid binary predictions.")
    tp = int(np.sum((labels == 1) & (predictions == 1)))
    fp = int(np.sum((labels == 0) & (predictions == 1)))
    alerts = tp + fp
    result.update(tp=tp, fp=fp, fn=positives - tp, tn=negatives - fp, alerts=alerts)
    denominators = {"alert_rate": (alerts, n), "tpr": (tp, positives),
                    "fpr": (fp, negatives), "precision": (tp, alerts)}
    for name, (num, den) in denominators.items():
        low, high = _wilson(num, den, protocol["uncertainty"]["z"])
        valid = name == "alert_rate" or (supported and (
            name != "precision" or alerts >= protocol["precision_minimum_alerts"]))
        result.update({name: _ratio(num, den) if valid else None, f"{name}_numerator": num,
                       f"{name}_denominator": den, f"{name}_ci_low": low,
                       f"{name}_ci_high": high})
        if not valid:
            result[f"{name}_ci_low"] = result[f"{name}_ci_high"] = None
        result[f"{name}_status"] = "defined" if valid and den else "undefined"
    result["precision_support"] = (
        "supported" if supported and alerts >= protocol["precision_minimum_alerts"]
        else "insufficient evidence"
    )
    return result


def _subsets(frame: pd.DataFrame) -> Iterator[tuple[int, str, str, pd.DataFrame]]:
    for fold in range(6):  # 0 is pooled; not a new fold or batch
        scope = frame if fold == 0 else frame.loc[frame.fold == fold]
        for attribute, names in GROUPS.items():
            for name in names:
                yield fold, attribute, name, scope.loc[scope[attribute] == name]


def _stability(table: pd.DataFrame, keys: list[str], metrics: tuple[str, ...]) -> pd.DataFrame:
    rows = []
    for key, part in table.loc[table.fold != 0].groupby(keys, sort=False):
        row = dict(zip(keys, key, strict=True))
        row["supported_folds"] = int(part.support.eq("supported").sum())
        for metric in metrics:
            values = part[metric].dropna().to_numpy(dtype=float)
            row.update({
                f"{metric}_valid_folds": len(values),
                f"{metric}_mean": float(values.mean()) if len(values) else None,
                f"{metric}_sample_sd": float(values.std(ddof=1)) if len(values) > 1 else None,
                f"{metric}_min": float(values.min()) if len(values) else None,
                f"{metric}_max": float(values.max()) if len(values) else None,
                f"{metric}_positive_folds": int((values > 0).sum()),
            })
        rows.append(row)
    return pd.DataFrame(rows)


def summarize(raw: pd.DataFrame, policy: pd.DataFrame, protocol: dict[str, Any]
              ) -> dict[str, pd.DataFrame]:
    """Read saved decisions; produce only aggregate tables, no new policy or fit."""
    validate_protocol(protocol)
    probabilities, operations = [], []
    for feature in FEATURES:
        for fold, attr, group, rows in _subsets(raw.loc[raw.feature_set == feature]):
            probabilities.append({
                "feature_set": feature, "fold": fold, "attribute": attr, "group": group,
                **measure(rows.Attrition.to_numpy(), rows.probability.to_numpy(), protocol),
            })
        for capacity in CAPACITIES:
            subset = policy.loc[(policy.feature_set == feature)
                                & (policy.capacity_percent == capacity)]
            for fold, attr, group, rows in _subsets(subset):
                operations.append({
                    "feature_set": feature, "capacity_percent": capacity, "fold": fold,
                    "attribute": attr, "group": group,
                    **measure(rows.Attrition.to_numpy(), rows.probability.to_numpy(),
                              protocol, rows.prediction.to_numpy()),
                })
    prob, ops = pd.DataFrame(probabilities), pd.DataFrame(operations)
    # Each demographic partition must reconcile, without counting attributes as extra employees.
    for table, keys in ((prob, ["feature_set", "fold"]),
                        (ops, ["feature_set", "capacity_percent", "fold"])):
        counts = ["n", "positives", "negatives"] + (
            ["alerts", "tp", "fp", "fn", "tn"] if "alerts" in table else []
        )
        for _, scope in table.groupby(keys):
            overall = scope.loc[scope.attribute == "Overall", counts].iloc[0].to_numpy()
            for attr in ("AgeGroup", "Gender", "MaritalStatus"):
                if not np.array_equal(scope.loc[scope.attribute == attr, counts].sum(), overall):
                    raise ValueError("Subgroup totals do not reconcile.")
    disparities = []
    thresholds = protocol["screening_thresholds"]
    for (feature_key, capacity_key, fold_key, attr_key), scope in ops.loc[
        ops.attribute != "Overall"
    ].groupby(["feature_set", "capacity_percent", "fold", "attribute"]):
        row: dict[str, Any] = {
            "feature_set": feature_key, "capacity_percent": capacity_key, "fold": fold_key,
            "attribute": attr_key, "groups": len(scope), "supported_groups":
                int(scope.support.eq("supported").sum()),
        }
        for metric in RATES:
            values = scope[metric].dropna().to_numpy(dtype=float)
            row[f"{metric}_gap"] = (
                float(np.ptp(values)) if len(values) == len(scope) and len(values) > 1 else None
            )
        rates = scope.alert_rate.dropna().to_numpy(dtype=float)
        row["selection_rate_ratio"] = (
            float(rates.min() / rates.max()) if len(rates) > 1 and rates.max() else None
        )
        raw_scope = prob.loc[(prob.feature_set == feature_key) & (prob.fold == fold_key)
                             & (prob.attribute == attr_key)]
        row["max_absolute_calibration_gap"] = float(raw_scope.calibration_gap.abs().max())
        supported = scope.support.eq("supported").all()
        defined = all(row[k] is not None for k in ("tpr_gap", "fpr_gap", "selection_rate_ratio"))
        flags = []
        if defined:
            for metric, key in (("tpr_gap", "tpr_difference"), ("fpr_gap", "fpr_difference"),
                                ("max_absolute_calibration_gap", "calibration_gap_absolute")):
                if row[metric] > thresholds[key]:
                    flags.append(metric)
            if row["selection_rate_ratio"] < thresholds["selection_rate_ratio"]:
                flags.append("selection_rate_ratio")
        row["descriptive_flags"] = ";".join(flags)
        row["status"] = ("INSUFFICIENT_EVIDENCE" if not supported or not defined else
                         "REVIEW" if flags else "NO_SCREENING_FLAG_NOT_APPROVAL")
        row["support"] = "supported" if supported else "insufficient evidence"
        disparities.append(row)
    paired: list[dict[str, Any]] = []
    paired_raw: list[dict[str, Any]] = []
    for table, keys, metrics, destination in (
        (prob, ["fold", "attribute", "group"], PROB_METRICS, paired_raw),
        (ops, ["capacity_percent", "fold", "attribute", "group"], (*RATES, "tp", "fp", "fn"),
         paired),
    ):
        left = table.loc[table.feature_set == FEATURES[0]].set_index(keys)
        right = table.loc[table.feature_set == FEATURES[1]].set_index(keys)
        for key in left.index:
            a, b = left.loc[key], right.loc[key]
            row = dict(zip(keys, key, strict=True))
            row["support"] = a["support"]
            for metric in metrics:
                row[f"delta_{metric}"] = (
                    float(b[metric]) - float(a[metric])
                    if pd.notna(a[metric]) and pd.notna(b[metric]) else None
                )
            if "capacity_percent" in row:
                selected = policy.loc[policy.capacity_percent == row["capacity_percent"]]
                if row["fold"]:
                    selected = selected.loc[selected.fold == row["fold"]]
                selected = selected.loc[selected[row["attribute"]] == row["group"]]
                x = selected.loc[selected.feature_set == FEATURES[0]].sort_values("source_row")
                y = selected.loc[selected.feature_set == FEATURES[1]].sort_values("source_row")
                px, py = x.prediction.to_numpy() == 1, y.prediction.to_numpy() == 1
                positive = x.Attrition.to_numpy() == 1
                row.update(
                    common_alerts=int(np.sum(px & py)), common_tp=int(np.sum(px & py & positive)),
                    only_16_tp=int(np.sum(px & ~py & positive)),
                    only_17_tp=int(np.sum(~px & py & positive)),
                )
            destination.append(row)
    gaps = pd.DataFrame(disparities)
    paired_table, paired_prob = pd.DataFrame(paired), pd.DataFrame(paired_raw)
    return {
        "probability": prob, "policy": ops, "disparities": gaps,
        "paired_policy": paired_table, "paired_probability": paired_prob,
        "probability_stability": _stability(
            prob, ["feature_set", "attribute", "group"], PROB_METRICS),
        "policy_stability": _stability(
            ops, ["feature_set", "capacity_percent", "attribute", "group"], RATES),
        "disparity_stability": _stability(
            gaps, ["feature_set", "capacity_percent", "attribute"],
            ("tpr_gap", "fpr_gap", "alert_rate_gap")),
        "paired_stability": _stability(
            paired_table, ["capacity_percent", "attribute", "group"],
            ("delta_tp", "delta_tpr", "delta_fpr", "delta_precision")),
    }
