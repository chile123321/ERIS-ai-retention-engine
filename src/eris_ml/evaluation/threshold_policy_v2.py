"""Cross-fitted capacity policies for fixed raw Logistic baselines, no new model selection."""

from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
import pandas as pd

from eris_ml.evaluation.feature_ablation import LOGISTIC_PARAMETERS, validate_definitions
from eris_ml.evaluation.model_comparison import IDS, validate_reference
from eris_ml.evaluation.thresholding import confusion_metrics, threshold_curve
from eris_ml.features.definitions import FeatureDefinition
from eris_ml.models.calibration_v2 import FEATURES, calibration_splits, check_raw
from eris_ml.models.factory import create_estimator
from eris_ml.models.training import build_model_pipeline

CAPACITIES = (5, 10, 15)
NO_ALERT = float(np.nextafter(1.0, np.inf))
PROTOCOL: dict[str, Any] = {
    "protocol_version": "logistic-raw-threshold-policy-v2",
    "feature_sets": list(FEATURES),
    "capacity_percent": list(CAPACITIES),
    "business_approved": False,
    "production_approved": False,
    "status": "hypothetical_capacity_analysis",
    "base_config": "configs/models/logistic_regression.yaml",
    "outer_assignment": "artifacts/predictions/baseline_feature_ablation_v2_assignments.csv",
    "inner_cv": {"n_splits": 4, "shuffle": True, "random_state": 43},
    "threshold": {
        "comparison": "probability_greater_than_or_equal",
        "candidates": "unique_inner_probabilities_plus_0_1_and_no_alert_sentinel",
        "no_alert_sentinel": "nextafter_1_toward_positive_infinity",
        "budget_rounding": "floor_integer_percent_times_rows",
        "selection_order": ["maximize_recall", "maximize_precision", "maximize_threshold"],
    },
    "top_k": {
        "batch": "each_outer_validation_fold",
        "k_rounding": "floor_integer_percent_times_rows",
        "order": ["probability_descending", "source_row_ascending"],
    },
    "reference_threshold": 0.5,
    "reviews_per_true_positive_when_tp_zero": "undefined",
    "cost_assumptions": "none",
    "final_test_used": False,
}


def budget(rows: int, percent: int) -> int:
    if type(percent) is not int or percent not in CAPACITIES or rows < 1:
        raise ValueError("Expected a positive batch and capacity 5, 10 or 15 percent.")
    return rows * percent // 100


def operational(target: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    if (
        target.ndim != 1
        or target.shape != prediction.shape
        or not len(target)
        or not set(np.unique(target)).issubset({0, 1})
        or not set(np.unique(prediction)).issubset({0, 1})
    ):
        raise ValueError("Aligned binary target/prediction vectors required.")
    tp = int(((target == 1) & (prediction == 1)).sum())
    fp = int(((target == 0) & (prediction == 1)).sum())
    fn = int(((target == 1) & (prediction == 0)).sum())
    tn = int(((target == 0) & (prediction == 0)).sum())
    metrics = confusion_metrics(tn, fp, fn, tp)
    metrics["reviews_per_tp"] = metrics["alerts"] / tp if tp else None
    metrics["reviews_per_tp_status"] = "defined" if tp else "undefined_no_tp"
    return metrics


def select_capacity(target: np.ndarray, probability: np.ndarray, percent: int) -> dict[str, Any]:
    """Use inner labels only; all ties at a threshold are included, never split."""
    limit = budget(len(target), percent)
    curve = threshold_curve(target, probability)
    # A finite sentinel above 1 represents an explicit no-alert policy, even for scores of 1.
    none = {
        "threshold": NO_ALERT,
        **confusion_metrics(int((target == 0).sum()), 0, int((target == 1).sum()), 0),
    }
    curve = pd.concat([curve, pd.DataFrame([none])], ignore_index=True)
    feasible = curve.loc[curve.alerts <= limit]
    best = feasible.sort_values(["recall", "precision", "threshold"], ascending=False).iloc[0]
    return {
        "threshold": float(best.threshold),
        "inner_budget": limit,
        "inner_alerts": int(best.alerts),
        "inner_recall": float(best.recall),
        "inner_precision": float(best.precision),
        "inner_alert_rate": float(best.alert_rate),
        "no_alert": bool(best.threshold > 1),
    }


def top_k(probability: np.ndarray, source_row: np.ndarray, percent: int) -> np.ndarray:
    """Batch rank policy; labels are deliberately absent from this interface."""
    if (
        probability.ndim != 1
        or probability.shape != source_row.shape
        or len(np.unique(source_row)) != len(source_row)
        or not np.isfinite(probability).all()
        or not np.isfinite(source_row).all()
        or ((probability < 0) | (probability > 1)).any()
    ):
        raise ValueError("Invalid ranking probabilities or stable row IDs.")
    selected = np.zeros(len(probability), dtype=int)
    order = np.lexsort((source_row, -probability))
    selected[order[: budget(len(probability), percent)]] = 1
    return selected


def validate_inputs(
    development: pd.DataFrame,
    assignment: pd.DataFrame,
    baseline: pd.DataFrame,
    calibration_raw: pd.DataFrame,
) -> pd.DataFrame:
    validate_reference(development, assignment, baseline)
    if (
        set(calibration_raw.columns) != {*IDS, "feature_set", "method", "probability"}
        or set(calibration_raw.feature_set) != set(FEATURES)
        or set(calibration_raw.method) != {"raw"}
        or calibration_raw.isna().any().any()
    ):
        raise ValueError("Only the two calibration raw references are allowed.")
    references = []
    expected = assignment[IDS].sort_values("source_row").reset_index(drop=True)
    for feature in FEATURES:
        raw = calibration_raw.loc[calibration_raw.feature_set == feature].sort_values("source_row")
        if not raw[IDS].reset_index(drop=True).equals(expected):
            raise ValueError("Calibration raw IDs/targets/folds mismatch.")
        previous = baseline.loc[
            (baseline.feature_set == feature) & (baseline.model == "logistic_regression")
        ].sort_values("source_row")
        check_raw(previous.probability.to_numpy(), raw.probability.to_numpy())
        references.append(
            {
                "feature_set": feature,
                "max_abs_difference": float(
                    np.max(np.abs(previous.probability.to_numpy() - raw.probability.to_numpy()))
                ),
            }
        )
    return pd.DataFrame(references)


def validate_policy_oof(oof: pd.DataFrame, assignment: pd.DataFrame) -> None:
    expected = {
        (f, policy, c)
        for f in FEATURES
        for policy in ("inner_threshold", "top_k")
        for c in CAPACITIES
    }
    expected |= {(f, "reference_0.5", 0) for f in FEATURES}
    if (
        oof.isna().any().any()
        or set(zip(oof.feature_set, oof.policy, oof.capacity_percent, strict=True)) != expected
    ):
        raise ValueError("Incomplete policy OOF coverage.")
    identity = assignment[IDS].sort_values("source_row").reset_index(drop=True)
    for _, part in oof.groupby(["feature_set", "policy", "capacity_percent"]):
        if not part[IDS].sort_values("source_row").reset_index(drop=True).equals(identity):
            raise ValueError("Policy OOF ID/target/fold mismatch.")
        if (
            not np.isfinite(part.probability).all()
            or not part.probability.between(0, 1).all()
            or not part.prediction.isin([0, 1]).all()
        ):
            raise ValueError("Invalid policy OOF probability/prediction.")


def evaluate(
    development: pd.DataFrame,
    definitions: Mapping[str, FeatureDefinition],
    assignment: pd.DataFrame,
    baseline: pd.DataFrame,
    calibration_raw: pd.DataFrame,
    logistic: dict[str, Any],
    progress: Callable[[str], None] | None = None,
) -> dict[str, pd.DataFrame]:
    validate_definitions(definitions)
    regression = validate_inputs(development, assignment, baseline, calibration_raw)
    if (
        logistic.get("model") != "logistic_regression"
        or logistic.get("random_seed") != 42
        or logistic.get("parameters") != LOGISTIC_PARAMETERS
    ):
        raise ValueError("Only the unchanged Logistic baseline is permitted.")
    outer, target = assignment.fold.to_numpy(), development.Attrition.to_numpy()
    predictions, selections, inner_predictions = [], [], []
    for feature in FEATURES:
        features = development.loc[:, list(definitions[feature].all_features)].copy()
        previous = baseline.loc[
            (baseline.feature_set == feature) & (baseline.model == "logistic_regression")
        ].set_index("source_row")
        for fold in range(1, 6):
            if progress:
                progress(f"{feature}, outer fold {fold}/5: fixed baseline inner OOF")
            train, validation = np.flatnonzero(outer != fold), np.flatnonzero(outer == fold)
            inner_target = target[train]
            scores, seen = np.full(len(train), np.nan), np.zeros(len(train), dtype=int)
            inner_fold_ids = np.zeros(len(train), dtype=int)
            for inner_fold, (fit, held) in enumerate(calibration_splits(inner_target), 1):
                pipeline = build_model_pipeline(definitions[feature], create_estimator(logistic))
                pipeline.fit(features.iloc[train[fit]], inner_target[fit])
                scores[held] = pipeline.predict_proba(features.iloc[train[held]])[:, 1]
                seen[held] += 1
                inner_fold_ids[held] = inner_fold
            if not np.all(seen == 1) or not np.isfinite(scores).all():
                raise ValueError("Invalid inner OOF coverage.")
            inner = assignment.iloc[train][IDS[:3]].copy()
            inner["outer_fold"], inner["inner_fold"] = fold, inner_fold_ids
            inner["feature_set"], inner["probability"] = feature, scores
            inner_predictions.append(inner)
            # Policy selection sees only inner_target/scores. No outer labels are passed.
            policies = {
                percent: select_capacity(inner_target, scores, percent) for percent in CAPACITIES
            }
            for percent, choice in policies.items():
                selections.append(
                    {"feature_set": feature, "fold": fold, "capacity_percent": percent, **choice}
                )
            rows = assignment.iloc[validation][IDS].copy()
            probabilities = previous.loc[rows.source_row, "probability"].to_numpy()
            for policy, percent in [
                ("reference_0.5", 0),
                *(
                    (policy, percent)
                    for policy in ("inner_threshold", "top_k")
                    for percent in CAPACITIES
                ),
            ]:
                if policy == "top_k":
                    prediction = top_k(probabilities, rows.source_row.to_numpy(), percent)
                else:
                    threshold = 0.5 if percent == 0 else policies[percent]["threshold"]
                    prediction = (probabilities >= threshold).astype(int)
                table = rows.copy()
                table["feature_set"], table["policy"], table["capacity_percent"] = (
                    feature,
                    policy,
                    percent,
                )
                table["probability"], table["prediction"] = probabilities, prediction
                predictions.append(table)
    oof = pd.concat(predictions, ignore_index=True)
    validate_policy_oof(oof, assignment)
    return {
        "oof": oof,
        "selections": pd.DataFrame(selections),
        "regression": regression,
        "inner_oof": pd.concat(inner_predictions, ignore_index=True),
    }


def summarize(oof: pd.DataFrame, selections: pd.DataFrame) -> dict[str, pd.DataFrame]:
    summaries, folds = [], []
    keys = ["feature_set", "policy", "capacity_percent"]
    for (feature, policy, percent), rows in oof.groupby(keys, sort=False):
        local = []
        for fold, part in rows.groupby("fold"):
            metrics = operational(part.Attrition.to_numpy(), part.prediction.to_numpy())
            allowed = budget(len(part), percent) if percent else len(part)
            record = {
                "feature_set": feature,
                "policy": policy,
                "capacity_percent": percent,
                "fold": fold,
                "batch_rows": len(part),
                "budget_applicable": bool(percent),
                "budget_alerts": allowed,
                "budget_exceeded": bool(metrics["alerts"] > allowed),
                "excess_alerts": max(0, metrics["alerts"] - allowed),
                **metrics,
            }
            local.append(record)
            folds.append(record)
        summaries.append(
            {
                "feature_set": feature,
                "policy": policy,
                "capacity_percent": percent,
                **operational(rows.Attrition.to_numpy(), rows.prediction.to_numpy()),
                "exceeded_folds": sum(row["budget_exceeded"] for row in local),
                "excess_alerts": sum(row["excess_alerts"] for row in local),
                "sum_batch_budget": sum(row["budget_alerts"] for row in local),
            }
        )
    paired = []
    for (policy, percent), rows in oof.groupby(["policy", "capacity_percent"], sort=False):
        for fold in [0, 1, 2, 3, 4, 5]:
            part = rows if fold == 0 else rows.loc[rows.fold == fold]
            first = part.loc[part.feature_set == FEATURES[0]].sort_values("source_row")
            second = part.loc[part.feature_set == FEATURES[1]].sort_values("source_row")
            if not first[IDS].reset_index(drop=True).equals(second[IDS].reset_index(drop=True)):
                raise ValueError("Paired feature-set alignment failed.")
            y = first.Attrition.to_numpy()
            a, b = first.prediction.to_numpy(), second.prediction.to_numpy()
            ma, mb = operational(y, a), operational(y, b)
            paired.append(
                {
                    "policy": policy,
                    "capacity_percent": percent,
                    "fold": fold,
                    "jobrole_only_true_positives": int(((a == 0) & (b == 1) & (y == 1)).sum()),
                    "general_only_true_positives": int(((a == 1) & (b == 0) & (y == 1)).sum()),
                    **{
                        f"delta_{metric}_17_minus_16": mb[metric] - ma[metric]
                        for metric in ("tp", "fp", "fn", "tn", "alerts", "precision", "recall")
                    },
                }
            )
    stability = (
        selections.groupby(["feature_set", "capacity_percent"])
        .agg(
            threshold_mean=("threshold", "mean"),
            threshold_median=("threshold", "median"),
            threshold_std=("threshold", "std"),
            threshold_min=("threshold", "min"),
            threshold_max=("threshold", "max"),
        )
        .reset_index()
    )
    return {
        "summary": pd.DataFrame(summaries),
        "folds": pd.DataFrame(folds),
        "paired": pd.DataFrame(paired),
        "stability": stability,
    }
