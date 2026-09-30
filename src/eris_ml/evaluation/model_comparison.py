"""Fixed tree baselines on persisted V2 folds, with unmodified Logistic references."""

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone

from eris_ml.evaluation.feature_ablation import (
    METRICS,
    REMOVED,
    AblationResult,
    probability_metrics,
    validate_definitions,
    validate_fold_assignment,
    validate_oof,
)
from eris_ml.features.definitions import FeatureDefinition
from eris_ml.models.factory import create_estimator
from eris_ml.models.training import build_model_pipeline

TREE_MODELS = ("random_forest", "xgboost")
ALL_MODELS = ("logistic_regression", *TREE_MODELS)
IDS = ["source_row", "EmployeeNumber", "Attrition", "fold"]


def validate_reference(
    development: pd.DataFrame, assignments: pd.DataFrame, baseline: pd.DataFrame,
) -> None:
    """Reject drift before fitting; never generate a replacement fold assignment."""
    if (set(assignments.columns) != {*IDS, "development_position"}
        or assignments.isna().any().any() or assignments.columns.has_duplicates
        or not assignments.source_row.is_unique or not assignments.EmployeeNumber.is_unique
        or not np.array_equal(assignments.development_position, np.arange(len(development)))
        or not assignments[IDS[:3]].reset_index(drop=True).equals(
            development[IDS[:3]].reset_index(drop=True))):
        raise ValueError("Saved assignment differs from development IDs/target/order.")
    validate_fold_assignment(development.Attrition.to_numpy(), assignments.fold.to_numpy())
    validate_oof(baseline, assignments)


def validate_comparison_oof(oof: pd.DataFrame, assignments: pd.DataFrame) -> None:
    """Reuse the baseline validator for each tree plus unchanged Logistic reference."""
    if set(oof.model) != set(ALL_MODELS):
        raise ValueError("Incomplete model comparison.")
    for model in TREE_MODELS:
        pair = oof.loc[oof.model.isin(["logistic_regression", model])].copy()
        pair.loc[pair.model == model, "model"] = "dummy_prior"
        validate_oof(pair, assignments)


def summarize(oof: pd.DataFrame, assignments: pd.DataFrame) -> AblationResult:
    """Recompute comparable pooled and fold metrics from probabilities, including Logistic."""
    validate_comparison_oof(oof, assignments)
    folds, summaries = [], []
    for (feature, model), rows in oof.groupby(["feature_set", "model"], sort=False):
        row: dict[str, Any] = {
            "feature_set": feature, "model": model,
            **probability_metrics(rows.Attrition.to_numpy(), rows.probability.to_numpy()),
        }
        local = []
        for fold, part in rows.groupby("fold"):
            metrics = probability_metrics(part.Attrition.to_numpy(), part.probability.to_numpy())
            record = {"feature_set": feature, "model": model, "fold": fold,
                      "validation_rows": len(part), **metrics}
            local.append(record)
            folds.append(record)
        for metric in METRICS:
            values = [part[metric] for part in local]
            row[f"{metric}_mean"] = float(np.mean(values))
            row[f"{metric}_std"] = float(np.std(values, ddof=1))
        summaries.append(row)
    return AblationResult(assignments.copy(), oof, pd.DataFrame(folds), pd.DataFrame(summaries))


def compare_models(
    development: pd.DataFrame, definitions: Mapping[str, FeatureDefinition],
    assignments: pd.DataFrame, baseline: pd.DataFrame, configs: Mapping[str, dict[str, Any]],
) -> AblationResult:
    """Forty train-only fits; saved Logistic probabilities are never refitted."""
    validate_definitions(definitions)
    validate_reference(development, assignments, baseline)
    if set(configs) != set(TREE_MODELS):
        raise ValueError("Exactly two fixed tree configs are required.")
    estimators = {}
    for model, config in configs.items():
        if config.get("model") != model or config.get("random_seed") != 42:
            raise ValueError("Model identity/seed mismatch.")
        estimators[model] = create_estimator(config)  # factory enforces V1 baseline parameters
    tables = [baseline.loc[baseline.model == "logistic_regression"].copy()]
    target, shared = development.Attrition.to_numpy(), assignments.fold.to_numpy()
    for name in REMOVED:
        definition = definitions[name]
        features = development.loc[:, list(definition.all_features)].copy()
        for model in TREE_MODELS:
            template = build_model_pipeline(definition, estimators[model])
            scores = np.full(len(development), np.nan)
            seen = np.zeros(len(development), dtype=int)
            for fold in range(1, 6):
                train, validation = np.flatnonzero(shared != fold), np.flatnonzero(shared == fold)
                pipeline = clone(template)
                pipeline.fit(features.iloc[train], target[train])
                positive = np.flatnonzero(np.asarray(pipeline.classes_) == 1)
                if len(positive) != 1:
                    raise ValueError("Expected one positive-class probability column.")
                prediction = pipeline.predict_proba(features.iloc[validation])[:, positive[0]]
                scores[validation] = prediction
                seen[validation] += 1
            if not np.all(seen == 1):
                raise ValueError("OOF coverage is not exactly once.")
            table = assignments[IDS].copy()
            table["feature_set"], table["model"], table["probability"] = name, model, scores
            tables.append(table)
    return summarize(pd.concat(tables, ignore_index=True), assignments)


def comparison_deltas(result: AblationResult) -> pd.DataFrame:
    """Pooled and paired-fold deltas for feature removal and model-family comparisons."""
    summary = result.summary.set_index(["feature_set", "model"])
    fold_index = result.folds.set_index(["feature_set", "model", "fold"])
    pairs = []
    for model in ALL_MODELS:
        for feature in REMOVED:
            if feature != "v2_full_18":
                pairs.append((feature, model, "v2_full_18", model))
        for reference in ("v2_no_department_17", "v2_no_jobrole_17"):
            pairs.append(("v2_general_16", model, reference, model))
    for model in TREE_MODELS:
        pairs.extend((feature, model, feature, "logistic_regression") for feature in REMOVED)
    records = []
    for feature, model, ref_feature, ref_model in pairs:
        for metric in METRICS:
            changes = np.array([
                fold_index.loc[(feature, model, fold), metric]
                - fold_index.loc[(ref_feature, ref_model, fold), metric] for fold in range(1, 6)
            ])
            records.append({
                "feature_set": feature, "model": model, "reference_feature": ref_feature,
                "reference_model": ref_model, "metric": metric,
                "pooled_delta": summary.loc[(feature, model), metric]
                - summary.loc[(ref_feature, ref_model), metric],
                "improved_folds": int(np.sum(changes < 0 if metric == "brier_score"
                                              else changes > 0)),
                **{f"fold_{i}": float(value) for i, value in enumerate(changes, 1)},
            })
    return pd.DataFrame(records)
