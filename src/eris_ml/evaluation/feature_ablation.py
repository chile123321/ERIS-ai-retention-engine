"""Fixed-baseline V2 ablation on one shared, validated development fold assignment."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from eris_ml.features.definitions import FeatureDefinition, validate_feature_columns
from eris_ml.models.factory import create_estimator
from eris_ml.models.training import build_model_pipeline

METRICS = ("average_precision", "roc_auc", "brier_score")
MODELS = ("dummy_prior", "logistic_regression")
REMOVED = {
    "v2_full_18": set(),
    "v2_no_department_17": {"Department"},
    "v2_no_jobrole_17": {"JobRole"},
    "v2_general_16": {"Department", "JobRole"},
}
GROUPS = {
    "nominal": ("Department", "JobRole", "OverTime"),
    "ordinal": ("Education", "EnvironmentSatisfaction", "JobInvolvement", "JobLevel",
                "JobSatisfaction", "PerformanceRating"),
    "numeric": ("Age", "DistanceFromHome", "MonthlyIncome", "TotalWorkingYears",
                "TrainingTimesLastYear", "YearsAtCompany", "YearsInCurrentRole",
                "YearsSinceLastPromotion", "YearsWithCurrManager"),
}
LOGISTIC_PARAMETERS = {
    "solver": "liblinear", "C": 1.0, "max_iter": 2000, "class_weight": None,
}


@dataclass
class AblationResult:
    """Tables only; fitted estimators never leave the fold loop."""

    assignments: pd.DataFrame
    oof: pd.DataFrame
    folds: pd.DataFrame
    summary: pd.DataFrame


def validate_definitions(definitions: Mapping[str, FeatureDefinition]) -> None:
    """Fail before fitting on count, name, type-group, or exclusion drift."""
    if set(definitions) != set(REMOVED):
        raise ValueError("Expected the four locked V2 feature sets.")
    for name, definition in definitions.items():
        if definition.version != name:
            raise ValueError(f"Feature-set version mismatch: {name}.")
        for kind, full in GROUPS.items():
            expected = tuple(column for column in full if column not in REMOVED[name])
            if getattr(definition, kind) != expected:
                raise ValueError(f"Locked feature group changed: {name}/{kind}.")
        if (not REMOVED[name].issubset(definition.excluded)
            or set(definition.all_features) & set(definition.non_features)):
            raise ValueError(f"Excluded features entered model: {name}.")


def validate_fold_assignment(target: np.ndarray, assignment: np.ndarray) -> None:
    """Prove exactly one validation fold per row and complete disjoint partitions."""
    if (target.ndim != 1 or assignment.shape != target.shape or len(target) == 0
        or set(np.unique(target)) != {0, 1}
        or assignment.dtype.kind not in "iu" or set(np.unique(assignment)) != set(range(1, 6))):
        raise ValueError("Invalid binary target or five-fold assignment.")
    seen = np.zeros(len(target), dtype=int)
    for fold in range(1, 6):
        train = np.flatnonzero(assignment != fold)
        validation = np.flatnonzero(assignment == fold)
        if (np.intersect1d(train, validation).size
            or len(train) + len(validation) != len(target)
            or set(np.unique(target[train])) != {0, 1}
            or set(np.unique(target[validation])) != {0, 1}):
            raise ValueError("Fold leakage, incomplete partition or missing target class.")
        seen[validation] += 1
    if not np.all(seen == 1):
        raise ValueError("Every row must appear exactly once in validation.")


def make_shared_folds(target: np.ndarray) -> np.ndarray:
    """Generate once in original development row order, independent of feature set."""
    if (target.ndim != 1 or set(np.unique(target)) != {0, 1}
        or min(np.count_nonzero(target == value) for value in (0, 1)) < 5):
        raise ValueError("Five stratified folds require at least five rows per class.")
    assignment = np.zeros(len(target), dtype=int)
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (_, validation) in enumerate(splitter.split(np.zeros(len(target)), target), 1):
        assignment[validation] = fold
    validate_fold_assignment(target, assignment)
    return assignment


def probability_metrics(target: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    """PR-AUC means average precision; no operational threshold is selected."""
    if (scores.shape != target.shape or not np.isfinite(scores).all()
        or ((scores < 0) | (scores > 1)).any()):
        raise ValueError("Invalid OOF probabilities.")
    return {
        "average_precision": float(average_precision_score(target, scores)),
        "roc_auc": float(roc_auc_score(target, scores)),
        "brier_score": float(brier_score_loss(target, scores)),
    }


def validate_oof(oof: pd.DataFrame, assignments: pd.DataFrame) -> None:
    """Reject missing/duplicate rows, candidates, IDs, targets, folds or invalid scores."""
    expected_columns = {
        "source_row", "EmployeeNumber", "Attrition", "fold", "feature_set", "model",
        "probability",
    }
    expected_candidates = {(feature, model) for feature in REMOVED for model in MODELS}
    if (set(oof.columns) != expected_columns or oof.isna().any().any()
        or len(oof) != len(assignments) * len(expected_candidates)
        or set(zip(oof["feature_set"], oof["model"], strict=True)) != expected_candidates):
        raise ValueError("Incomplete OOF table/candidate coverage.")
    columns = ["source_row", "EmployeeNumber", "Attrition", "fold"]
    expected = assignments[columns].sort_values("source_row").reset_index(drop=True)
    for _, rows in oof.groupby(["feature_set", "model"]):
        if (len(rows) != len(assignments) or not rows.source_row.is_unique
            or not rows.EmployeeNumber.is_unique):
            raise ValueError("Missing or duplicate OOF row IDs.")
        actual = rows[columns].sort_values("source_row").reset_index(drop=True)
        if not actual.equals(expected):
            raise ValueError("OOF ID/target/fold alignment differs from shared assignment.")
        probability_metrics(rows.Attrition.to_numpy(), rows.probability.to_numpy())


def evaluate_feature_ablation(
    development: pd.DataFrame, definitions: Mapping[str, FeatureDefinition],
    logistic_config: dict[str, Any],
) -> AblationResult:
    """Fit 4 x 2 x 5 fresh pipelines using only the columns and rows of each fold."""
    validate_definitions(definitions)
    if (logistic_config.get("model") != "logistic_regression"
        or logistic_config.get("random_seed") != 42
        or logistic_config.get("parameters") != LOGISTIC_PARAMETERS):
        raise ValueError("Use the unchanged, unweighted V1 Logistic baseline config.")
    required = {name for columns in GROUPS.values() for name in columns}
    required |= {"source_row", "EmployeeNumber", "Attrition"}
    if (development.columns.has_duplicates or not required.issubset(development.columns)
        or development[["source_row", "EmployeeNumber", "Attrition"]].isna().any().any()
        or not development.source_row.is_unique or not development.EmployeeNumber.is_unique):
        raise ValueError("Development schema or stable IDs are invalid.")
    target = development.Attrition.to_numpy()
    shared = make_shared_folds(target)
    assignments = development[["source_row", "EmployeeNumber", "Attrition"]].copy()
    assignments["fold"] = shared
    assignments["development_position"] = np.arange(len(development))
    all_oof: list[pd.DataFrame] = []
    folds: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    configs = {
        "dummy_prior": {"model": "dummy", "parameters": {"strategy": "prior"}},
        "logistic_regression": logistic_config,
    }
    for name in REMOVED:
        definition = definitions[name]
        # Project before any preprocessing or fit, including Dummy's pipeline.
        features = development.loc[:, list(definition.all_features)].copy()
        validate_feature_columns(features.columns, definition)
        for model, config in configs.items():
            template = build_model_pipeline(definition, create_estimator(config))
            scores = np.full(len(development), np.nan)
            seen = np.zeros(len(development), dtype=int)
            candidate_folds: list[dict[str, Any]] = []
            for fold in range(1, 6):
                train = np.flatnonzero(shared != fold)
                validation = np.flatnonzero(shared == fold)
                pipeline = clone(template)
                pipeline.fit(features.iloc[train], target[train])
                positive = np.flatnonzero(np.asarray(pipeline.classes_) == 1)
                if len(positive) != 1:
                    raise ValueError("Classifier has no unique positive-class column.")
                prediction = pipeline.predict_proba(features.iloc[validation])[:, positive[0]]
                scores[validation] = prediction
                seen[validation] += 1
                row = {
                    "feature_set": name, "model": model, "fold": fold,
                    "train_rows": len(train), "validation_rows": len(validation),
                    "train_positives": int(target[train].sum()),
                    "validation_positives": int(target[validation].sum()),
                    **probability_metrics(target[validation], prediction),
                }
                candidate_folds.append(row)
                folds.append(row)
            if not np.all(seen == 1):
                raise ValueError("OOF prediction coverage is not exactly once.")
            summary: dict[str, Any] = {
                "feature_set": name, "feature_count": len(definition.all_features),
                "model": model, **probability_metrics(target, scores),
            }
            for metric in METRICS:
                values = [row[metric] for row in candidate_folds]
                summary[f"{metric}_mean"] = float(np.mean(values))
                summary[f"{metric}_std"] = float(np.std(values, ddof=1))
                summary[f"{metric}_min"] = float(np.min(values))
                summary[f"{metric}_max"] = float(np.max(values))
            summaries.append(summary)
            table = assignments.drop(columns="development_position").copy()
            table["feature_set"], table["model"], table["probability"] = name, model, scores
            all_oof.append(table)
    oof = pd.concat(all_oof, ignore_index=True)
    validate_oof(oof, assignments)
    summary_table = pd.DataFrame(summaries)
    for model in MODELS:
        reference = summary_table.loc[
            (summary_table.feature_set == "v2_full_18") & (summary_table.model == model)
        ].iloc[0]
        mask = summary_table.model == model
        for metric in METRICS:
            summary_table.loc[mask, f"{metric}_delta_vs_full"] = (
                summary_table.loc[mask, metric] - reference[metric]
            )
    return AblationResult(assignments, oof, pd.DataFrame(folds), summary_table)
