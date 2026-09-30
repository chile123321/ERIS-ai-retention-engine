"""Development-only nested Logistic search on an immutable external outer assignment."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, ParameterGrid, StratifiedKFold

from eris_ml.evaluation.feature_ablation import METRICS, probability_metrics, validate_definitions
from eris_ml.evaluation.model_comparison import IDS, validate_reference
from eris_ml.features.definitions import FeatureDefinition
from eris_ml.models.training import build_model_pipeline
from eris_ml.models.tuning import validate_search_config

FEATURES = ("v2_general_16", "v2_no_department_17")


@dataclass
class NestedResult:
    oof: pd.DataFrame
    summary: pd.DataFrame
    folds: pd.DataFrame
    selected: pd.DataFrame
    search: pd.DataFrame
    inner_assignments: pd.DataFrame
    deltas: pd.DataFrame


def select_best(results: Mapping[str, Any]) -> int:
    """Predeclared numerical tie: within 1e-12 of max AP, smaller C then L2."""
    scores = np.asarray(results["mean_test_score"], dtype=float)
    if not np.isfinite(scores).all():
        raise ValueError("Non-finite inner search score.")
    tied = np.flatnonzero(scores >= scores.max() - 1e-12)
    return int(min(tied, key=lambda i: (
        results["params"][i]["model__C"], results["params"][i]["model__penalty"] != "l2",
    )))


def make_inner_splits(target: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """Indices local to outer-train; never receive outer-validation rows."""
    if min(np.count_nonzero(target == value) for value in (0, 1)) < 4:
        raise ValueError("Each outer-training class needs at least four rows.")
    splits = list(StratifiedKFold(n_splits=4, shuffle=True, random_state=43).split(
        np.zeros(len(target)), target))
    seen = np.zeros(len(target), dtype=int)
    for train, validation in splits:
        if np.intersect1d(train, validation).size or len(train) + len(validation) != len(target):
            raise ValueError("Invalid inner partition.")
        seen[validation] += 1
    if not np.all(seen == 1):
        raise ValueError("Incomplete inner validation coverage.")
    return splits


def validate_nested_oof(oof: pd.DataFrame, assignment: pd.DataFrame) -> None:
    expected = {(feature, model) for feature in FEATURES for model in ("baseline", "tuned")}
    if (set(oof.columns) != {*IDS, "feature_set", "model", "probability"}
        or oof.isna().any().any()
        or set(zip(oof.feature_set, oof.model, strict=True)) != expected):
        raise ValueError("Incomplete nested OOF candidates/schema.")
    identity = assignment[IDS].sort_values("source_row").reset_index(drop=True)
    for _, part in oof.groupby(["feature_set", "model"]):
        actual = part[IDS].sort_values("source_row").reset_index(drop=True)
        if (not part.source_row.is_unique or not part.EmployeeNumber.is_unique
            or not actual.equals(identity)):
            raise ValueError("Nested OOF row/target/fold mismatch.")
        probability_metrics(part.Attrition.to_numpy(), part.probability.to_numpy())


def metric_tables(oof: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summaries, folds = [], []
    for (feature, model), rows in oof.groupby(["feature_set", "model"], sort=False):
        local = []
        for fold, part in rows.groupby("fold"):
            row = {"feature_set": feature, "model": model, "fold": fold,
                   **probability_metrics(part.Attrition.to_numpy(), part.probability.to_numpy())}
            local.append(row)
            folds.append(row)
        summary = {"feature_set": feature, "model": model,
                   **probability_metrics(rows.Attrition.to_numpy(), rows.probability.to_numpy())}
        for metric in METRICS:
            values = [row[metric] for row in local]
            summary[f"{metric}_mean"] = float(np.mean(values))
            summary[f"{metric}_std"] = float(np.std(values, ddof=1))
        summaries.append(summary)
    summary_frame, fold_frame = pd.DataFrame(summaries), pd.DataFrame(folds)
    indexed = summary_frame.set_index(["feature_set", "model"])
    fold_index = fold_frame.set_index(["feature_set", "model", "fold"])
    pairs = [(name, "tuned", name, "baseline") for name in FEATURES]
    pairs.append((FEATURES[0], "tuned", FEATURES[1], "tuned"))
    deltas = []
    for feature, model, reference, reference_model in pairs:
        for metric in METRICS:
            values = [float(fold_index.loc[(feature, model, fold), metric]
                            - fold_index.loc[(reference, reference_model, fold), metric])
                      for fold in range(1, 6)]
            deltas.append({
                "feature_set": feature, "model": model, "reference": reference,
                "reference_model": reference_model, "metric": metric,
                "pooled_delta": indexed.loc[(feature, model), metric]
                - indexed.loc[(reference, reference_model), metric],
                "improved_folds": sum(v < 0 if metric == "brier_score" else v > 0 for v in values),
                **{f"fold_{i}": value for i, value in enumerate(values, 1)},
            })
    return summary_frame, fold_frame, pd.DataFrame(deltas)


def run_nested(
    development: pd.DataFrame, definitions: Mapping[str, FeatureDefinition],
    assignment: pd.DataFrame, baseline: pd.DataFrame, search_config: dict[str, Any],
    progress: Callable[[str], None] | None = None,
) -> NestedResult:
    validate_definitions(definitions)
    validate_reference(development, assignment, baseline)
    validate_search_config(search_config, "logistic_regression")
    target, outer = development.Attrition.to_numpy(), assignment.fold.to_numpy()
    tables, selected, search_rows, inner_rows = [], [], [], []
    # Construct inner assignments once per outer fold, shared between feature sets.
    splits = {}
    for fold in range(1, 6):
        train = np.flatnonzero(outer != fold)
        splits[fold] = make_inner_splits(target[train])
        for inner_fold, (_, validation) in enumerate(splits[fold], 1):
            part = assignment.iloc[train[validation]][IDS[:3]].copy()
            part["outer_fold"], part["inner_fold"] = fold, inner_fold
            inner_rows.append(part)
    for feature in FEATURES:
        features = development.loc[:, list(definitions[feature].all_features)].copy()
        base = baseline.loc[(baseline.feature_set == feature)
                            & (baseline.model == "logistic_regression")].copy()
        base["model"] = "baseline"
        tables.append(base)
        scores = np.full(len(development), np.nan)
        seen = np.zeros(len(development), dtype=int)
        for fold in range(1, 6):
            if progress:
                progress(f"{feature}, outer fold {fold}/5: inner 4-fold search")
            train, validation = np.flatnonzero(outer != fold), np.flatnonzero(outer == fold)
            pipeline = build_model_pipeline(
                definitions[feature], LogisticRegression(**search_config["fixed_parameters"]))
            grid = {f"model__{key}": values
                    for key, values in search_config["search_space"].items()}
            search = GridSearchCV(pipeline, grid, scoring="average_precision", cv=splits[fold],
                                  refit=select_best, n_jobs=1, error_score="raise")
            search.fit(features.iloc[train], target[train])
            best = search.best_index_
            params = search.best_params_
            selected.append({"feature_set": feature, "outer_fold": fold,
                             "C": params["model__C"], "penalty": params["model__penalty"],
                             "inner_average_precision": float(search.cv_results_[
                                 "mean_test_score"][best]), "candidate_index": best})
            for i, parameters in enumerate(ParameterGrid(grid)):
                search_rows.append({
                    "feature_set": feature, "outer_fold": fold, "candidate_index": i,
                    "C": parameters["model__C"], "penalty": parameters["model__penalty"],
                    "selected": i == best,
                    "mean_inner_ap": float(search.cv_results_["mean_test_score"][i]),
                    **{f"inner_{j + 1}_ap": float(search.cv_results_[f"split{j}_test_score"][i])
                       for j in range(4)},
                })
            positive = np.flatnonzero(np.asarray(search.classes_) == 1)
            if len(positive) != 1:
                raise ValueError("No unique positive class.")
            scores[validation] = search.predict_proba(features.iloc[validation])[:, positive[0]]
            seen[validation] += 1
        if not np.all(seen == 1):
            raise ValueError("Incomplete outer prediction coverage.")
        table = assignment[IDS].copy()
        table["feature_set"], table["model"], table["probability"] = feature, "tuned", scores
        tables.append(table)
    oof = pd.concat(tables, ignore_index=True)
    validate_nested_oof(oof, assignment)
    summary, folds, deltas = metric_tables(oof)
    return NestedResult(oof, summary, folds, pd.DataFrame(selected), pd.DataFrame(search_rows),
                        pd.concat(inner_rows, ignore_index=True), deltas)
