"""Cross-fitted calibration of fixed V2 Logistic baselines, never tuned estimators."""

from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold

from eris_ml.evaluation.feature_ablation import (
    LOGISTIC_PARAMETERS,
    probability_metrics,
    validate_definitions,
)
from eris_ml.evaluation.model_comparison import IDS, validate_reference
from eris_ml.features.definitions import FeatureDefinition
from eris_ml.models.factory import create_estimator
from eris_ml.models.training import build_model_pipeline

FEATURES = ("v2_general_16", "v2_no_department_17")
METHODS = ("raw", "sigmoid", "isotonic")
PROTOCOL: dict[str, Any] = {
    "protocol_version": "logistic-baseline-calibration-v2", "feature_sets": list(FEATURES),
    "base_config": "configs/models/logistic_regression.yaml",
    "outer_assignment": "artifacts/predictions/baseline_feature_ablation_v2_assignments.csv",
    "calibration_cv": {"n_splits": 4, "shuffle": True, "random_state": 43},
    "methods": list(METHODS), "ensemble": False, "n_jobs": 1,
    "raw_regression": {"rtol": 1e-7, "atol": 1e-9},
    "reliability": {"bins": 10, "strategy": "quantile", "low_count": 30, "low_positives": 10},
    "selection": {
        "max_auc_drop": .005, "minimum_improved_folds": {"sigmoid": 3, "isotonic": 4},
        "max_fold_brier_increase": .005,
        "positive_brier_gain_after_leaving_any_fold_out": True, "sigmoid_tie_tolerance": .0001,
    },
    "final_test_used": False, "production_approved": False,
}


def validate_protocol(protocol: dict[str, Any]) -> None:
    if protocol != PROTOCOL:
        raise ValueError("Calibration V2 protocol differs from the locked baseline-only protocol.")


def calibration_splits(target: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    if min(np.count_nonzero(target == value) for value in (0, 1)) < 4:
        raise ValueError("Insufficient outer-training rows per class for calibration CV.")
    splits = list(StratifiedKFold(n_splits=4, shuffle=True, random_state=43).split(
        np.zeros(len(target)), target))
    seen = np.zeros(len(target), dtype=int)
    for train, validation in splits:
        if np.intersect1d(train, validation).size or len(train) + len(validation) != len(target):
            raise ValueError("Invalid calibration partition.")
        seen[validation] += 1
    if not np.all(seen == 1):
        raise ValueError("Incomplete calibration OOF coverage.")
    return splits


def check_raw(previous: np.ndarray, current: np.ndarray) -> float:
    if (previous.shape != current.shape or not np.isfinite(current).all()
        or not np.allclose(previous, current, rtol=1e-7, atol=1e-9)):
        raise ValueError("Raw regression failed: stop calibration comparison; baseline preserved.")
    return float(np.max(np.abs(previous - current)))


def validate_calibration_oof(oof: pd.DataFrame, assignment: pd.DataFrame) -> None:
    expected = {(feature, method) for feature in FEATURES for method in METHODS}
    if (set(oof.columns) != {*IDS, "feature_set", "method", "probability"}
        or oof.isna().any().any()
        or set(zip(oof.feature_set, oof.method, strict=True)) != expected):
        raise ValueError("Incomplete calibration OOF schema/candidate coverage.")
    identity = assignment[IDS].sort_values("source_row").reset_index(drop=True)
    for _, part in oof.groupby(["feature_set", "method"]):
        if (not part.source_row.is_unique or not part.EmployeeNumber.is_unique
            or not part[IDS].sort_values("source_row").reset_index(drop=True).equals(identity)):
            raise ValueError("Calibration OOF ID/target/fold mismatch.")
        probability_metrics(part.Attrition.to_numpy(), part.probability.to_numpy())


def run_calibration(
    development: pd.DataFrame, definitions: Mapping[str, FeatureDefinition],
    assignment: pd.DataFrame, baseline: pd.DataFrame, logistic: dict[str, Any],
    progress: Callable[[str], None] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    validate_definitions(definitions)
    validate_reference(development, assignment, baseline)
    if (logistic.get("model") != "logistic_regression" or logistic.get("random_seed") != 42
        or logistic.get("parameters") != LOGISTIC_PARAMETERS):
        raise ValueError("Only the unchanged Logistic baseline is permitted, not tuned configs.")
    target, outer = development.Attrition.to_numpy(), assignment.fold.to_numpy()
    tables, regression, memberships = [], [], []
    splits = {}
    for fold in range(1, 6):
        train = np.flatnonzero(outer != fold)
        splits[fold] = calibration_splits(target[train])
        for inner_fold, (_, validation) in enumerate(splits[fold], 1):
            part = assignment.iloc[train[validation]][IDS[:3]].copy()
            part["outer_fold"], part["calibration_fold"] = fold, inner_fold
            memberships.append(part)
    # All raw folds must pass before any calibrator is fit.
    for feature in FEATURES:
        features = development.loc[:, list(definitions[feature].all_features)].copy()
        template = build_model_pipeline(definitions[feature], create_estimator(logistic))
        prior = baseline.loc[(baseline.feature_set == feature)
                             & (baseline.model == "logistic_regression")].set_index("source_row")
        previous = prior.loc[assignment.source_row, "probability"].to_numpy()
        scores = np.full(len(development), np.nan)
        for fold in range(1, 6):
            train, validation = np.flatnonzero(outer != fold), np.flatnonzero(outer == fold)
            raw = clone(template).fit(features.iloc[train], target[train])
            scores[validation] = raw.predict_proba(features.iloc[validation])[:, 1]
            difference = check_raw(previous[validation], scores[validation])
            regression.append({"feature_set": feature, "fold": fold,
                               "max_abs_difference": difference})
        table = assignment[IDS].copy()
        table["feature_set"], table["method"], table["probability"] = feature, "raw", scores
        tables.append(table)
    if progress:
        progress("All 10 raw regression checks passed; starting train-only calibration.")
    for feature in FEATURES:
        features = development.loc[:, list(definitions[feature].all_features)].copy()
        template = build_model_pipeline(definitions[feature], create_estimator(logistic))
        for method in ("sigmoid", "isotonic"):
            scores = np.full(len(development), np.nan)
            seen = np.zeros(len(development), dtype=int)
            for fold in range(1, 6):
                if progress:
                    progress(f"{feature}, {method}, outer fold {fold}/5")
                train, validation = np.flatnonzero(outer != fold), np.flatnonzero(outer == fold)
                calibrated = CalibratedClassifierCV(
                    estimator=clone(template), method=method, cv=splits[fold],
                    ensemble=False, n_jobs=1,
                )
                calibrated.fit(features.iloc[train], target[train])
                scores[validation] = calibrated.predict_proba(features.iloc[validation])[:, 1]
                seen[validation] += 1
            if not np.all(seen == 1):
                raise ValueError("Calibration outer-validation coverage must be exactly once.")
            table = assignment[IDS].copy()
            table["feature_set"], table["method"], table["probability"] = feature, method, scores
            tables.append(table)
    oof = pd.concat(tables, ignore_index=True)
    validate_calibration_oof(oof, assignment)
    return oof, pd.DataFrame(regression), pd.concat(memberships, ignore_index=True)
