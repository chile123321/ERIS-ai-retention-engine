"""Leakage-safe cross-validation interfaces."""

from collections.abc import Mapping
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold

from eris_ml.evaluation.metrics import calculate_metrics

SCALAR_METRICS = (
    "accuracy",
    "balanced_accuracy",
    "average_precision",
    "roc_auc",
    "precision",
    "recall",
    "f1",
    "f2",
    "brier_score",
)

DIAGNOSTIC_METRICS = (
    "train_average_precision",
    "validation_average_precision",
    "overfitting_gap",
    "fit_time_seconds",
    "prediction_time_seconds",
)


def _validated_cv_config(config: Mapping[str, Any]) -> tuple[int, bool, int]:
    allowed = {"n_splits", "shuffle", "random_seed"}
    unexpected = sorted(set(config) - allowed)
    if unexpected:
        raise ValueError(f"Unexpected cross-validation settings: {unexpected}.")
    n_splits = config.get("n_splits", 5)
    shuffle = config.get("shuffle", True)
    random_seed = config.get("random_seed", 42)
    if n_splits != 5 or shuffle is not True or random_seed != 42:
        raise ValueError("Baseline CV must use 5 folds, shuffle=True, and random_seed=42.")
    return int(n_splits), bool(shuffle), int(random_seed)


def run_cross_validation(
    pipeline: BaseEstimator,
    features: pd.DataFrame,
    target: pd.Series | np.ndarray,
    cross_validation: Mapping[str, Any],
    *,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Evaluate a complete pipeline using development folds only."""
    if not isinstance(features, pd.DataFrame):
        raise TypeError("features must be a pandas DataFrame.")
    targets = np.asarray(target)
    if targets.ndim != 1 or len(features) != targets.size:
        raise ValueError("Features and target must contain the same non-zero number of rows.")
    if targets.size == 0:
        raise ValueError("Development data must not be empty.")

    n_splits, shuffle, random_seed = _validated_cv_config(cross_validation)
    if set(np.unique(targets).tolist()) != {0, 1}:
        raise ValueError("Target must contain binary classes 0 and 1.")
    class_counts = np.bincount(targets.astype(int), minlength=2)
    if class_counts.min() < n_splits:
        raise ValueError("Target must contain at least five observations from each binary class.")

    splitter = StratifiedKFold(
        n_splits=n_splits,
        shuffle=shuffle,
        random_state=random_seed,
    )
    probabilities = np.full(targets.size, np.nan, dtype=float)
    fold_assignments = np.zeros(targets.size, dtype=int)
    fold_metrics: list[dict[str, Any]] = []

    for fold, (train_indices, validation_indices) in enumerate(
        splitter.split(features, targets), start=1
    ):
        fold_pipeline = clone(pipeline)
        fit_started = perf_counter()
        fold_pipeline.fit(features.iloc[train_indices], targets[train_indices])
        fit_time = perf_counter() - fit_started
        classes = np.asarray(getattr(fold_pipeline, "classes_", []))
        positive_positions = np.flatnonzero(classes == 1)
        if positive_positions.size != 1:
            raise ValueError("Classifier must expose a probability column for class 1.")
        train_probabilities = fold_pipeline.predict_proba(features.iloc[train_indices])[
            :, positive_positions[0]
        ]
        prediction_started = perf_counter()
        fold_probabilities = fold_pipeline.predict_proba(features.iloc[validation_indices])[
            :, positive_positions[0]
        ]
        prediction_time = perf_counter() - prediction_started
        probabilities[validation_indices] = fold_probabilities
        fold_assignments[validation_indices] = fold
        metrics = calculate_metrics(
            targets[validation_indices], fold_probabilities, threshold=threshold
        )
        train_average_precision = float(
            average_precision_score(targets[train_indices], train_probabilities)
        )
        validation_average_precision = metrics["average_precision"]
        fold_metrics.append(
            {
                "fold": fold,
                "train_rows": int(train_indices.size),
                "validation_rows": int(validation_indices.size),
                "train_average_precision": train_average_precision,
                "validation_average_precision": validation_average_precision,
                "overfitting_gap": train_average_precision - validation_average_precision,
                "fit_time_seconds": fit_time,
                "prediction_time_seconds": prediction_time,
                **metrics,
            }
        )

    if np.isnan(probabilities).any() or (fold_assignments == 0).any():
        raise RuntimeError("Every development row must receive one OOF prediction.")

    fold_summary = {
        metric: {
            "mean": float(np.mean([row[metric] for row in fold_metrics])),
            "standard_deviation": float(
                np.std([row[metric] for row in fold_metrics], ddof=1)
            ),
        }
        for metric in SCALAR_METRICS + DIAGNOSTIC_METRICS
    }
    return {
        "fold_metrics": fold_metrics,
        "fold_summary": fold_summary,
        "oof_metrics": calculate_metrics(targets, probabilities, threshold=threshold),
        "oof_probabilities": probabilities,
        "fold_assignments": fold_assignments,
    }
