"""Evaluation metric interfaces."""

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def calculate_metrics(
    y_true: Any, probabilities: Any, threshold: float = 0.5
) -> dict[str, Any]:
    """Calculate approved ranking, classification, and calibration metrics."""
    targets = np.asarray(y_true)
    scores = np.asarray(probabilities, dtype=float)
    if targets.ndim != 1 or scores.ndim != 1:
        raise ValueError("y_true and probabilities must be one-dimensional.")
    if targets.size == 0:
        raise ValueError("y_true and probabilities must not be empty.")
    if targets.size != scores.size:
        raise ValueError("y_true and probabilities must have the same length.")
    if not np.isfinite(scores).all():
        raise ValueError("Probabilities must be finite.")
    if ((scores < 0.0) | (scores > 1.0)).any():
        raise ValueError("Probabilities must be in [0, 1].")
    if not isinstance(threshold, (int, float)) or not np.isfinite(threshold):
        raise ValueError("Threshold must be a finite number in [0, 1].")
    if not 0.0 <= float(threshold) <= 1.0:
        raise ValueError("Threshold must be in [0, 1].")

    unique_targets = set(np.unique(targets).tolist())
    if not unique_targets.issubset({0, 1}):
        raise ValueError("y_true must contain only binary values 0 and 1.")
    if unique_targets != {0, 1}:
        raise ValueError("y_true must contain both classes 0 and 1.")

    predictions = (scores >= float(threshold)).astype(int)
    tn, fp, fn, tp = confusion_matrix(targets, predictions, labels=[0, 1]).ravel()
    predicted_positive_count = int(tp + fp)
    return {
        "accuracy": float(accuracy_score(targets, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(targets, predictions)),
        "average_precision": float(average_precision_score(targets, scores)),
        "roc_auc": float(roc_auc_score(targets, scores)),
        "precision": float(precision_score(targets, predictions, zero_division=0)),
        "recall": float(recall_score(targets, predictions, zero_division=0)),
        "f1": float(f1_score(targets, predictions, zero_division=0)),
        "f2": float(fbeta_score(targets, predictions, beta=2, zero_division=0)),
        "brier_score": float(brier_score_loss(targets, scores)),
        "predicted_positive_count": predicted_positive_count,
        "alert_rate": float(predicted_positive_count / targets.size),
        "false_positive_count": int(fp),
        "false_negative_count": int(fn),
        "false_positives_per_true_positive": None if tp == 0 else float(fp / tp),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
    }
