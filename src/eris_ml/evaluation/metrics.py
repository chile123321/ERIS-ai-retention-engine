"""Evaluation metric interfaces."""

from typing import Any


def calculate_metrics(y_true: Any, probabilities: Any, threshold: float) -> dict[str, Any]:
    """Calculate approved ranking, classification, and calibration metrics."""
    raise NotImplementedError("Metric calculation has not been implemented.")
