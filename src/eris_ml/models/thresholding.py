"""Decision-threshold selection interfaces."""

from typing import Any


def select_threshold(y_true: Any, probabilities: Any, protocol: dict[str, Any]) -> float:
    """Select a threshold from development predictions only."""
    raise NotImplementedError("Threshold selection has not been implemented.")
