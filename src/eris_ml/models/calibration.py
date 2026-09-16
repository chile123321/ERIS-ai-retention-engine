"""Probability calibration interfaces."""

from typing import Any


def calibrate_model(model: Any, development_data: Any) -> Any:
    """Calibrate a model using development data only."""
    raise NotImplementedError("Calibration has not been implemented.")
