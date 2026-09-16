"""Estimator factory interfaces."""

from typing import Any


def create_estimator(config: dict[str, Any]) -> Any:
    """Create an estimator from approved configuration."""
    raise NotImplementedError("Estimator creation has not been implemented.")
