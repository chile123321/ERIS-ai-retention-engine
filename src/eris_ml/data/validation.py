"""Data validation interfaces."""

from typing import Any


def validate_frame(frame: Any, contract: dict[str, Any]) -> None:
    """Validate a frame against the versioned contract."""
    raise NotImplementedError("Data validation has not been implemented.")
