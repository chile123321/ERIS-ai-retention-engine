"""Input and prediction drift interfaces."""

from typing import Any


def assess_drift(reference: Any, current: Any) -> dict[str, Any]:
    """Compare approved reference and current distributions."""
    raise NotImplementedError("Drift monitoring has not been implemented.")
