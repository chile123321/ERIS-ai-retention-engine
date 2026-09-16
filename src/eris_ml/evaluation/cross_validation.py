"""Leakage-safe cross-validation interfaces."""

from typing import Any


def run_cross_validation(pipeline: Any, development_data: Any) -> dict[str, Any]:
    """Evaluate a complete pipeline using development folds only."""
    raise NotImplementedError("Cross-validation has not been implemented.")
