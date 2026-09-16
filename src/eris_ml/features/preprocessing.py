"""Leakage-safe preprocessing pipeline construction."""

from typing import Any


def build_preprocessing_pipeline(config: dict[str, Any]) -> Any:
    """Build preprocessing to be fitted only inside training folds."""
    raise NotImplementedError("Preprocessing design is outside the scaffold milestone.")
