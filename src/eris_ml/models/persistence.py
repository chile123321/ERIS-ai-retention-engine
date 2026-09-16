"""Versioned model-bundle persistence interfaces."""

from pathlib import Path
from typing import Any


def save_model_bundle(bundle: Any, path: Path, metadata: dict[str, Any]) -> None:
    """Persist preprocessing, model, threshold, and release metadata together."""
    raise NotImplementedError("Model bundle persistence has not been implemented.")


def load_model_bundle(path: Path) -> Any:
    """Load an approved bundle explicitly, never as an import side effect."""
    raise NotImplementedError("Model bundle loading has not been implemented.")
