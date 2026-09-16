"""Model-training orchestration interfaces."""

from typing import Any


def train_model(config: dict[str, Any]) -> Any:
    """Train a preprocessing-plus-model pipeline on development folds."""
    raise NotImplementedError("Training is intentionally unavailable in the scaffold.")
