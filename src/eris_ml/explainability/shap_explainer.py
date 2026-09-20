"""Compatibility interface for non-causal bundle-local SHAP explanations."""

from typing import Any

from eris_ml.models.bundle import ModelBundle


def explain_predictions(model: ModelBundle, rows: Any) -> list[dict[str, Any]]:
    """Delegate to checked original-feature raw-log-odds explanations."""
    return model.explain(rows)
