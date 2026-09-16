"""SHAP integration interfaces; explanations are not causal evidence."""

from typing import Any


def explain_predictions(model: Any, rows: Any) -> Any:
    """Explain model behavior for supplied rows."""
    raise NotImplementedError("SHAP explainability has not been implemented.")
