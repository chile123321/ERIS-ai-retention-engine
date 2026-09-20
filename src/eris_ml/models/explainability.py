"""Tree SHAP explanations on development-trained XGBoost raw margins."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from scipy.sparse import issparse

if TYPE_CHECKING:
    from eris_ml.models.bundle import ModelBundle

DISCLAIMER = (
    "SHAP describes how the model’s prediction changes relative to its baseline. "
    "It does not establish causality and must not be interpreted as proof that "
    "changing a feature will prevent employee attrition."
)
ADDITIVITY_TOLERANCE = 1e-4


def encoded_feature_mapping(bundle: ModelBundle) -> tuple[list[str], list[str]]:
    """Map each transformed column back to its one original configured feature."""
    transformer = bundle.pipeline.named_steps["preprocessor"]
    encoded = list(transformer.get_feature_names_out())
    originals = bundle.metadata["feature_order"]
    mapped: list[str] = []
    for name in encoded:
        matches = [feature for feature in originals
                   if name == feature or name.startswith(f"{feature}_")]
        if len(matches) != 1:
            raise ValueError(f"Cannot uniquely map transformed feature {name!r}.")
        mapped.append(matches[0])
    if set(mapped) != set(originals):
        raise ValueError("Transformed feature mapping does not cover all 25 originals.")
    return encoded, mapped


def _shap_matrix(bundle: ModelBundle, frame: pd.DataFrame) -> tuple[np.ndarray, float, float]:
    """Return raw-margin SHAP matrix, scalar baseline and maximum additivity gap."""
    import shap

    preprocessor = bundle.pipeline.named_steps["preprocessor"]
    model = bundle.pipeline.named_steps["model"]
    transformed = preprocessor.transform(frame)
    matrix = transformed.toarray() if issparse(transformed) else np.asarray(transformed)
    explainer = shap.TreeExplainer(
        model, model_output="raw", feature_perturbation="tree_path_dependent",
    )
    values = np.asarray(explainer.shap_values(matrix), dtype=float)
    baseline = float(np.asarray(explainer.expected_value).reshape(-1)[0])
    if values.shape != matrix.shape or not np.isfinite(values).all():
        raise ValueError("Unexpected SHAP shape or non-finite contribution.")
    margins = np.asarray(model.predict(matrix, output_margin=True), dtype=float).reshape(-1)
    error = float(np.max(np.abs(baseline + values.sum(axis=1) - margins)))
    if error > ADDITIVITY_TOLERANCE:
        raise ValueError(f"SHAP raw-margin additivity failed: {error:.6g}.")
    return values, baseline, error


def global_importance(bundle: ModelBundle, frame: pd.DataFrame,
                      *, sample_size: int) -> tuple[pd.DataFrame, float]:
    """Aggregate absolute one-hot contributions to exactly 25 original features."""
    if len(frame) != sample_size or sample_size < 1:
        raise ValueError("Global SHAP sample size differs from the supplied rows.")
    validated = bundle.validate_input(frame)
    values, _, error = _shap_matrix(bundle, validated)
    _, mapping = encoded_feature_mapping(bundle)
    importance = {name: 0.0 for name in bundle.metadata["feature_order"]}
    for index, original in enumerate(mapping):
        importance[original] += float(np.abs(values[:, index]).mean())
    rows = pd.DataFrame({
        "feature": list(importance), "mean_abs_shap": list(importance.values()),
    }).sort_values(["mean_abs_shap", "feature"], ascending=[False, True])
    rows["sample_size"] = sample_size
    rows["seed"] = 42
    rows["model_output_space"] = "raw_log_odds"
    return rows.reset_index(drop=True), error


def local_explanations(bundle: ModelBundle, frame: pd.DataFrame,
                       *, top_k: int = 5) -> list[dict[str, Any]]:
    """Return deterministic signed original-feature contributions per record."""
    if not isinstance(top_k, int) or not 1 <= top_k <= 25:
        raise ValueError("top_k must be an integer in [1,25].")
    values, baseline, error = _shap_matrix(bundle, frame)
    _, mapping = encoded_feature_mapping(bundle)
    features = bundle.metadata["feature_order"]
    result: list[dict[str, Any]] = []
    for row_index in range(len(frame)):
        aggregate = {name: 0.0 for name in features}
        for index, original in enumerate(mapping):
            aggregate[original] += float(values[row_index, index])
        ranked = sorted(aggregate, key=lambda name: (-abs(aggregate[name]), name))[:top_k]
        factors = []
        for name in ranked:
            value = frame.iloc[row_index][name]
            safe_value: Any = None if pd.isna(value) else (
                value.item() if isinstance(value, np.generic) else value
            )
            contribution = aggregate[name]
            factors.append({
                "feature": name, "feature_value": safe_value,
                "shap_value": contribution,
                "direction": "increase" if contribution > 0 else (
                    "decrease" if contribution < 0 else "neutral"
                ),
            })
        result.append({
            "base_value": baseline, "model_output_space": "raw_log_odds",
            "top_factors": factors, "additivity_max_error": error,
            "disclaimer": DISCLAIMER,
        })
    return result
