"""Synthetic-only SHAP mapping, deterministic ranking and additivity tests."""

import pandas as pd
import pytest

from eris_ml.models.explainability import (
    ADDITIVITY_TOLERANCE,
    DISCLAIMER,
    encoded_feature_mapping,
    global_importance,
)


def test_global_shap_has_25_original_features_and_additivity(
    synthetic_bundle: tuple[object, pd.DataFrame, object],
) -> None:
    bundle, frame, definition = synthetic_bundle
    encoded, mapped = encoded_feature_mapping(bundle)
    assert len(encoded) >= 25 and set(mapped) == set(definition.all_features)
    importance, error = global_importance(bundle, frame.iloc[:20], sample_size=20)
    assert len(importance) == 25 and importance["feature"].is_unique
    assert set(importance["feature"]) == set(definition.all_features)
    assert importance["mean_abs_shap"].is_monotonic_decreasing
    assert error < ADDITIVITY_TOLERANCE
    assert importance["model_output_space"].eq("raw_log_odds").all()


def test_local_shap_is_deterministic_and_noncausal(
    synthetic_bundle: tuple[object, pd.DataFrame, object],
) -> None:
    bundle, frame, definition = synthetic_bundle
    first = bundle.explain(frame.iloc[:2], top_k=5)
    assert first == bundle.explain(frame.iloc[:2], top_k=5)
    assert len(first) == 2 and first[0]["model_output_space"] == "raw_log_odds"
    assert first[0]["additivity_max_error"] < ADDITIVITY_TOLERANCE
    assert first[0]["disclaimer"] == DISCLAIMER
    assert len(first[0]["top_factors"]) == 5
    assert {factor["feature"] for factor in first[0]["top_factors"]}.issubset(
        definition.all_features
    )
    with pytest.raises(ValueError, match="top_k"):
        bundle.explain(frame.iloc[:1], top_k=0)
