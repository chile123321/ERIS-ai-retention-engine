"""Persistence tests awaiting the model-bundle schema."""

import pytest


@pytest.mark.skip(reason="Model bundle persistence is not implemented in the scaffold.")
def test_model_bundle_round_trip() -> None:
    """Verify model and metadata round-trip after implementation."""
