"""API dependency providers."""

from fastapi import Request

from eris_ml.models.bundle import ModelBundle


def get_model_bundle(request: Request) -> ModelBundle | None:
    """Return a bundle loaded by application lifespan, if one is available."""
    return getattr(request.app.state, "model_bundle", None)
