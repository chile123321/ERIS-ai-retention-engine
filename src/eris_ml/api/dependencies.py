"""API dependency providers."""

from typing import Any

from fastapi import Request


def get_model_bundle(request: Request) -> Any | None:
    """Return a bundle loaded by application lifespan, if one is available."""
    return getattr(request.app.state, "model_bundle", None)
