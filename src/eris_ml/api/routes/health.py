"""Liveness and readiness routes."""

from fastapi import APIRouter, Request

from eris_ml.api.errors import ApiError
from eris_ml.api.schemas import HealthResponse, ReadyResponse

router = APIRouter(tags=["operations"])


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Report that the API process is alive."""
    return HealthResponse()


@router.get("/ready", response_model=ReadyResponse)
def ready(request: Request) -> ReadyResponse:
    """Report model readiness; never disclose local paths or startup exceptions."""
    if not request.app.state.ready:
        code = request.app.state.load_error_code or "MODEL_NOT_READY"
        raise ApiError(code, "Model is not ready.", 503)
    metadata = request.app.state.metadata
    return ReadyResponse(candidate_version=metadata["candidate_version"],
                         bundle_version=metadata["bundle_version"])
