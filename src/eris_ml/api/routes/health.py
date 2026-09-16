"""Liveness route."""

from fastapi import APIRouter

from eris_ml.api.schemas import HealthResponse

router = APIRouter(tags=["operations"])


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Report that the API process is alive."""
    return HealthResponse()
