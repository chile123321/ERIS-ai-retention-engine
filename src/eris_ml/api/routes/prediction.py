"""Prediction route, intentionally unavailable without a model."""

from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, status

from eris_ml.api.dependencies import get_model_bundle
from eris_ml.api.schemas import PredictionRequest, PredictionResponse

router = APIRouter(tags=["prediction"])


@router.post("/predict", response_model=PredictionResponse)
def predict(
    request: PredictionRequest,
    bundle: Annotated[Any | None, Depends(get_model_bundle)],
) -> PredictionResponse:
    """Reject prediction until a validated bundle and feature schema exist."""
    del request
    if bundle is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Prediction unavailable: no approved model bundle is loaded.",
        )
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Prediction logic has not been implemented.",
    )
