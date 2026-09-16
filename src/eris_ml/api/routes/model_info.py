"""Model readiness route."""

from typing import Annotated, Any

from fastapi import APIRouter, Depends

from eris_ml.api.dependencies import get_model_bundle
from eris_ml.api.schemas import ModelInfoResponse

router = APIRouter(tags=["model"])


@router.get("/model-info", response_model=ModelInfoResponse)
def model_info(bundle: Annotated[Any | None, Depends(get_model_bundle)]) -> ModelInfoResponse:
    """Return readiness without triggering model loading."""
    if bundle is None:
        return ModelInfoResponse(ready=False, detail="No approved model bundle is loaded.")
    return ModelInfoResponse(ready=True, detail="An approved model bundle is loaded.")
