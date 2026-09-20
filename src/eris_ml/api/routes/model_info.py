"""Allowlisted research-model information route."""

from typing import Annotated

from fastapi import APIRouter, Depends

from eris_ml.api.dependencies import get_model_bundle
from eris_ml.api.errors import ApiError
from eris_ml.api.schemas import ModelInfoResponse
from eris_ml.models.bundle import ModelBundle

router = APIRouter(tags=["model"])


@router.get("/model-info", response_model=ModelInfoResponse)
def model_info(
    bundle: Annotated[ModelBundle | None, Depends(get_model_bundle)],
) -> ModelInfoResponse:
    """Return safe metadata without reloading or exposing internal configuration."""
    if bundle is None:
        raise ApiError("MODEL_NOT_READY", "Model is not ready.", 503)
    metadata = bundle.metadata
    return ModelInfoResponse(**{key: metadata[key] for key in (
        "candidate_version", "bundle_version", "model_family", "feature_schema_version",
        "feature_count", "threshold", "probability_type", "calibration",
        "training_scope", "fairness_status", "holdout_status", "production_approved",
    )})
