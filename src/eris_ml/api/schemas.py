"""Public API response schemas."""

from typing import Any

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Basic liveness response."""

    status: str = "ok"
    service: str = "eris-ai-retention-engine"


class ModelInfoResponse(BaseModel):
    """Current model readiness and metadata."""

    ready: bool
    detail: str
    metadata: dict[str, Any] | None = None


class PredictionRequest(BaseModel):
    """Placeholder request until approved feature schemas exist."""

    features: dict[str, Any] = Field(description="Approved model features; schema pending")


class PredictionResponse(BaseModel):
    """Future prediction response contract."""

    attrition_probability: float
    predicted_class: int
    model_version: str
