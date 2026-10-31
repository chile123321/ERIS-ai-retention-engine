"""Read-only dispatch to frozen B2 and s1 inference interfaces, without fallback."""

import hashlib
import logging
from threading import RLock
from typing import Any
from uuid import UUID

from eris_ml.api_b2.adaboost import AdaBoostAdapter
from eris_ml.api_b2.schemas import (
    ADABOOST_ID,
    FEATURE_NAMES,
    FEATURE_SCHEMA_VERSION,
    MODEL_ID,
    MODEL_SHA256,
    BatchPredictRequest,
    BatchPredictResponse,
    ErrorCode,
    ExplainRequest,
    ExplainResponse,
    ModelMetadata,
    PredictionResult,
    PredictRequest,
    PredictResponse,
    ShapContribution,
)
from eris_ml.api_b2.settings import Settings
from eris_ml.models import bundle_v3_b2 as b2

LOGGER = logging.getLogger(__name__)


class ServiceError(Exception):
    def __init__(self, code: ErrorCode, status: int) -> None:
        super().__init__(code)
        self.code = code
        self.status = status


class ModelService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.bundle: b2.B2Bundle | None = None
        self.adaboost: AdaBoostAdapter | None = None
        self.adaboost_failure: ErrorCode = "MODEL_NOT_READY"
        self.failure: ErrorCode = "MODEL_NOT_READY"
        self._started = False
        self._lock = RLock()

    def start(self) -> None:
        with self._lock:
            if self._started:
                return
            self._started = True
            try:
                directory = self.settings.bundle_path
                if directory.is_symlink() or (directory / "pipeline.joblib").is_symlink():
                    raise ValueError("Untrusted bundle path")
                if self.settings.expected_model_sha256 != MODEL_SHA256:
                    raise ValueError("Expected hash differs from locked candidate")
                with (directory / "pipeline.joblib").open("rb") as stream:
                    digest = hashlib.file_digest(stream, "sha256").hexdigest()
                if digest != MODEL_SHA256:
                    raise ValueError("Model checksum mismatch")
                loaded = b2.load_bundle(directory)
                if (
                    loaded.metadata.get("model_version") != MODEL_ID
                    or loaded.metadata.get("feature_set") != FEATURE_SCHEMA_VERSION
                    or tuple(loaded.metadata.get("feature_order", ())) != FEATURE_NAMES
                ):
                    raise ValueError("Candidate identity/schema mismatch")
                self.bundle = loaded
            except FileNotFoundError:
                self.failure = "MODEL_NOT_READY"
                LOGGER.warning("B2 unavailable: MODEL_NOT_READY")
            except Exception:
                self.failure = "BUNDLE_INTEGRITY_ERROR"
                LOGGER.warning("B2 unavailable: BUNDLE_INTEGRITY_ERROR")
            try:
                self.adaboost = AdaBoostAdapter(self.settings.adaboost_bundle_path)
            except FileNotFoundError:
                LOGGER.warning("AdaBoost unavailable: MODEL_NOT_READY")
            except Exception:
                self.adaboost_failure = "BUNDLE_INTEGRITY_ERROR"
                LOGGER.warning("AdaBoost unavailable: BUNDLE_INTEGRITY_ERROR")

    def select(self, model_id: str) -> None:
        if model_id not in (MODEL_ID, ADABOOST_ID):
            raise ServiceError("MODEL_UNKNOWN", 422)
        if not self.settings.service_api_key.get_secret_value().strip():
            raise ServiceError("MODEL_NOT_READY", 503)
        if model_id == MODEL_ID:
            self.require_ready()
        elif self.adaboost is None:
            raise ServiceError(self.adaboost_failure, 503)

    def require_ready(self) -> b2.B2Bundle:
        if self.bundle is None:
            raise ServiceError(self.failure, 503)
        if not self.settings.service_api_key.get_secret_value().strip():
            raise ServiceError("MODEL_NOT_READY", 503)
        return self.bundle

    @staticmethod
    def result(reference: str, raw: dict[str, Any]) -> PredictionResult:
        if raw["model_version"] != MODEL_ID:
            raise ValueError("Unexpected model output")
        return PredictionResult.model_validate(
            {
                "reference_id": reference,
                "estimated_probability": raw["estimated_probability"],
                "imputed_features": raw["imputed_features"],
                "metadata": ModelMetadata(),
            }
        )

    def predict(self, request: PredictRequest, request_id: UUID) -> PredictResponse:
        self.select(request.model_id)
        try:
            if request.model_id == ADABOOST_ID:
                assert self.adaboost is not None
                return PredictResponse(request_id=request_id, result=self.adaboost.predict(
                    [request.features], [request.reference_id]
                )[0])
            raw = b2.predict_one(
                self.require_ready(), request.features.model_dump(exclude_none=True)
            )
            return PredictResponse(
                request_id=request_id, result=self.result(request.reference_id, raw)
            )
        except Exception:
            raise ServiceError("INFERENCE_ERROR", 500) from None

    def batch(self, request: BatchPredictRequest, request_id: UUID) -> BatchPredictResponse:
        self.select(request.model_id)
        if len(request.items) > self.settings.max_batch_size:
            # API-1 reserves BATCH_TOO_LARGE's fixed message for >100 items.
            raise ServiceError("VALIDATION_ERROR", 422)
        try:
            if request.model_id == ADABOOST_ID:
                assert self.adaboost is not None
                return BatchPredictResponse(request_id=request_id, results=self.adaboost.predict(
                    [item.features for item in request.items],
                    [item.reference_id for item in request.items],
                ))
            records = [item.features.model_dump(exclude_none=True) for item in request.items]
            raw = b2.predict_batch(self.require_ready(), records)
            return BatchPredictResponse(
                request_id=request_id,
                results=[
                    self.result(item.reference_id, result)
                    for item, result in zip(request.items, raw, strict=True)
                ],
            )
        except Exception:
            raise ServiceError("INFERENCE_ERROR", 500) from None

    def explain(self, request: ExplainRequest, request_id: UUID) -> ExplainResponse:
        self.select(request.model_id)
        try:
            if request.model_id == ADABOOST_ID:
                assert self.adaboost is not None
                return self.adaboost.explain(request.features, request_id=request_id,
                                            reference_id=request.reference_id)
            original = request.features.model_dump()
            with self._lock:
                raw = b2.explain_one(
                    self.require_ready(), request.features.model_dump(exclude_none=True)
                )
            if raw["model_version"] != MODEL_ID or set(raw["contributions"]) != set(FEATURE_NAMES):
                raise ValueError("Unexpected explanation output")
            contributions = []
            for name in FEATURE_NAMES:
                value = raw["contributions"][name]
                contributions.append(
                    ShapContribution.model_validate(
                        {
                            "feature": name,
                            "feature_value": original[name],
                            "shap_value": value,
                            "direction": "increase"
                            if value > 0
                            else "decrease"
                            if value < 0
                            else "neutral",
                        }
                    )
                )
            return ExplainResponse(
                request_id=request_id,
                reference_id=request.reference_id,
                estimated_probability=raw["estimated_probability"],
                base_value=raw["base_value"],
                contributions=contributions,
                top_factors=sorted(contributions, key=lambda item: -abs(item.shap_value))[:5],
                additivity_error=raw["absolute_additivity_error"],
                metadata=ModelMetadata(),
            )
        except Exception:
            raise ServiceError("EXPLANATION_ERROR", 500) from None
