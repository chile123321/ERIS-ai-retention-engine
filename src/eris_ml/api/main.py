"""Single FastAPI app; trusted bundle loads once in lifespan, never at import."""

from __future__ import annotations

import json
import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from uuid import UUID, uuid4

from fastapi import Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import Response

from eris_ml.api.auth import require_service_token, service_token_configured
from eris_ml.api.contract import load_contract, validate_record
from eris_ml.api.errors import (
    ApiError,
    api_error_handler,
    unexpected_error_handler,
    validation_error_handler,
)
from eris_ml.api.monitoring import OperationalMetrics, safe_endpoint
from eris_ml.api.monitoring import router as metrics_router
from eris_ml.api.routes import health, model_info, prediction
from eris_ml.api.schemas import SYNTHETIC_EXAMPLE, PredictionRecord
from eris_ml.models.persistence import load_bundle, sidecar_paths
from eris_ml.settings import Settings

ROOT = Path(__file__).resolve().parents[3]
LOGGER = logging.getLogger("eris_ml.api")


def request_id_or_new(value: str | None) -> str:
    """Accept only canonical UUIDs, so a supplied identifier cannot contain PII."""
    if value is not None:
        try:
            parsed = UUID(value)
            if value == str(parsed):
                return value
        except (ValueError, AttributeError):
            pass
    return str(uuid4())


def _artifact_path(path: Path) -> Path:
    """Resolve a local, non-symlinked artifact path without opening data CSVs."""
    resolved = path if path.is_absolute() else ROOT / path
    if resolved.is_symlink():
        raise ValueError("Symlinked model artifacts are prohibited.")
    return resolved


def create_app(settings: Settings | None = None) -> FastAPI:
    """Create the app with dependency injection for local/synthetic testing."""
    config = settings or Settings()
    LOGGER.setLevel(getattr(logging, config.api_log_level.upper(), logging.INFO))

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        application.state.model_bundle = None
        application.state.metadata = None
        application.state.contract = None
        application.state.ready = False
        application.state.load_count = 0
        application.state.load_error_code = "MODEL_NOT_READY"
        if service_token_configured(config.service_token):
            try:
                path = _artifact_path(config.model_bundle_path)
                metadata_path, checksum_path = sidecar_paths(path)
                if (_artifact_path(config.model_metadata_path) != metadata_path
                    or _artifact_path(config.model_checksum_path) != checksum_path):
                    raise ValueError("Bundle sidecar paths do not match the model artifact.")
                bundle = load_bundle(path)  # checksum verified before trusted joblib load
                contract = load_contract(ROOT, bundle)
                synthetic = PredictionRecord.model_validate(SYNTHETIC_EXAMPLE)
                features, _ = validate_record(synthetic, contract, bundle)
                probability = bundle.predict_proba(features)
                if len(probability) != 1 or not 0 <= probability[0] <= 1:
                    raise ValueError("Bundle startup smoke test failed.")
                application.state.model_bundle = bundle
                application.state.metadata = bundle.metadata
                application.state.contract = contract
                application.state.ready = True
                application.state.load_count = 1
                application.state.load_error_code = None
                application.state.metrics.set_ready(True)
            except Exception:
                application.state.load_error_code = "BUNDLE_INTEGRITY_ERROR"
                application.state.metrics.increment("bundle_load_failures")
                LOGGER.error("bundle_startup_failed")
        else:
            LOGGER.error("service_auth_configuration_invalid")
        yield
        application.state.model_bundle = None
        application.state.ready = False
        application.state.metrics.set_ready(False)

    application = FastAPI(
        title="ERIS Attrition Research API", version="0.12.0", lifespan=lifespan,
        description=(
            "Research-only decision support; production_approved=false. "
            "The frozen threshold is 0.345651. Alerts do not predict certainty of resignation. "
            "SHAP factors are raw log-odds, not causal explanations."
        ),
    )
    application.state.model_bundle = None
    application.state.settings = config
    application.state.ready = False
    application.state.load_count = 0
    application.state.load_error_code = "MODEL_NOT_READY"
    application.state.metrics = OperationalMetrics()
    origins = [item.strip() for item in config.cors_origins.split(",") if item.strip()]
    if origins:
        if "*" in origins:
            raise ValueError("Wildcard CORS is prohibited for this research API.")
        application.add_middleware(CORSMiddleware, allow_origins=origins,
                                   allow_credentials=False, allow_methods=["GET", "POST"],
                                   allow_headers=["Content-Type", "X-Request-ID", "Authorization"])

    @application.middleware("http")
    async def request_audit(request: Request, call_next: object) -> Response:
        request.state.request_id = request_id_or_new(request.headers.get("X-Request-ID"))
        request.state.batch_size = 0
        started = time.perf_counter()
        response: Response = await call_next(request)  # type: ignore[operator]
        response.headers["X-Request-ID"] = request.state.request_id
        endpoint = safe_endpoint(request.url.path)
        elapsed = time.perf_counter() - started
        request.app.state.metrics.record_request(
            endpoint, request.method, response.status_code, elapsed
        )
        LOGGER.info(json.dumps({
            "request_id": request.state.request_id, "endpoint": endpoint,
            "method": request.method if request.method in {"GET", "POST"} else "OTHER",
            "status_code": response.status_code,
            "latency_ms": round(elapsed * 1000, 2),
            "candidate_version": (
                "eris-xgboost-v1" if request.app.state.ready else "unavailable"
            ),
            "bundle_version": "bundle-v1" if request.app.state.ready else "unavailable",
        }))
        return response

    application.add_exception_handler(ApiError, api_error_handler)  # type: ignore[arg-type]
    application.add_exception_handler(
        RequestValidationError, validation_error_handler  # type: ignore[arg-type]
    )
    application.add_exception_handler(Exception, unexpected_error_handler)
    application.include_router(health.router)
    application.include_router(metrics_router)
    application.include_router(model_info.router, prefix="/api/v1",
                               dependencies=[Depends(require_service_token)])
    application.include_router(prediction.router, prefix="/api/v1",
                               dependencies=[Depends(require_service_token)])
    return application


app = create_app()
