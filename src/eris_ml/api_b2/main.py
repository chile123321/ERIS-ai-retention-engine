"""B2 local service. Run uvicorn eris_ml.api_b2.main:app --host 127.0.0.1."""

import logging
import secrets
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from uuid import UUID, uuid4

from fastapi import FastAPI, Request, Response, Security
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader

from eris_ml.api_b2.schemas import (
    SAFE_ERROR_MESSAGES,
    BatchPredictRequest,
    BatchPredictResponse,
    ErrorCode,
    ErrorResponse,
    ExplainRequest,
    ExplainResponse,
    ModelMetadata,
    PredictRequest,
    PredictResponse,
)
from eris_ml.api_b2.service import ModelService, ServiceError
from eris_ml.api_b2.settings import Settings


def error_response(code: ErrorCode, status: int, request_id: UUID | None) -> JSONResponse:
    error = ErrorResponse(request_id=request_id, code=code, message=SAFE_ERROR_MESSAGES[code])
    return JSONResponse(status_code=status, content=error.model_dump(mode="json"))


def create_app(settings: Settings | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        config = settings if settings is not None else Settings()
        logging.getLogger("eris_ml.api_b2").setLevel(config.log_level)
        service = ModelService(config)
        application.state.service = service
        service.start()
        try:
            yield
        finally:
            service.bundle = None

    application = FastAPI(
        title="ERIS V3 B2 Model API",
        version="1.0.0",
        lifespan=lifespan,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    # Documentation-only header dependency. Authentication remains in boundary().
    api_key_header = APIKeyHeader(
        name="X-ERIS-Service-Key", scheme_name="ERISServiceKey", auto_error=False
    )

    @application.middleware("http")
    async def boundary(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        request.state.request_id = uuid4()
        if request.url.path.startswith("/v1/"):
            service: ModelService | None = getattr(application.state, "service", None)
            if service is None:
                return error_response("MODEL_NOT_READY", 503, request.state.request_id)
            try:
                service.require_ready()
            except ServiceError as exc:
                return error_response(exc.code, exc.status, request.state.request_id)
            key = request.headers.get("X-ERIS-Service-Key")
            if key is None:
                return error_response("AUTH_MISSING", 401, request.state.request_id)
            expected = service.settings.service_api_key.get_secret_value()
            if not secrets.compare_digest(key.encode("utf-8"), expected.encode("utf-8")):
                return error_response("AUTH_INVALID", 401, request.state.request_id)
        return await call_next(request)

    @application.exception_handler(ServiceError)
    async def service_error(request: Request, exc: ServiceError) -> JSONResponse:
        return error_response(exc.code, exc.status, request.state.request_id)

    @application.exception_handler(RequestValidationError)
    async def invalid_request(request: Request, exc: RequestValidationError) -> JSONResponse:
        # Never echo Pydantic input/context or exception text.
        code: ErrorCode = "VALIDATION_ERROR"
        if isinstance(exc.body, dict):
            try:
                request.state.request_id = UUID(str(exc.body.get("request_id")))
            except (ValueError, TypeError, AttributeError):
                pass
            items = exc.body.get("items")
            if request.url.path == "/v1/predict/batch" and isinstance(items, list):
                if len(items) > 100:
                    code = "BATCH_TOO_LARGE"
        return error_response(code, 422, request.state.request_id)

    @application.exception_handler(Exception)
    async def unexpected_error(request: Request, exc: Exception) -> JSONResponse:
        code: ErrorCode = (
            "EXPLANATION_ERROR" if request.url.path == "/v1/explain" else "INFERENCE_ERROR"
        )
        return error_response(code, 500, getattr(request.state, "request_id", None))

    @application.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @application.get("/ready", responses={503: {"model": ErrorResponse}})
    def ready() -> dict[str, str]:
        application.state.service.require_ready()
        return {"status": "ready"}

    @application.get(
        "/v1/model/info",
        response_model=ModelMetadata,
        dependencies=[Security(api_key_header)],
        responses={401: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
    )
    def model_info() -> ModelMetadata:
        application.state.service.require_ready()
        return ModelMetadata()

    @application.post(
        "/v1/predict",
        response_model=PredictResponse,
        dependencies=[Security(api_key_header)],
        responses={
            401: {"model": ErrorResponse},
            422: {"model": ErrorResponse},
            503: {"model": ErrorResponse},
            500: {"model": ErrorResponse},
        },
    )
    def predict(body: PredictRequest, request: Request) -> PredictResponse:
        request.state.request_id = body.request_id or request.state.request_id
        service: ModelService = application.state.service
        return service.predict(body, request.state.request_id)

    @application.post(
        "/v1/predict/batch",
        response_model=BatchPredictResponse,
        dependencies=[Security(api_key_header)],
        responses={
            401: {"model": ErrorResponse},
            422: {"model": ErrorResponse},
            503: {"model": ErrorResponse},
            500: {"model": ErrorResponse},
        },
    )
    def batch(body: BatchPredictRequest, request: Request) -> BatchPredictResponse:
        request.state.request_id = body.request_id or request.state.request_id
        service: ModelService = application.state.service
        return service.batch(body, request.state.request_id)

    @application.post(
        "/v1/explain",
        response_model=ExplainResponse,
        dependencies=[Security(api_key_header)],
        responses={
            401: {"model": ErrorResponse},
            422: {"model": ErrorResponse},
            503: {"model": ErrorResponse},
            500: {"model": ErrorResponse},
        },
    )
    def explain(body: ExplainRequest, request: Request) -> ExplainResponse:
        request.state.request_id = body.request_id or request.state.request_id
        service: ModelService = application.state.service
        return service.explain(body, request.state.request_id)

    return application


app = create_app()
