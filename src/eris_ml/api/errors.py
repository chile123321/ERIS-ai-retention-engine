"""Sanitized API errors; never echo employee payloads or exception objects."""

from __future__ import annotations

from typing import Any

from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse


class ApiError(Exception):
    """A public error with a controlled code, status and field-only details."""

    def __init__(self, code: str, message: str, status_code: int,
                 details: list[dict[str, str]] | None = None) -> None:
        self.code = code
        self.message = message
        self.status_code = status_code
        self.details = details or []


def error_response(request: Request, code: str, message: str, status_code: int,
                   details: list[dict[str, str]] | None = None) -> JSONResponse:
    """Return the same privacy-safe error envelope for all API failures."""
    return JSONResponse(
        status_code=status_code,
        headers={"WWW-Authenticate": "Bearer"} if status_code == 401 else None,
        content={"error": {"code": code, "message": message, "details": details or [],
                           "request_id": getattr(request.state, "request_id", "unavailable")}},
    )


async def api_error_handler(request: Request, exc: ApiError) -> JSONResponse:
    """Render controlled API errors."""
    return error_response(request, exc.code, exc.message, exc.status_code, exc.details)


async def validation_error_handler(request: Request,
                                   exc: RequestValidationError) -> JSONResponse:
    """Expose only invalid field paths and generic reasons, never input values."""
    details: list[dict[str, str]] = []
    for error in exc.errors():
        location: Any = error.get("loc", ())
        field = ".".join(str(part) for part in location if part != "body") or "body"
        details.append({"field": field, "reason": "Invalid type, missing field or extra field."})
    return error_response(request, "VALIDATION_ERROR", "Input validation failed.", 422, details)


async def unexpected_error_handler(request: Request, exc: Exception) -> JSONResponse:
    """Suppress tracebacks, paths, secrets and payloads on unexpected errors."""
    del exc
    return error_response(request, "PREDICTION_ERROR", "Request could not be completed.", 500)
