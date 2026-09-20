"""Fail-closed service authentication for backend-to-ML-API calls."""

from __future__ import annotations

import hmac
from typing import Annotated

from fastapi import Request, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from eris_ml.api.errors import ApiError

BEARER = HTTPBearer(auto_error=False, scheme_name="ServiceBearer")


def service_token_configured(token: str | None) -> bool:
    """Reject absent, short, whitespace-padded or example credentials."""
    return bool(
        token and len(token) >= 32 and token == token.strip()
        and not any(character.isspace() for character in token)
        and "<" not in token and ">" not in token
    )


def require_service_token(
    request: Request,
    credentials: Annotated[HTTPAuthorizationCredentials | None, Security(BEARER)],
) -> None:
    """Authenticate a configured service token without exposing failure details."""
    expected: str | None = request.app.state.settings.service_token
    supplied = credentials.credentials if credentials is not None else ""
    if (not service_token_configured(expected) or credentials is None
        or credentials.scheme.lower() != "bearer"
        or not hmac.compare_digest(supplied, expected or "")):
        raise ApiError("UNAUTHORIZED", "Service authentication required.", 401)
