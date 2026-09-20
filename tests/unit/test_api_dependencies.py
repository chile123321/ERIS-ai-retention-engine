"""Dependency injection stays state-only and never opens bundle artifacts."""

from types import SimpleNamespace
from typing import Any

from eris_ml.api.dependencies import get_model_bundle


def test_dependency_returns_only_app_state_bundle() -> None:
    marker = object()
    request: Any = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(
        model_bundle=marker,
    )))
    assert get_model_bundle(request) is marker
    request.app.state.model_bundle = None
    assert get_model_bundle(request) is None
