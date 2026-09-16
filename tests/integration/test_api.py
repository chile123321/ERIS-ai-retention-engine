"""API integration tests that do not require a trained model."""

from fastapi.testclient import TestClient

from eris_ml.api.main import create_app


def test_health_endpoint() -> None:
    """Health remains available before a model is loaded."""
    response = TestClient(create_app()).get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_prediction_is_unavailable_without_model() -> None:
    """The service returns 503 instead of a fabricated prediction."""
    response = TestClient(create_app()).post("/predict", json={"features": {}})
    assert response.status_code == 503
