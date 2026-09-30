"""Step 12C service auth and operational monitoring, using synthetic input only."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from secrets import token_hex
from uuid import UUID, uuid4

import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from xgboost import XGBClassifier

from eris_ml.api.main import create_app
from eris_ml.api.schemas import SYNTHETIC_EXAMPLE
from eris_ml.settings import Settings

ROOT = Path(__file__).resolve().parents[2]
TOKEN = "synthetic-test-only-service-credential-0002"
PATHS = {
    "model_bundle_path": ROOT / "artifacts/models/eris_xgboost_v1.joblib",
    "model_metadata_path": ROOT / "artifacts/models/eris_xgboost_v1.metadata.json",
    "model_checksum_path": ROOT / "artifacts/models/eris_xgboost_v1.sha256",
}


def _app(token: str = TOKEN) -> FastAPI:
    return create_app(Settings(_env_file=None, service_token=token, **PATHS))


def _value(metrics: str, name: str) -> int:
    return int(next(line.split()[-1] for line in metrics.splitlines()
                    if line.startswith(name + " ")))


def test_fail_closed_without_auth_configuration() -> None:
    app = _app("")
    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
        ready = client.get("/ready")
        assert ready.status_code == 503
        assert "token" not in ready.text.lower()
        assert client.get("/api/v1/model-info").status_code == 401
        assert client.get("/metrics").status_code == 401
        assert app.state.load_count == 0


def test_random_64_character_token_from_environment_and_no_settings_cache(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
) -> None:
    """Exercise real Settings, Bearer parsing and routes without auth overrides."""
    first_token = token_hex(32)
    second_token = token_hex(32)
    assert first_token != second_token
    monkeypatch.setenv("ERIS_SERVICE_TOKEN", first_token)
    first_app = create_app()
    with TestClient(first_app) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/ready").status_code == 200
        assert client.get("/api/v1/model-info").status_code == 401
        assert client.get("/api/v1/model-info", headers={
            "Authorization": f"Bearer {first_token}",
        }).status_code == 200
        prediction = client.post("/api/v1/predict?include_explanation=false", headers={
            "Authorization": f"Bearer {first_token}",
        }, json=SYNTHETIC_EXAMPLE)
        assert prediction.status_code == 200
        assert prediction.json()["threshold"] == 0.345651
        for malformed in (
            "", "Bearer", "Bearer ", f"Basic {first_token}",
            f"Bearer Bearer {first_token}", f"Bearer {second_token}",
        ):
            assert client.get("/api/v1/model-info", headers={
                "Authorization": malformed,
            }).status_code == 401

        spec = client.get("/openapi.json").json()
        scheme = spec["components"]["securitySchemes"]["ServiceBearer"]
        assert scheme["type"] == "http" and scheme["scheme"] == "bearer"
        for path, method in (
            ("/api/v1/model-info", "get"), ("/api/v1/predict", "post"),
            ("/api/v1/predict/batch", "post"), ("/metrics", "get"),
        ):
            assert {"ServiceBearer": []} in spec["paths"][path][method]["security"]
        assert "security" not in spec["paths"]["/health"]["get"]

    monkeypatch.setenv("ERIS_SERVICE_TOKEN", second_token)
    second_app = create_app()
    with TestClient(second_app) as client:
        assert client.get("/ready").status_code == 200
        assert client.get("/api/v1/model-info", headers={
            "Authorization": f"Bearer {first_token}",
        }).status_code == 401
        assert client.get("/api/v1/model-info", headers={
            "Authorization": f"Bearer {second_token}",
        }).status_code == 200
    assert first_app.state.settings is not second_app.state.settings
    assert first_token not in caplog.text and second_token not in caplog.text
    assert "MonthlyIncome" not in caplog.text


def test_auth_tracing_monitoring_and_unchanged_prediction(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
) -> None:
    def forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("API attempted data CSV access or model training")

    monkeypatch.setattr(pd, "read_csv", forbidden)
    monkeypatch.setattr(XGBClassifier, "fit", forbidden)
    app = _app()
    auth = {"Authorization": f"Bearer {TOKEN}"}
    sentinel = "PRIVATE_SYNTHETIC_ID_999"
    record = {**SYNTHETIC_EXAMPLE, "record_id": sentinel}
    with TestClient(app) as client:
        assert client.get("/health").json() == {
            "status": "ok", "service": "eris-attrition-model-api",
        }
        assert client.get("/ready").status_code == 200
        no_auth = client.get("/api/v1/model-info")
        assert no_auth.status_code == 401
        assert no_auth.headers["WWW-Authenticate"] == "Bearer"
        assert client.get("/api/v1/model-info", headers={
            "Authorization": "Bearer incorrect-token",
        }).status_code == 401
        assert client.get("/metrics").status_code == 401
        assert client.post("/api/v1/predict", json=record).status_code == 401
        assert client.post("/api/v1/predict/batch", json={
            "records": [record],
        }).status_code == 401
        assert client.get("/api/v1/model-info", headers=auth).status_code == 200
        valid_id = str(uuid4())
        response = client.post("/api/v1/predict?include_explanation=false", json=record,
                               headers={**auth, "X-Request-ID": valid_id})
        assert response.status_code == 200
        result = response.json()
        assert response.headers["X-Request-ID"] == valid_id == result["request_id"]
        assert result["record_id"] == sentinel
        assert result["feature_schema_version"] == "v1-full"
        assert result["candidate_version"] == "eris-xgboost-v1"
        assert result["bundle_version"] == "bundle-v1"
        assert result["threshold"] == 0.345651
        assert 0 <= result["probability"] <= 1
        assert result["alert"] == (result["probability"] >= 0.345651)
        assert result["predicted_at_utc"].endswith("Z")
        assert datetime.fromisoformat(result["predicted_at_utc"]).tzinfo is not None
        invalid_id = client.post("/api/v1/predict?include_explanation=false", json=record,
                                 headers={**auth, "X-Request-ID": sentinel})
        assert invalid_id.status_code == 200
        assert invalid_id.headers["X-Request-ID"] != sentinel
        assert str(UUID(invalid_id.headers["X-Request-ID"])) == invalid_id.headers["X-Request-ID"]
        unknown = client.post("/api/v1/predict?include_explanation=false", json={
            **record, "Department": "Unknown Unit",
        }, headers=auth)
        assert unknown.status_code == 200
        assert "Department" in unknown.json()["warnings"][0]
        bad = client.post("/api/v1/predict?include_explanation=false", json={
            **record, "Age": 999,
        }, headers=auth)
        assert bad.status_code == 422
        assert "999" not in bad.text
        batch = client.post("/api/v1/predict/batch", headers=auth, json={
            "records": [record, {**record, "record_id": "DEMO-SECOND"}],
        })
        assert batch.status_code == 200
        assert [row["record_id"] for row in batch.json()["results"]] == [
            sentinel, "DEMO-SECOND",
        ]
        assert client.post("/api/v1/predict/batch", headers=auth, json={
            "records": [record] * 101,
        }).status_code == 413
        assert client.post("/api/v1/predict/batch", headers=auth, json={
            "records": [record] * 21, "include_explanation": True,
        }).status_code == 413
        metrics = client.get("/metrics", headers=auth)
        assert metrics.status_code == 200
        assert _value(metrics.text, "eris_authentication_failures_total") >= 5
        assert _value(metrics.text, "eris_validation_failures_total") >= 1
        assert _value(metrics.text, "eris_prediction_requests_total") >= 1
        assert _value(metrics.text, "eris_batch_records_total") == 2
        assert _value(metrics.text, "eris_unknown_category_warnings_total") == 1
        assert "eris_http_request_duration_seconds_sum" in metrics.text
        assert 'endpoint="/api/v1/predict",method="POST",status="200"' in metrics.text
        assert app.state.load_count == 1
        openapi = client.get("/openapi.json").json()
        assert "ServiceBearer" in openapi["components"]["securitySchemes"]
        for path, method in (("/api/v1/model-info", "get"),
                             ("/api/v1/predict", "post"),
                             ("/api/v1/predict/batch", "post"), ("/metrics", "get")):
            assert {"ServiceBearer": []} in openapi["paths"][path][method]["security"]
        assert client.get("/docs").status_code == 200
    assert TOKEN not in caplog.text
    assert sentinel not in caplog.text
    assert "MonthlyIncome" not in caplog.text
    assert "6200" not in caplog.text
    assert TOKEN not in metrics.text and sentinel not in metrics.text
    assert "MonthlyIncome" not in metrics.text
    assert "record_id" not in metrics.text
    assert "request_id" not in metrics.text
    assert "probability" not in metrics.text
    assert "shap_value" not in metrics.text
    assert "Authorization" not in metrics.text
