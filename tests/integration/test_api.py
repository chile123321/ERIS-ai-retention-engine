"""Local checked-bundle API tests; no data CSV or final-evaluation access."""

import json
from pathlib import Path

import pandas as pd
import pytest
import yaml
from fastapi.testclient import TestClient
from xgboost import XGBClassifier

from eris_ml.api.main import create_app
from eris_ml.api.schemas import SYNTHETIC_EXAMPLE
from eris_ml.models.bundle import ModelBundle
from eris_ml.settings import Settings
from eris_ml.utils.hashing import sha256_file

ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / "artifacts/models/eris_xgboost_v1.joblib"
METADATA = ROOT / "artifacts/models/eris_xgboost_v1.metadata.json"
CHECKSUM = ROOT / "artifacts/models/eris_xgboost_v1.sha256"
TEST_TOKEN = "synthetic-test-only-service-credential-0001"


def _settings(**changes: object) -> Settings:
    values: dict[str, object] = {
        "model_bundle_path": MODEL, "model_metadata_path": METADATA,
        "model_checksum_path": CHECKSUM, "max_batch_size": 100,
        "service_token": TEST_TOKEN,
    }
    values.update(changes)
    return Settings(_env_file=None, **values)


def _post(client: TestClient, data: dict[str, object], *, explain: bool = False) -> object:
    flag = "true" if explain else "false"
    return client.post(f"/api/v1/predict?include_explanation={flag}", json=data)


def test_health_readiness_model_info_and_load_once() -> None:
    app = create_app(_settings())
    with TestClient(app) as client:
        client.headers.update({"Authorization": f"Bearer {TEST_TOKEN}"})
        assert client.get("/health").json() == {
            "status": "ok", "service": "eris-attrition-model-api",
        }
        ready = client.get("/ready")
        assert ready.status_code == 200 and ready.json()["candidate_version"] == "eris-xgboost-v1"
        info = client.get("/api/v1/model-info")
        assert info.status_code == 200
        assert info.json()["threshold"] == 0.345651
        assert info.json()["production_approved"] is False
        assert info.json()["prediction_horizon_validated"] is False
        assert "path" not in json.dumps(info.json()).lower()
        assert app.state.load_count == 1
        _post(client, SYNTHETIC_EXAMPLE)
        assert app.state.load_count == 1
        openapi = client.get("/openapi.json").json()
        assert "/api/v1/predict" in openapi["paths"]
        assert client.get("/docs").status_code == 200
        assert "artifacts/models" not in json.dumps(openapi)
        assert "illustrative_alert" in json.dumps(openapi)
        assert "illustrative_no_alert" in json.dumps(openapi)


def test_single_batch_validation_explanation_and_order() -> None:
    with TestClient(create_app(_settings())) as client:
        client.headers.update({"Authorization": f"Bearer {TEST_TOKEN}"})
        result = _post(client, SYNTHETIC_EXAMPLE, explain=True)
        assert result.status_code == 200
        body = result.json()
        assert 0 <= body["probability"] <= 1
        assert body["alert"] == (body["probability"] >= 0.345651)
        assert body["candidate_version"] == "eris-xgboost-v1"
        assert body["threshold"] == 0.345651
        assert len(body["top_factors"]) == 5
        assert body["explanation_space"] == "raw_log_odds"
        assert "causality" in body["shap_disclaimer"]
        assert {item["feature"] for item in body["top_factors"]}.issubset(
            set(SYNTHETIC_EXAMPLE) - {"record_id"}
        )
        second = {**SYNTHETIC_EXAMPLE, "record_id": "DEMO-SECOND", "OverTime": "Yes"}
        batch = client.post("/api/v1/predict/batch", json={
            "records": [SYNTHETIC_EXAMPLE, second], "include_explanation": False,
        })
        assert batch.status_code == 200 and batch.json()["count"] == 2
        assert [row["record_id"] for row in batch.json()["results"]] == [
            "DEMO-LOW-001", "DEMO-SECOND",
        ]
        assert all(row["top_factors"] == [] for row in batch.json()["results"])
        bad = _post(client, {**SYNTHETIC_EXAMPLE, "Age": 999})
        assert bad.status_code == 422 and bad.json()["error"]["code"] == "VALIDATION_ERROR"
        assert bad.json()["error"]["details"][0]["field"].endswith("Age")
        assert _post(client, {**SYNTHETIC_EXAMPLE, "Education": 999}).status_code == 422
        assert _post(client, {**SYNTHETIC_EXAMPLE, "EmployeeNumber": 5}).status_code == 422
        assert _post(client, {**SYNTHETIC_EXAMPLE, "Age": "NaN"}).status_code == 422
        assert _post(client, {key: value for key, value in SYNTHETIC_EXAMPLE.items()
                              if key != "Age"}).status_code == 422
        unknown = _post(client, {**SYNTHETIC_EXAMPLE, "Department": "Unknown Unit"})
        assert unknown.status_code == 200 and "Department" in unknown.json()["warnings"][0]
        assert client.post("/api/v1/predict/batch", json={"records": []}).status_code == 413
        too_large = client.post("/api/v1/predict/batch", json={
            "records": [SYNTHETIC_EXAMPLE] * 21, "include_explanation": True,
        })
        assert too_large.status_code == 413
        assert client.post("/api/v1/predict?top_k=11", json=SYNTHETIC_EXAMPLE).status_code == 422
        examples = json.loads((ROOT / "tests/fixtures/manual_prediction_examples.json").read_text(
            encoding="utf-8"
        ))
        assert len(examples) == 3
        assert all(_post(client, example).status_code == 200 for example in examples.values())


def test_not_ready_for_missing_or_tampered_bundle(tmp_path: Path) -> None:
    missing = tmp_path / "missing.joblib"
    with TestClient(create_app(_settings(
        model_bundle_path=missing, model_metadata_path=missing.with_suffix(".metadata.json"),
        model_checksum_path=missing.with_suffix(".sha256"),
    ))) as client:
        client.headers.update({"Authorization": f"Bearer {TEST_TOKEN}"})
        assert client.get("/health").status_code == 200
        response = client.get("/ready")
        assert response.status_code == 503
        assert response.json()["error"]["code"] == "BUNDLE_INTEGRITY_ERROR"
        assert _post(client, SYNTHETIC_EXAMPLE).status_code == 503
    for source in (MODEL, METADATA, CHECKSUM):
        (tmp_path / source.name).write_bytes(source.read_bytes())
    copied = tmp_path / MODEL.name
    copied.write_bytes(copied.read_bytes() + b"tampered")
    with TestClient(create_app(_settings(
        model_bundle_path=copied,
        model_metadata_path=tmp_path / METADATA.name,
        model_checksum_path=tmp_path / CHECKSUM.name,
    ))) as client:
        client.headers.update({"Authorization": f"Bearer {TEST_TOKEN}"})
        assert client.get("/ready").status_code == 503
        assert "C:" not in json.dumps(client.get("/ready").json())
    copied.write_bytes(MODEL.read_bytes())
    metadata_path = tmp_path / METADATA.name
    changed = json.loads(metadata_path.read_text(encoding="utf-8"))
    changed["threshold"] = 0.5
    metadata_path.write_text(json.dumps(changed), encoding="utf-8")
    (tmp_path / CHECKSUM.name).write_text(
        f"{sha256_file(copied)}  {copied.name}\n"
        f"{sha256_file(metadata_path)}  {metadata_path.name}\n", encoding="ascii",
    )
    with TestClient(create_app(_settings(
        model_bundle_path=copied, model_metadata_path=metadata_path,
        model_checksum_path=tmp_path / CHECKSUM.name,
    ))) as client:
        client.headers.update({"Authorization": f"Bearer {TEST_TOKEN}"})
        assert client.get("/ready").status_code == 503


def test_explanation_disabled_does_not_call_shap_and_error_is_sanitized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with TestClient(create_app(_settings())) as client:
        client.headers.update({"Authorization": f"Bearer {TEST_TOKEN}"})
        def broken(*args: object, **kwargs: object) -> object:
            raise RuntimeError("SECRET_LOCAL_PATH")

        monkeypatch.setattr(ModelBundle, "explain", broken)
        assert _post(client, SYNTHETIC_EXAMPLE, explain=False).status_code == 200
        response = _post(client, SYNTHETIC_EXAMPLE, explain=True)
        assert response.status_code == 500
        assert response.json()["error"]["code"] == "EXPLANATION_ERROR"
        assert "SECRET_LOCAL_PATH" not in response.text


def test_api_does_not_read_data_csv_or_train_and_logs_no_payload(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
) -> None:
    ledger_path = ROOT / "configs/release/final_evaluation_record_v1.yaml"
    before = ledger_path.read_bytes()

    def forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("API attempted data access or training")

    monkeypatch.setattr(pd, "read_csv", forbidden)
    monkeypatch.setattr(XGBClassifier, "fit", forbidden)
    sentinel = "PRIVATE_RECORD_999"
    request = {**SYNTHETIC_EXAMPLE, "record_id": sentinel}
    with TestClient(create_app(_settings())) as client:
        client.headers.update({"Authorization": f"Bearer {TEST_TOKEN}"})
        assert _post(client, request).status_code == 200
    assert sentinel not in caplog.text
    assert "MonthlyIncome" not in caplog.text
    assert "6200" not in caplog.text
    assert ledger_path.read_bytes() == before
    assert yaml.safe_load(before)["run_count"] == 1
