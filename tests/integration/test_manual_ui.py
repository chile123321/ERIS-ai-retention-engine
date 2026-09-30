"""Standalone localhost UI uses the frozen synthetic-only inference path."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from eris_ml.api.main import create_app as create_production_app
from eris_ml.manual_ui import app as ui_module
from eris_ml.models.bundle import ModelBundle
from eris_ml.models.inference import infer_records
from eris_ml.settings import Settings

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = json.loads((ROOT / "tests/fixtures/manual_prediction_examples.json").read_text(
    encoding="utf-8"
))


def test_page_schema_examples_and_no_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ERIS_SERVICE_TOKEN", raising=False)
    with TestClient(ui_module.create_app()) as client:
        page = client.get("/")
        assert page.status_code == 200
        assert "Load Low Example" in page.text and "Load High Example" in page.text
        assert 'name="Attrition"' not in page.text
        assert client.get("/health").status_code == 200
        schema = client.get("/schema").json()
        assert schema["feature_count"] == len(schema["fields"]) == 25
        assert len({field["name"] for field in schema["fields"]}) == 25
        assert "Attrition" not in {field["name"] for field in schema["fields"]}
        assert schema["threshold"] == 0.345651
        assert schema["candidate_version"] == "eris-xgboost-v1"
        assert schema["bundle_version"] == "bundle-v1"
        assert schema["production_approved"] is False
        assert all(field["label"] for field in schema["fields"])
        for signal in ("low", "high"):
            assert client.get(f"/examples/{signal}").json() == EXAMPLES[
                f"synthetic_{signal}_signal"
            ]
        assert client.get("/openapi.json").status_code == 404


def test_prediction_uses_shared_inference_and_shap(monkeypatch: pytest.MonkeyPatch) -> None:
    called = 0
    actual = ui_module.infer_records

    def tracked(bundle: ModelBundle, rows: list[dict[str, Any]]) -> Any:
        nonlocal called
        called += 1
        return actual(bundle, rows)

    monkeypatch.setattr(ui_module, "infer_records", tracked)
    with TestClient(ui_module.create_app()) as client:
        for signal, expected_alert in (("low", False), ("high", True)):
            response = client.post("/predict", json=EXAMPLES[f"synthetic_{signal}_signal"])
            assert response.status_code == 200, response.text
            result = response.json()
            assert 0 <= result["probability"] <= 1
            assert result["threshold"] == 0.345651
            assert result["alert"] is expected_alert
            assert result["decision_label"] == (
                "REVIEW_RECOMMENDED" if expected_alert else "NO_REVIEW_ALERT"
            )
            assert result["explanation_space"] == "raw_log_odds"
            assert len(result["top_factors"]) == 5
            assert all({"feature", "feature_value", "direction", "shap_value"}
                       <= set(factor) for factor in result["top_factors"])
            assert result["shap_disclaimer"].endswith("non-causal.")
            assert result["production_approved"] is False
        assert called == 2


def test_field_validation_and_explanation_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    with TestClient(ui_module.create_app()) as client:
        base = EXAMPLES["synthetic_low_signal"]
        missing = dict(base)
        missing.pop("Age")
        assert "Age" in client.post("/predict", json=missing).json()["errors"]
        wrong_type = dict(base, Age="34")
        assert "Age" in client.post("/predict", json=wrong_type).json()["errors"]
        out_of_range = dict(base, Age=999)
        assert "Age" in client.post("/predict", json=out_of_range).json()["errors"]
        unknown = dict(base, Department="Unknown Department")
        assert "Department" in client.post("/predict", json=unknown).json()["errors"]
        assert client.post("/predict", json=missing).status_code == 422

        def broken_explain(self: ModelBundle, records: Any, *, top_k: int = 5) -> Any:
            raise RuntimeError("synthetic SHAP failure")

        monkeypatch.setattr(ModelBundle, "explain", broken_explain)
        response = client.post("/predict", json=base)
        assert response.status_code == 200
        assert response.json()["probability"] > 0
        assert response.json()["explanation_error"]
        assert response.json()["top_factors"] == []


def test_production_auth_and_openapi_remain_separate() -> None:
    token = "synthetic-test-only-service-credential-0004"
    settings = Settings(
        _env_file=None, service_token=token,
        model_bundle_path=ROOT / "artifacts/models/eris_xgboost_v1.joblib",
        model_metadata_path=ROOT / "artifacts/models/eris_xgboost_v1.metadata.json",
        model_checksum_path=ROOT / "artifacts/models/eris_xgboost_v1.sha256",
    )
    with TestClient(create_production_app(settings)) as client:
        assert client.post("/api/v1/predict", json=EXAMPLES["synthetic_low_signal"]
                           ).status_code == 401
        spec = client.get("/openapi.json").json()
        assert "/examples/{signal}" not in spec["paths"]
        assert "/schema" not in spec["paths"]
        assert spec["components"]["securitySchemes"]["ServiceBearer"]["scheme"] == "bearer"


def test_local_only_and_no_employee_data_reads(monkeypatch: pytest.MonkeyPatch) -> None:
    for host in ("0.0.0.0", "localhost", "192.168.1.2", "::1"):
        with pytest.raises(ValueError, match="127.0.0.1"):
            ui_module.create_app(host=host)
    monkeypatch.setenv("ERIS_MANUAL_UI_HOST", "0.0.0.0")
    with pytest.raises(ValueError, match="127.0.0.1"):
        with TestClient(ui_module.create_app()):
            pass
    monkeypatch.delenv("ERIS_MANUAL_UI_HOST")
    original = Path.open

    def no_data_csv(self: Path, *args: Any, **kwargs: Any) -> Any:
        if self.suffix == ".csv" and ("development" in self.name or "final_test" in self.name):
            raise AssertionError("Protected employee data file was accessed")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", no_data_csv)
    with TestClient(ui_module.create_app()) as client:
        assert client.post("/predict", json=EXAMPLES["synthetic_low_signal"]
                           ).status_code == 200
        assert client.get("/", headers={"Host": "external.example"}).status_code == 400


def test_local_ui_matches_cli_and_production_service() -> None:
    from scripts.manual_model_test import load_frozen_bundle, predict_local

    payload = dict(EXAMPLES["synthetic_low_signal"], BusinessTravel="Non-Travel")
    bundle, contract = load_frozen_bundle()
    cli = predict_local(payload, bundle, contract, include_explanation=False)
    with TestClient(ui_module.create_app()) as client:
        ui = client.post("/predict", json=payload).json()
    features = {name: payload[name] for name in bundle.metadata["feature_order"]}
    service = infer_records(bundle, [features])[0]
    assert ui["probability"] == cli["probability"] == service.probability
    assert ui["threshold"] == cli["threshold"] == service.threshold
    assert ui["alert"] == cli["alert"] == service.alert
    assert ui["decision_label"] == cli["decision_label"] == service.decision_label


def test_missing_or_wrong_bundle_fails_before_serving(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="frozen bundle"):
        with TestClient(ui_module.create_app(bundle_path=tmp_path / "missing.joblib")):
            pass
    wrong = tmp_path / "wrong.joblib"
    wrong.write_bytes(b"not a valid bundle")
    with pytest.raises(RuntimeError, match="frozen bundle"):
        with TestClient(ui_module.create_app(bundle_path=wrong)):
            pass
