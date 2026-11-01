"""API-3 contract/export checks; never start the app or deserialize a model."""

import importlib.util
import json
from pathlib import Path

import pytest

from eris_ml.api_b2.main import app
from eris_ml.api_b2.schemas import FEATURE_NAMES, MODEL_ID, MODEL_SHA256
from eris_ml.models import bundle_v3_b2

ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = ROOT / "docs/model-api-v1.openapi.json"
EXPECTED_PATHS = {
    "/health",
    "/ready",
    "/v1/model/info",
    "/v1/predict",
    "/v1/predict/batch",
    "/v1/explain",
}


@pytest.fixture
def exporter(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("OpenAPI must not load/infer/start a bundle")

    for name in ("load_bundle", "predict_one", "predict_batch", "explain_one"):
        monkeypatch.setattr(bundle_v3_b2, name, forbidden)
    spec = importlib.util.spec_from_file_location(
        "export_api_b2_openapi", ROOT / "scripts/export_api_b2_openapi.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_export_deterministic_matches_live_openapi_without_state_change(exporter):
    state = dict(app.state._state)
    cache = app.openapi_schema
    first = exporter.contract_bytes()
    assert exporter.contract_bytes() == first == SPEC_PATH.read_bytes()
    assert app.openapi_schema is cache
    assert dict(app.state._state) == state
    assert json.loads(first) == app.openapi()
    assert exporter.export() == SPEC_PATH


def test_safe_export_no_clobber(exporter, monkeypatch, tmp_path):
    monkeypatch.setattr(exporter, "ROOT", tmp_path)
    (tmp_path / "docs").mkdir()
    output = tmp_path / "docs/contract.json"
    exporter.export(output)
    assert output.read_bytes() == exporter.contract_bytes()
    original = output.read_bytes()
    exporter.export(output)
    assert output.read_bytes() == original
    output.write_bytes(b"existing-unrelated-file")
    with pytest.raises(FileExistsError):
        exporter.export(output)
    assert output.read_bytes() == b"existing-unrelated-file"
    for path in (tmp_path / "other.json", tmp_path / "docs/file.txt", Path("docs/../other.json")):
        with pytest.raises(ValueError):
            exporter.export(path)


def test_endpoint_and_security_metadata():
    spec = json.loads(SPEC_PATH.read_bytes())
    assert set(spec["paths"]) == EXPECTED_PATHS
    assert spec["components"]["securitySchemes"] == {
        "ERISServiceKey": {"type": "apiKey", "in": "header", "name": "X-ERIS-Service-Key"}
    }
    assert "security" not in spec
    for path, methods in spec["paths"].items():
        assert set(methods) == {
            "get" if path in ("/health", "/ready", "/v1/model/info") else "post"
        }
        operation = next(iter(methods.values()))
        if path.startswith("/v1/"):
            assert operation["security"] == [{"ERISServiceKey": []}]
            for status in ("401", "503"):
                assert operation["responses"][status]["content"]["application/json"]["schema"] == {
                    "$ref": "#/components/schemas/ErrorResponse"
                }
        else:
            assert not operation.get("security")
        if path == "/ready":
            assert "503" in operation["responses"]
        if "post" in methods:
            assert operation["responses"]["422"]["content"]["application/json"]["schema"] == {
                "$ref": "#/components/schemas/ErrorResponse"
            }
        elif path == "/v1/model/info":
            assert "422" not in operation["responses"]  # no request validation on this GET


def test_schema_fidelity_and_no_sensitive_documentation():
    spec = json.loads(SPEC_PATH.read_bytes())
    schemas = spec["components"]["schemas"]
    features = schemas["EmployeeFeatures"]
    assert set(features["properties"]) == set(FEATURE_NAMES)
    assert set(features["required"]) == set(FEATURE_NAMES) - {
        "TrainingTimesLastYear",
        "YearsWithCurrManager",
    }
    assert len(features["required"]) == 13
    for name in ("TrainingTimesLastYear", "YearsWithCurrManager"):
        assert {branch["type"] for branch in features["properties"][name]["anyOf"]} == {
            "integer",
            "null",
        }
    items = schemas["BatchPredictRequest"]["properties"]["items"]
    assert items["minItems"] == 1 and items["maxItems"] == 100
    for schema in schemas.values():
        if schema.get("type") == "object":
            assert schema["additionalProperties"] is False
    meta = schemas["ModelMetadata"]["properties"]
    assert meta["model_id"]["const"] == MODEL_ID
    assert meta["model_sha256"]["const"] == MODEL_SHA256
    assert meta["model_status"]["const"] == "research_candidate"
    assert meta["decision_support_only"]["const"] is True
    assert meta["independent_validation"]["const"] is False
    text = SPEC_PATH.read_text(encoding="utf-8")
    for forbidden in (
        "HTTPValidationError",
        '"threshold"',
        '"risk_level"',
        '"will_leave"',
        "C:\\",
        "/app/",
        "artifacts/models",
        "ERIS_SERVICE_API_KEY",
        "Traceback",
    ):
        assert forbidden not in text
