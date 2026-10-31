"""B2 service checks: synthetic requests, existing bundle, no fit or employee data."""

import hashlib
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from eris_ml.api_b2.main import create_app
from eris_ml.api_b2.schemas import FEATURE_NAMES, MODEL_ID, MODEL_SHA256, ExplainResponse
from eris_ml.api_b2.service import ModelService
from eris_ml.api_b2.settings import Settings
from eris_ml.models import bundle_v3_b2 as b2

KEY = "synthetic-test-key-not-a-deployment-secret"
HEADERS = {"X-ERIS-Service-Key": KEY}


def sample():
    return {
        "reference_id": "synthetic-one",
        "features": {
            "Age": 35,
            "Department": "Research & Development",
            "DistanceFromHome": 5,
            "Education": 3,
            "EnvironmentSatisfaction": 3,
            "JobLevel": 2,
            "JobSatisfaction": 3,
            "MonthlyIncome": 5000,
            "OverTime": "No",
            "TotalWorkingYears": 10,
            "TrainingTimesLastYear": 3,
            "YearsAtCompany": 5,
            "YearsInCurrentRole": 3,
            "YearsSinceLastPromotion": 1,
            "YearsWithCurrManager": 3,
        },
    }


@pytest.fixture(scope="module")
def client():
    settings = Settings(service_api_key=KEY)
    directory = settings.bundle_path
    if not directory.is_dir():
        pytest.skip("Local B2 artifact unavailable; real-bundle service checks not run")
    before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in directory.iterdir()}
    assert before["pipeline.joblib"] == MODEL_SHA256
    with TestClient(create_app(settings), raise_server_exceptions=False) as connection:
        assert connection.get("/ready").status_code == 200
        yield connection
    after = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in directory.iterdir()}
    assert before == after


def test_health_readiness_info_and_no_browser_surface(client):
    assert client.get("/health").json() == {"status": "ok"}
    assert client.get("/ready").json() == {"status": "ready"}
    response = client.get("/v1/model/info", headers=HEADERS)
    assert response.status_code == 200
    assert response.json()["model_id"] == MODEL_ID
    assert response.json()["model_sha256"] == MODEL_SHA256
    assert response.json()["independent_validation"] is False
    for path in ("/docs", "/redoc", "/openapi.json"):
        assert client.get(path).status_code == 404
    assert "access-control-allow-origin" not in response.headers


@pytest.mark.parametrize(
    "headers,code", [({}, "AUTH_MISSING"), ({"X-ERIS-Service-Key": "incorrect"}, "AUTH_INVALID")]
)
@pytest.mark.parametrize(
    "path", ["/v1/model/info", "/v1/predict", "/v1/predict/batch", "/v1/explain"]
)
def test_authentication(client, headers, code, path):
    response = client.request("GET" if path.endswith("info") else "POST", path, headers=headers)
    assert response.status_code == 401
    assert response.json()["code"] == code
    UUID(response.json()["request_id"])


def test_single_matches_direct_and_request_uuid(client):
    body = sample()
    body["request_id"] = str(uuid4())
    direct = b2.predict_one(client.app.state.service.bundle, body["features"])
    response = client.post("/v1/predict", json=body, headers=HEADERS)
    assert response.status_code == 200
    assert response.json()["request_id"] == body["request_id"]
    result = response.json()["result"]
    assert result["estimated_probability"] == pytest.approx(
        direct["estimated_probability"], abs=1e-12
    )
    assert 0 <= result["estimated_probability"] <= 1
    assert set(result) == {"reference_id", "estimated_probability", "imputed_features", "metadata"}
    repeated = client.post("/v1/predict", json=body, headers=HEADERS).json()
    assert repeated["request_id"] == response.json()["request_id"]
    repeated_result = repeated["result"]
    # Frozen RF uses n_jobs=4: parallel summation can differ at machine precision.
    assert repeated_result.pop("estimated_probability") == pytest.approx(
        result["estimated_probability"], abs=1e-12, rel=0
    )
    assert repeated_result == {k: v for k, v in result.items() if k != "estimated_probability"}


@pytest.mark.parametrize(
    "missing",
    [
        [],
        ["TrainingTimesLastYear"],
        ["YearsWithCurrManager"],
        ["TrainingTimesLastYear", "YearsWithCurrManager"],
    ],
)
def test_optional_missing_and_null_delegated(client, monkeypatch, missing):
    original = b2.predict_one
    calls = []

    def observed(bundle, record):
        calls.append(record)
        return original(bundle, record)

    monkeypatch.setattr(b2, "predict_one", observed)
    body = sample()
    for name in missing:
        body["features"][name] = None
    first = client.post("/v1/predict", json=body, headers=HEADERS)
    for name in missing:
        body["features"].pop(name)
    second = client.post("/v1/predict", json=body, headers=HEADERS)
    assert first.status_code == second.status_code == 200
    left, right = first.json()["result"], second.json()["result"]
    assert left.pop("estimated_probability") == pytest.approx(
        right.pop("estimated_probability"), abs=1e-12, rel=0
    )
    assert left == right
    assert first.json()["result"]["imputed_features"] == missing
    assert all(set(record) == set(FEATURE_NAMES) - set(missing) for record in calls)


def test_batch_uses_batch_interface_once_preserves_order(client, monkeypatch):
    original = b2.predict_batch
    calls = []

    def observed(bundle, records):
        calls.append(records)
        return original(bundle, records)

    def forbidden(*args, **kwargs):
        pytest.fail("Batch must not call predict_one")

    monkeypatch.setattr(b2, "predict_batch", observed)
    monkeypatch.setattr(b2, "predict_one", forbidden)
    items = [{**sample(), "reference_id": str(i)} for i in range(100)]
    items[1]["features"]["OverTime"] = "Yes"
    response = client.post("/v1/predict/batch", json={"items": items}, headers=HEADERS)
    assert response.status_code == 200
    assert len(calls) == 1
    assert [r["reference_id"] for r in response.json()["results"]] == [str(i) for i in range(100)]
    direct = original(client.app.state.service.bundle, calls[0])
    assert [r["estimated_probability"] for r in response.json()["results"]] == pytest.approx(
        [r["estimated_probability"] for r in direct], abs=1e-12
    )


@pytest.mark.parametrize("kind", ["empty", "oversize", "duplicate", "invalid_item"])
def test_batch_atomic_validation(client, monkeypatch, kind):
    def forbidden(*args, **kwargs):
        pytest.fail("Invalid batch must never infer")

    monkeypatch.setattr(b2, "predict_batch", forbidden)
    items = [sample()]
    if kind == "empty":
        items = []
    elif kind == "oversize":
        items = [{**sample(), "reference_id": str(i)} for i in range(101)]
    elif kind == "duplicate":
        items += [sample()]
    else:
        items += [{**sample(), "reference_id": "invalid"}]
        items[1]["features"]["Age"] = 41.0
    response = client.post("/v1/predict/batch", json={"items": items}, headers=HEADERS)
    assert response.status_code == 422
    assert response.json()["code"] == (
        "BATCH_TOO_LARGE" if kind == "oversize" else "VALIDATION_ERROR"
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("Age", 41.0),
        ("Age", True),
        ("Age", -1),
        ("Age", "35"),
        ("Age", 2147483648),
        ("Department", "private-invalid-input"),
        ("email", "private-invalid-input"),
        ("OverTime", "sometimes"),
    ],
)
def test_safe_validation_errors(client, field, value):
    body = sample()
    body["request_id"] = str(uuid4())
    body["features"][field] = value
    response = client.post("/v1/predict", json=body, headers=HEADERS)
    assert response.status_code == 422
    assert response.json()["request_id"] == body["request_id"]
    assert "private-invalid-input" not in response.text
    assert "traceback" not in response.text.lower()
    assert "features" not in response.text


def test_malformed_json_and_nonfinite_are_safe(client):
    for raw in ('{"features":', '{"reference_id":"private", "features":{"Age":NaN}}'):
        response = client.post(
            "/v1/predict",
            content=raw,
            headers={
                **HEADERS,
                "Content-Type": "application/json",
            },
        )
        assert response.status_code == 422
        assert "private" not in response.text


def test_explain_real_probability_space(client):
    body = sample()
    body["features"]["TrainingTimesLastYear"] = None
    response = client.post("/v1/explain", json=body, headers=HEADERS)
    assert response.status_code == 200
    result = ExplainResponse.model_validate(response.json())
    assert len(result.contributions) == 15 and len(result.top_factors) == 5
    assert result.contributions[10].feature_value is None
    assert result.base_value + sum(c.shap_value for c in result.contributions) == pytest.approx(
        result.estimated_probability, abs=1e-6
    )
    direct = b2.explain_one(
        client.app.state.service.bundle,
        {k: v for k, v in body["features"].items() if v is not None},
    )
    assert result.estimated_probability == pytest.approx(direct["estimated_probability"], abs=1e-12)
    assert {c.feature: c.shap_value for c in result.contributions} == direct["contributions"]


@pytest.mark.parametrize(
    "method,path,code",
    [
        ("predict_one", "/v1/predict", "INFERENCE_ERROR"),
        ("predict_batch", "/v1/predict/batch", "INFERENCE_ERROR"),
        ("explain_one", "/v1/explain", "EXPLANATION_ERROR"),
    ],
)
def test_internal_errors_redacted(client, monkeypatch, method, path, code):
    def fail(*args, **kwargs):
        raise RuntimeError("C:/private-secret-path employee-input private-secret")

    monkeypatch.setattr(b2, method, fail)
    body = {"items": [sample()]} if method == "predict_batch" else sample()
    response = client.post(path, json=body, headers=HEADERS)
    assert response.status_code == 500
    assert response.json()["code"] == code
    assert "private" not in response.text


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -0.1, 1.1])
def test_invalid_output_probability_rejected(client, monkeypatch, value):
    monkeypatch.setattr(
        b2,
        "predict_one",
        lambda *args: {
            "model_version": MODEL_ID,
            "estimated_probability": value,
            "imputed_features": [],
        },
    )
    assert client.post("/v1/predict", json=sample(), headers=HEADERS).status_code == 500


@pytest.mark.parametrize("mode", ["absent", "wrong_hash", "identity", "loader_failure"])
def test_failed_startup_keeps_health_and_safe_503(monkeypatch, tmp_path, mode):
    options = {"service_api_key": KEY}
    if mode == "absent":
        options["bundle_path"] = tmp_path / "absent"
    elif mode == "wrong_hash":
        options["expected_model_sha256"] = "0" * 64
    elif mode == "identity":
        monkeypatch.setattr(
            b2,
            "load_bundle",
            lambda path: SimpleNamespace(
                metadata={
                    "model_version": "wrong-model",
                }
            ),
        )
    else:

        def fail(path):
            raise ValueError("private-path in bundle failure")

        monkeypatch.setattr(b2, "load_bundle", fail)
    with TestClient(create_app(Settings(**options))) as connection:
        assert connection.get("/health").json() == {"status": "ok"}
        for path in ("/ready", "/v1/model/info", "/v1/predict", "/v1/predict/batch", "/v1/explain"):
            response = connection.request(
                "GET" if path in ("/ready", "/v1/model/info") else "POST", path, headers=HEADERS
            )
            assert response.status_code == 503
            assert "private" not in response.text


def test_empty_key_and_configured_batch_cap(client, monkeypatch):
    service = client.app.state.service
    # Explicit empty value, independent of a caller's configured environment key.
    from pydantic import SecretStr

    monkeypatch.setattr(service.settings, "service_api_key", SecretStr(""))
    assert client.get("/health").status_code == 200
    assert client.get("/ready").status_code == 503
    assert client.post("/v1/predict", json=sample(), headers=HEADERS).status_code == 503
    monkeypatch.setattr(service.settings, "service_api_key", SecretStr(KEY))
    monkeypatch.setattr(service.settings, "max_batch_size", 1)
    items = [sample(), {**sample(), "reference_id": "second"}]
    response = client.post("/v1/predict/batch", json={"items": items}, headers=HEADERS)
    assert response.status_code == 422 and response.json()["code"] == "VALIDATION_ERROR"


def test_load_once_no_import_load_and_settings_environment(client, monkeypatch):
    count = []
    bundle = client.app.state.service.bundle
    monkeypatch.setattr(b2, "load_bundle", lambda path: count.append(path) or bundle)
    service = ModelService(Settings(service_api_key=KEY))
    service.start()
    service.start()
    assert len(count) == 1
    command = (
        "from eris_ml.models import bundle_v3_b2 as b; "
        "b.load_bundle=lambda p: (_ for _ in ()).throw(AssertionError('import loaded bundle')); "
        "import eris_ml.api_b2.main"
    )
    completed = subprocess.run([sys.executable, "-B", "-c", command], capture_output=True)
    assert completed.returncode == 0, completed.stderr.decode()
    monkeypatch.setenv("ERIS_SERVICE_API_KEY", KEY)
    monkeypatch.setenv("ERIS_MAX_BATCH_SIZE", "7")
    monkeypatch.setenv("ERIS_LOG_LEVEL", "WARNING")
    monkeypatch.setenv("ERIS_BUNDLE_PATH", str(Path("synthetic-path")))
    monkeypatch.setenv("ERIS_EXPECTED_MODEL_SHA256", MODEL_SHA256)
    settings = Settings()
    assert settings.max_batch_size == 7 and settings.log_level == "WARNING"
    assert settings.bundle_path == Path("synthetic-path")
    assert KEY not in repr(settings)
    assert settings.model_config["env_file"] is None
    for invalid in (0, 101):
        with pytest.raises(ValidationError):
            Settings(max_batch_size=invalid)
    assert "ERIS_SERVICE_API_KEY" in os.environ
