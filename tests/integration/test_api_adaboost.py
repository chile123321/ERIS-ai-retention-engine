"""Synthetic multi-model API checks; no research data, fit, or artifact writes."""

import asyncio
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event

import pytest
from fastapi.testclient import TestClient

from eris_ml.api_b2.explanation_worker import ExplanationWorker
from eris_ml.api_b2.main import create_app
from eris_ml.api_b2.schemas import ADABOOST_ID, FEATURE_NAMES, MODEL_ID, EmployeeFeatures
from eris_ml.api_b2.service import ServiceError
from eris_ml.api_b2.settings import Settings
from eris_ml.models.adaboost_v3 import positive_probability

HEADERS = {"X-ERIS-Service-Key": "synthetic-test-key"}


def body():
    return {
        "model_id": ADABOOST_ID,
        "reference_id": "synthetic",
        "features": dict(zip(FEATURE_NAMES, [
            35, "Sales", 5, 3, 3, 2, 3, 5000, "No", 10, 3, 5, 3, 1, 3,
        ], strict=True)),
    }


@pytest.fixture(scope="module")
def client():
    config = Settings(service_api_key=HEADERS["X-ERIS-Service-Key"])
    paths = [p for directory in (config.bundle_path, config.adaboost_bundle_path)
             for p in directory.iterdir() if p.is_file()]
    before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    original = Path.open

    def guarded(path, *args, **kwargs):
        if path.suffix.lower() == ".csv" or "data" in path.parts:
            raise AssertionError("API may not read research datasets")
        return original(path, *args, **kwargs)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(Path, "open", guarded)
        with TestClient(create_app(config)) as connection:
            assert connection.app.state.service.adaboost is not None
            yield connection
    assert before == {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def test_models_auth_and_no_fallback(client, monkeypatch):
    assert client.get("/v1/models").status_code == 401
    result = client.get("/v1/models", headers=HEADERS).json()
    assert result["default_model_id"] == MODEL_ID
    assert [m["metadata"]["model_id"] for m in result["models"]] == [MODEL_ID, ADABOOST_ID]
    assert all(m["ready"] and m["supports_explain"] for m in result["models"])
    unknown = client.post("/v1/predict", headers=HEADERS, json={**body(), "model_id": "other"})
    assert unknown.status_code == 422 and unknown.json()["code"] == "MODEL_UNKNOWN"
    monkeypatch.setattr(client.app.state.service, "adaboost", None)
    failed = client.post("/v1/predict", headers=HEADERS, json=body())
    assert failed.status_code == 503
    legacy = body()
    legacy.pop("model_id")
    assert client.post("/v1/predict", headers=HEADERS, json=legacy).json()[
        "result"]["metadata"]["model_id"] == MODEL_ID


def test_real_predict_batch_optional_and_no_shap(client, monkeypatch):
    adapter = client.app.state.service.adaboost

    def forbidden(*args, **kwargs):
        pytest.fail("Prediction must not explain")

    monkeypatch.setattr(adapter.explainer, "explain", forbidden)
    payload = body()
    payload["features"]["TrainingTimesLastYear"] = None
    payload["features"].pop("YearsWithCurrManager")
    frame = adapter.frame([EmployeeFeatures.model_validate(payload["features"])])
    expected = float(positive_probability(adapter.bundle.model, frame)[0])
    result = client.post("/v1/predict", headers=HEADERS, json=payload).json()["result"]
    assert result["estimated_probability"] == pytest.approx(expected, abs=1e-12)
    assert result["imputed_features"] == ["TrainingTimesLastYear", "YearsWithCurrManager"]
    items = [{"reference_id": str(i), "features": payload["features"]} for i in range(3)]
    results = client.post("/v1/predict/batch", headers=HEADERS,
                          json={"model_id": ADABOOST_ID, "items": items}).json()["results"]
    assert [r["reference_id"] for r in results] == ["0", "1", "2"]
    assert all(r["estimated_probability"] == pytest.approx(expected) for r in results)
    payload["features"]["Department"] = "secret-invalid-category"
    response = client.post("/v1/predict", headers=HEADERS, json=payload)
    assert response.status_code == 422 and "secret-invalid" not in response.text


def test_real_explain_and_separate_timings(client):
    payload = body()
    payload["features"]["YearsWithCurrManager"] = None
    predictions = []
    for _ in range(6):
        start = time.perf_counter()
        response = client.post("/v1/predict", headers=HEADERS, json=payload)
        predictions.append(time.perf_counter() - start)
        assert response.status_code == 200
    probability = response.json()["result"]["estimated_probability"]
    explanations = []
    for _ in range(2):
        start = time.perf_counter()
        response = client.post("/v1/explain", headers=HEADERS, json=payload)
        explanations.append(time.perf_counter() - start)
        assert response.status_code == 200, response.text
        result = response.json()
        assert result["metadata"]["model_id"] == ADABOOST_ID
        assert result["estimated_probability"] == pytest.approx(probability, abs=1e-12)
        assert result["explanation_space"] == "probability"
        assert [c["feature"] for c in result["contributions"]] == list(FEATURE_NAMES)
        assert result["contributions"][-1]["feature_value"] is None
        assert result["base_value"] + sum(c["shap_value"] for c in result["contributions"]) == (
            pytest.approx(probability, abs=1e-8)
        )
    print("\nAPI synthetic latency seconds:", json.dumps({
        "predict_first": predictions[0], "predict_warm_mean": sum(predictions[1:]) / 5,
        "explain_first": explanations[0], "explain_warm": explanations[1],
        "probability": probability,
    }))


def test_explain_errors_redacted(client, monkeypatch):
    def fail(*args, **kwargs):
        raise ValueError("private employee input C:/secret")

    monkeypatch.setattr(client.app.state.service.adaboost.explainer, "explain", fail)
    response = client.post("/v1/explain", headers=HEADERS, json=body())
    assert response.status_code == 500 and response.json()["code"] == "EXPLANATION_ERROR"
    assert "private" not in response.text and "secret" not in response.text


def test_busy_timeout_retains_slot_predict_responsive(client, monkeypatch):
    entered, release = Event(), Event()
    worker = ExplanationWorker(queue_size=0, timeout=0.2)
    monkeypatch.setattr(client.app.state, "explanation_worker", worker)

    def slow(*args):
        entered.set()
        release.wait(5)
        raise ServiceError("EXPLANATION_ERROR", 500)

    monkeypatch.setattr(client.app.state.service, "explain", slow)
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            pending = pool.submit(client.post, "/v1/explain", headers=HEADERS, json=body())
            assert entered.wait(2)
            assert client.get("/health").status_code == 200
            assert client.post("/v1/predict", headers=HEADERS, json=body()).status_code == 200
            assert client.post("/v1/explain", headers=HEADERS, json=body()).status_code == 429
            timed_out = pending.result(2)
            assert timed_out.status_code == 504
            assert timed_out.json()["code"] == "EXPLANATION_TIMEOUT"
            # The timed-out computation is still active, so it must continue occupying capacity.
            assert client.post("/v1/explain", headers=HEADERS, json=body()).status_code == 429
    finally:
        release.set()
        worker.close()


def test_queue_limit_cancel_and_serial_execution():
    entered, release = Event(), Event()
    calls = []
    worker = ExplanationWorker(queue_size=1, timeout=2)

    def slow():
        calls.append(1)
        entered.set()
        release.wait(3)
        raise ServiceError("EXPLANATION_ERROR", 500)

    async def run():
        first = asyncio.create_task(worker.run(slow))
        assert await asyncio.to_thread(entered.wait, 1)
        second = asyncio.create_task(worker.run(slow))
        await asyncio.sleep(0)  # Allow submission to the bounded queue.
        with pytest.raises(ServiceError, match="EXPLANATION_BUSY"):
            await worker.run(slow)
        second.cancel()
        with pytest.raises(asyncio.CancelledError):
            await second
        # Abandoned queued work must still occupy capacity until actually dequeued.
        with pytest.raises(ServiceError, match="EXPLANATION_BUSY"):
            await worker.run(slow)
        release.set()
        with pytest.raises(ServiceError, match="EXPLANATION_ERROR"):
            await first

    try:
        asyncio.run(run())
    finally:
        release.set()
        worker.close()
    assert calls == [1]  # Cancelled queued work never entered the explainer.


def test_independent_bundle_failure_and_load_once(monkeypatch, tmp_path):
    from eris_ml.api_b2 import service as module

    real = module.AdaBoostAdapter
    calls = []

    def observe(path):
        calls.append(path)
        return real(path)

    monkeypatch.setattr(module, "AdaBoostAdapter", observe)
    config = Settings(
        service_api_key=HEADERS["X-ERIS-Service-Key"], bundle_path=tmp_path / "absent"
    )
    with TestClient(create_app(config)) as connection:
        connection.app.state.service.start()
        assert len(calls) == 1
        assert connection.get("/ready").status_code == 503  # Legacy readiness is B2 readiness.
        assert connection.post("/v1/predict", headers=HEADERS, json=body()).status_code == 200
        legacy = body()
        legacy.pop("model_id")
        assert connection.post("/v1/predict", headers=HEADERS, json=legacy).status_code == 503


def test_pinned_adaboost_hash_before_deserialization(monkeypatch, tmp_path):
    from eris_ml.api_b2 import adaboost

    monkeypatch.setattr(adaboost, "sha256_file", lambda path: "0" * 64)
    monkeypatch.setattr(adaboost, "load_bundle", lambda path: pytest.fail("Untrusted deserialize"))
    with pytest.raises(ValueError, match="checksum"):
        adaboost.AdaBoostAdapter(tmp_path)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -0.1, 1.1])
def test_adaboost_output_boundary(client, monkeypatch, value):
    from eris_ml.api_b2 import adaboost

    monkeypatch.setattr(adaboost, "positive_probability", lambda *args: [value])
    response = client.post("/v1/predict", headers=HEADERS, json=body())
    assert response.status_code == 500 and response.json()["code"] == "INFERENCE_ERROR"
