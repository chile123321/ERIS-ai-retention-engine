"""Local UI contract tests; real bundle smoke is optional and never fits."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from eris_ml.manual_ui import adaboost_app as ui
from eris_ml.models.adaboost_v3 import positive_probability
from eris_ml.utils.hashing import sha256_file


class FastExplainer:
    def explain(self, frame: pd.DataFrame, cycles: int, seed: int) -> dict[str, Any]:
        names = list(frame)
        return {
            "estimated_probability": 0.25,
            "base_value": 0.15,
            "feature_values": frame.iloc[0].to_dict(),
            "feature_order": names,
            "contributions_percentage_points": [10.0] + [0.0] * 14,
            "seconds": 0.02,
            "additivity_error": 0.0,
        }


@pytest.fixture
def state(monkeypatch):
    if not ui.BUNDLE_PATH.is_dir():
        pytest.skip("Local AdaBoost s1 bundle required")
    bundle = ui.load_bundle(ui.BUNDLE_PATH)
    cold = {
        "bundle_load_seconds": 0.1,
        "explainer_init_seconds": 0.01,
        "explainer_warmup_seconds": 0.2,
        "total_cold_start_seconds": 0.31,
    }
    monkeypatch.setattr(ui, "_load_state", lambda _: (bundle, FastExplainer(), cold))
    return bundle


def test_page_schema_and_local_only(state, monkeypatch):
    with TestClient(ui.create_app()) as client:
        page = client.get("/")
        assert page.status_code == 200
        assert "Dự đoán" in page.text and "Xem giải thích" in page.text
        assert "input đã thay đổi" in page.text.lower()
        assert "không phải nguyên nhân" in page.text
        schema = client.get("/schema").json()
        assert schema["feature_count"] == len(schema["fields"]) == 15
        assert sum(not f["required"] for f in schema["fields"]) == 2
        assert schema["feature_set"] == "v3_final_15"
        assert schema["model_id"] == "adaboost-v3-final15-baseline-sigmoid-s1"
        assert schema["production_approved"] is False
        assert client.get("/openapi.json").status_code == 404
        assert client.get("/", headers={"Host": "external.example"}).status_code == 400
    for host in ("0.0.0.0", "localhost", "::1"):
        with pytest.raises(ValueError, match="127.0.0.1"):
            ui.create_app(host=host)
    monkeypatch.setenv("ERIS_ADABOOST_UI_HOST", "0.0.0.0")
    with pytest.raises(ValueError, match="127.0.0.1"):
        with TestClient(ui.create_app()):
            pass


def test_predict_has_no_shap_and_matches_bundle(state, monkeypatch):
    sample = state.metadata["manual_sample"]

    def forbidden(*args, **kwargs):
        raise AssertionError("Predict must not calculate SHAP")

    monkeypatch.setattr(FastExplainer, "explain", forbidden)
    with TestClient(ui.create_app()) as client:
        response = client.post("/predict", json=sample)
        assert response.status_code == 200
        result = response.json()
        direct = float(positive_probability(state.model, pd.DataFrame([sample]))[0])
        assert result["estimated_probability"] == pytest.approx(direct, abs=1e-15)
        assert result["prediction_seconds"] >= 0
        assert len(result["prediction_token"]) == 64
        assert "base_value" not in result and "contributions" not in result


def test_explain_requires_same_predicted_profile_and_returns_all_features(state):
    sample = state.metadata["manual_sample"]
    with TestClient(ui.create_app()) as client:
        prediction = client.post("/predict", json=sample).json()
        changed = dict(sample, Age=sample["Age"] + 1)
        mismatch = client.post(
            "/explain", json={"record": changed, "prediction_token": prediction["prediction_token"]}
        )
        assert mismatch.status_code == 409
        result = client.post(
            "/explain", json={"record": sample, "prediction_token": prediction["prediction_token"]}
        ).json()
        assert len(result["contributions"]) == 15
        assert {x["feature"] for x in result["contributions"]} == set(sample)
        assert result["base_value"] == 0.15 and result["estimated_probability"] == 0.25
        assert "không phải nguyên nhân" in result["disclaimer"]
        assert result["explanation_seconds"] == 0.02


def test_validation_optional_missing_and_no_employee_data(state, monkeypatch):
    sample = state.metadata["manual_sample"]
    original = Path.open

    def no_data(self: Path, *args: Any, **kwargs: Any) -> Any:
        assert "final_test" not in self.name and "development_raw" not in self.name
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", no_data)
    with TestClient(ui.create_app()) as client:
        missing = dict(sample, TrainingTimesLastYear=None, YearsWithCurrManager="Missing")
        result = client.post("/predict", json=missing)
        assert result.status_code == 200
        assert result.json()["imputed_features"] == [
            "TrainingTimesLastYear",
            "YearsWithCurrManager",
        ]
        for field, value in (("Department", "Unknown"), ("Age", "1e999")):
            response = client.post("/predict", json={**sample, field: value})
            assert response.status_code == 422
            assert field in response.json()["errors"]
        incomplete = dict(sample)
        incomplete.pop("Age")
        assert client.post("/predict", json=incomplete).status_code == 422


def test_real_bundle_load_predict_explain_smoke_without_changes():
    if not ui.BUNDLE_PATH.is_dir():
        pytest.skip("Local AdaBoost s1 bundle required")
    files = list(ui.BUNDLE_PATH.iterdir())
    before = {path: sha256_file(path) for path in files}
    bundle = ui.load_bundle(ui.BUNDLE_PATH)
    sample = bundle.metadata["manual_sample"]
    predicted = float(positive_probability(bundle.model, pd.DataFrame([sample]))[0])
    explanation = bundle.explainer().explain(pd.DataFrame([sample]), cycles=2, seed=42)
    assert explanation["estimated_probability"] == pytest.approx(predicted, abs=1e-15)
    assert explanation["additivity_error"] < 1e-8
    assert len(explanation["contributions"]) == 15
    assert before == {path: sha256_file(path) for path in files}
