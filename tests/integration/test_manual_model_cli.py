"""Local wrapper/CLI and protected API agree on the same synthetic records."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

from fastapi.testclient import TestClient

from eris_ml.api.main import create_app
from eris_ml.settings import Settings

ROOT = Path(__file__).resolve().parents[2]
WRAPPER = ROOT / "scripts/manual_model_test.ps1"
EXAMPLES = json.loads((ROOT / "tests/fixtures/manual_prediction_examples.json").read_text(
    encoding="utf-8"
))


def _run_wrapper(*args: str, input_text: str | None = None) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.pop("ERIS_SERVICE_TOKEN", None)
    env["PYTHONIOENCODING"] = "utf-8"
    return subprocess.run(
        ["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
         str(WRAPPER), *args],
        input=input_text, cwd=ROOT, env=env, text=True, encoding="utf-8",
        capture_output=True, timeout=45, check=False,
    )


def test_low_high_examples_and_json_without_service_token() -> None:
    for signal in ("low", "high"):
        process = _run_wrapper("-Example", signal, "-NoExplanation", "-JsonOutput")
        assert process.returncode == 0, process.stderr
        result = json.loads(process.stdout)
        assert result["record_id"] == EXAMPLES[f"synthetic_{signal}_signal"]["record_id"]
        assert result["candidate_version"] == "eris-xgboost-v1"
        assert result["bundle_version"] == "bundle-v1"
        assert result["feature_schema_version"] == "v1-full"
        assert result["threshold"] == 0.345651
        assert result["alert"] == (result["probability"] >= result["threshold"])
        assert 0 <= result["probability"] <= 1
        assert result["top_factors"] == []
        assert result["production_approved"] is False


def test_local_cli_matches_protected_api_for_same_synthetic_record() -> None:
    process = _run_wrapper("-Example", "low", "-NoExplanation", "-JsonOutput")
    assert process.returncode == 0, process.stderr
    local = json.loads(process.stdout)
    token = "synthetic-test-only-service-credential-0003"
    with TestClient(create_app(Settings(
        _env_file=None, service_token=token,
        model_bundle_path=ROOT / "artifacts/models/eris_xgboost_v1.joblib",
        model_metadata_path=ROOT / "artifacts/models/eris_xgboost_v1.metadata.json",
        model_checksum_path=ROOT / "artifacts/models/eris_xgboost_v1.sha256",
    ))) as client:
        response = client.post("/api/v1/predict?include_explanation=false",
                               json=EXAMPLES["synthetic_low_signal"],
                               headers={"Authorization": f"Bearer {token}"})
        assert response.status_code == 200
        api = response.json()
    for name in ("probability", "threshold", "alert", "decision_label",
                 "candidate_version", "bundle_version", "feature_schema_version"):
        assert local[name] == api[name]


def test_cli_input_file_validation_and_no_protected_data_access(tmp_path: Path) -> None:
    good = tmp_path / "example.json"
    good.write_text(json.dumps(EXAMPLES["synthetic_low_signal"]), encoding="utf-8")
    assert _run_wrapper("-InputFile", str(good), "-NoExplanation").returncode == 0
    bad = tmp_path / "missing.json"
    bad.write_text(json.dumps({"Age": 34}), encoding="utf-8")
    process = _run_wrapper("-InputFile", str(bad), "-NoExplanation")
    assert process.returncode != 0 and "Invalid or missing" in process.stderr
    prohibited = _run_wrapper("-InputFile", "data/processed/final_test_raw_v1.csv")
    assert prohibited.returncode != 0
    assert "regular JSON" in prohibited.stderr


def test_interactive_wrapper_changes_a_feature_before_prediction() -> None:
    # Menu 1, change BusinessTravel from the low example, accept the other 24,
    # confirm, then leave the menu. The subprocess has no service token.
    answers = "1\n1\n" + ("\n" * 24) + "y\n5\n"
    process = _run_wrapper("-NoExplanation", input_text=answers)
    assert process.returncode == 0, process.stderr
    assert "Review all 25 feature values" in process.stdout
    assert "BusinessTravel: Non-Travel" in process.stdout
    assert "ERIS Prediction Result" in process.stdout
    assert "Threshold: 0.345651" in process.stdout
    assert "Production approved: false" in process.stdout
