"""Synthetic candidate freeze/report round-trip, without real employee data."""

from pathlib import Path
from shutil import copyfile

import pytest
import yaml

import eris_ml.utils.freeze as freeze
from eris_ml.utils.freeze import (
    CALIBRATION_CONFIG,
    FAIRNESS_PROTOCOL,
    FEATURE_CONFIG,
    MODEL_CONFIG,
    THRESHOLD_CONFIG,
    build_freeze_manifest,
    validate_freeze_manifest,
    verify_manifest_hashes,
)
from eris_ml.utils.freeze_reporting import write_freeze_report
from eris_ml.utils.hashing import sha256_file

ROOT = Path(__file__).resolve().parents[2]
VERSIONS = {"python": "3.12", "pandas": "test", "scikit-learn": "test",
            "xgboost": "test", "numpy": "test", "matplotlib": "test"}


def _yaml(root: Path, name: str) -> dict:
    return yaml.safe_load((root / name).read_text(encoding="utf-8"))


def test_synthetic_freeze_report_and_hash_drift(tmp_path: Path,
                                                monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (FEATURE_CONFIG, MODEL_CONFIG, CALIBRATION_CONFIG,
                 THRESHOLD_CONFIG, FAIRNESS_PROTOCOL):
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        copyfile(ROOT / name, target)
    development = tmp_path / freeze.DEVELOPMENT_CSV
    development.parent.mkdir(parents=True, exist_ok=True)
    development.write_text("source_row,EmployeeNumber,Attrition\n0,101,0\n", encoding="utf-8")
    fairness = tmp_path / freeze.FAIRNESS_CSV
    fairness.write_text("synthetic development-only fixture\n", encoding="utf-8")
    monkeypatch.setattr(freeze, "KNOWN_DEVELOPMENT_SHA256", sha256_file(development))
    manifest = build_freeze_manifest(
        tmp_path, _yaml(tmp_path, MODEL_CONFIG), _yaml(tmp_path, CALIBRATION_CONFIG),
        _yaml(tmp_path, THRESHOLD_CONFIG), _yaml(tmp_path, FAIRNESS_PROTOCOL),
        VERSIONS, created_at_utc="2026-09-20T00:00:00+00:00",
    )
    assert len(manifest["artifact_hashes"]) == 11
    assert len(manifest["missing_artifacts"]) == 4
    assert manifest["dataset"]["raw_data_hash_status"] == "missing_from_approved_evidence"
    verify_manifest_hashes(manifest, tmp_path)
    yaml_path = tmp_path / "candidate_v1.yaml"
    yaml_path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    validate_freeze_manifest(_yaml(tmp_path, "candidate_v1.yaml"))
    report = tmp_path / "candidate_freeze_v1.md"
    write_freeze_report(report, manifest)
    content = report.read_text(encoding="utf-8")
    assert "Step 10: COMPLETE" in content
    assert "retained_with_protocol_deviation" in content
    assert "AgeGroup" in content and "0.517" in content
    assert "Raw data SHA-256" in content and "missing" in content
    assert "Final-test evaluation authorized: **NO**" in content
    assert not list(tmp_path.rglob("*.joblib"))
    (tmp_path / FEATURE_CONFIG).write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="changed"):
        verify_manifest_hashes(manifest, tmp_path)
