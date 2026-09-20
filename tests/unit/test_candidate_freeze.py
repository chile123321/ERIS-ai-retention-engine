"""Synthetic candidate-freeze scope, manifest and hash validation tests."""

from argparse import Namespace
from copy import deepcopy
from pathlib import Path

import pytest
import yaml

import eris_ml.utils.freeze as freeze
from eris_ml.utils.freeze import (
    FEATURE_CONFIG,
    HASH_PATHS,
    KNOWN_DEVELOPMENT_SHA256,
    MODEL_CONFIG,
    assert_freeze_path,
    build_freeze_manifest,
    hash_approved_artifacts,
    validate_freeze_manifest,
)
from scripts.freeze_candidate import INPUTS, OUTPUTS, validate_paths

VERSIONS = {"python": "3.12", "pandas": "test", "scikit-learn": "test",
            "xgboost": "test", "numpy": "test", "matplotlib": "test"}


def _candidate() -> dict:
    return {
        "threshold": 0.3456506431102752,
        "expected_alert_rate": 0.14540816326530612,
        "expected_precision": 0.6549707602339181,
        "expected_recall": 0.5894736842105263,
        "cross_fitted_alert_rate": 0.14625850340136054,
        "cross_fitted_precision": 0.6453488372093024,
        "cross_fitted_recall": 0.5842105263157895,
        "feasible_folds": 5,
    }


@pytest.fixture
def manifest(monkeypatch: pytest.MonkeyPatch) -> dict:
    monkeypatch.setattr(freeze, "validate_sources", lambda *args: _candidate())
    hashes = {name: {"status": "present", "sha256": "a" * 64} for name in HASH_PATHS}
    hashes[freeze.DEVELOPMENT_CSV]["sha256"] = KNOWN_DEVELOPMENT_SHA256
    monkeypatch.setattr(freeze, "hash_approved_artifacts", lambda *args: hashes)
    return build_freeze_manifest(Path("synthetic"), {}, {}, {}, {}, VERSIONS,
                                 created_at_utc="2026-09-20T00:00:00+00:00")


def test_manifest_has_research_only_scope_and_explicit_incident(manifest: dict) -> None:
    validate_freeze_manifest(manifest)
    assert manifest["candidate_name"] == "eris-xgboost-v1"
    assert manifest["feature_schema"]["feature_count"] == 25
    assert manifest["calibration"]["method"] == "none"
    assert manifest["threshold"]["value"] == 0.345651
    assert manifest["fairness"]["status"] == "review_known_limitation"
    assert manifest["holdout"]["mixed_fairness_file_loaded_during_schema_inspection"] is True
    assert manifest["approvals"] == {
        "capstone_evaluation_approved": True, "business_approved": False,
        "production_approved": False, "final_test_evaluation_authorized": False,
    }
    assert manifest["dataset"]["raw_data_sha256"] is None
    validate_freeze_manifest(yaml.safe_load(yaml.safe_dump(manifest)))


@pytest.mark.parametrize(("path", "value"), [
    (("feature_schema", "feature_count"), 24),
    (("feature_schema", "config"), "configs/features/other.yaml"),
    (("model", "family"), "logistic"),
    (("model", "config"), "configs/models/xgboost.yaml"),
    (("calibration", "method"), "sigmoid"),
    (("threshold", "value"), 0.4),
    (("threshold", "policy"), "capacity_20_percent"),
    (("fairness", "status"), "pass"),
    (("fairness", "age_group"), {"status": "PASS"}),
    (("holdout", "status"), "clean"),
    (("holdout", "mixed_fairness_file_loaded_during_schema_inspection"), False),
    (("approvals", "business_approved"), True),
    (("approvals", "production_approved"), True),
    (("approvals", "final_test_evaluation_authorized"), True),
    (("scope", "production"), True),
])
def test_rejects_any_release_scope_drift(
    manifest: dict, path: tuple[str, str], value: object
) -> None:
    changed = deepcopy(manifest)
    changed[path[0]][path[1]] = value
    with pytest.raises(ValueError):
        validate_freeze_manifest(changed)


def test_hash_is_deterministic_and_missing_artifact_is_explicit(tmp_path: Path) -> None:
    path = tmp_path / FEATURE_CONFIG
    path.parent.mkdir(parents=True)
    path.write_text("synthetic feature config", encoding="utf-8")
    first = hash_approved_artifacts(tmp_path, (FEATURE_CONFIG, MODEL_CONFIG))
    assert first == hash_approved_artifacts(tmp_path, (FEATURE_CONFIG, MODEL_CONFIG))
    assert first[FEATURE_CONFIG]["status"] == "present"
    assert len(first[FEATURE_CONFIG]["sha256"]) == 64
    assert first[MODEL_CONFIG] == {"status": "missing", "sha256": None}


def test_forbidden_paths_rejected_before_file_access(tmp_path: Path) -> None:
    for name in ("data/processed/final_test_raw_v1.csv",
                 "data/quarantine/fairness_audit_mixed_v1.csv"):
        with pytest.raises(ValueError, match="prohibited"):
            assert_freeze_path(Path(name), tmp_path, allowed=HASH_PATHS)
    args = Namespace(**{**INPUTS, **OUTPUTS})
    args.feature_config = Path("data/quarantine/fairness_audit_mixed_v1.csv")
    with pytest.raises(ValueError, match="Quarantined"):
        validate_paths(args)
    args.feature_config = Path("data/processed/final_test_raw_v1.csv")
    with pytest.raises(ValueError, match="prohibited"):
        validate_paths(args)
