"""Synthetic-only orchestration tests; no real locked final-test access."""

import json
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from eris_ml.evaluation import final_evaluation as evaluation
from eris_ml.features.definitions import FeatureDefinition


def _frames() -> tuple[pd.DataFrame, pd.DataFrame, FeatureDefinition]:
    names = tuple(f"feature_{index}" for index in range(25))
    definition = FeatureDefinition(
        version="v1-full", nominal=(), ordinal=(), numeric=names,
        target=("Attrition",), identifiers=("EmployeeNumber",),
        provenance=("source_row",), excluded=(),
    )

    def make_frame(rows: int, positives: int, offset: int) -> pd.DataFrame:
        frame = pd.DataFrame({name: np.full(rows, index, dtype=float)
                              for index, name in enumerate(names)})
        frame["source_row"] = np.arange(offset, offset + rows)
        frame["EmployeeNumber"] = np.arange(offset + 10000, offset + 10000 + rows)
        frame["Attrition"] = np.array([0] * (rows - positives) + [1] * positives)
        return frame

    return make_frame(1176, 190, 0), make_frame(294, 47, 2000), definition


def _setup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
           *, fail_fit: bool = False, fail_after_access: bool = False,
           candidate: str = evaluation.CANDIDATE_NAME,
           threshold: float = evaluation.THRESHOLD) -> Mock:
    development, final, definition = _frames()
    ledger = tmp_path / evaluation.LEDGER_PATH
    evaluation.atomic_yaml(ledger, evaluation.initial_ledger())
    manifest_path = tmp_path / evaluation.MANIFEST_PATH
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text("synthetic: true\n", encoding="utf-8")
    auth = evaluation.authorization_record("a" * 64, "2026-09-20T03:00:00+00:00")
    manifest = {
        "candidate_name": candidate,
        "feature_schema": {"feature_count": 25},
        "model": {"family": "xgboost", "parameters": evaluation.MODEL_PARAMETERS},
        "calibration": {"method": "none", "probability_type": "raw"},
        "threshold": {"value": threshold, "policy": "capacity_15_percent"},
        "fairness": {"status": "review_known_limitation"},
        "holdout": {"status": "retained_with_protocol_deviation"},
        "approvals": {"business_approved": False, "production_approved": False},
    }
    monkeypatch.setattr(evaluation, "preflight", lambda root: (
        manifest, auth, development, definition,
        development["Attrition"],
    ))
    fit = Mock()
    if fail_fit:
        fit.side_effect = ValueError("synthetic fit failure")
    else:
        pipeline = Mock()
        pipeline.predict_proba.return_value = np.column_stack((
            np.full(294, 0.9), np.full(294, 0.1),
        ))
        fit.return_value = pipeline
    monkeypatch.setattr(evaluation, "fit_frozen", fit)

    def synthetic_read(path: Path) -> tuple[pd.DataFrame, str]:
        assert path == tmp_path / evaluation.FINAL_TEST_PATH
        assert evaluation.load_yaml(ledger)["status"] == "started"
        assert fit.called
        if fail_after_access:
            raise ValueError("synthetic schema failure")
        return final, "f" * 64

    monkeypatch.setattr(evaluation, "read_final_once", synthetic_read)
    monkeypatch.setattr(evaluation, "manifest_sha256", lambda path: "a" * 64)
    monkeypatch.setattr(evaluation, "stratified_bootstrap", lambda y, p: {
        "method": "synthetic", "confidence_level": 0.95,
        "iterations_requested": 2, "valid_iterations": 2, "random_state": 42,
        "intervals": {name: [0.0, 1.0] for name in evaluation.CI_METRICS},
    })
    return fit


def test_synthetic_one_time_run_outputs_and_rerun_guard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _setup(tmp_path, monkeypatch)
    result = evaluation.execute_once(tmp_path)
    ledger = evaluation.load_yaml(tmp_path / evaluation.LEDGER_PATH)
    assert ledger["run_count"] == 1 and ledger["status"] == "completed"
    assert result["row_count"] == 294 and result["business_approved"] is False
    assert result["result"] == "DOES_NOT_MEET_PREDECLARED_GUARDRAILS"
    report = (tmp_path / evaluation.REPORT_PATH).read_text(encoding="utf-8")
    assert "retained_with_protocol_deviation" in report
    assert "Final-test results must not be used to revise this candidate" in report
    json_result = json.loads((tmp_path / evaluation.METRICS_PATH).read_text(encoding="utf-8"))
    assert json_result["result"] == result["result"]
    rows = pd.read_csv(tmp_path / evaluation.PREDICTIONS_PATH)
    assert len(rows) == 294 and rows["threshold"].eq(evaluation.THRESHOLD).all()
    assert not list(tmp_path.rglob("*.joblib")) and not list(tmp_path.rglob("*.pkl"))
    with pytest.raises(ValueError, match="already used"):
        evaluation.execute_once(tmp_path)


def test_synthetic_failure_before_access(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _setup(tmp_path, monkeypatch, fail_fit=True)
    with pytest.raises(ValueError, match="fit failure"):
        evaluation.execute_once(tmp_path)
    ledger = evaluation.load_yaml(tmp_path / evaluation.LEDGER_PATH)
    assert ledger["run_count"] == 0 and ledger["status"] == "failed_before_access"


def test_synthetic_failure_after_access(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _setup(tmp_path, monkeypatch, fail_after_access=True)
    with pytest.raises(ValueError, match="schema failure"):
        evaluation.execute_once(tmp_path)
    ledger = evaluation.load_yaml(tmp_path / evaluation.LEDGER_PATH)
    assert ledger["run_count"] == 1 and ledger["status"] == "failed_after_access"


@pytest.mark.parametrize("drift", ["candidate", "threshold"])
def test_synthetic_frozen_drift_rejected_before_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, drift: str,
) -> None:
    kwargs = ({"candidate": "wrong-model"} if drift == "candidate"
              else {"threshold": 0.4})
    fit = _setup(tmp_path, monkeypatch, **kwargs)
    with pytest.raises(ValueError, match="Runtime candidate differs"):
        evaluation.execute_once(tmp_path)
    assert not fit.called
    ledger = evaluation.load_yaml(tmp_path / evaluation.LEDGER_PATH)
    assert ledger["run_count"] == 0 and ledger["status"] == "failed_before_access"
