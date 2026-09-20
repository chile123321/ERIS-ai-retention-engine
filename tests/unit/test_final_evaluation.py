"""Synthetic-only tests for one-time authorization, metrics and ledger logic."""

from copy import deepcopy

import numpy as np
import pytest

from eris_ml.evaluation.final_evaluation import (
    GUARDRAILS,
    THRESHOLD,
    assert_unused_ledger,
    atomic_yaml,
    authorization_record,
    final_metrics,
    guardrail_decision,
    initial_ledger,
    load_yaml,
    mark_failure,
    mark_started,
    stratified_bootstrap,
    validate_authorization,
)


def test_authorization_is_bound_to_manifest_and_registered_guardrails() -> None:
    record = authorization_record("a" * 64, "2026-09-20T03:00:00+00:00")
    validate_authorization(record, "a" * 64)
    assert record["pre_registered_guardrails"] == GUARDRAILS
    assert record["restrictions"]["candidate_comparison_allowed"] is False
    with pytest.raises(ValueError, match="Authorization differs"):
        validate_authorization(record, "b" * 64)
    changed = deepcopy(record)
    changed["pre_registered_guardrails"]["recall_minimum"] = 0.40
    with pytest.raises(ValueError, match="Authorization differs"):
        validate_authorization(changed, "a" * 64)


def test_atomic_one_time_ledger_and_terminal_states(tmp_path: object) -> None:
    from pathlib import Path

    path = Path(str(tmp_path)) / "record.yaml"
    atomic_yaml(path, initial_ledger())
    started = mark_started(path, load_yaml(path))
    assert load_yaml(path) == started
    assert started["run_count"] == 1 and started["status"] == "started"
    with pytest.raises(ValueError, match="already used"):
        mark_started(path, load_yaml(path))
    mark_failure(path, started, after_access=True)
    assert load_yaml(path)["status"] == "failed_after_access"
    with pytest.raises(ValueError, match="already used"):
        assert_unused_ledger(load_yaml(path))


def test_pre_access_failure_is_not_counted_as_final_access(tmp_path: object) -> None:
    from pathlib import Path

    path = Path(str(tmp_path)) / "record.yaml"
    atomic_yaml(path, initial_ledger())
    mark_failure(path, load_yaml(path), after_access=False)
    record = load_yaml(path)
    assert record["run_count"] == 0
    assert record["status"] == "failed_before_access"


def test_metrics_f2_zero_denominator_and_frozen_threshold() -> None:
    scores = final_metrics([0, 0, 1, 1], [0.01, 0.02, 0.03, 0.04])
    assert scores["tp"] == 0 and scores["fp_per_tp"] is None
    assert scores["fn_per_tp"] is None and scores["f2"] == 0
    perfect = final_metrics([0, 0, 1, 1], [0.01, 0.02, 0.9, 0.8])
    assert perfect["f2"] == 1 and perfect["brier"] > 0
    with pytest.raises(ValueError, match="frozen threshold"):
        final_metrics([0, 1], [0.1, 0.9], threshold=0.5)
    assert THRESHOLD == 0.345651


def test_probability_validation_and_deterministic_paired_bootstrap() -> None:
    y = np.array([0, 0, 0, 1, 1, 1])
    p = np.array([0.1, 0.2, 0.3, 0.6, 0.7, 0.8])
    first = stratified_bootstrap(y, p, iterations=15, seed=42)
    assert first == stratified_bootstrap(y, p, iterations=15, seed=42)
    assert first["valid_iterations"] == 15
    assert set(first["intervals"]) == {
        "pr_auc", "roc_auc", "brier", "precision", "recall", "f1", "f2", "alert_rate",
    }
    with pytest.raises(ValueError, match="Probabilities"):
        final_metrics([0, 1], [0.2, float("nan")])


def test_guardrail_result_follows_all_six_registered_checks() -> None:
    passing = {
        "pr_auc": 0.6, "roc_auc": 0.8, "brier": 0.1,
        "precision": 0.5, "recall": 0.5, "alert_rate": 0.15,
    }
    result, checks = guardrail_decision(passing)
    assert result == "PASS_WITH_KNOWN_LIMITATIONS" and all(checks.values())
    failing = {**passing, "recall": 0.44}
    result, checks = guardrail_decision(failing)
    assert result == "DOES_NOT_MEET_PREDECLARED_GUARDRAILS"
    assert checks["recall_minimum"] is False
