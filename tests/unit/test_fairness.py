"""Synthetic fairness metric, disparity, bootstrap and scope-guard tests."""

from argparse import Namespace
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from eris_ml.evaluation.fairness import (
    add_audit_groups,
    bootstrap_attribute,
    disparity_metrics,
    group_metrics,
    screen_disparity,
    validate_protocol,
)
from scripts.audit_fairness import INPUTS, OUTPUTS, validate_paths

ROOT = Path(__file__).resolve().parents[2]
SUPPORT = {"total_rows": 2, "positive_rows": 1, "negative_rows": 1}


def _protocol() -> dict:
    with (ROOT / "configs/evaluation/fairness_protocol_v1.yaml").open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def test_frozen_protocol_rejects_scope_or_incident_changes() -> None:
    protocol = validate_protocol(_protocol())
    assert protocol["holdout_status"] == "retained_with_protocol_deviation"
    for key, value in (("business_approved", True), ("working_threshold", 0.4),
                       ("mixed_fairness_file_loaded_during_schema_inspection", False)):
        bad = deepcopy(protocol)
        bad[key] = value
        with pytest.raises(ValueError):
            validate_protocol(bad)


def test_fixed_age_bands_and_rejected_age_domain() -> None:
    frame = pd.DataFrame({"Age": [18, 29, 30, 39, 40, 49, 50, 80],
                          "Gender": ["A"] * 8})
    assert add_audit_groups(frame)["AgeGroup"].tolist() == [
        "18-29", "18-29", "30-39", "30-39", "40-49", "40-49", "50+", "50+"
    ]
    frame.loc[0, "Age"] = 17
    with pytest.raises(ValueError, match="Age"):
        add_audit_groups(frame)


def test_confusion_probability_metrics_and_insufficient_support() -> None:
    rows = group_metrics([1, 1, 0, 0], [1, 0, 1, 0], [0.8, 0.4, 0.6, 0.2],
                         support=SUPPORT)
    assert (rows["tn"], rows["fp"], rows["fn"], rows["tp"]) == (1, 1, 1, 1)
    assert rows["tpr"] == rows["fpr"] == rows["fnr"] == rows["tnr"] == 0.5
    assert rows["alert_rate"] == rows["precision"] == rows["npv"] == 0.5
    assert rows["brier"] == pytest.approx(0.2)
    assert rows["calibration_gap"] == pytest.approx(0)
    assert rows["ece"] is not None
    small = group_metrics([1], [0], [0.2], support=SUPPORT)
    assert small["support_status"] == "insufficient support"
    assert small["ece"] is None and small["fpr"] is None
    assert small["precision"] is None


def test_disparity_ratio_gap_and_zero_denominator() -> None:
    groups = [
        {"alert_rate": 0.2, "tpr": 0.4, "fpr": 0.1, "precision": 0.5,
         "fnr": 0.6, "brier": 0.2, "calibration_gap": -0.01,
         "support_status": "supported"},
        {"alert_rate": 0.4, "tpr": 0.7, "fpr": 0.3, "precision": 0.6,
         "fnr": 0.3, "brier": 0.1, "calibration_gap": 0.04,
         "support_status": "supported"},
    ]
    disparity = disparity_metrics(groups)
    assert disparity["selection_rate_difference"] == pytest.approx(0.2)
    assert disparity["selection_rate_ratio"] == 0.5
    assert disparity["equalized_odds_gap"] == pytest.approx(0.3)
    assert disparity["calibration_gap_difference"] == pytest.approx(0.05)
    for group in groups:
        group["alert_rate"] = 0
    assert disparity_metrics(groups)["selection_rate_ratio"] is None


def test_screening_never_passes_unsupported_groups() -> None:
    groups = [{"support_status": "insufficient support", "calibration_gap": 0.1}]
    disparity = {"supported_groups": 0, "group_count": 1}
    assert screen_disparity(groups, disparity, {},
                            _protocol()["screening_thresholds"]) == "INSUFFICIENT_EVIDENCE"


def test_paired_stratified_bootstrap_is_deterministic() -> None:
    frame = pd.DataFrame({
        "Gender": ["A"] * 4 + ["B"] * 4,
        "Attrition": [0, 0, 1, 1] * 2,
        "prediction": [0, 1, 0, 1, 0, 0, 1, 1],
        "probability": [0.1, 0.6, 0.4, 0.8, 0.2, 0.3, 0.7, 0.9],
    })
    first = bootstrap_attribute(frame, "Gender", "prediction", "probability", SUPPORT,
                                iterations=40, seed=42)
    second = bootstrap_attribute(frame, "Gender", "prediction", "probability", SUPPORT,
                                 iterations=40, seed=42)
    assert first == second
    assert first[0]["A"]["tpr"][2] == 40
    assert first[1]["tpr_difference"][2] == 40


def test_cli_rejects_quarantine_and_final_test_paths_before_opening() -> None:
    args = Namespace(**{**INPUTS, **OUTPUTS})
    args.fairness_data = Path("data/quarantine/fairness_audit_mixed_v1.csv")
    with pytest.raises(ValueError, match="Quarantined"):
        validate_paths(args)
    args.fairness_data = Path("data/processed/final_test_raw_v1.csv")
    with pytest.raises(ValueError, match="prohibited"):
        validate_paths(args)


@pytest.mark.parametrize("column", ["Gender", "AgeGroup", "MaritalStatus",
                                      "Gender_AgeGroup"])
def test_invalid_bootstrap_data_does_not_accept_missing_attribute(column: str) -> None:
    frame = pd.DataFrame({"Attrition": [0, 1], "prediction": [0, 1],
                          "probability": np.array([0.2, 0.8])})
    with pytest.raises(KeyError):
        bootstrap_attribute(frame, column, "prediction", "probability", SUPPORT,
                            iterations=2)
