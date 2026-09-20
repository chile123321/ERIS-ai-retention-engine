"""Synthetic end-to-end fairness reporting; never access real or locked-test rows."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from eris_ml.evaluation.fairness import join_audit_inputs, run_fairness_audit
from eris_ml.evaluation.fairness_reporting import plot_fairness_dashboard, write_fairness_report

ROOT = Path(__file__).resolve().parents[2]


def _fixture():
    count = 12
    source = np.arange(count)
    employee = source + 1000
    target = np.array([0, 1] * 6)
    ages = [24, 25, 35, 36, 45, 46, 51, 52, 24, 36, 45, 52]
    development = pd.DataFrame({"source_row": source, "EmployeeNumber": employee,
                                "Age": ages, "Attrition": target})
    fairness = development.copy()
    fairness["split"] = "development"
    fairness["Gender"] = ["Female", "Male"] * 6
    fairness["MaritalStatus"] = ["Single", "Married", "Divorced"] * 4
    xgb = np.array([0.2, 0.8, 0.4, 0.7, 0.3, 0.6, 0.1, 0.9, 0.2, 0.8, 0.4, 0.7])
    logistic = np.array([0.3, 0.7, 0.45, 0.65, 0.35, 0.55,
                         0.2, 0.8, 0.3, 0.7, 0.45, 0.65])
    calibration = development[["source_row", "EmployeeNumber", "Attrition"]].copy()
    calibration["outer_fold"] = source % 3 + 1
    calibration["xgboost_raw_probability"] = xgb
    calibration["logistic_raw_probability"] = logistic
    policy = calibration.copy()
    for model, scores in (("xgboost", xgb), ("logistic", logistic)):
        policy[f"{model}_capacity_15_percent_threshold"] = 0.5
        policy[f"{model}_capacity_15_percent_prediction"] = (scores >= 0.5).astype(int)
    with (ROOT / "configs/evaluation/fairness_protocol_v1.yaml").open(encoding="utf-8") as stream:
        protocol = yaml.safe_load(stream)
    return development, fairness, calibration, policy, protocol


def test_fairness_audit_end_to_end_without_training(tmp_path: Path) -> None:
    development, fairness, calibration, policy, protocol = _fixture()
    frame = join_audit_inputs(development, fairness, calibration, policy,
                              expected_rows=12, expected_target_counts={0: 6, 1: 6},
                              forbidden_employee_numbers={9999})
    assert len(frame) == 12
    assert set(frame["AgeGroup"]) == {"18-29", "30-39", "40-49", "50+"}
    results = run_fairness_audit(frame, protocol, bootstrap_iterations=12)
    assert len(results["metrics"]) == 2 * (2 + 4 + 3 + 8)
    assert len(results["sensitivity"]) == 8
    assert set(results["disparities"]["status"]) == {"INSUFFICIENT_EVIDENCE"}
    assert results["comparison"]["xgboost"]["overall"]["tp"] == 6
    assert results["comparison"]["xgboost"]["overall"]["fp"] == 0
    report = tmp_path / "fairness_audit_v1.md"
    dashboard = tmp_path / "fairness_dashboard_v1.png"
    write_fairness_report(report, frame, results, protocol, dashboard.name)
    plot_fairness_dashboard(dashboard, results)
    text = report.read_text(encoding="utf-8")
    assert "retained_with_protocol_deviation" in text
    assert "synthetic benchmark" in text
    assert "Fixed-threshold sensitivity" in text
    assert "| 0.345651 | AgeGroup |" in text
    assert dashboard.is_file() and dashboard.stat().st_size > 0


def test_join_rejects_probability_mismatch_or_missing_row() -> None:
    development, fairness, calibration, policy, _ = _fixture()
    policy.loc[0, "xgboost_raw_probability"] += 0.01
    with pytest.raises(ValueError, match="probabilities differ"):
        join_audit_inputs(development, fairness, calibration, policy,
                          expected_rows=12, expected_target_counts={0: 6, 1: 6})
    policy = policy.iloc[:-1]
    with pytest.raises(ValueError, match="row count"):
        join_audit_inputs(development, fairness, calibration, policy,
                          expected_rows=12, expected_target_counts={0: 6, 1: 6})


def test_join_rejects_non_development_fairness_rows() -> None:
    development, fairness, calibration, policy, _ = _fixture()
    fairness.loc[0, "split"] = "final_test"
    with pytest.raises(ValueError, match="split=development"):
        join_audit_inputs(development, fairness, calibration, policy,
                          expected_rows=12, expected_target_counts={0: 6, 1: 6})
