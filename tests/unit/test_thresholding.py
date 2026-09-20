"""Unit tests for business constraints, threshold arithmetic, and safety."""

from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from eris_ml.data.splitting import assert_development_path
from eris_ml.evaluation.thresholding import (
    confusion_metrics,
    metrics_at_threshold,
    per_thousand,
    scenarios,
    select_threshold,
    threshold_curve,
    threshold_stability,
    validate_policy,
    validate_probability_vector,
)
from eris_ml.models.tuning import load_yaml_config

ROOT = Path(__file__).resolve().parents[2]


def test_policy_contains_only_unapproved_scenarios() -> None:
    policy = validate_policy(load_yaml_config(ROOT / "configs/threshold/threshold_policy_v1.yaml"))
    assert len(scenarios(policy)) == 9
    assert policy["business_approved"] is False
    invalid = deepcopy(policy)
    invalid["business_approved"] = True
    with pytest.raises(ValueError, match="unapproved"):
        validate_policy(invalid)
    invalid = deepcopy(policy)
    invalid["capacity_scenarios"][0]["max_alert_rate"] = 1.1
    with pytest.raises(ValueError, match="Capacity"):
        validate_policy(invalid)


def test_boundary_thresholds_and_zero_denominators() -> None:
    labels, scores = [0, 1, 0, 1], [0.1, 0.4, 0.4, 0.9]
    at_zero = metrics_at_threshold(labels, scores, 0)
    at_one = metrics_at_threshold(labels, scores, 1)
    assert (at_zero["alerts"], at_zero["tp"], at_zero["fp"]) == (4, 2, 2)
    assert (at_one["alerts"], at_one["precision"], at_one["recall"]) == (0, 0, 0)
    assert at_one["fp_per_tp"] is None and at_one["fn_per_tp"] is None
    assert confusion_metrics(0, 0, 2, 0)["specificity"] == 0
    curve = threshold_curve(labels, scores)
    assert set([0.0, 1.0, *scores]) == set(curve["threshold"])
    assert len(curve) == 5  # two duplicate 0.4 values share one boundary
    for _, row in curve.iterrows():
        direct = metrics_at_threshold(labels, scores, row["threshold"])
        assert (row["tn"], row["fp"], row["fn"], row["tp"]) == (
            direct["tn"], direct["fp"], direct["fn"], direct["tp"]
        )


def test_capacity_recall_cost_and_tie_breaking() -> None:
    curve = threshold_curve([0, 1, 0, 1], [0.1, 0.2, 0.8, 0.9])
    capacity = select_threshold(curve, {"kind": "capacity", "value": 0.25})
    assert capacity is not None and capacity["threshold"] == 0.9
    recall = select_threshold(curve, {"kind": "recall", "value": 1.0})
    assert recall is not None and recall["threshold"] == 0.2
    cost = select_threshold(curve, {"kind": "cost", "value": 2.0})
    assert cost is not None
    assert cost["total_cost"] == cost["fp"] + 2 * cost["fn"]
    assert cost["normalized_cost"] == cost["total_cost"] / 4
    fake = pd.DataFrame([
        {"threshold": 0.2, "recall": 0.5, "precision": 0.5, "alert_rate": 0.1,
         "fp": 1, "fn": 1, "tn": 1, "tp": 1},
        {"threshold": 0.4, "recall": 0.5, "precision": 0.5, "alert_rate": 0.1,
         "fp": 1, "fn": 1, "tn": 1, "tp": 1},
    ])
    tied = select_threshold(fake, {"kind": "capacity", "value": 0.1})
    assert tied is not None and tied["threshold"] == 0.4
    assert select_threshold(fake, {"kind": "capacity", "value": 0.05}) is None


@pytest.mark.parametrize(
    "scenario",
    [
        {"kind": "capacity", "value": -0.1},
        {"kind": "recall", "value": 1.1},
        {"kind": "cost", "value": 0},
        {"kind": "cost", "value": np.inf},
    ],
)
def test_invalid_constraints_are_rejected(scenario) -> None:
    curve = threshold_curve([0, 1], [0.2, 0.8])
    with pytest.raises(ValueError):
        select_threshold(curve, scenario)


@pytest.mark.parametrize("scores", [[np.nan, 0.5], [-0.1, 0.5], [np.inf, 0.5]])
def test_invalid_probabilities_are_rejected(scores) -> None:
    with pytest.raises(ValueError):
        validate_probability_vector([0, 1], scores)


def test_threshold_stability_and_per_thousand() -> None:
    assert threshold_stability([0.2, 0.3, 0.4])["median"] == pytest.approx(0.3)
    assert threshold_stability([0.1, 0.9])["unstable"] is True
    rates = per_thousand({"tn": 70, "fp": 10, "fn": 15, "tp": 5})
    assert rates == {
        "alerts": 150, "true_positives": 50,
        "false_positives": 100, "missed_attrition": 150,
    }


def test_final_test_guard_rejects_path_before_opening() -> None:
    with pytest.raises(ValueError, match="prohibited"):
        assert_development_path(Path("data/processed/final_test_raw_v1.csv"))
