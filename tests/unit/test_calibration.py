"""Unit checks for strict calibration configuration and probability metrics."""

from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import log_loss

from eris_ml.data.splitting import assert_development_path
from eris_ml.evaluation.calibration_reporting import (
    calibration_metrics,
    candidate_configuration,
    reliability_bins,
    select_method,
    validate_probabilities,
)
from eris_ml.models.calibration import (
    load_outer_best_parameters,
    validate_calibration_protocol,
    validate_xgboost_parameters,
)
from eris_ml.models.tuning import load_yaml_config

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "configs/calibration/calibration_protocol_v1.yaml"
PARAMETERS = {
    "n_estimators": 5, "max_depth": 1, "learning_rate": 0.1,
    "min_child_weight": 1, "subsample": 0.8, "colsample_bytree": 0.8,
    "gamma": 0.0, "reg_alpha": 0.1, "reg_lambda": 3.0,
}


def test_protocol_requires_approved_methods_seeds_and_bins() -> None:
    protocol = validate_calibration_protocol(load_yaml_config(PROTOCOL))
    assert protocol["methods"] == ["raw", "sigmoid", "isotonic"]
    assert protocol["outer_cv"]["n_splits"] == 5
    assert protocol["calibration_cv"]["n_splits"] == 4
    for section, key, value in (
        (None, "methods", ["raw", "sigmoid"]),
        (None, "methods", ["raw", "sigmoid", "platt"]),
        ("calibration_cv", "random_state", 42),
        ("calibration_bins", "strategy", "uniform"),
        (None, "operational_threshold", 0.4),
    ):
        invalid = deepcopy(protocol)
        target = invalid if section is None else invalid[section]
        target[key] = value
        with pytest.raises(ValueError):
            validate_calibration_protocol(invalid)


def test_brier_log_loss_ece_mce_and_extreme_probabilities() -> None:
    labels = [0, 1, 0, 1]
    scores = [0.1, 0.4, 0.6, 0.9]
    metrics = calibration_metrics(labels, scores, bins=2)
    assert metrics["brier_score"] == pytest.approx(0.185)
    assert metrics["log_loss"] == pytest.approx(log_loss(labels, scores))
    assert metrics["expected_calibration_error"] == pytest.approx(0.25)
    assert metrics["maximum_calibration_error"] == pytest.approx(0.25)
    extreme = calibration_metrics([0, 1], [0, 1])
    assert np.isfinite(extreme["log_loss"])
    assert extreme["brier_score"] == 0


def test_quantile_bins_keep_duplicates_and_skip_empty_bins() -> None:
    rows = reliability_bins([0, 0, 1, 1], [0.2, 0.2, 0.2, 0.8], count=10)
    assert sum(row["count"] for row in rows) == 4
    assert len(rows) <= 2
    assert all(np.isfinite(row["absolute_gap"]) for row in rows)
    single = reliability_bins([0, 1], [0.5, 0.5], count=10)
    assert single == [{
        "bin_id": 1, "count": 2, "mean_predicted_probability": 0.5,
        "observed_positive_rate": 0.5, "absolute_gap": 0.0,
    }]


@pytest.mark.parametrize("scores", [[-0.1, 0.5], [0.5, 1.1], [np.nan, 0.4], [np.inf, 0.4]])
def test_probability_validation_rejects_invalid_scores(scores) -> None:
    with pytest.raises(ValueError):
        validate_probabilities([0, 1], scores)


def test_selection_keeps_raw_when_guardrail_or_fold_stability_fails() -> None:
    raw = {
        "brier_score": 0.10, "log_loss": 0.30,
        "expected_calibration_error": 0.06, "average_precision": 0.65, "roc_auc": 0.84,
    }
    sigmoid = {**raw, "brier_score": 0.09, "log_loss": 0.29,
               "expected_calibration_error": 0.04}
    isotonic = {**sigmoid, "brier_score": 0.08, "average_precision": 0.63}
    folds = {
        "raw": [{"brier_score": 0.10}] * 5,
        "sigmoid": [{"brier_score": 0.09}] * 4 + [{"brier_score": 0.11}],
        "isotonic": [{"brier_score": 0.08}] * 5,
    }
    decision = select_method({"raw": raw, "sigmoid": sigmoid, "isotonic": isotonic}, folds)
    assert decision["method"] == "sigmoid"
    assert decision["improved_folds"]["sigmoid"] == 4
    folds["sigmoid"] = [{"brier_score": 0.09}] * 2 + [{"brier_score": 0.11}] * 3
    assert select_method(
        {"raw": raw, "sigmoid": sigmoid, "isotonic": isotonic}, folds
    )["method"] == "raw"


def test_xgboost_outer_parameters_require_unique_rank_one_and_allowlist() -> None:
    import json

    records = pd.DataFrame([
        {"outer_fold": 1, "rank": 1, "parameters": json.dumps(PARAMETERS)},
        {"outer_fold": 2, "rank": 1, "parameters": json.dumps(PARAMETERS)},
        {"outer_fold": "full", "rank": 1, "parameters": json.dumps(PARAMETERS)},
    ])
    selected = load_outer_best_parameters(records, 2)
    assert set(selected) == {1, 2}
    with pytest.raises(ValueError, match="exactly one"):
        load_outer_best_parameters(pd.concat([records, records.iloc[[0]]]), 2)
    with pytest.raises(ValueError, match="allowlist"):
        validate_xgboost_parameters({**PARAMETERS, "scale_pos_weight": 2.0})
    with pytest.raises(ValueError, match="max_depth"):
        validate_xgboost_parameters({**PARAMETERS, "max_depth": 2})
    records["model"] = "logistic_regression"
    with pytest.raises(ValueError, match="non-XGBoost"):
        load_outer_best_parameters(records, 2)


def test_final_test_path_guard_and_unfitted_candidate_config() -> None:
    with pytest.raises(ValueError, match="prohibited"):
        assert_development_path(Path("data/processed/final_test_raw_v1.csv"))
    candidate = candidate_configuration("xgboost", "raw")
    assert candidate["calibration"]["method"] == "none"
    assert candidate["selection"]["production_approved"] is False
    assert "predictions" not in candidate
