"""Unit tests for approved binary-classification metrics."""

import numpy as np
import pytest

from eris_ml.evaluation.metrics import calculate_metrics


def test_calculate_metrics_returns_expected_classification_values() -> None:
    metrics = calculate_metrics([0, 0, 1, 1], [0.1, 0.6, 0.4, 0.9])

    assert metrics["accuracy"] == pytest.approx(0.5)
    assert metrics["balanced_accuracy"] == pytest.approx(0.5)
    assert metrics["precision"] == pytest.approx(0.5)
    assert metrics["recall"] == pytest.approx(0.5)
    assert metrics["f1"] == pytest.approx(0.5)
    assert metrics["f2"] == pytest.approx(0.5)
    assert metrics["confusion_matrix"] == {"tn": 1, "fp": 1, "fn": 1, "tp": 1}
    assert metrics["predicted_positive_count"] == 2
    assert metrics["alert_rate"] == pytest.approx(0.5)
    assert metrics["false_positive_count"] == 1
    assert metrics["false_negative_count"] == 1
    assert metrics["false_positives_per_true_positive"] == pytest.approx(1.0)


def test_false_positive_ratio_is_none_when_there_are_no_true_positives() -> None:
    metrics = calculate_metrics([0, 0, 1, 1], [0.1, 0.2, 0.3, 0.4])

    assert metrics["predicted_positive_count"] == 0
    assert metrics["false_positives_per_true_positive"] is None


@pytest.mark.parametrize("probabilities", [[-0.1, 0.2], [0.2, 1.1]])
def test_probability_outside_unit_interval_is_rejected(probabilities) -> None:
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        calculate_metrics([0, 1], probabilities)


@pytest.mark.parametrize("invalid", [[np.nan, 0.5], [np.inf, 0.5], [-np.inf, 0.5]])
def test_non_finite_probability_is_rejected(invalid) -> None:
    with pytest.raises(ValueError, match="finite"):
        calculate_metrics([0, 1], invalid)


@pytest.mark.parametrize("threshold", [-0.01, 1.01, np.nan, np.inf])
def test_invalid_threshold_is_rejected(threshold) -> None:
    with pytest.raises(ValueError, match="Threshold"):
        calculate_metrics([0, 1], [0.1, 0.9], threshold=threshold)
