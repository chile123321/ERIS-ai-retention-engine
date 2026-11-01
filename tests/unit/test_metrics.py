import numpy as np
import pytest

from eris_ml.evaluation.metrics import calculate_metrics


def test_calculate_metrics():
    result = calculate_metrics([0, 1, 0, 1], [0.1, 0.8, 0.6, 0.4])

    assert result["accuracy"] == pytest.approx(0.5)
    assert result["average_precision"] == pytest.approx(5 / 6)
    assert result["roc_auc"] == pytest.approx(0.75)
    assert result["brier_score"] == pytest.approx(0.1925)
    assert result["confusion_matrix"] == {"tn": 1, "fp": 1, "fn": 1, "tp": 1}


@pytest.mark.parametrize("probabilities", [[-0.1, 0.8], [0.1, 1.1]])
def test_probability_outside_unit_interval_is_rejected(probabilities):
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        calculate_metrics([0, 1], probabilities)


def test_nan_probability_is_rejected():
    with pytest.raises(ValueError, match="finite"):
        calculate_metrics([0, 1], np.array([0.1, np.nan]))
