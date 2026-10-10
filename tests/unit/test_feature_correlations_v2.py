"""Synthetic correlation/alignment checks; no training or holdout access."""

import numpy as np
import pandas as pd
import pytest

from scripts.feature_correlations_v2 import align, build, correlations


@pytest.fixture
def frames():
    development = pd.DataFrame({
        "source_row": range(5), "EmployeeNumber": range(10, 15),
        "Attrition": [0, 0, 1, 1, 1], "Age": [20, 30, 40, 50, 60],
    })
    assignment = development.drop(columns="Age").assign(fold=range(1, 6))
    oof = assignment.assign(probability=[.1, .2, .3, .4, .5])
    return development, assignment, oof


def test_alignment_is_by_ids_not_position(frames):
    development, assignment, oof = frames
    result = align(development, assignment.iloc[::-1], oof.iloc[::-1])
    assert result.probability.tolist() == [.1, .2, .3, .4, .5]


@pytest.mark.parametrize("column,value", [
    ("source_row", 99), ("EmployeeNumber", 99), ("Attrition", 1),
    ("fold", 2), ("probability", np.nan), ("probability", 1.1),
])
def test_alignment_rejects_wrong_provenance(frames, column, value):
    development, assignment, oof = frames
    oof.loc[0, column] = value
    with pytest.raises(ValueError):
        align(development, assignment, oof)


def test_exact_correlations_and_constant_undefined():
    frame = pd.DataFrame({"Attrition": [0, 0, 1, 1], "probability": [0., 0., 1., 1.],
                          "OverTime": ["No", "No", "Yes", "Yes"], "constant": [2] * 4})
    table = correlations(frame, ["OverTime", "constant"]).set_index("Feature")
    assert table.loc["OverTime", "Pearson / nhãn"] == pytest.approx(1)
    assert table.loc["OverTime", "Spearman / OOF"] == pytest.approx(1)
    assert pd.isna(table.loc["constant", "Pearson / nhãn"])
    frame.loc[0, "OverTime"] = "Unknown"
    with pytest.raises(ValueError, match="OverTime"):
        correlations(frame, ["OverTime"])


def test_output_guards_before_any_input_read(tmp_path):
    with pytest.raises(ValueError, match="final-test"):
        build(tmp_path, tmp_path / "final_test_raw_v1.csv")
    output = tmp_path / "existing.html"
    output.write_text("keep", encoding="utf-8")
    with pytest.raises(ValueError, match="overwrite"):
        build(tmp_path, output)
    assert output.read_text(encoding="utf-8") == "keep"
