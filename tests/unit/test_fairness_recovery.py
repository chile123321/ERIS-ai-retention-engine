"""Safety tests for development-only fairness inputs and recovery extraction."""

import csv
from pathlib import Path

import pandas as pd
import pytest

from eris_ml.data.fairness_recovery import (
    extract_development_audit_rows,
    validate_development_fairness_frame,
)


def _development() -> pd.DataFrame:
    return pd.DataFrame({
        "source_row": [0, 2], "EmployeeNumber": [101, 103],
        "Age": [25, 35], "Attrition": [0, 1],
    })


def _audit() -> pd.DataFrame:
    return pd.DataFrame({
        "source_row": [0, 2], "EmployeeNumber": [101, 103],
        "split": ["development", "development"], "Age": [25, 35],
        "Gender": ["Female", "Male"], "MaritalStatus": ["Single", "Married"],
        "Attrition": [0, 1],
    })


@pytest.mark.parametrize("other_split", ["final_test", "test", "", None])
def test_rejects_every_non_development_fairness_split(other_split: str | None) -> None:
    audit = _audit()
    audit.loc[1, "split"] = other_split
    with pytest.raises(ValueError, match="split=development"):
        validate_development_fairness_frame(
            audit, _development(), expected_rows=2, expected_target_counts={0: 1, 1: 1}
        )


def test_rejects_locked_test_id_overlap() -> None:
    with pytest.raises(ValueError, match="overlaps locked-test IDs"):
        validate_development_fairness_frame(
            _audit(), _development(), expected_rows=2, expected_target_counts={0: 1, 1: 1},
            forbidden_employee_numbers={103},
        )


def test_rejects_fractional_id_instead_of_truncating_it() -> None:
    audit = _audit()
    audit["source_row"] = ["0.5", "2"]
    with pytest.raises(ValueError, match="non-integer"):
        validate_development_fairness_frame(
            audit, _development(), expected_rows=2, expected_target_counts={0: 1, 1: 1}
        )


def test_streaming_extraction_retains_only_approved_rows(tmp_path: Path) -> None:
    raw = tmp_path / "synthetic_raw.csv"
    with raw.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=["EmployeeNumber", "Age", "Gender", "MaritalStatus", "Attrition"]
        )
        writer.writeheader()
        writer.writerows([
            {"EmployeeNumber": 101, "Age": 25, "Gender": "Female",
             "MaritalStatus": "Single", "Attrition": "No"},
            {"EmployeeNumber": 102, "Age": 40, "Gender": "Male",
             "MaritalStatus": "Divorced", "Attrition": "No"},
            {"EmployeeNumber": 103, "Age": 35, "Gender": "Male",
             "MaritalStatus": "Married", "Attrition": "Yes"},
        ])
    audit = extract_development_audit_rows(raw, _development())
    validate_development_fairness_frame(
        audit, _development(), expected_rows=2, expected_target_counts={0: 1, 1: 1},
        forbidden_source_rows={1}, forbidden_employee_numbers={102},
    )
    assert audit["source_row"].tolist() == [0, 2]
    assert audit["split"].tolist() == ["development", "development"]
