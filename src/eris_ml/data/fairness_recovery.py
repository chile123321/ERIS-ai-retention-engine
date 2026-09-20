"""Recover a development-only fairness attribute file without using locked-test rows."""

import csv
from pathlib import Path

import pandas as pd

AUDIT_COLUMNS = (
    "source_row", "EmployeeNumber", "split", "Age", "Gender", "MaritalStatus", "Attrition"
)
RAW_COLUMNS = {"EmployeeNumber", "Age", "Gender", "MaritalStatus", "Attrition"}


def load_manifest_ids(
    path: Path,
) -> tuple[dict[tuple[int, int], int], set[int], set[int]]:
    """Read development keys/labels and locked-test IDs, never locked-test attributes."""
    development: dict[tuple[int, int], int] = {}
    other_source_rows: set[int] = set()
    other_employee_numbers: set[int] = set()
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if not {"source_row", "EmployeeNumber", "Attrition", "split"}.issubset(
            reader.fieldnames or ()
        ):
            raise ValueError("Split manifest has missing required columns.")
        for row in reader:
            split = row["split"]
            source_row = int(row["source_row"])
            employee_number = int(row["EmployeeNumber"])
            if split == "development":
                key = (source_row, employee_number)
                if key in development:
                    raise ValueError("Duplicate development key in split manifest.")
                development[key] = int(row["Attrition"])
            elif split == "final_test":
                # IDs are used only for the requested no-overlap assertion.
                other_source_rows.add(source_row)
                other_employee_numbers.add(employee_number)
            else:
                raise ValueError("Split manifest contains an unexpected split.")
    return development, other_source_rows, other_employee_numbers


def extract_development_audit_rows(raw_path: Path, development: pd.DataFrame) -> pd.DataFrame:
    """Stream raw rows; retain attributes only when the source row is approved development."""
    approved = development.set_index("source_row")
    if not approved.index.is_unique:
        raise ValueError("Development source_row is not unique.")
    selected: dict[int, dict[str, str | int]] = {}
    with raw_path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if not RAW_COLUMNS.issubset(reader.fieldnames or ()):
            raise ValueError("Raw source has missing audit columns.")
        for source_row, row in enumerate(reader):
            if source_row not in approved.index:
                continue
            expected = approved.loc[source_row]
            employee_number = int(row["EmployeeNumber"])
            if employee_number != int(expected["EmployeeNumber"]):
                raise ValueError("Raw and development EmployeeNumber do not align.")
            raw_label = {"No": 0, "Yes": 1}.get(row["Attrition"])
            if raw_label is None or raw_label != int(expected["Attrition"]):
                raise ValueError("Raw and development Attrition do not align.")
            age = int(row["Age"])
            if age != int(expected["Age"]):
                raise ValueError("Raw and development Age do not align.")
            selected[source_row] = {
                "source_row": source_row,
                "EmployeeNumber": employee_number,
                "split": "development",
                "Age": age,
                "Gender": row["Gender"],
                "MaritalStatus": row["MaritalStatus"],
                "Attrition": raw_label,
            }
    if len(selected) != len(development):
        raise ValueError("Not every development row was found in the raw source.")
    return pd.DataFrame(
        [selected[int(source_row)] for source_row in development["source_row"]],
        columns=AUDIT_COLUMNS,
    )


def validate_development_fairness_frame(
    frame: pd.DataFrame,
    development: pd.DataFrame,
    *,
    expected_rows: int = 1176,
    expected_target_counts: dict[int, int] | None = None,
    forbidden_source_rows: set[int] | None = None,
    forbidden_employee_numbers: set[int] | None = None,
) -> None:
    """Reject any non-development split and verify IDs, target, and audit attributes."""
    if expected_target_counts is None:
        expected_target_counts = {0: 986, 1: 190}
    if not set(AUDIT_COLUMNS).issubset(frame.columns):
        raise ValueError("Fairness input is missing required columns.")
    if len(frame) != expected_rows or len(development) != expected_rows:
        raise ValueError("Fairness and development row counts must match expected rows.")
    if frame["split"].isna().any() or not frame["split"].eq("development").all():
        raise ValueError("Fairness input must contain only split=development.")
    if frame[list(AUDIT_COLUMNS)].isna().any().any():
        raise ValueError("Fairness input contains missing required values.")
    if frame["source_row"].duplicated().any() or frame["EmployeeNumber"].duplicated().any():
        raise ValueError("Fairness input contains duplicate IDs.")
    if any(frame[column].astype(str).str.strip().eq("").any()
           for column in ("Gender", "MaritalStatus")):
        raise ValueError("Fairness input contains empty audit attributes.")
    compared = ("source_row", "EmployeeNumber", "Age", "Attrition")
    try:
        actual = frame.loc[:, compared].apply(pd.to_numeric, errors="raise")
        expected = development.loc[:, compared].apply(pd.to_numeric, errors="raise")
    except (TypeError, ValueError) as exc:
        raise ValueError("Fairness input contains non-integer IDs, Age, or target.") from exc
    if (actual.mod(1).ne(0).any().any() or expected.mod(1).ne(0).any().any()):
        raise ValueError("Fairness input contains non-integer IDs, Age, or target.")
    actual = actual.astype("int64").sort_values("source_row")
    expected = expected.astype("int64").sort_values("source_row")
    if not actual.reset_index(drop=True).equals(expected.reset_index(drop=True)):
        raise ValueError("Fairness IDs, Age, or target do not match development.")
    if actual["Attrition"].value_counts().sort_index().to_dict() != expected_target_counts:
        raise ValueError("Fairness target distribution does not match development contract.")
    if forbidden_source_rows and not set(actual["source_row"]).isdisjoint(forbidden_source_rows):
        raise ValueError("Fairness source_row overlaps locked-test IDs.")
    if forbidden_employee_numbers and not set(actual["EmployeeNumber"]).isdisjoint(
        forbidden_employee_numbers
    ):
        raise ValueError("Fairness EmployeeNumber overlaps locked-test IDs.")
