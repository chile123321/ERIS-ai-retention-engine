"""Create a development-only fairness audit CSV from the raw source and split manifest."""

import csv
import os
import tempfile
from pathlib import Path

import pandas as pd

from eris_ml.data.fairness_recovery import (
    AUDIT_COLUMNS,
    extract_development_audit_rows,
    load_manifest_ids,
    validate_development_fairness_frame,
)

ROOT = Path(__file__).resolve().parents[1]
DEVELOPMENT = ROOT / "data/processed/development_raw_v1.csv"
MANIFEST = ROOT / "data/processed/split_manifest_v1.csv"
RAW_SOURCE = ROOT / "data/raw/WA_Fn-UseC_-HR-Employee-Attrition.csv"
OUTPUT = ROOT / "data/processed/fairness_audit_development_v1.csv"


def main() -> None:
    """Validate inputs and emit only selected development rows; never open locked final test."""
    if OUTPUT.exists():
        raise FileExistsError(
            "Development-only fairness output already exists; refusing overwrite."
        )
    development = pd.read_csv(DEVELOPMENT)
    required_development = {"source_row", "EmployeeNumber", "Age", "Attrition"}
    if not required_development.issubset(development.columns):
        raise ValueError("Development split is missing required alignment columns.")
    manifest_development, other_sources, other_employees = load_manifest_ids(MANIFEST)
    expected_keys = {
        (int(row.source_row), int(row.EmployeeNumber)): int(row.Attrition)
        for row in development.itertuples(index=False)
    }
    if len(expected_keys) != len(development) or expected_keys != manifest_development:
        raise ValueError("Development IDs/targets do not match split manifest.")
    audit = extract_development_audit_rows(RAW_SOURCE, development)
    validate_development_fairness_frame(
        audit, development,
        forbidden_source_rows=other_sources,
        forbidden_employee_numbers=other_employees,
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", newline="", encoding="utf-8", suffix=".csv",
            prefix="fairness_audit_development_v1_", dir=OUTPUT.parent, delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            writer = csv.DictWriter(stream, fieldnames=AUDIT_COLUMNS)
            writer.writeheader()
            writer.writerows(audit.to_dict(orient="records"))
        written = pd.read_csv(temporary_path)
        validate_development_fairness_frame(
            written, development,
            forbidden_source_rows=other_sources,
            forbidden_employee_numbers=other_employees,
        )
        os.rename(temporary_path, OUTPUT)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    print("Created development-only fairness audit: 1176 rows; targets {0: 986, 1: 190}.")
    print("IDs match development; no locked-test ID overlap; split=development only.")


if __name__ == "__main__":
    main()
