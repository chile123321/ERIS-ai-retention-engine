"""Verify v2 schema against IBM header and immutable development evidence."""

from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd
import yaml

from eris_ml.features.definitions import load_feature_definition
from eris_ml.utils.hashing import sha256_file

ROOT = Path(__file__).resolve().parents[2]
DEVELOPMENT = ROOT / "data/processed/development_raw_v1.csv"
SPLIT_MANIFEST = ROOT / "data/processed/split_manifest_v1.csv"
IBM_SOURCE = ROOT / "data/raw/WA_Fn-UseC_-HR-Employee-Attrition.csv"
REGISTRY = ROOT / "configs/features/feature_set_v2_practical.yaml"


def test_all_v2_features_have_exact_ibm_names_and_development_types() -> None:
    definition = load_feature_definition(
        ROOT / "configs/features/feature_set_v2_full_18.yaml"
    )
    # Read only the IBM header: schema verification must not inspect source rows.
    with IBM_SOURCE.open(encoding="utf-8", newline="") as stream:
        ibm_columns = next(csv.reader(stream))
    assert len(ibm_columns) == 35
    assert set(definition.all_features) <= set(ibm_columns)
    assert "Attrition" in ibm_columns

    development = pd.read_csv(DEVELOPMENT)
    assert development.shape == (1176, 28)
    assert set(definition.all_features) <= set(development.columns)
    assert pd.api.types.is_string_dtype(development["Department"])
    assert pd.api.types.is_string_dtype(development["JobRole"])
    assert pd.api.types.is_string_dtype(development["OverTime"])
    for name in set(definition.all_features) - {"Department", "JobRole", "OverTime"}:
        assert pd.api.types.is_integer_dtype(development[name]), name
    assert pd.api.types.is_integer_dtype(development["Attrition"])
    assert development["Attrition"].value_counts().sort_index().to_dict() == {
        0: 986, 1: 190,
    }
    assert development["source_row"].is_unique
    assert development["EmployeeNumber"].is_unique


def test_existing_split_membership_fingerprints_are_unchanged_without_final_access() -> None:
    registry = yaml.safe_load(REGISTRY.read_text(encoding="utf-8"))
    fingerprints = registry["comparison_protocol"]["immutable_split_fingerprints"]
    assert sha256_file(SPLIT_MANIFEST) == fingerprints["split_manifest_sha256"]
    assert sha256_file(DEVELOPMENT) == fingerprints["development_sha256"]
    assert fingerprints["development_rows"] == 1176
    assert fingerprints["development_target_distribution"] == {0: 986, 1: 190}

    manifest = pd.read_csv(SPLIT_MANIFEST)
    assert list(manifest.columns) == ["source_row", "EmployeeNumber", "Attrition", "split"]
    assert len(manifest) == 1470
    assert manifest["source_row"].is_unique and manifest["EmployeeNumber"].is_unique
    assert manifest["split"].value_counts().to_dict() == {
        "development": 1176, "final_test": 294,
    }
    development_manifest = manifest.loc[manifest["split"] == "development"]
    final_manifest = manifest.loc[manifest["split"] == "final_test"]
    assert set(development_manifest["source_row"]).isdisjoint(final_manifest["source_row"])
    assert set(development_manifest["EmployeeNumber"]).isdisjoint(
        final_manifest["EmployeeNumber"]
    )
    development = pd.read_csv(DEVELOPMENT, usecols=[
        "source_row", "EmployeeNumber", "Attrition",
    ])
    assert set(development["source_row"]) == set(development_manifest["source_row"])
    assert set(development["EmployeeNumber"]) == set(development_manifest["EmployeeNumber"])
    assert final_manifest["Attrition"].value_counts().sort_index().to_dict() == {
        0: 247, 1: 47,
    }

    # The existing ledger is evidence only; this test never opens final_test_raw_v1.csv.
    record = yaml.safe_load(
        (ROOT / "configs/release/final_evaluation_record_v1.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert record["run_count"] == 1 and record["status"] == "completed"
    assert record["row_count"] == fingerprints["final_test_rows_from_existing_record"]
    assert record["target_distribution"] == fingerprints[
        "final_test_target_distribution_from_existing_record"
    ]
