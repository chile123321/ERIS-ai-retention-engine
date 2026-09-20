"""Dataset-derived API contract; no employee data file is opened at runtime."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import yaml

from eris_ml.api.schemas import PredictionRecord
from eris_ml.features.definitions import load_feature_definition
from eris_ml.models.bundle import ModelBundle


class ContractError(ValueError):
    """Internal contract mismatch or sanitized request validation failure."""


def load_contract(root: Path, bundle: ModelBundle) -> dict[str, Any]:
    """Reconcile config, schema and checksummed bundle metadata on startup."""
    with (root / "configs/data/data_contract_v1.yaml").open(encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    if not isinstance(raw, dict) or not isinstance(raw.get("api_input"), dict):
        raise ContractError("API input contract is missing.")
    contract: dict[str, Any] = raw["api_input"]
    definition = load_feature_definition(root / "configs/features/feature_set_v1_full.yaml")
    metadata = bundle.metadata
    if (contract.get("contract_status") != "dataset_derived"
        or contract.get("hr_business_approved") is not False
        or contract.get("production_approved") is not False
        or contract.get("feature_schema_version") != "v1-full"
        or contract.get("feature_count") != 25
        or contract.get("unknown_nominal") != "accept_with_warning"
        or contract.get("missing_values") != "reject_at_api"
        or metadata["feature_order"] != list(definition.all_features)
        or set(PredictionRecord.model_fields) != set(definition.all_features) | {"record_id"}
        or set(contract.get("nominal_categories", {})) != set(definition.nominal)
        or set(contract.get("numeric_safety_ranges", {})) != set(definition.numeric)
        or set(metadata["ordinal_domains"]) != set(definition.ordinal)):
        raise ContractError("API input contract differs from the frozen bundle schema.")
    for name, values in contract["nominal_categories"].items():
        if not isinstance(values, list) or not values or not all(
            isinstance(value, str) for value in values
        ):
            raise ContractError(f"Nominal category contract is invalid for {name}.")
    for name, bounds in contract["numeric_safety_ranges"].items():
        if (not isinstance(bounds, list) or len(bounds) != 2
            or not all(isinstance(value, (int, float)) and math.isfinite(value)
                       for value in bounds) or bounds[0] > bounds[1]):
            raise ContractError(f"Numeric safety range is invalid for {name}.")
    return contract


def validate_record(record: PredictionRecord, contract: dict[str, Any],
                    bundle: ModelBundle) -> tuple[dict[str, Any], list[str]]:
    """Validate one record without echoing values or retaining identifiers."""
    features = record.model_dump(exclude={"record_id"})
    details: list[dict[str, str]] = []
    warnings: list[str] = []
    for name, kind in bundle.metadata["feature_types"].items():
        value = features[name]
        if value is None:
            details.append({
                "field": name, "reason": "Missing values are not accepted by this API."
            })
        elif kind == "numeric":
            bounds = contract["numeric_safety_ranges"][name]
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                details.append({"field": name, "reason": "Value must be finite numeric."})
            elif not bounds[0] <= value <= bounds[1]:
                details.append({"field": name,
                                "reason": "Value must be within the configured safety range."})
        elif kind == "ordinal":
            if value not in bundle.metadata["ordinal_domains"][name]:
                details.append({"field": name, "reason": "Value is outside the ordinal domain."})
        elif value not in contract["nominal_categories"][name]:
            warnings.append(f"{name}: unknown category was encoded as unseen; review input.")
    if details:
        raise ContractError(details)
    bundle.validate_input(features)
    return features, warnings
