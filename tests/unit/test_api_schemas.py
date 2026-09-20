"""Explicit 25-feature API schema and dataset-derived contract tests."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from eris_ml.api.contract import ContractError, load_contract, validate_record
from eris_ml.api.schemas import SYNTHETIC_EXAMPLE, PredictionRecord

ROOT = Path(__file__).resolve().parents[2]


def test_record_has_exact_25_features_and_rejects_extra_or_missing() -> None:
    record = PredictionRecord.model_validate(SYNTHETIC_EXAMPLE)
    assert len(record.model_dump(exclude={"record_id"})) == 25
    with pytest.raises(ValidationError):
        PredictionRecord.model_validate({**SYNTHETIC_EXAMPLE, "EmployeeNumber": 123})
    without_age = {key: value for key, value in SYNTHETIC_EXAMPLE.items() if key != "Age"}
    with pytest.raises(ValidationError):
        PredictionRecord.model_validate(without_age)
    with pytest.raises(ValidationError):
        PredictionRecord.model_validate({**SYNTHETIC_EXAMPLE, "Age": "34"})


def test_contract_ranges_ordinals_and_unknown_category_warning(
    synthetic_bundle: tuple[object, object, object],
) -> None:
    bundle, _, _ = synthetic_bundle
    contract = load_contract(ROOT, bundle)
    example = dict(SYNTHETIC_EXAMPLE)
    for name in bundle.metadata["ordinal_domains"]:
        example[name] = 2 if 2.0 in bundle.metadata["ordinal_domains"][name] else 3
    record = PredictionRecord.model_validate(example)
    features, warnings = validate_record(record, contract, bundle)
    assert len(features) == 25 and warnings == []
    changed = PredictionRecord.model_validate({**example, "Department": "Never Seen Unit"})
    _, warnings = validate_record(changed, contract, bundle)
    assert len(warnings) == 1 and "Department" in warnings[0]
    for changed_value in (999, float("inf"), float("nan")):
        changed = PredictionRecord.model_validate({**example, "Age": changed_value})
        with pytest.raises(ContractError):
            validate_record(changed, contract, bundle)
    changed = PredictionRecord.model_validate({**example, "Education": 999})
    with pytest.raises(ContractError):
        validate_record(changed, contract, bundle)


def test_contract_marks_dataset_derived_and_not_hr_approved(
    synthetic_bundle: tuple[object, object, object],
) -> None:
    bundle, _, _ = synthetic_bundle
    contract = load_contract(ROOT, bundle)
    assert contract["contract_status"] == "dataset_derived"
    assert contract["hr_business_approved"] is False
