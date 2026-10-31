"""Schema-only checks: synthetic records, no model imports, bundle load or fit."""

from __future__ import annotations

import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any
from uuid import UUID

import pytest
import yaml
from pydantic import ValidationError

from eris_ml.api_b2.schemas import (
    FEATURE_NAMES,
    MODEL_ID,
    MODEL_SHA256,
    SAFE_ERROR_MESSAGES,
    TECHNICAL_MAXIMUM,
    BatchItem,
    BatchPredictRequest,
    BatchPredictResponse,
    EmployeeFeatures,
    ErrorResponse,
    ExplainRequest,
    ExplainResponse,
    ModelMetadata,
    PredictionResult,
    PredictRequest,
    PredictResponse,
)

ROOT = Path(__file__).resolve().parents[2]
REQUEST_ID = "46c7df60-03bb-4f78-85b7-1388ba103faf"
OPTIONAL = {"TrainingTimesLastYear", "YearsWithCurrManager"}
NUMERIC = [name for name in FEATURE_NAMES if name not in {"Department", "OverTime"}]
SYNTHETIC = {
    "Age": 35, "Department": "Sales", "DistanceFromHome": 5, "Education": 3,
    "EnvironmentSatisfaction": 3, "JobLevel": 2, "JobSatisfaction": 3,
    "MonthlyIncome": 6000, "OverTime": "No", "TotalWorkingYears": 10,
    "TrainingTimesLastYear": 2, "YearsAtCompany": 6, "YearsInCurrentRole": 3,
    "YearsSinceLastPromotion": 1, "YearsWithCurrManager": 2,
}


def request(reference: str = "demo-001") -> dict[str, Any]:
    return {"reference_id": reference, "features": dict(SYNTHETIC)}


def prediction(reference: str = "demo-001") -> dict[str, Any]:
    return {
        "reference_id": reference, "estimated_probability": 0.25,
        "imputed_features": [], "metadata": ModelMetadata().model_dump(),
    }


def explanation() -> dict[str, Any]:
    contributions = [
        {"feature": name, "feature_value": SYNTHETIC[name], "shap_value": 0.0,
         "direction": "neutral"}
        for name in FEATURE_NAMES
    ]
    return {
        "request_id": REQUEST_ID, "reference_id": "demo-001",
        "estimated_probability": 0.25, "base_value": 0.25,
        "contributions": contributions, "top_factors": deepcopy(contributions[:5]),
        "additivity_error": 0.0, "metadata": ModelMetadata().model_dump(),
    }


def test_complete_record_and_locked_yaml_alignment() -> None:
    result = EmployeeFeatures.model_validate(SYNTHETIC)
    assert result.model_dump() == SYNTHETIC
    config = yaml.safe_load(
        (ROOT / "configs/features/feature_set_v3_final_15.yaml").read_text(encoding="utf-8")
    )
    assert tuple(EmployeeFeatures.model_fields) == tuple(f["name"] for f in config["features"])
    assert tuple(EmployeeFeatures.model_fields) == FEATURE_NAMES
    assert len(FEATURE_NAMES) == 15
    assert sum(field.is_required() for field in EmployeeFeatures.model_fields.values()) == 13
    assert {f["name"] for f in config["features"] if not f["required"]} == OPTIONAL
    for feature in config["features"]:
        for value in feature.get("vocabulary", feature.get("values", [])):
            EmployeeFeatures.model_validate({**SYNTHETIC, feature["name"]: value})


@pytest.mark.parametrize("name", [name for name in FEATURE_NAMES if name not in OPTIONAL])
@pytest.mark.parametrize("missing", [True, False], ids=["omitted", "null"])
def test_required_feature_cannot_be_omitted_or_null(name: str, missing: bool) -> None:
    data: dict[str, Any] = dict(SYNTHETIC)
    if missing:
        data.pop(name)
    else:
        data[name] = None
    with pytest.raises(ValidationError):
        EmployeeFeatures.model_validate(data)


@pytest.mark.parametrize("explicit_null", [False, True])
def test_optional_missing_preserved_without_imputation(explicit_null: bool) -> None:
    data: dict[str, Any] = {k: v for k, v in SYNTHETIC.items() if k not in OPTIONAL}
    if explicit_null:
        data.update(dict.fromkeys(OPTIONAL))
    parsed = EmployeeFeatures.model_validate(data)
    assert parsed.TrainingTimesLastYear is None and parsed.YearsWithCurrManager is None
    assert not OPTIONAL & set(parsed.model_dump(exclude_none=True))
    assert (OPTIONAL <= parsed.model_fields_set) is explicit_null


@pytest.mark.parametrize("name", NUMERIC)
@pytest.mark.parametrize(
    "value", ["1", 1.0, 1.5, True, False, -1, TECHNICAL_MAXIMUM + 1,
              10**400, float("nan"), float("inf"), float("-inf")],
    ids=["numeric-string", "integral-float", "fraction", "true", "false", "negative",
         "overflow", "huge-integer", "nan", "inf", "negative-inf"],
)
def test_all_numeric_inputs_are_strict_and_bounded(name: str, value: object) -> None:
    with pytest.raises(ValidationError):
        EmployeeFeatures.model_validate({**SYNTHETIC, name: value})


def test_age_41_float_rejected_in_python_and_json() -> None:
    data = {**SYNTHETIC, "Age": 41.0}
    with pytest.raises(ValidationError):
        EmployeeFeatures.model_validate(data)
    with pytest.raises(ValidationError):
        EmployeeFeatures.model_validate_json(json.dumps(data))


@pytest.mark.parametrize("name", [n for n in NUMERIC if n not in {
    "Education", "EnvironmentSatisfaction", "JobLevel", "JobSatisfaction"
}])
@pytest.mark.parametrize("value", [0, TECHNICAL_MAXIMUM])
def test_technical_integer_boundaries(name: str, value: int) -> None:
    parsed = EmployeeFeatures.model_validate({**SYNTHETIC, name: value})
    assert parsed.model_dump()[name] == value


@pytest.mark.parametrize("name,value", [
    ("Education", 0), ("Education", 6), ("JobLevel", 0), ("JobLevel", 6),
    ("EnvironmentSatisfaction", 0), ("EnvironmentSatisfaction", 5),
    ("JobSatisfaction", 0), ("JobSatisfaction", 5),
    ("Department", "Unknown"), ("Department", "Sales "),
    ("OverTime", "yes"), ("OverTime", True),
])
def test_declared_codes_and_categories(name: str, value: object) -> None:
    with pytest.raises(ValidationError):
        EmployeeFeatures.model_validate({**SYNTHETIC, name: value})


@pytest.mark.parametrize("field", ["name", "email", "phone", "EmployeeNumber", "source_row"])
def test_pii_and_identifiers_forbidden_at_both_input_levels(field: str) -> None:
    with pytest.raises(ValidationError):
        EmployeeFeatures.model_validate({**SYNTHETIC, field: "synthetic"})
    with pytest.raises(ValidationError):
        PredictRequest.model_validate({**request(), field: "synthetic"})
    with pytest.raises(ValidationError):
        BatchItem.model_validate({**request(), field: "synthetic"})


@pytest.mark.parametrize("reference", ["", "x" * 129, 123, True])
def test_invalid_reference_id(reference: object) -> None:
    with pytest.raises(ValidationError):
        PredictRequest.model_validate({**request(), "reference_id": reference})


def test_uuid_and_reference_roundtrip_without_generation() -> None:
    assert PredictRequest.model_validate(request()).request_id is None
    for cls in (PredictRequest, ExplainRequest):
        parsed = cls.model_validate_json(json.dumps({**request("x" * 128),
                                                   "request_id": REQUEST_ID}))
        assert parsed.request_id == UUID(REQUEST_ID)
        assert len(parsed.reference_id) == 128
        with pytest.raises(ValidationError):
            cls.model_validate({**request(), "request_id": "not-a-uuid"})


@pytest.mark.parametrize("count", [0, 101])
def test_invalid_batch_length(count: int) -> None:
    with pytest.raises(ValidationError):
        BatchPredictRequest.model_validate({"items": [request(str(i)) for i in range(count)]})


def test_batch_100_order_duplicates_and_atomic_validation() -> None:
    items = [request(str(i)) for i in range(100)]
    parsed = BatchPredictRequest.model_validate({"items": items})
    assert [item.reference_id for item in parsed.items] == [str(i) for i in range(100)]
    assert parsed.request_id is None
    with pytest.raises(ValidationError):
        BatchPredictRequest.model_validate({"items": [request(), request()]})
    broken = deepcopy(items)
    broken[23]["features"]["Age"] = True
    with pytest.raises(ValidationError) as exc:
        BatchPredictRequest.model_validate({"items": broken})
    assert exc.value.errors(include_input=False)[0]["loc"] == ("items", 23, "features", "Age")


@pytest.mark.parametrize("probability", [-0.01, 1.01, float("nan"), float("inf"), "0.5", True])
def test_invalid_response_probability(probability: object) -> None:
    with pytest.raises(ValidationError):
        PredictionResult.model_validate({**prediction(), "estimated_probability": probability})
    with pytest.raises(ValidationError):
        ExplainResponse.model_validate({**explanation(), "estimated_probability": probability})


def test_prediction_response_order_and_optional_names() -> None:
    single = PredictResponse.model_validate({"request_id": REQUEST_ID, "result": prediction()})
    assert single.result.metadata.model_id == MODEL_ID
    batch = BatchPredictResponse.model_validate({
        "request_id": REQUEST_ID, "results": [prediction("z"), prediction("a")],
    })
    assert [result.reference_id for result in batch.results] == ["z", "a"]
    for invalid in (["Age"], ["TrainingTimesLastYear"] * 2):
        with pytest.raises(ValidationError):
            PredictionResult.model_validate({**prediction(), "imputed_features": invalid})
    for value in (0.0, 1.0):
        PredictionResult.model_validate({**prediction(), "estimated_probability": value})


@pytest.mark.parametrize("count", [0, 14, 16])
def test_shap_contribution_count(count: int) -> None:
    data = explanation()
    data["contributions"] = (data["contributions"] * 2)[:count]
    with pytest.raises(ValidationError):
        ExplainResponse.model_validate(data)


def test_shap_coverage_top_factors_and_direction() -> None:
    good = ExplainResponse.model_validate(explanation())
    assert good.explanation_space == "probability" and len(good.contributions) == 15
    assert "causality" in good.warning
    changes = []
    duplicate = explanation()
    duplicate["contributions"][-1] = deepcopy(duplicate["contributions"][0])
    changes.append(duplicate)
    too_many = explanation()
    too_many["top_factors"] = too_many["contributions"][:6]
    changes.append(too_many)
    wrong_top = explanation()
    wrong_top["top_factors"] = wrong_top["contributions"][1:6]
    changes.append(wrong_top)
    wrong_direction = explanation()
    wrong_direction["contributions"][0]["direction"] = "increase"
    changes.append(wrong_direction)
    for data in changes:
        with pytest.raises(ValidationError):
            ExplainResponse.model_validate(data)


@pytest.mark.parametrize("field,value", [
    ("model_id", "rf-v3-final15-b1"), ("model_sha256", "a" * 64),
    ("feature_schema_version", "v2_general_16"), ("model_status", "production"),
    ("decision_support_only", False), ("decision_support_only", 1),
    ("independent_validation", True), ("independent_validation", 0),
])
def test_metadata_constants_cannot_be_overridden(field: str, value: object) -> None:
    with pytest.raises(ValidationError):
        ModelMetadata.model_validate({field: value})


def test_metadata_defaults_are_locked() -> None:
    assert ModelMetadata().model_dump() == {
        "model_id": MODEL_ID, "model_sha256": MODEL_SHA256,
        "feature_schema_version": "v3_final_15", "model_status": "research_candidate",
        "decision_support_only": True, "independent_validation": False,
    }


def test_error_contract_excludes_input_and_exception_text() -> None:
    for code, message in SAFE_ERROR_MESSAGES.items():
        assert ErrorResponse.model_validate({"code": code, "message": message}).request_id is None
    data: dict[str, Any] = {
        "code": "VALIDATION_ERROR", "message": SAFE_ERROR_MESSAGES["VALIDATION_ERROR"],
        "details": [{"field": "Age", "item_index": 1, "issue": "invalid_type"}],
    }
    ErrorResponse.model_validate(data)
    for change in ({"message": "Traceback: private input"}, {"traceback": "private"},
                   {"details": [{"field": "Age", "issue": "invalid_type", "input": SYNTHETIC}]}):
        with pytest.raises(ValidationError):
            ErrorResponse.model_validate({**data, **change})


def test_schema_import_does_not_import_ml_or_create_app() -> None:
    code = (
        "import sys; import eris_ml.api_b2.schemas; "
        "blocked=('eris_ml.models','sklearn','joblib','shap','fastapi'); "
        "assert not any(n==p or n.startswith(p+'.') for n in sys.modules for p in blocked)"
    )
    subprocess.run([sys.executable, "-B", "-c", code], cwd=ROOT, check=True)
