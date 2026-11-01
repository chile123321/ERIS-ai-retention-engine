"""Pure Pydantic v2 contract for the frozen B2 research candidate."""

from __future__ import annotations

from typing import Annotated, Final, Literal, Self, get_args
from uuid import UUID

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    StrictInt,
    StrictStr,
    field_validator,
    model_validator,
)

MODEL_ID: Final = "rf-v3-final15-unweighted-b2"
MODEL_SHA256: Final = "0c72841a8505a910d31c2a687fbcbf5259e3afb60f9eddcb7586801519b5628f"
FEATURE_SCHEMA_VERSION: Final = "v3_final_15"
TECHNICAL_MAXIMUM: Final = 2_147_483_647  # Representation guardrail, not a business bound.
SHAP_WARNING: Final = "SHAP explains model behavior; it does not establish causality."

FeatureName = Literal[
    "Age", "Department", "DistanceFromHome", "Education", "EnvironmentSatisfaction",
    "JobLevel", "JobSatisfaction", "MonthlyIncome", "OverTime", "TotalWorkingYears",
    "TrainingTimesLastYear", "YearsAtCompany", "YearsInCurrentRole",
    "YearsSinceLastPromotion", "YearsWithCurrManager",
]
FEATURE_NAMES: tuple[str, ...] = get_args(FeatureName)
OptionalFeatureName = Literal["TrainingTimesLastYear", "YearsWithCurrManager"]


def _strict_integer(value: object) -> object:
    # Pydantic Literal equality can accept True/1.0 as 1 without this boundary.
    if type(value) is not int:
        raise ValueError("Expected an integer, without coercion.")
    return value


def _strict_boolean(value: object) -> object:
    if type(value) is not bool:
        raise ValueError("Expected a boolean, without coercion.")
    return value


TechnicalInteger = Annotated[int, Field(strict=True, ge=0, le=TECHNICAL_MAXIMUM)]
OrdinalFive = Annotated[Literal[1, 2, 3, 4, 5], BeforeValidator(_strict_integer)]
OrdinalFour = Annotated[Literal[1, 2, 3, 4], BeforeValidator(_strict_integer)]
ReferenceId = Annotated[StrictStr, Field(min_length=1, max_length=128)]
FiniteNumber = Annotated[float, Field(strict=True, allow_inf_nan=False)]
Probability = Annotated[FiniteNumber, Field(ge=0, le=1)]


class ContractModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid", validate_default=True, hide_input_in_errors=True
    )


class EmployeeFeatures(ContractModel):
    """Ordered IBM-coded inputs; optional absence/null is preserved, never imputed here."""

    Age: TechnicalInteger
    Department: Literal["Human Resources", "Research & Development", "Sales"]
    DistanceFromHome: TechnicalInteger
    Education: OrdinalFive
    EnvironmentSatisfaction: OrdinalFour
    JobLevel: OrdinalFive
    JobSatisfaction: OrdinalFour
    MonthlyIncome: TechnicalInteger
    OverTime: Literal["No", "Yes"]
    TotalWorkingYears: TechnicalInteger
    TrainingTimesLastYear: TechnicalInteger | None = None
    YearsAtCompany: TechnicalInteger
    YearsInCurrentRole: TechnicalInteger
    YearsSinceLastPromotion: TechnicalInteger
    YearsWithCurrManager: TechnicalInteger | None = None


class PredictRequest(ContractModel):
    request_id: UUID | None = None
    reference_id: ReferenceId
    features: EmployeeFeatures


class BatchItem(ContractModel):
    reference_id: ReferenceId
    features: EmployeeFeatures


class BatchPredictRequest(ContractModel):
    request_id: UUID | None = None
    items: Annotated[list[BatchItem], Field(min_length=1, max_length=100)]

    @model_validator(mode="after")
    def unique_references(self) -> Self:
        references = [item.reference_id for item in self.items]
        if len(references) != len(set(references)):
            raise ValueError("Duplicate reference_id within batch.")
        return self


class ExplainRequest(PredictRequest):
    """Same request shape as single prediction."""


class ModelMetadata(ContractModel):
    model_id: Literal["rf-v3-final15-unweighted-b2"] = MODEL_ID
    model_sha256: Literal[
        "0c72841a8505a910d31c2a687fbcbf5259e3afb60f9eddcb7586801519b5628f"
    ] = MODEL_SHA256
    feature_schema_version: Literal["v3_final_15"] = FEATURE_SCHEMA_VERSION
    model_status: Literal["research_candidate"] = "research_candidate"
    decision_support_only: Annotated[Literal[True], BeforeValidator(_strict_boolean)] = True
    independent_validation: Annotated[Literal[False], BeforeValidator(_strict_boolean)] = False


class PredictionResult(ContractModel):
    reference_id: ReferenceId
    estimated_probability: Probability
    imputed_features: Annotated[list[OptionalFeatureName], Field(max_length=2)]
    metadata: ModelMetadata

    @field_validator("imputed_features")
    @classmethod
    def unique_imputed_features(cls, value: list[OptionalFeatureName]) -> list[OptionalFeatureName]:
        if len(value) != len(set(value)):
            raise ValueError("Duplicate imputed feature.")
        return value


class PredictResponse(ContractModel):
    request_id: UUID
    result: PredictionResult


class BatchPredictResponse(ContractModel):
    """List order is preserved; future service must match it to request item order."""

    request_id: UUID
    results: Annotated[list[PredictionResult], Field(min_length=1, max_length=100)]

    @model_validator(mode="after")
    def unique_references(self) -> Self:
        references = [result.reference_id for result in self.results]
        if len(references) != len(set(references)):
            raise ValueError("Duplicate result reference_id.")
        return self


class ShapContribution(ContractModel):
    feature: FeatureName
    feature_value: StrictInt | FiniteNumber | StrictStr | None
    shap_value: FiniteNumber
    direction: Literal["increase", "decrease", "neutral"]

    @model_validator(mode="after")
    def matching_direction(self) -> Self:
        expected = (
            "increase" if self.shap_value > 0 else "decrease" if self.shap_value < 0 else "neutral"
        )
        if self.direction != expected:
            raise ValueError("SHAP direction must match the contribution sign.")
        return self


class ExplainResponse(ContractModel):
    request_id: UUID
    reference_id: ReferenceId
    estimated_probability: Probability
    base_value: Probability
    contributions: Annotated[list[ShapContribution], Field(min_length=15, max_length=15)]
    top_factors: Annotated[list[ShapContribution], Field(max_length=5)]
    additivity_error: Annotated[FiniteNumber, Field(ge=0)]
    explanation_space: Literal["probability"] = "probability"
    warning: Literal[
        "SHAP explains model behavior; it does not establish causality."
    ] = SHAP_WARNING
    metadata: ModelMetadata

    @model_validator(mode="after")
    def complete_contributions(self) -> Self:
        if tuple(item.feature for item in self.contributions) != FEATURE_NAMES:
            raise ValueError("Contributions must cover all 15 features once, in schema order.")
        ranked = sorted(self.contributions, key=lambda item: -abs(item.shap_value))
        if self.top_factors != ranked[: len(self.top_factors)]:
            raise ValueError("Top factors must match largest absolute contributions in order.")
        return self


ErrorCode = Literal[
    "AUTH_MISSING", "AUTH_INVALID", "VALIDATION_ERROR", "BATCH_TOO_LARGE",
    "MODEL_NOT_READY", "BUNDLE_INTEGRITY_ERROR", "INFERENCE_ERROR", "EXPLANATION_ERROR",
]
SAFE_ERROR_MESSAGES: dict[ErrorCode, str] = {
    "AUTH_MISSING": "Service key is required.",
    "AUTH_INVALID": "Service key is invalid.",
    "VALIDATION_ERROR": "Request validation failed.",
    "BATCH_TOO_LARGE": "Batch exceeds 100 items.",
    "MODEL_NOT_READY": "Model is not ready.",
    "BUNDLE_INTEGRITY_ERROR": "Model bundle integrity check failed.",
    "INFERENCE_ERROR": "Prediction could not be completed.",
    "EXPLANATION_ERROR": "Explanation could not be completed.",
}


class ErrorDetail(ContractModel):
    """Allowlisted locations/reasons only: no submitted values or exception text."""

    field: FeatureName | Literal["body", "request_id", "reference_id", "features", "items"]
    item_index: Annotated[int, Field(strict=True, ge=0, le=99)] | None = None
    issue: Literal[
        "required", "invalid_type", "out_of_range", "invalid_value", "extra_fields",
        "duplicate_reference_id", "empty_batch", "batch_too_large",
    ]


class ErrorResponse(ContractModel):
    request_id: UUID | None = None
    code: ErrorCode
    message: StrictStr
    details: Annotated[list[ErrorDetail], Field(min_length=1, max_length=100)] | None = None

    @model_validator(mode="after")
    def safe_message(self) -> Self:
        if self.message != SAFE_ERROR_MESSAGES[self.code]:
            raise ValueError("Use the safe message defined for this error code.")
        return self
