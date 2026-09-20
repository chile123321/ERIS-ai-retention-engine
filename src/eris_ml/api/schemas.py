"""Canonical, explicit 25-feature request and safe response schemas."""

from datetime import datetime
from typing import Any, cast

from pydantic import BaseModel, ConfigDict, Field, StrictFloat, StrictInt, StrictStr

Number = StrictInt | StrictFloat
SYNTHETIC_EXAMPLE = {
    "record_id": "DEMO-LOW-001", "Age": 34, "BusinessTravel": "Travel_Rarely",
    "Department": "Research & Development", "DistanceFromHome": 4,
    "Education": 3, "EducationField": "Life Sciences", "EnvironmentSatisfaction": 4,
    "JobInvolvement": 3, "JobLevel": 2, "JobRole": "Research Scientist",
    "JobSatisfaction": 4, "MonthlyIncome": 6200, "NumCompaniesWorked": 2,
    "OverTime": "No", "PercentSalaryHike": 15, "PerformanceRating": 3,
    "RelationshipSatisfaction": 3, "StockOptionLevel": 1, "TotalWorkingYears": 10,
    "TrainingTimesLastYear": 3, "WorkLifeBalance": 3, "YearsAtCompany": 6,
    "YearsInCurrentRole": 4, "YearsSinceLastPromotion": 2,
    "YearsWithCurrManager": 4,
}


class HealthResponse(BaseModel):
    """Process liveness; independent of model readiness."""

    status: str = "ok"
    service: str = "eris-attrition-model-api"


class ReadyResponse(BaseModel):
    """Model readiness without internal paths or failure details."""

    status: str = "ready"
    model_loaded: bool = True
    candidate_version: str
    bundle_version: str


class ModelInfoResponse(BaseModel):
    """Allowlisted, research-only model metadata."""

    candidate_version: str
    bundle_version: str
    model_family: str
    feature_schema_version: str
    feature_count: int
    threshold: float
    probability_type: str
    calibration: str
    training_scope: str
    fairness_status: str
    holdout_status: str
    production_approved: bool
    prediction_horizon_validated: bool = False


class PredictionRecord(BaseModel):
    """Exactly 25 canonical features and an optional non-model passthrough ID."""

    model_config = ConfigDict(
        extra="forbid", json_schema_extra=cast(Any, {"examples": [SYNTHETIC_EXAMPLE]})
    )

    record_id: StrictStr | None = Field(
        default=None, max_length=64, description="Not a model feature"
    )
    Age: Number
    BusinessTravel: StrictStr
    Department: StrictStr
    DistanceFromHome: Number
    Education: StrictInt
    EducationField: StrictStr
    EnvironmentSatisfaction: StrictInt
    JobInvolvement: StrictInt
    JobLevel: StrictInt
    JobRole: StrictStr
    JobSatisfaction: StrictInt
    MonthlyIncome: Number
    NumCompaniesWorked: Number
    OverTime: StrictStr
    PercentSalaryHike: Number
    PerformanceRating: StrictInt
    RelationshipSatisfaction: StrictInt
    StockOptionLevel: StrictInt
    TotalWorkingYears: Number
    TrainingTimesLastYear: Number
    WorkLifeBalance: StrictInt
    YearsAtCompany: Number
    YearsInCurrentRole: Number
    YearsSinceLastPromotion: Number
    YearsWithCurrManager: Number


class BatchPredictionRequest(BaseModel):
    """Ordered batch; limit and contract are checked before inference."""

    model_config = ConfigDict(extra="forbid")
    records: list[PredictionRecord]
    include_explanation: bool = False


class FactorResponse(BaseModel):
    """Signed original-feature SHAP factor in raw log-odds."""

    feature: str
    feature_value: str | int | float | None
    shap_value: float
    direction: str


class PredictionResponse(BaseModel):
    """Research-only decision-support signal, never an employment decision."""

    request_id: str
    record_id: str | None
    candidate_version: str
    bundle_version: str
    feature_schema_version: str
    predicted_at_utc: datetime
    probability: float
    threshold: float
    alert: bool
    decision_label: str
    top_factors: list[FactorResponse]
    explanation_space: str | None
    warnings: list[str]
    disclaimer: str
    shap_disclaimer: str | None = None


class BatchPredictionResponse(BaseModel):
    """Ordered results for an all-or-nothing batch."""

    candidate_version: str
    count: int
    results: list[PredictionResponse]
