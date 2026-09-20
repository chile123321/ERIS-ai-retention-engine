"""Single and bounded-batch inference from the loaded frozen bundle."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Query, Request

from eris_ml.api.contract import ContractError, validate_record
from eris_ml.api.dependencies import get_model_bundle
from eris_ml.api.errors import ApiError
from eris_ml.api.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    PredictionRecord,
    PredictionResponse,
)
from eris_ml.models.bundle import ModelBundle

router = APIRouter(tags=["prediction"])
DECISION_DISCLAIMER = (
    "This research model provides decision-support only and must not be used for "
    "automated employment decisions. An alert is not a certainty of resignation."
)
SHAP_DISCLAIMER = (
    "SHAP describes how the model prediction differs from its baseline. It does not "
    "establish causality and does not prove that changing a feature will prevent attrition."
)


def _bundle_or_error(bundle: ModelBundle | None) -> ModelBundle:
    if bundle is None:
        raise ApiError("MODEL_NOT_READY", "Model is not ready.", 503)
    return bundle


def _validated(records: list[PredictionRecord], request: Request,
               bundle: ModelBundle) -> tuple[list[dict[str, Any]], list[list[str]]]:
    features: list[dict[str, Any]] = []
    warnings: list[list[str]] = []
    for index, record in enumerate(records):
        try:
            row, row_warnings = validate_record(record, request.app.state.contract, bundle)
        except ContractError as exc:
            raw_details = exc.args[0] if exc.args else []
            details = raw_details if isinstance(raw_details, list) else []
            safe = [{"field": f"records.{index}.{item['field']}",
                     "reason": item["reason"]} for item in details]
            raise ApiError("VALIDATION_ERROR", "Input validation failed.", 422, safe) from None
        features.append(row)
        warnings.append(row_warnings)
    return features, warnings


def _results(records: list[PredictionRecord], request: Request, bundle: ModelBundle,
             *, include_explanation: bool, top_k: int) -> list[PredictionResponse]:
    features, warnings = _validated(records, request, bundle)
    try:
        probability = bundle.predict_proba(features)
    except Exception:
        raise ApiError("PREDICTION_ERROR", "Prediction could not be completed.", 500) from None
    explanations: list[dict[str, Any]] | None = None
    if include_explanation:
        try:
            explanations = bundle.explain(features, top_k=top_k)
        except Exception:
            raise ApiError(
                "EXPLANATION_ERROR", "Explanation could not be completed.", 500
            ) from None
    output: list[PredictionResponse] = []
    for index, record in enumerate(records):
        score = float(probability[index])
        alert = score >= 0.345651
        factors = explanations[index]["top_factors"] if explanations is not None else []
        output.append(PredictionResponse(
            request_id=request.state.request_id, record_id=record.record_id,
            candidate_version=bundle.metadata["candidate_version"],
            bundle_version=bundle.metadata["bundle_version"],
            feature_schema_version=bundle.metadata["feature_schema_version"],
            predicted_at_utc=datetime.now(UTC), probability=score,
            threshold=0.345651, alert=alert,
            decision_label="REVIEW_RECOMMENDED" if alert else "NO_REVIEW_ALERT",
            top_factors=factors,
            explanation_space="raw_log_odds" if explanations is not None else None,
            warnings=warnings[index], disclaimer=DECISION_DISCLAIMER,
            shap_disclaimer=SHAP_DISCLAIMER if explanations is not None else None,
        ))
    request.app.state.metrics.increment("prediction_alerts", sum(item.alert for item in output))
    request.app.state.metrics.increment(
        "unknown_category_warnings", sum(len(item) for item in warnings)
    )
    return output


@router.post(
    "/predict", response_model=PredictionResponse,
    description="Research-only raw probability at frozen threshold 0.345651. "
                "SHAP is raw log-odds and non-causal; production_approved=false.",
    responses={200: {"content": {"application/json": {"examples": {
        "illustrative_alert": {"summary": "Illustrative review signal (not a measured case)",
                               "value": {"probability": 0.41, "threshold": 0.345651,
                                         "alert": True, "decision_label": "REVIEW_RECOMMENDED"}},
        "illustrative_no_alert": {"summary": "Illustrative no-alert signal",
                                  "value": {"probability": 0.12, "threshold": 0.345651,
                                            "alert": False, "decision_label": "NO_REVIEW_ALERT"}},
    }}}}},
)
def predict(
    record: PredictionRecord,
    request: Request,
    bundle: Annotated[ModelBundle | None, Depends(get_model_bundle)],
    include_explanation: bool = Query(default=True),
    top_k: int = Query(default=5, ge=1, le=10),
) -> PredictionResponse:
    """Estimate one decision-support signal without persistence or retraining."""
    return _results([record], request, _bundle_or_error(bundle),
                    include_explanation=include_explanation, top_k=top_k)[0]


@router.post("/predict/batch", response_model=BatchPredictionResponse,
             description="Ordered all-or-nothing batch; 100 default maximum, 20 with SHAP.")
def predict_batch(
    payload: BatchPredictionRequest,
    request: Request,
    bundle: Annotated[ModelBundle | None, Depends(get_model_bundle)],
    top_k: int = Query(default=5, ge=1, le=10),
) -> BatchPredictionResponse:
    """Predict a bounded batch, preserving record order and privacy."""
    model = _bundle_or_error(bundle)
    size = len(payload.records)
    request.state.batch_size = size
    maximum = request.app.state.settings.max_batch_size
    if size < 1 or size > maximum or (payload.include_explanation and size > 20):
        raise ApiError("BATCH_LIMIT_EXCEEDED", "Batch size exceeds the configured limit.", 413)
    results = _results(payload.records, request, model,
                       include_explanation=payload.include_explanation, top_k=top_k)
    request.app.state.metrics.increment("batch_records", size)
    return BatchPredictionResponse(candidate_version=model.metadata["candidate_version"],
                                   count=size, results=results)
