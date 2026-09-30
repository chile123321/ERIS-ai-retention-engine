"""Standalone loopback-only manual UI for the frozen research bundle."""

from __future__ import annotations

import json
import os
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any

from fastapi import Body, FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import ValidationError
from starlette.middleware.trustedhost import TrustedHostMiddleware

from eris_ml.api.contract import ContractError, load_contract, validate_record
from eris_ml.api.schemas import PredictionRecord
from eris_ml.models.bundle import ModelBundle
from eris_ml.models.inference import infer_records
from eris_ml.models.persistence import load_bundle
from eris_ml.utils.hashing import sha256_file

ROOT = Path(__file__).resolve().parents[3]
BUNDLE_PATH = ROOT / "artifacts/models/eris_xgboost_v1.joblib"
EXAMPLES_PATH = ROOT / "tests/fixtures/manual_prediction_examples.json"
TEMPLATE_PATH = Path(__file__).parent / "templates/index.html"
EXPECTED_SHA256 = "bb9c00307a36763f91de58d598caf952cca3f451e7928ca39d1a22b1763dac21"
SHAP_NOTE = "SHAP values are local explanations in raw log-odds and are non-causal."
LABELS = {
    "Age": "Tuổi", "BusinessTravel": "Tần suất công tác",
    "Department": "Phòng ban", "DistanceFromHome": "Khoảng cách tới nơi làm việc",
    "Education": "Trình độ học vấn", "EducationField": "Lĩnh vực học vấn",
    "EnvironmentSatisfaction": "Hài lòng với môi trường làm việc",
    "JobInvolvement": "Mức độ gắn kết công việc", "JobLevel": "Cấp bậc công việc",
    "JobRole": "Vai trò công việc", "JobSatisfaction": "Mức độ hài lòng với công việc",
    "MonthlyIncome": "Thu nhập hằng tháng", "NumCompaniesWorked": "Số công ty từng làm",
    "OverTime": "Làm thêm giờ", "PercentSalaryHike": "Tỷ lệ tăng lương (%)",
    "PerformanceRating": "Đánh giá hiệu suất",
    "RelationshipSatisfaction": "Hài lòng với quan hệ công việc",
    "StockOptionLevel": "Mức quyền chọn cổ phiếu",
    "TotalWorkingYears": "Tổng số năm làm việc",
    "TrainingTimesLastYear": "Số lần đào tạo năm trước",
    "WorkLifeBalance": "Cân bằng công việc và cuộc sống",
    "YearsAtCompany": "Số năm tại công ty", "YearsInCurrentRole": "Số năm trong vai trò hiện tại",
    "YearsSinceLastPromotion": "Số năm từ lần thăng chức gần nhất",
    "YearsWithCurrManager": "Số năm làm việc với quản lý hiện tại",
}


def require_loopback(host: str) -> None:
    """Reject wildcard, external and alternate loopback bindings."""
    if host != "127.0.0.1":
        raise ValueError("Manual UI must bind to 127.0.0.1 only.")


def _configured_host() -> str:
    """Catch a direct `uvicorn --host ...` launch as well as the local setting."""
    host = os.environ.get("ERIS_MANUAL_UI_HOST", "127.0.0.1")
    for index, value in enumerate(sys.argv):
        if value == "--host" and index + 1 < len(sys.argv):
            host = sys.argv[index + 1]
        elif value.startswith("--host="):
            host = value.partition("=")[2]
    return host


def _load_state(bundle_path: Path, examples_path: Path) -> tuple[ModelBundle, dict[str, Any],
                                                                  dict[str, dict[str, Any]]]:
    """Hash-check before deserialization; never inspect employee data files."""
    if not bundle_path.is_file() or bundle_path.is_symlink():
        raise ValueError("Frozen bundle is missing or is a symbolic link.")
    if sha256_file(bundle_path) != EXPECTED_SHA256:
        raise ValueError("Frozen bundle SHA-256 differs from bundle-v1.")
    bundle = load_bundle(bundle_path)
    contract = load_contract(ROOT, bundle)
    if examples_path.is_symlink():
        raise ValueError("Synthetic fixture must not be a symbolic link.")
    examples: Any = json.loads(examples_path.read_text(encoding="utf-8"))
    if not isinstance(examples, dict):
        raise ValueError("Synthetic fixture is invalid.")
    for signal in ("low", "high"):
        record = examples.get(f"synthetic_{signal}_signal")
        if not isinstance(record, dict):
            raise ValueError("Synthetic fixture is incomplete.")
        parsed = PredictionRecord.model_validate(record)
        features, warnings = validate_record(parsed, contract, bundle)
        if warnings or len(features) != 25:
            raise ValueError("Synthetic fixture differs from the bundle contract.")
    return bundle, contract, examples


def _field_errors(payload: dict[str, Any], bundle: ModelBundle,
                  contract: dict[str, Any]) -> tuple[PredictionRecord | None,
                                                     dict[str, Any] | None,
                                                     dict[str, str]]:
    errors: dict[str, str] = {}
    try:
        record = PredictionRecord.model_validate(payload)
    except ValidationError as exc:
        for item in exc.errors():
            field = str(item["loc"][0]) if item["loc"] else "record"
            errors[field] = "Required or invalid type." if item["type"] != "extra_forbidden" \
                else "Unexpected field."
        return None, None, errors
    try:
        features, warnings = validate_record(record, contract, bundle)
    except ContractError as exc:
        details = exc.args[0] if exc.args and isinstance(exc.args[0], list) else []
        for item in details:
            errors[str(item["field"])] = str(item["reason"])
        return None, None, errors or {"record": "Input contract validation failed."}
    except ValueError:
        return None, None, {"record": "Input contract validation failed."}
    for warning in warnings:
        errors[warning.split(":", 1)[0]] = "Choose one of the listed categories."
    return (None, None, errors) if errors else (record, features, {})


def create_app(*, host: str = "127.0.0.1", bundle_path: Path = BUNDLE_PATH,
               examples_path: Path = EXAMPLES_PATH) -> FastAPI:
    """Build an independent UI; no production auth or routes are changed."""
    require_loopback(host)

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        require_loopback(_configured_host())
        try:
            bundle, contract, examples = _load_state(bundle_path, examples_path)
        except Exception as exc:
            raise RuntimeError(
                "Manual UI startup failed: frozen bundle or fixture invalid."
            ) from exc
        application.state.bundle = bundle
        application.state.contract = contract
        application.state.examples = examples
        yield
        application.state.bundle = None

    application = FastAPI(title="ERIS local manual test", docs_url=None,
                          redoc_url=None, openapi_url=None, lifespan=lifespan)
    application.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "testserver"])

    @application.middleware("http")
    async def reject_non_loopback_server(request: Request, call_next: Any) -> Any:
        """Fail closed even if the app is programmatically bound to a wildcard host."""
        server = request.scope.get("server")
        if server and server[0] not in {"127.0.0.1", "testserver"}:
            return JSONResponse({"message": "Local-only service."}, status_code=403)
        return await call_next(request)

    @application.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ready"}

    @application.get("/", response_class=HTMLResponse)
    def index() -> HTMLResponse:
        return HTMLResponse(TEMPLATE_PATH.read_text(encoding="utf-8"), headers={
            "Cache-Control": "no-store",
            "Content-Security-Policy": (
                "default-src 'self'; script-src 'self' 'unsafe-inline'; "
                "style-src 'self' 'unsafe-inline'; connect-src 'self'; "
                "img-src 'self' data:; object-src 'none'; frame-ancestors 'none'"
            ),
        })

    @application.get("/schema")
    def schema(request: Request) -> dict[str, Any]:
        bundle: ModelBundle = request.app.state.bundle
        contract: dict[str, Any] = request.app.state.contract
        fields: list[dict[str, Any]] = []
        for name in bundle.metadata["feature_order"]:
            kind = bundle.metadata["feature_types"][name]
            field: dict[str, Any] = {"name": name, "label": LABELS[name], "kind": kind}
            if kind == "nominal":
                field["options"] = contract["nominal_categories"][name]
            elif kind == "ordinal":
                field["options"] = bundle.metadata["ordinal_domains"][name]
            else:
                field["minimum"], field["maximum"] = contract["numeric_safety_ranges"][name]
            fields.append(field)
        return {"fields": fields, "feature_count": len(fields),
                "candidate_version": bundle.metadata["candidate_version"],
                "bundle_version": bundle.metadata["bundle_version"],
                "threshold": bundle.metadata["threshold"],
                "probability_type": bundle.metadata["probability_type"],
                "production_approved": False, "research_only": True}

    @application.get("/examples/{signal}")
    def example(signal: str, request: Request) -> JSONResponse:
        if signal not in {"low", "high"}:
            return JSONResponse({"message": "Only low and high examples are available."},
                                status_code=404)
        return JSONResponse(request.app.state.examples[f"synthetic_{signal}_signal"],
                            headers={"Cache-Control": "no-store"})

    @application.post("/predict")
    def predict(request: Request, payload: Annotated[Any, Body()]) -> JSONResponse:
        if not isinstance(payload, dict):
            return JSONResponse({"message": "One JSON record is required.",
                                 "errors": {"record": "Expected a JSON object."}}, status_code=422)
        bundle: ModelBundle = request.app.state.bundle
        record, features, errors = _field_errors(payload, bundle, request.app.state.contract)
        if errors or record is None or features is None:
            return JSONResponse({"message": "Please correct the highlighted fields.",
                                 "errors": errors}, status_code=422)
        try:
            result = infer_records(bundle, [features])[0]
        except Exception:
            return JSONResponse({"message": "Prediction could not be completed."},
                                status_code=500)
        factors: list[dict[str, Any]] = []
        explanation_error: str | None = None
        try:
            factors = bundle.explain(features, top_k=5)[0]["top_factors"]
        except Exception:
            explanation_error = "SHAP explanation unavailable; prediction remains valid."
        output = {
            "record_id": record.record_id or datetime.now(UTC).strftime("MANUAL-%Y%m%d-%H%M%S"),
            "candidate_version": bundle.metadata["candidate_version"],
            "bundle_version": bundle.metadata["bundle_version"],
            "probability": result.probability, "threshold": result.threshold,
            "alert": result.alert, "decision_label": result.decision_label,
            "top_factors": factors, "explanation_space": "raw_log_odds" if factors else None,
            "explanation_error": explanation_error,
            "shap_disclaimer": SHAP_NOTE, "production_approved": False,
            "research_only": True,
        }
        return JSONResponse(output, headers={"Cache-Control": "no-store"})

    return application


app = create_app()
