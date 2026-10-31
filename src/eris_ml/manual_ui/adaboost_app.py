"""Loopback-only manual UI for the frozen AdaBoost baseline+sigmoid research bundle."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any

import pandas as pd
from fastapi import Body, FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from starlette.middleware.trustedhost import TrustedHostMiddleware

from eris_ml.explainability.adaboost_shap import Bundle, Explainer, load_bundle, parse_inputs
from eris_ml.models.adaboost_v3 import positive_probability

ROOT = Path(__file__).resolve().parents[3]
BUNDLE_PATH = ROOT / "artifacts/models/adaboost_v3/s1"
TEMPLATE_PATH = Path(__file__).parent / "templates/adaboost_v3.html"
LABELS = {
    "Age": "Tuổi (năm)",
    "Department": "Phòng ban (mã IBM)",
    "DistanceFromHome": "Khoảng cách từ nhà (đơn vị IBM)",
    "Education": "Học vấn (mã IBM 1–5)",
    "EnvironmentSatisfaction": "Hài lòng môi trường (mã IBM 1–4)",
    "JobLevel": "Cấp bậc (mã IBM 1–5)",
    "JobSatisfaction": "Hài lòng công việc (mã IBM 1–4)",
    "MonthlyIncome": "Thu nhập tháng (giá trị IBM)",
    "OverTime": "Làm thêm giờ (No/Yes)",
    "TotalWorkingYears": "Tổng số năm làm việc",
    "TrainingTimesLastYear": "Số lần đào tạo năm trước",
    "YearsAtCompany": "Số năm tại công ty",
    "YearsInCurrentRole": "Số năm trong vai trò hiện tại",
    "YearsSinceLastPromotion": "Số năm từ lần thăng chức gần nhất",
    "YearsWithCurrManager": "Số năm với quản lý hiện tại",
}


def require_loopback(host: str) -> None:
    if host != "127.0.0.1":
        raise ValueError("AdaBoost manual UI must bind to 127.0.0.1 only.")


def _configured_host() -> str:
    host = os.environ.get("ERIS_ADABOOST_UI_HOST", "127.0.0.1")
    for index, value in enumerate(sys.argv):
        if value == "--host" and index + 1 < len(sys.argv):
            host = sys.argv[index + 1]
        elif value.startswith("--host="):
            host = value.partition("=")[2]
    return host


def _token(frame: pd.DataFrame) -> str:
    payload = json.dumps(
        {name: None if pd.isna(value) else value for name, value in frame.iloc[0].items()},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _parse(payload: Any, bundle: Bundle) -> tuple[pd.DataFrame | None, dict[str, str]]:
    if not isinstance(payload, dict) or set(payload) != set(bundle.metadata["feature_order"]):
        return None, {"record": "Biểu mẫu phải chứa đúng 15 feature của v3_final_15."}
    try:
        values = {str(name): "" if value is None else str(value) for name, value in payload.items()}
        return parse_inputs(values, bundle.metadata["feature_config"]), {}
    except (ValueError, TypeError, OverflowError) as error:
        message = str(error)
        field = next(
            (
                name
                for name in bundle.metadata["feature_order"]
                if message.startswith(name) or f"{name}:" in message
            ),
            "record",
        )
        return None, {field: message}


def _load_state(bundle_path: Path) -> tuple[Bundle, Explainer, dict[str, Any]]:
    started = time.perf_counter()
    bundle = load_bundle(bundle_path)
    loaded = time.perf_counter()
    explainer = bundle.explainer()
    initialized = time.perf_counter()
    # One minimal permutation cycle compiles/caches the first explanation path.
    sample = pd.DataFrame([bundle.metadata["manual_sample"]])
    explainer.explain(sample, cycles=1, seed=42)
    warmed = time.perf_counter()
    return (
        bundle,
        explainer,
        {
            "bundle_load_seconds": loaded - started,
            "explainer_init_seconds": initialized - loaded,
            "explainer_warmup_seconds": warmed - initialized,
            "total_cold_start_seconds": warmed - started,
        },
    )


def create_app(*, host: str = "127.0.0.1", bundle_path: Path = BUNDLE_PATH) -> FastAPI:
    """Create an independent local UI; it never changes the B2/API application."""
    require_loopback(host)

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        require_loopback(_configured_host())
        try:
            bundle, explainer, cold_start = _load_state(bundle_path)
        except Exception as error:
            raise RuntimeError(
                "AdaBoost UI startup failed: bundle/integrity check failed."
            ) from error
        application.state.bundle = bundle
        application.state.explainer = explainer
        application.state.cold_start = cold_start
        yield
        application.state.bundle = None
        application.state.explainer = None

    application = FastAPI(
        title="ERIS AdaBoost local manual test",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
        lifespan=lifespan,
    )
    application.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "testserver"])

    @application.middleware("http")
    async def local_only(request: Request, call_next: Any) -> Any:
        server = request.scope.get("server")
        if server and server[0] not in {"127.0.0.1", "testserver"}:
            return JSONResponse({"message": "Local-only UI."}, status_code=403)
        return await call_next(request)

    @application.get("/", response_class=HTMLResponse)
    def index() -> HTMLResponse:
        return HTMLResponse(
            TEMPLATE_PATH.read_text(encoding="utf-8"),
            headers={
                "Cache-Control": "no-store",
                "Content-Security-Policy": (
                    "default-src 'self'; script-src 'self' 'unsafe-inline'; "
                    "style-src 'self' 'unsafe-inline'; connect-src 'self'; "
                    "img-src 'self' data:; object-src 'none'; frame-ancestors 'none'"
                ),
            },
        )

    @application.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ready"}

    @application.get("/schema")
    def schema(request: Request) -> dict[str, Any]:
        bundle: Bundle = request.app.state.bundle
        fields = []
        for spec in bundle.metadata["feature_config"]["features"]:
            field = {
                "name": spec["name"],
                "label": LABELS[spec["name"]],
                "kind": spec["type"],
                "required": spec["required"],
            }
            if spec["type"] == "categorical":
                field["options"] = spec["vocabulary"]
            elif "values" in spec:
                field["options"] = spec["values"]
            else:
                field["minimum"] = spec.get("minimum", 0)
            fields.append(field)
        manifest = json.loads((bundle_path / "manifest.json").read_text(encoding="utf-8"))
        return {
            "model_id": bundle.metadata["model_id"],
            "feature_set": "v3_final_15",
            "feature_count": 15,
            "fields": fields,
            "sample": bundle.metadata["manual_sample"],
            "artifact_sha256": manifest["pipeline.joblib"],
            "cold_start": request.app.state.cold_start,
            "explanation_cycles": 64,
            "production_approved": False,
        }

    @application.post("/predict")
    def predict(request: Request, payload: Annotated[Any, Body()]) -> JSONResponse:
        bundle: Bundle = request.app.state.bundle
        frame, errors = _parse(payload, bundle)
        if frame is None:
            return JSONResponse(
                {"message": "Hãy sửa các trường được đánh dấu.", "errors": errors},
                status_code=422,
            )
        try:
            started = time.perf_counter()
            probability = float(positive_probability(bundle.model, frame)[0])
            elapsed = time.perf_counter() - started
        except Exception:
            return JSONResponse({"message": "Không thể dự đoán với bundle s1."}, status_code=500)
        return JSONResponse(
            {
                "model_id": bundle.metadata["model_id"],
                "estimated_probability": probability,
                "prediction_seconds": elapsed,
                "prediction_token": _token(frame),
                "imputed_features": [
                    name
                    for name in bundle.metadata["feature_order"]
                    if pd.isna(frame.iloc[0][name])
                ],
                "predicted_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
                "cold_start": request.app.state.cold_start,
            },
            headers={"Cache-Control": "no-store"},
        )

    @application.post("/explain")
    def explain(request: Request, payload: Annotated[Any, Body()]) -> JSONResponse:
        if not isinstance(payload, dict) or set(payload) != {"record", "prediction_token"}:
            return JSONResponse({"message": "Cần hồ sơ đã dự đoán và token."}, status_code=422)
        bundle: Bundle = request.app.state.bundle
        frame, errors = _parse(payload["record"], bundle)
        if frame is None:
            return JSONResponse(
                {"message": "Input không hợp lệ.", "errors": errors}, status_code=422
            )
        if payload["prediction_token"] != _token(frame):
            return JSONResponse(
                {"message": "Input đã thay đổi; hãy dự đoán lại trước khi giải thích."},
                status_code=409,
            )
        try:
            explainer: Explainer = request.app.state.explainer
            result = explainer.explain(frame, cycles=64, seed=42)
        except Exception:
            return JSONResponse(
                {"message": "Không thể tạo SHAP; kết quả dự đoán vẫn hợp lệ."}, status_code=500
            )
        contributions = [
            {
                "feature": name,
                "value": result["feature_values"][name],
                "contribution_percentage_points": contribution,
                "direction": "increase"
                if contribution > 0
                else "decrease"
                if contribution < 0
                else "neutral",
            }
            for name, contribution in zip(
                result["feature_order"], result["contributions_percentage_points"], strict=True
            )
        ]
        return JSONResponse(
            {
                "model_id": bundle.metadata["model_id"],
                "estimated_probability": result["estimated_probability"],
                "base_value": result["base_value"],
                "contributions": contributions,
                "explanation_seconds": result["seconds"],
                "additivity_error": result["additivity_error"],
                "cold_start": request.app.state.cold_start,
                "disclaimer": (
                    "SHAP phản ánh cách model thay đổi xác suất quanh background đã lưu; "
                    "không phải nguyên nhân nghỉ việc hoặc khuyến nghị nhân sự."
                ),
            },
            headers={"Cache-Control": "no-store"},
        )

    return application


app = create_app()
