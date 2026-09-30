"""Local, authenticated-API-independent manual test of the frozen ERIS bundle.

This script reads only its checked bundle, versioned feature contract, synthetic
examples or an explicitly supplied JSON object. It never starts a server.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import traceback
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from eris_ml.api.contract import ContractError, load_contract, validate_record
from eris_ml.api.schemas import PredictionRecord
from eris_ml.models.bundle import ModelBundle
from eris_ml.models.inference import infer_records
from eris_ml.models.persistence import load_bundle
from eris_ml.utils.hashing import sha256_file

ROOT = Path(__file__).resolve().parents[1]
BUNDLE_PATH = ROOT / "artifacts/models/eris_xgboost_v1.joblib"
EXAMPLES_PATH = ROOT / "tests/fixtures/manual_prediction_examples.json"
EXPECTED_BUNDLE_SHA256 = "bb9c00307a36763f91de58d598caf952cca3f451e7928ca39d1a22b1763dac21"
SHAP_NOTE = "SHAP values are local model explanations in raw log-odds and are non-causal."
DESCRIPTIONS = {
    "Age": "Tuổi (năm)",
    "BusinessTravel": "Tần suất đi công tác",
    "Department": "Phòng ban",
    "DistanceFromHome": "Khoảng cách đến nơi làm việc (đơn vị demo)",
    "Education": "Trình độ học vấn (mã bậc demo)",
    "EducationField": "Lĩnh vực học vấn",
    "EnvironmentSatisfaction": "Mức hài lòng với môi trường làm việc",
    "JobInvolvement": "Mức gắn kết với công việc",
    "JobLevel": "Cấp bậc công việc (mã demo)",
    "JobRole": "Vai trò công việc",
    "JobSatisfaction": "Mức hài lòng với công việc",
    "MonthlyIncome": "Thu nhập hàng tháng (đơn vị demo)",
    "NumCompaniesWorked": "Số công ty từng làm việc",
    "OverTime": "Có làm thêm giờ hay không",
    "PercentSalaryHike": "Tỷ lệ tăng lương (phần trăm)",
    "PerformanceRating": "Mức đánh giá hiệu suất (mã demo)",
    "RelationshipSatisfaction": "Mức hài lòng với quan hệ công việc",
    "StockOptionLevel": "Mức quyền chọn cổ phiếu (mã demo)",
    "TotalWorkingYears": "Tổng số năm đi làm",
    "TrainingTimesLastYear": "Số lần đào tạo năm trước",
    "WorkLifeBalance": "Mức cân bằng công việc/cuộc sống",
    "YearsAtCompany": "Số năm tại công ty",
    "YearsInCurrentRole": "Số năm ở vai trò hiện tại",
    "YearsSinceLastPromotion": "Số năm từ lần thăng chức gần nhất",
    "YearsWithCurrManager": "Số năm làm việc với quản lý hiện tại",
}


class ManualTestError(ValueError):
    """Expected local input or frozen-artifact failure without a traceback."""


def load_frozen_bundle() -> tuple[ModelBundle, dict[str, Any]]:
    """Verify the exact approved artifact before trusted joblib deserialization."""
    if not BUNDLE_PATH.is_file() or BUNDLE_PATH.is_symlink():
        raise ManualTestError("Frozen bundle is missing or is a symbolic link.")
    if sha256_file(BUNDLE_PATH) != EXPECTED_BUNDLE_SHA256:
        raise ManualTestError("Frozen bundle SHA-256 differs from bundle-v1.")
    try:
        bundle = load_bundle(BUNDLE_PATH)
        contract = load_contract(ROOT, bundle)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise ManualTestError("Frozen bundle or its contract failed validation.") from exc
    return bundle, contract


def load_examples() -> dict[str, dict[str, Any]]:
    """Read only the pre-existing synthetic fixture, never a dataset row."""
    try:
        examples: Any = json.loads(EXAMPLES_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ManualTestError("Synthetic example fixture is unavailable.") from exc
    if not isinstance(examples, dict) or not all(
        isinstance(examples.get(f"synthetic_{signal}_signal"), dict)
        for signal in ("low", "high")
    ):
        raise ManualTestError("Synthetic low/high examples are missing.")
    return examples


def load_input_file(path: Path) -> dict[str, Any]:
    """Accept a single local JSON record, never the protected data directories."""
    parts = [part.casefold() for part in path.parts]
    if (path.suffix.casefold() != ".json" or "data" in parts
        or "quarantine" in parts or "predictions" in parts
        or "final_test" in path.name.casefold()
        or "development" in path.name.casefold()
        or any(item.is_symlink() for item in (path, *path.parents))):
        raise ManualTestError("Input must be a regular JSON file outside protected data paths.")
    try:
        payload: Any = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ManualTestError("Input JSON file was not found.") from exc
    except json.JSONDecodeError as exc:
        raise ManualTestError("Input JSON has invalid syntax.") from exc
    except OSError as exc:
        raise ManualTestError("Input JSON could not be read.") from exc
    if not isinstance(payload, dict):
        raise ManualTestError("Input JSON must contain one object with 25 features.")
    return payload


def validate_payload(payload: dict[str, Any], bundle: ModelBundle,
                     contract: dict[str, Any]) -> tuple[PredictionRecord, dict[str, Any]]:
    """Reuse the API's 25-field Pydantic schema and checked contract validator."""
    try:
        record = PredictionRecord.model_validate(payload)
    except ValidationError as exc:
        fields = sorted({".".join(str(part) for part in error["loc"])
                         for error in exc.errors()})
        raise ManualTestError(f"Invalid or missing input fields: {', '.join(fields)}.") from None
    try:
        features, warnings = validate_record(record, contract, bundle)
    except ContractError as exc:
        details = exc.args[0] if exc.args and isinstance(exc.args[0], list) else []
        names = sorted({str(item["field"]) for item in details})
        raise ManualTestError(f"Input value outside contract: {', '.join(names)}.") from None
    except ValueError:
        raise ManualTestError("Input failed frozen bundle validation.") from None
    if warnings:
        names = sorted({warning.split(":", 1)[0] for warning in warnings})
        raise ManualTestError(f"Unknown category in: {', '.join(names)}.")
    return record, features


def predict_local(payload: dict[str, Any], bundle: ModelBundle, contract: dict[str, Any],
                  *, include_explanation: bool = True, top_k: int = 5) -> dict[str, Any]:
    """Call the same shared inference function as the protected API route."""
    if not 1 <= top_k <= 10:
        raise ManualTestError("top-k must be an integer from 1 to 10.")
    record, features = validate_payload(payload, bundle, contract)
    try:
        outcome = infer_records(bundle, [features])[0]
    except Exception:
        raise ManualTestError("Prediction failed; no result was produced.") from None
    explanation: dict[str, Any] | None = None
    explanation_error: str | None = None
    if include_explanation:
        try:
            explanation = bundle.explain(features, top_k=top_k)[0]
        except Exception:
            explanation_error = "SHAP explanation failed; prediction remains valid."
    return {
        "record_id": record.record_id or datetime.now(UTC).strftime("MANUAL-%Y%m%d-%H%M%S"),
        "candidate_version": bundle.metadata["candidate_version"],
        "bundle_version": bundle.metadata["bundle_version"],
        "feature_schema_version": bundle.metadata["feature_schema_version"],
        "probability": outcome.probability,
        "threshold": outcome.threshold,
        "alert": outcome.alert,
        "decision": "NO_REVIEW" if outcome.decision_label == "NO_REVIEW_ALERT"
                    else outcome.decision_label,
        "decision_label": outcome.decision_label,
        "production_approved": bundle.metadata["production_approved"],
        "top_factors": explanation["top_factors"] if explanation is not None else [],
        "explanation_space": explanation["model_output_space"]
                             if explanation is not None else None,
        "explanation_error": explanation_error,
        "shap_disclaimer": SHAP_NOTE if explanation is not None else None,
        "warnings": [],
        "research_only": True,
    }


def print_result(result: dict[str, Any]) -> None:
    """Display local research output in a compact, non-causal format."""
    print("\nERIS Prediction Result")
    print(f"Record ID: {result['record_id']}")
    print(f"Model: {result['candidate_version']}")
    print(f"Bundle: {result['bundle_version']}")
    print(f"Probability: {result['probability']:.4f} ({result['probability']:.2%})")
    print(f"Threshold: {result['threshold']:.6f}")
    print(f"Alert: {str(result['alert']).lower()}")
    print(f"Decision: {result['decision']}")
    print(f"Production approved: {str(result['production_approved']).lower()}")
    if result["top_factors"]:
        print("\nTop contributing factors")
        for rank, factor in enumerate(result["top_factors"], start=1):
            print(f"{rank}. Feature: {factor['feature']}")
            print(f"   Value: {factor['feature_value']}")
            print(f"   Direction: {factor['direction']}")
            print(f"   SHAP value: {factor['shap_value']:.6f}")
        print(SHAP_NOTE)
    elif result["explanation_error"]:
        print(result["explanation_error"])
    else:
        print("SHAP explanation skipped.")
    print("Synthetic/manual research testing only; not an accuracy evaluation.")


def _manual_value(name: str, bundle: ModelBundle, contract: dict[str, Any],
                  default: Any) -> Any:
    kind = bundle.metadata["feature_types"][name]
    print(f"\n{name} — {DESCRIPTIONS[name]}")
    if kind == "nominal":
        choices = contract["nominal_categories"][name]
        print("Options: " + ", ".join(f"{index}: {value}" for index, value
                                    in enumerate(choices, start=1)))
    elif kind == "ordinal":
        choices = bundle.metadata["ordinal_domains"][name]
        print("Allowed levels: " + ", ".join(str(value) for value in choices)
              + " (demo-coded levels; higher is not an HR-approved rule)")
    else:
        bounds = contract["numeric_safety_ranges"][name]
        print(f"Allowed demo range: {bounds[0]}–{bounds[1]}")
    while True:
        raw = input(f"Value [Enter = {default}]: ").strip()
        if not raw:
            return default
        try:
            if kind == "nominal":
                if raw.isdigit() and 1 <= int(raw) <= len(choices):
                    return choices[int(raw) - 1]
                if raw in choices:
                    return raw
            elif kind == "ordinal":
                if str(int(raw)) == raw and int(raw) in choices:
                    return int(raw)
            else:
                number: int | float = float(raw) if any(char in raw for char in ".eE") else int(raw)
                if math.isfinite(number) and bounds[0] <= number <= bounds[1]:
                    return number
        except ValueError:
            pass
        print("Invalid value for this feature; try again or press Enter for the default.")


def prompt_record(bundle: ModelBundle, contract: dict[str, Any],
                  example: dict[str, Any]) -> dict[str, Any] | None:
    """Prompt exactly the frozen 25 model features and require confirmation."""
    print("Synthetic/manual research input only; do not use this to evaluate accuracy.")
    selected: dict[str, Any] = {}
    for name in bundle.metadata["feature_order"]:
        selected[name] = _manual_value(name, bundle, contract, example[name])
    selected["record_id"] = datetime.now(UTC).strftime("MANUAL-%Y%m%d-%H%M%S")
    print("\nReview all 25 feature values:")
    for name in bundle.metadata["feature_order"]:
        print(f"  {name}: {selected[name]}")
    confirm = input("Run prediction? [y/N]: ").strip().casefold()
    if confirm not in {"y", "yes"}:
        print("Prediction cancelled.")
        return None
    return selected


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Local manual test of the frozen ERIS bundle.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--interactive", action="store_true")
    mode.add_argument("--example", choices=("low", "high"))
    mode.add_argument("--input-file", type=Path)
    parser.add_argument("--no-explanation", action="store_true")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--json-output", action="store_true")
    parser.add_argument("--debug", action="store_true")
    return parser


def run(argv: list[str] | None = None) -> int:
    """Run menu or one direct test without server, token, network or data CSV."""
    args = _parser().parse_args(argv)
    if not 1 <= args.top_k <= 10:
        raise ManualTestError("top-k must be an integer from 1 to 10.")
    if args.json_output and not (args.example or args.input_file):
        raise ManualTestError("--json-output requires --example or --input-file.")
    bundle, contract = load_frozen_bundle()
    examples = load_examples()
    include_explanation = not args.no_explanation
    if args.example or args.input_file:
        payload = (examples[f"synthetic_{args.example}_signal"] if args.example
                   else load_input_file(args.input_file))
        result = predict_local(payload, bundle, contract,
                               include_explanation=include_explanation, top_k=args.top_k)
        if args.json_output:
            print(json.dumps(result, ensure_ascii=False, allow_nan=False))
        else:
            print("Synthetic example; not a real employee or accuracy evaluation."
                  if args.example else "Local JSON input; not an accuracy evaluation.")
            print_result(result)
        return 0
    while True:
        print("\nERIS Manual Model Test\n")
        print("1. Nhập dữ liệu nhân viên thủ công")
        print("2. Chạy hồ sơ mẫu nguy cơ thấp")
        print("3. Chạy hồ sơ mẫu nguy cơ cao")
        print("4. Đọc hồ sơ từ file JSON")
        print("5. Thoát")
        choice = input("Select 1–5: ").strip()
        if choice == "5":
            return 0
        if choice == "1":
            payload = prompt_record(bundle, contract, examples["synthetic_low_signal"])
            if payload is None:
                continue
        elif choice in {"2", "3"}:
            signal = "low" if choice == "2" else "high"
            payload = examples[f"synthetic_{signal}_signal"]
            print("Synthetic example only; not a real employee.")
        elif choice == "4":
            payload = load_input_file(Path(input("JSON file path: ").strip()))
        else:
            print("Choose a menu number from 1 to 5.")
            continue
        print_result(predict_local(payload, bundle, contract,
                                   include_explanation=include_explanation, top_k=args.top_k))


def main() -> None:
    """Keep ordinary errors concise while allowing opt-in debugging."""
    argv = sys.argv[1:]
    try:
        code = run(argv)
    except (ManualTestError, EOFError) as exc:
        print(f"ERIS manual test error: {exc or 'Input stream ended.'}", file=sys.stderr)
        if "--debug" in argv:
            traceback.print_exc()
        code = 1
    except Exception:
        print("ERIS manual test error: unexpected local failure.", file=sys.stderr)
        if "--debug" in argv:
            traceback.print_exc()
        code = 1
    raise SystemExit(code)


if __name__ == "__main__":
    main()
