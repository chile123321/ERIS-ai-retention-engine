"""Run hypothetical capacity policies on verified V2 Logistic raw OOF probabilities."""

import argparse
import json
import platform
import warnings
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn

from eris_ml.data.loaders import load_development_data
from eris_ml.data.splitting import assert_development_path
from eris_ml.evaluation.feature_ablation import validate_definitions
from eris_ml.evaluation.threshold_policy_v2 import PROTOCOL, evaluate, summarize, validate_inputs
from eris_ml.evaluation.threshold_policy_v2_reporting import build_report
from eris_ml.features.definitions import load_feature_definition
from eris_ml.models.calibration_v2 import FEATURES
from eris_ml.models.factory import create_estimator
from eris_ml.models.tuning import load_yaml_config
from eris_ml.utils.hashing import sha256_file

ROOT = Path(__file__).resolve().parents[1]
DEVELOPMENT = ROOT / "data/processed/development_raw_v1.csv"
BASE = "baseline_feature_ablation_v2"
OUTPUTS = {
    key: ROOT / f"artifacts/reports/threshold_policy_v2_{suffix}"
    for key, suffix in {
        "preflight": "preflight.json",
        "trace": "trace.json",
        "summary": "metrics.csv",
        "folds": "folds.csv",
        "selections": "thresholds.csv",
        "stability": "stability.csv",
        "paired": "paired.csv",
        "regression": "raw_regression.csv",
    }.items()
}
OUTPUTS["report"] = ROOT / "artifacts/reports/threshold_policy_v2.md"
OUTPUTS["oof"] = ROOT / "artifacts/predictions/threshold_policy_v2_oof.csv"
OUTPUTS["inner_oof"] = ROOT / "artifacts/predictions/threshold_policy_v2_inner_oof.csv"


def write_json(path: Path, value: dict[str, Any]) -> None:
    payload = json.dumps(value, indent=2, allow_nan=False)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(payload + "\n")


def run(development: Path = DEVELOPMENT) -> None:
    assert_development_path(development)
    if development.is_symlink() or development.resolve() != DEVELOPMENT.resolve():
        raise ValueError("Only the locked development path is allowed.")
    if any(path.exists() or path.is_symlink() for path in OUTPUTS.values()):
        raise ValueError("Refusing existing V2 threshold artifacts/preflight.")
    protocol_path = ROOT / "configs/threshold/threshold_policy_v2.yaml"
    protocol = load_yaml_config(protocol_path)
    if protocol != PROTOCOL:
        raise ValueError("Threshold policy differs from locked V2 protocol.")
    registry_path = ROOT / "configs/features/feature_set_v2_practical.yaml"
    registry = load_yaml_config(registry_path)
    feature_paths = {
        name: ROOT / path for name, path in registry["comparison_feature_sets"].items()
    }
    definitions = {name: load_feature_definition(path) for name, path in feature_paths.items()}
    validate_definitions(definitions)
    logistic_path = ROOT / protocol["base_config"]
    logistic = load_yaml_config(logistic_path)
    old_trace_path = ROOT / f"artifacts/reports/{BASE}_trace.json"
    old_trace = json.loads(old_trace_path.read_text(encoding="utf-8"))
    baseline_paths = {
        "assignments": ROOT / protocol["outer_assignment"],
        "oof": ROOT / f"artifacts/predictions/{BASE}_oof.csv",
        "summary": ROOT / f"artifacts/reports/{BASE}_metrics.csv",
        "fold_metrics": ROOT / f"artifacts/reports/{BASE}_folds.csv",
    }
    calibration_oof_path = ROOT / "artifacts/predictions/calibration_v2_oof.csv"
    calibration_trace_path = ROOT / "artifacts/reports/calibration_v2_trace.json"
    calibration_trace = json.loads(calibration_trace_path.read_text(encoding="utf-8"))
    if sha256_file(calibration_oof_path) != calibration_trace["output_sha256"]["oof"]:
        raise ValueError("Calibration OOF fingerprint mismatch.")
    manifest = ROOT / "data/processed/split_manifest_v1.csv"
    shared = [
        registry_path,
        development,
        manifest,
        logistic_path,
        *feature_paths.values(),
        ROOT / "src/eris_ml/features/preprocessing.py",
        ROOT / "src/eris_ml/models/training.py",
        ROOT / "src/eris_ml/models/factory.py",
    ]
    for path in shared:
        key = path.relative_to(ROOT).as_posix()
        digest = sha256_file(path)
        if (
            digest != old_trace["input_sha256"][key]
            or digest != calibration_trace["input_sha256"][key]
        ):
            raise ValueError(f"Baseline/calibration provenance mismatch: {path}")
    pinned = registry["comparison_protocol"]["immutable_split_fingerprints"]
    if (
        sha256_file(development) != pinned["development_sha256"]
        or sha256_file(manifest) != pinned["split_manifest_sha256"]
    ):
        raise ValueError("Locked split fingerprint mismatch.")
    for key, path in baseline_paths.items():
        if (
            sha256_file(path) != old_trace["output_sha256"][key]
            or sha256_file(path)
            != calibration_trace["input_sha256"][path.relative_to(ROOT).as_posix()]
        ):
            raise ValueError(f"Baseline artifact fingerprint mismatch: {key}")
    frame = load_development_data(development)
    if frame.shape != (1176, 28) or frame.Attrition.value_counts().to_dict() != {0: 986, 1: 190}:
        raise ValueError("Unexpected development shape/target.")
    assignment, baseline = (pd.read_csv(baseline_paths[key]) for key in ("assignments", "oof"))
    calibration = pd.read_csv(calibration_oof_path)
    raw = calibration.loc[calibration.method == "raw"].copy()
    regression = validate_inputs(frame, assignment, baseline, raw)
    sources = [
        "src/eris_ml/evaluation/threshold_policy_v2.py",
        "src/eris_ml/evaluation/threshold_policy_v2_reporting.py",
        "src/eris_ml/evaluation/thresholding.py",
        "src/eris_ml/models/calibration_v2.py",
        "src/eris_ml/evaluation/feature_ablation.py",
        "src/eris_ml/evaluation/model_comparison.py",
    ]
    hash_paths = [
        *shared,
        *baseline_paths.values(),
        old_trace_path,
        protocol_path,
        calibration_trace_path,
        calibration_oof_path,
        *(ROOT / path for path in sources),
        Path(__file__).resolve(),
    ]
    hashes = {path.relative_to(ROOT).as_posix(): sha256_file(path) for path in hash_paths}
    trace = {
        "experiment": "threshold-policy-v2",
        "started_at_utc": datetime.now(UTC).isoformat(),
        "protocol": protocol,
        "logistic_config": logistic,
        "resolved_parameters": create_estimator(logistic).get_params(),
        "development_rows": 1176,
        "target_distribution": {0: 986, 1: 190},
        "feature_sets": {name: list(definitions[name].all_features) for name in FEATURES},
        "input_sha256": hashes,
        "outer_assignment_regenerated": False,
        "outer_probabilities": "unchanged saved baseline, verified against calibration raw",
        "baseline_calibration_raw_max_difference": float(regression.max_abs_difference.max()),
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "baseline_versions": old_trace["versions"],
        "data_version": "IBM-existing-v1",
        "final_test_accessed": False,
        "business_approved": False,
        "model_serialized": False,
    }
    for path in OUTPUTS.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    write_json(OUTPUTS["preflight"], trace)
    preflight_hash = sha256_file(OUTPUTS["preflight"])
    print("Alignment, hashes and raw regression passed; protocol frozen before fits.", flush=True)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        tables = evaluate(
            frame,
            definitions,
            assignment,
            baseline,
            raw,
            logistic,
            progress=lambda message: print(message, flush=True),
        )
    if (
        any(sha256_file(ROOT / path) != digest for path, digest in hashes.items())
        or sha256_file(OUTPUTS["preflight"]) != preflight_hash
    ):
        raise ValueError("Input/protocol drift during evaluation.")
    tables.update(summarize(tables["oof"], tables["selections"]))
    counts: dict[str, int] = {}
    for warning in caught:
        key = f"{warning.category.__name__}: {warning.message}"
        counts[key] = counts.get(key, 0) + 1
    for key, table in tables.items():
        with OUTPUTS[key].open("x", encoding="utf-8", newline="") as stream:
            table.to_csv(stream, index=False)
    trace.update(
        {
            "completed_at_utc": datetime.now(UTC).isoformat(),
            "warnings": counts,
            "preflight_sha256": preflight_hash,
            "output_sha256": {key: sha256_file(OUTPUTS[key]) for key in tables},
        }
    )
    write_json(OUTPUTS["trace"], trace)
    with OUTPUTS["report"].open("x", encoding="utf-8") as stream:
        stream.write(build_report(tables, trace))
    print(tables["summary"].to_string(index=False))
    print(tables["selections"].to_string(index=False))
    print(f"Warnings: {counts}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development", type=Path, default=DEVELOPMENT)
    args = parser.parse_args()
    try:
        run(args.development)
    except (ValueError, KeyError, OSError) as exc:
        parser.exit(1, f"V2 threshold evaluation stopped: {exc}\n")
