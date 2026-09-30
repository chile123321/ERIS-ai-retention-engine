"""Run fixed RF/XGBoost baselines on saved V2 development folds; never overwrite outputs."""

from __future__ import annotations

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
import xgboost
import yaml

from eris_ml.data.loaders import load_development_data
from eris_ml.data.splitting import assert_development_path
from eris_ml.evaluation.feature_ablation import validate_definitions
from eris_ml.evaluation.model_comparison import (
    TREE_MODELS,
    compare_models,
    comparison_deltas,
    validate_reference,
)
from eris_ml.evaluation.model_comparison_reporting import build_report
from eris_ml.features.definitions import load_feature_definition
from eris_ml.models.factory import create_estimator
from eris_ml.utils.hashing import sha256_file

ROOT = Path(__file__).resolve().parents[1]
DEVELOPMENT = ROOT / "data/processed/development_raw_v1.csv"
REGISTRY = ROOT / "configs/features/feature_set_v2_practical.yaml"
BASELINE = {
    "assignments": ROOT / "artifacts/predictions/baseline_feature_ablation_v2_assignments.csv",
    "oof": ROOT / "artifacts/predictions/baseline_feature_ablation_v2_oof.csv",
    "summary": ROOT / "artifacts/reports/baseline_feature_ablation_v2_metrics.csv",
    "fold_metrics": ROOT / "artifacts/reports/baseline_feature_ablation_v2_folds.csv",
    "trace": ROOT / "artifacts/reports/baseline_feature_ablation_v2_trace.json",
    "report": ROOT / "artifacts/reports/baseline_feature_ablation_v2.md",
}

OUTPUTS = {
    "report": ROOT / "artifacts/reports/baseline_model_comparison_v2.md",
    "summary": ROOT / "artifacts/reports/baseline_model_comparison_v2_metrics.csv",
    "fold_metrics": ROOT / "artifacts/reports/baseline_model_comparison_v2_folds.csv",
    "deltas": ROOT / "artifacts/reports/baseline_model_comparison_v2_deltas.csv",
    "trace": ROOT / "artifacts/reports/baseline_model_comparison_v2_trace.json",
    "oof": ROOT / "artifacts/predictions/baseline_model_comparison_v2_oof.csv",
}


def load_yaml(path: Path) -> dict[str, Any]:
    value: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Expected YAML mapping.")
    return value


def json_parameters(value: Any) -> Any:
    """Preserve non-finite estimator sentinels as explicit metadata strings, not JSON NaN."""
    if isinstance(value, dict):
        return {key: json_parameters(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_parameters(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return str(value)
    return value


def run(development: Path = DEVELOPMENT) -> None:
    assert_development_path(development)  # reject locked path before any filesystem inspection
    if development.is_symlink() or development.resolve() != DEVELOPMENT.resolve():
        raise ValueError("Only the locked development path is accepted.")
    for path in OUTPUTS.values():
        if path.exists() or path.is_symlink():
            raise ValueError(f"Refusing to overwrite {path}")
    registry = load_yaml(REGISTRY)
    protocol = registry["comparison_protocol"]
    fingerprints = protocol["immutable_split_fingerprints"]
    manifest = ROOT / "data/processed/split_manifest_v1.csv"
    if (sha256_file(development) != fingerprints["development_sha256"]
        or sha256_file(manifest) != fingerprints["split_manifest_sha256"]):
        raise ValueError("Locked split fingerprint changed.")
    paths = {name: ROOT / value for name, value in registry["comparison_feature_sets"].items()}
    definitions = {name: load_feature_definition(path) for name, path in paths.items()}
    validate_definitions(definitions)
    old_trace = json.loads(BASELINE["trace"].read_text(encoding="utf-8"))
    # Only verify explicitly approved inputs; never follow arbitrary paths from a trace.
    shared_inputs = [REGISTRY, development, manifest, *paths.values(),
                     ROOT / "src/eris_ml/features/preprocessing.py",
                     ROOT / "src/eris_ml/models/factory.py",
                     ROOT / "src/eris_ml/models/training.py",
                     ROOT / "configs/models/logistic_regression.yaml"]
    for path in shared_inputs:
        if sha256_file(path) != old_trace["input_sha256"][path.relative_to(ROOT).as_posix()]:
            raise ValueError(f"Logistic reference provenance mismatch: {path}")
    for key in ("assignments", "oof", "summary", "fold_metrics"):
        if sha256_file(BASELINE[key]) != old_trace["output_sha256"][key]:
            raise ValueError(f"Logistic artifact hash mismatch: {key}")
    frame = load_development_data(development)
    if frame.shape != (1176, 28) or frame.Attrition.value_counts().to_dict() != {0: 986, 1: 190}:
        raise ValueError("Unexpected development shape or target.")
    assignments = pd.read_csv(BASELINE["assignments"])
    baseline = pd.read_csv(BASELINE["oof"])
    validate_reference(frame, assignments, baseline)
    config_paths = {model: ROOT / f"configs/models/{model}.yaml" for model in TREE_MODELS}
    configs = {model: load_yaml(path) for model, path in config_paths.items()}
    resolved = {model: create_estimator(config).get_params(deep=False)
                for model, config in configs.items()}
    # Resolve native XGBoost defaults on synthetic data, without retaining/serializing a model.
    probe = create_estimator(configs["xgboost"])
    probe.fit(np.array([[0.], [1.], [2.], [3.]]), np.array([0, 0, 1, 1]))
    native = json.loads(probe.get_booster().save_config())
    del probe
    hash_paths = [*shared_inputs, *BASELINE.values(), *config_paths.values(),
                  ROOT / "src/eris_ml/evaluation/model_comparison.py",
                  ROOT / "src/eris_ml/evaluation/model_comparison_reporting.py",
                  Path(__file__).resolve()]
    hashes = {path.relative_to(ROOT).as_posix(): sha256_file(path) for path in hash_paths}
    trace = {
        "experiment": "baseline-model-comparison-v2",
        "started_at_utc": datetime.now(UTC).isoformat(),
        "development_rows": 1176, "target_distribution": {0: 986, 1: 190},
        "cv": old_trace["cv"], "fold_assignment_regenerated": False,
        "data_version": "IBM-existing-v1", "configs": configs,
        "resolved_parameters": json_parameters(resolved),
        "xgboost_native_defaults_synthetic_probe": native,
        "logistic_reference_parameters": old_trace["logistic_parameters"],
        "logistic_reference_versions": old_trace["versions"],
        "feature_sets": {name: list(value.all_features) for name, value in definitions.items()},
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "pandas": pd.__version__, "scikit_learn": sklearn.__version__,
                     "xgboost": xgboost.__version__},
        "input_sha256": hashes, "final_test_accessed": False, "model_serialized": False,
    }
    json.dumps(trace, allow_nan=False)  # fail before expensive fitting/output creation
    print("Preflight passed: IDs/targets/folds and Logistic provenance hashes match.", flush=True)
    print(json.dumps({"fixed_configs": configs, "versions": trace["versions"]}), flush=True)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = compare_models(frame, definitions, assignments, baseline, configs)
    if any(sha256_file(ROOT / path) != digest for path, digest in hashes.items()):
        raise ValueError("Input/config/source drift during fitting.")
    warning_counts: dict[str, int] = {}
    for warning in caught:
        message = f"{warning.category.__name__}: {warning.message}"
        warning_counts[message] = warning_counts.get(message, 0) + 1
    trace["warnings"] = warning_counts
    trace["completed_at_utc"] = datetime.now(UTC).isoformat()
    for path in OUTPUTS.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    for key, table in (("summary", result.summary), ("fold_metrics", result.folds),
                       ("oof", result.oof), ("deltas", comparison_deltas(result))):
        with OUTPUTS[key].open("x", encoding="utf-8", newline="") as stream:
            table.to_csv(stream, index=False)
    trace["output_sha256"] = {key: sha256_file(OUTPUTS[key])
                              for key in ("summary", "fold_metrics", "oof", "deltas")}
    with OUTPUTS["trace"].open("x", encoding="utf-8") as stream:
        json.dump(trace, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    with OUTPUTS["report"].open("x", encoding="utf-8") as stream:
        stream.write(build_report(result, trace))
    print(result.summary.to_string(index=False))
    print(f"Warnings: {warning_counts}; OOF rows: {len(result.oof)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development", type=Path, default=DEVELOPMENT)
    arguments = parser.parse_args()
    try:
        run(arguments.development)
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(1, f"V2 comparison stopped: {exc}\n")
