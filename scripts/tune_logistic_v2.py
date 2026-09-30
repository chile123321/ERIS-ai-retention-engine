"""Freeze the V2 Logistic nested protocol, then train only on verified development rows."""

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
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ParameterGrid

from eris_ml.data.loaders import load_development_data
from eris_ml.data.splitting import assert_development_path
from eris_ml.evaluation.feature_ablation import validate_definitions
from eris_ml.evaluation.logistic_nested_reporting import build_report
from eris_ml.evaluation.model_comparison import validate_reference
from eris_ml.features.definitions import load_feature_definition
from eris_ml.models.logistic_nested_v2 import FEATURES, run_nested
from eris_ml.models.tuning import load_yaml_config, validate_search_config
from eris_ml.utils.hashing import sha256_file

ROOT = Path(__file__).resolve().parents[1]
DEVELOPMENT = ROOT / "data/processed/development_raw_v1.csv"
BASE = "baseline_feature_ablation_v2"
OUTPUTS = {key: ROOT / f"artifacts/reports/logistic_nested_v2_{suffix}" for key, suffix in {
    "preflight": "preflight.json", "trace": "trace.json", "summary": "metrics.csv",
    "folds": "folds.csv", "selected": "parameters.csv", "search": "search.csv",
    "deltas": "deltas.csv",
}.items()}
OUTPUTS["report"] = ROOT / "artifacts/reports/logistic_nested_v2.md"
OUTPUTS["oof"] = ROOT / "artifacts/predictions/logistic_nested_v2_oof.csv"
OUTPUTS["inner_assignments"] = ROOT / "artifacts/predictions/logistic_nested_v2_inner_folds.csv"


def write_json(path: Path, value: dict[str, Any]) -> None:
    payload = json.dumps(value, indent=2, allow_nan=False)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(payload + "\n")


def run(development: Path = DEVELOPMENT) -> None:
    assert_development_path(development)
    if development.is_symlink() or development.resolve() != DEVELOPMENT.resolve():
        raise ValueError("Only locked development path is allowed.")
    if any(path.exists() or path.is_symlink() for path in OUTPUTS.values()):
        raise ValueError("Refusing existing V2 nested outputs, including preflight trace.")
    registry_path = ROOT / "configs/features/feature_set_v2_practical.yaml"
    registry = load_yaml_config(registry_path)
    feature_paths = {key: ROOT / value
                     for key, value in registry["comparison_feature_sets"].items()}
    definitions = {key: load_feature_definition(path) for key, path in feature_paths.items()}
    validate_definitions(definitions)
    protocol_path = ROOT / "configs/tuning/logistic_nested_v2.yaml"
    protocol = load_yaml_config(protocol_path)
    if protocol != {
        "protocol_version": "logistic-nested-v2", "feature_sets": list(FEATURES),
        "outer_assignment": f"artifacts/predictions/{BASE}_assignments.csv",
        "inner_cv": {"n_splits": 4, "shuffle": True, "random_state": 43},
        "search_config": "configs/tuning/logistic_regression_search_v1.yaml",
        "selection_metric": "average_precision",
        "tie_break": {"absolute_tolerance": 1e-12, "order": ["smaller_C", "l2_before_l1"]},
        "execution": {"outer_jobs": 1, "search_jobs": 1}, "full_development_search": False,
        "final_test_used": False, "production_approved": False,
    }:
        raise ValueError("Nested protocol drift.")
    search_path = ROOT / protocol["search_config"]
    search_config = validate_search_config(load_yaml_config(search_path), "logistic_regression")
    if search_config["search_space"] != {
        "penalty": ["l1", "l2"], "C": [.001, .01, .03, .1, .3, 1., 3., 10., 30., 100.],
    }:
        raise ValueError("Expected unchanged V1 20-candidate grid.")
    old_trace_path = ROOT / f"artifacts/reports/{BASE}_trace.json"
    old_trace = json.loads(old_trace_path.read_text(encoding="utf-8"))
    baseline_paths = {
        "assignments": ROOT / protocol["outer_assignment"],
        "oof": ROOT / f"artifacts/predictions/{BASE}_oof.csv",
        "summary": ROOT / f"artifacts/reports/{BASE}_metrics.csv",
        "fold_metrics": ROOT / f"artifacts/reports/{BASE}_folds.csv",
    }
    manifest = ROOT / "data/processed/split_manifest_v1.csv"
    shared = [registry_path, development, manifest, *feature_paths.values(),
              ROOT / "configs/models/logistic_regression.yaml",
              ROOT / "src/eris_ml/features/preprocessing.py",
              ROOT / "src/eris_ml/models/training.py", ROOT / "src/eris_ml/models/factory.py"]
    for path in shared:
        if sha256_file(path) != old_trace["input_sha256"][path.relative_to(ROOT).as_posix()]:
            raise ValueError(f"Baseline input fingerprint mismatch: {path}")
    pinned = registry["comparison_protocol"]["immutable_split_fingerprints"]
    if (sha256_file(development) != pinned["development_sha256"]
        or sha256_file(manifest) != pinned["split_manifest_sha256"]):
        raise ValueError("Locked split fingerprint mismatch.")
    for key, path in baseline_paths.items():
        if sha256_file(path) != old_trace["output_sha256"][key]:
            raise ValueError(f"Baseline artifact fingerprint mismatch: {key}")
    frame = load_development_data(development)
    if frame.shape != (1176, 28) or frame.Attrition.value_counts().to_dict() != {0: 986, 1: 190}:
        raise ValueError("Unexpected development shape/target.")
    assignment, baseline = (pd.read_csv(baseline_paths[key]) for key in ("assignments", "oof"))
    validate_reference(frame, assignment, baseline)
    hash_paths = [*shared, *baseline_paths.values(), old_trace_path, protocol_path, search_path,
                  ROOT / "src/eris_ml/models/logistic_nested_v2.py",
                  ROOT / "src/eris_ml/models/tuning.py",
                  ROOT / "src/eris_ml/evaluation/feature_ablation.py",
                  ROOT / "src/eris_ml/evaluation/model_comparison.py",
                  ROOT / "src/eris_ml/evaluation/logistic_nested_reporting.py",
                  Path(__file__).resolve()]
    hashes = {path.relative_to(ROOT).as_posix(): sha256_file(path) for path in hash_paths}
    trace = {
        "experiment": "logistic-nested-v2", "started_at_utc": datetime.now(UTC).isoformat(),
        "protocol": protocol, "search_config": search_config,
        "ordered_grid": list(ParameterGrid(search_config["search_space"])),
        "resolved_estimator": LogisticRegression(**search_config["fixed_parameters"]).get_params(),
        "baseline_parameters": old_trace["logistic_parameters"],
        "baseline_versions": old_trace["versions"],
        "feature_sets": {name: list(definitions[name].all_features) for name in FEATURES},
        "development_rows": 1176, "target": {0: 986, 1: 190}, "data_version": "IBM-existing-v1",
        "outer_assignment_regenerated": False, "input_sha256": hashes,
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "pandas": pd.__version__, "scikit_learn": sklearn.__version__},
        "final_test_accessed": False, "production_model_serialized": False,
    }
    for path in OUTPUTS.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    write_json(OUTPUTS["preflight"], trace)  # on disk BEFORE first fit; never overwritten
    preflight_hash = sha256_file(OUTPUTS["preflight"])
    print("Preflight passed and protocol/grid/seed/tie rule frozen on disk.", flush=True)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = run_nested(frame, definitions, assignment, baseline, search_config,
                            progress=lambda message: print(message, flush=True))
    if (any(sha256_file(ROOT / path) != digest for path, digest in hashes.items())
        or sha256_file(OUTPUTS["preflight"]) != preflight_hash):
        raise ValueError("Input or protocol drift during training.")
    counts: dict[str, int] = {}
    for warning in caught:
        key = f"{warning.category.__name__}: {warning.message}"
        counts[key] = counts.get(key, 0) + 1
    trace.update({"warnings": counts, "completed_at_utc": datetime.now(UTC).isoformat(),
                  "preflight_sha256": preflight_hash})
    for key in ("summary", "folds", "selected", "search", "deltas", "oof", "inner_assignments"):
        with OUTPUTS[key].open("x", encoding="utf-8", newline="") as stream:
            getattr(result, key).to_csv(stream, index=False)
    trace["output_sha256"] = {key: sha256_file(path) for key, path in OUTPUTS.items()
                              if key not in ("trace", "report")}
    write_json(OUTPUTS["trace"], trace)
    with OUTPUTS["report"].open("x", encoding="utf-8") as stream:
        stream.write(build_report(result, trace))
    print(result.summary.to_string(index=False))
    print(result.selected.to_string(index=False))
    print(f"Warnings: {counts}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development", type=Path, default=DEVELOPMENT)
    arguments = parser.parse_args()
    try:
        run(arguments.development)
    except (ValueError, KeyError, OSError) as exc:
        parser.exit(1, f"V2 nested tuning stopped: {exc}\n")
