"""Reproduce the fixed V2 baseline comparison on the unchanged development split."""

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
import yaml

from eris_ml.data.loaders import load_development_data
from eris_ml.data.splitting import assert_development_path
from eris_ml.evaluation.feature_ablation import evaluate_feature_ablation, validate_definitions
from eris_ml.evaluation.feature_ablation_reporting import build_report
from eris_ml.features.definitions import load_feature_definition
from eris_ml.models.factory import create_estimator
from eris_ml.utils.hashing import sha256_file

ROOT = Path(__file__).resolve().parents[1]
DEVELOPMENT = ROOT / "data/processed/development_raw_v1.csv"
REGISTRY = ROOT / "configs/features/feature_set_v2_practical.yaml"
LOGISTIC = ROOT / "configs/models/logistic_regression.yaml"
OUTPUTS = {
    "report": ROOT / "artifacts/reports/baseline_feature_ablation_v2.md",
    "summary": ROOT / "artifacts/reports/baseline_feature_ablation_v2_metrics.csv",
    "fold_metrics": ROOT / "artifacts/reports/baseline_feature_ablation_v2_folds.csv",
    "trace": ROOT / "artifacts/reports/baseline_feature_ablation_v2_trace.json",
    "oof": ROOT / "artifacts/predictions/baseline_feature_ablation_v2_oof.csv",
    "assignments": ROOT / "artifacts/predictions/baseline_feature_ablation_v2_assignments.csv",
}


def load_yaml(path: Path) -> dict[str, Any]:
    value: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Expected YAML mapping.")
    return value


def run(development: Path = DEVELOPMENT) -> None:
    """Verify provenance before any fit; create only new V2 artifacts."""
    assert_development_path(development)  # before stat, resolve, hash or read
    if development.is_symlink() or development.resolve() != DEVELOPMENT.resolve():
        raise ValueError("Only the locked development_raw_v1.csv path is accepted.")
    for path in OUTPUTS.values():
        if path.exists() or path.is_symlink():
            raise ValueError(f"Refusing to overwrite output: {path.relative_to(ROOT)}")
    registry = load_yaml(REGISTRY)
    protocol = registry["comparison_protocol"]
    if (protocol["selection_data"] != "development_only" or protocol["final_test_used"]
        or protocol["shared_outer_folds"] != {
            "type": "StratifiedKFold", "n_splits": 5, "shuffle": True, "random_state": 42,
        }):
        raise ValueError("Locked development CV protocol changed.")
    paths = {name: ROOT / path for name, path in registry["comparison_feature_sets"].items()}
    definitions = {name: load_feature_definition(path) for name, path in paths.items()}
    validate_definitions(definitions)
    fingerprints = protocol["immutable_split_fingerprints"]
    manifest = ROOT / "data/processed/split_manifest_v1.csv"
    if (sha256_file(development) != fingerprints["development_sha256"]
        or sha256_file(manifest) != fingerprints["split_manifest_sha256"]):
        raise ValueError("Locked development or split manifest fingerprint changed.")
    frame = load_development_data(development)
    if (frame.shape != (1176, 28) or frame.Attrition.value_counts().to_dict() != {0: 986, 1: 190}
        or frame.isna().any().any()):
        raise ValueError("Unexpected locked development shape, target or missing data.")
    logistic = load_yaml(LOGISTIC)
    hash_paths = [REGISTRY, LOGISTIC, development, manifest, *paths.values(),
                  ROOT / "src/eris_ml/features/preprocessing.py",
                  ROOT / "src/eris_ml/models/factory.py",
                  ROOT / "src/eris_ml/models/training.py",
                  ROOT / "src/eris_ml/evaluation/feature_ablation.py",
                  ROOT / "src/eris_ml/evaluation/feature_ablation_reporting.py",
                  Path(__file__).resolve()]
    hashes = {path.relative_to(ROOT).as_posix(): sha256_file(path) for path in hash_paths}
    print("Preflight passed: 1176 development rows; four locked schemas; fingerprints match.")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = evaluate_feature_ablation(frame, definitions, logistic)
    # Fail if an input changed while fitting; never silently publish mixed provenance.
    if any(sha256_file(ROOT / path) != digest for path, digest in hashes.items()):
        raise ValueError("Input/config/source drift detected during the run.")
    warning_counts: dict[str, int] = {}
    for warning in caught:
        text = f"{warning.category.__name__}: {warning.message}"
        warning_counts[text] = warning_counts.get(text, 0) + 1
    trace = {
        "experiment": "baseline-feature-ablation-v2",
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "development_rows": len(frame), "target_distribution": {0: 986, 1: 190},
        "positive_rate": float(frame.Attrition.mean()), "data_version": "IBM-existing-v1",
        "split_version": protocol["split_version"], "cv": protocol["shared_outer_folds"],
        "logistic_config": logistic,
        "logistic_parameters": create_estimator(logistic).get_params(deep=False),
        "feature_sets": {name: list(definition.all_features)
                         for name, definition in definitions.items()},
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "pandas": pd.__version__, "scikit_learn": sklearn.__version__},
        "input_sha256": hashes, "warnings": warning_counts,
        "final_test_accessed": False, "model_serialized": False,
        "outputs": {key: path.relative_to(ROOT).as_posix() for key, path in OUTPUTS.items()},
    }
    report = build_report(result, trace)
    for path in OUTPUTS.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    for key, table in (("summary", result.summary), ("fold_metrics", result.folds),
                       ("oof", result.oof), ("assignments", result.assignments)):
        with OUTPUTS[key].open("x", encoding="utf-8", newline="") as stream:
            table.to_csv(stream, index=False)
    trace["output_sha256"] = {key: sha256_file(OUTPUTS[key])
                              for key in ("summary", "fold_metrics", "oof", "assignments")}
    with OUTPUTS["trace"].open("x", encoding="utf-8") as stream:
        json.dump(trace, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    with OUTPUTS["report"].open("x", encoding="utf-8") as stream:
        stream.write(report)
    print(result.summary[["feature_set", "model", "average_precision", "roc_auc",
                          "brier_score"]].to_string(index=False))
    print(f"OOF rows: {len(result.oof)}; 1176 per candidate. Report: {OUTPUTS['report']}")
    print(f"Warnings: {warning_counts}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development", type=Path, default=DEVELOPMENT)
    arguments = parser.parse_args()
    try:
        run(arguments.development)
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(1, f"V2 baseline stopped: {exc}\n")
