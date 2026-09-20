"""Freeze the approved-scope capstone candidate without model fitting or holdout access."""

import argparse
import os
import platform
from importlib.metadata import version
from pathlib import Path
from typing import Any

import yaml

from eris_ml.data.splitting import assert_development_path
from eris_ml.utils.freeze import (
    CALIBRATION_CONFIG,
    FAIRNESS_PROTOCOL,
    FEATURE_CONFIG,
    MODEL_CONFIG,
    THRESHOLD_CONFIG,
    build_freeze_manifest,
    validate_freeze_manifest,
    verify_manifest_hashes,
)
from eris_ml.utils.freeze_reporting import write_freeze_report

ROOT = Path(__file__).resolve().parents[1]
INPUTS = {
    "feature_config": ROOT / FEATURE_CONFIG,
    "model_config": ROOT / MODEL_CONFIG,
    "calibration_config": ROOT / CALIBRATION_CONFIG,
    "threshold_config": ROOT / THRESHOLD_CONFIG,
    "fairness_protocol": ROOT / FAIRNESS_PROTOCOL,
}
OUTPUTS = {
    "manifest": ROOT / "configs/release/candidate_v1.yaml",
    "report": ROOT / "artifacts/reports/candidate_freeze_v1.md",
}


def parse_args() -> argparse.Namespace:
    """Require explicit approved paths for the auditable one-time freeze command."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (*INPUTS, *OUTPUTS):
        parser.add_argument(f"--{name.replace('_', '-')}", type=Path, required=True)
    return parser.parse_args()


def validate_paths(args: argparse.Namespace) -> None:
    """Reject quarantine/final-test aliases before any metadata or content access to them."""
    for name, expected in {**INPUTS, **OUTPUTS}.items():
        path: Path = getattr(args, name)
        normalized = os.path.normcase(os.path.abspath(path)).replace("\\", "/")
        if "/data/quarantine/" in normalized.lower():
            raise ValueError("Quarantined data is prohibited in Step 10E.")
        assert_development_path(path)
        approved = os.path.normcase(os.path.abspath(expected)).replace("\\", "/")
        if normalized != approved:
            raise ValueError(f"Only the approved {name} path is permitted.")
        if path.is_symlink() or path.parent.is_symlink():
            raise ValueError("Freeze paths may not be symlinks.")
    for path in INPUTS.values():
        if not path.is_file():
            raise FileNotFoundError(f"Approved freeze input is missing: {path}")


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Freeze configuration is not a YAML mapping: {path}")
    return value


def _library_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "pandas": version("pandas"),
        "scikit-learn": version("scikit-learn"),
        "xgboost": version("xgboost"),
        "numpy": version("numpy"),
        "matplotlib": version("matplotlib"),
    }


def main() -> None:
    args = parse_args()
    validate_paths(args)
    manifest = build_freeze_manifest(
        ROOT,
        _load_yaml(args.model_config),
        _load_yaml(args.calibration_config),
        _load_yaml(args.threshold_config),
        _load_yaml(args.fairness_protocol),
        _library_versions(),
    )
    verify_manifest_hashes(manifest, ROOT)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(yaml.safe_dump(manifest, sort_keys=False, allow_unicode=True),
                             encoding="utf-8")
    validate_freeze_manifest(_load_yaml(args.manifest))
    write_freeze_report(args.report, manifest)
    print("Step 10: COMPLETE - candidate eris-xgboost-v1 frozen for capstone preparation.")
    print("Business/production/final-test evaluation approvals: false.")
    print("Raw-data SHA-256: missing from approved existing evidence; no raw data reopened.")


if __name__ == "__main__":
    main()
