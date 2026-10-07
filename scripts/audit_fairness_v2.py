"""Audit saved V2 Logistic raw/top-k artifacts without fitting or policy selection."""

import hashlib
import json
import platform
import runpy
import subprocess
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from eris_ml.data.splitting import assert_development_path
from eris_ml.evaluation.fairness_v2 import summarize, validate_inputs, validate_protocol
from eris_ml.evaluation.fairness_v2_reporting import build_report
from eris_ml.utils.hashing import sha256_file

ROOT = Path(__file__).resolve().parents[1]
PREFIX = "artifacts/reports/fairness_v2"
TABLES = ("probability", "policy", "disparities", "paired_policy", "paired_probability",
          "probability_stability", "policy_stability", "disparity_stability", "paired_stability")
OUTPUTS = {
    **{name: f"{PREFIX}_{name}.csv" for name in TABLES},
    "preflight": f"{PREFIX}_preflight.json", "trace": f"{PREFIX}_trace.json",
    "report": f"{PREFIX}.md",
}
INPUTS = {
    "development": "data/processed/development_raw_v1.csv",
    "assignment": "artifacts/predictions/baseline_feature_ablation_v2_assignments.csv",
    "audit": "data/processed/fairness_audit_development_v1.csv",
    "baseline": "artifacts/predictions/baseline_feature_ablation_v2_oof.csv",
    "calibration": "artifacts/predictions/calibration_v2_oof.csv",
    "policy": "artifacts/predictions/threshold_policy_v2_oof.csv",
}
PROTOCOL = "configs/evaluation/fairness_protocol_v2.yaml"
BRIDGE = "docs/quality/environment_audit_20261001_readonly.json"


def safe_path(root: Path, relative: str) -> Path:
    """Reject unsafe names before stat/open and symlinks before resolving."""
    rel = Path(relative)
    assert_development_path(rel)
    if rel.is_absolute() or ".." in rel.parts or any(
        text in relative.lower() for text in ("quarantine", "mixed", "fairness_audit_v1.csv")
    ):
        raise ValueError("Unsafe audit input path.")
    path = root / rel
    if any(p.is_symlink() for p in [path, *path.parents] if p != root.parent):
        raise ValueError("Symlink audit inputs are prohibited.")
    return path


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Expected JSON mapping.")
    return value


def verify_revision(root: Path, rel: str, historical: str, actual: str,
                    bridge: dict[str, Any], script: str) -> None:
    """Require exact reviewed hash pair and historical bytes; no generic drift allowance."""
    approved = bridge.get(rel)
    if (not approved or approved["current"] != actual
        or approved["historical"] != historical or script not in approved["traces"]):
        raise ValueError(f"Unreviewed provenance drift: {rel}")
    old_bytes = subprocess.check_output(["git", "show", f"HEAD:{rel}"], cwd=root)
    if hashlib.sha256(old_bytes).hexdigest() != historical:
        raise ValueError(f"Historical HEAD does not prove source revision: {rel}")
    if rel.endswith("feature_set_v2_practical.yaml"):
        old_config = yaml.safe_load(old_bytes)
        new_config = yaml.safe_load(safe_path(root, rel).read_text(encoding="utf-8"))
        for key in ("version", "preferred_feature_set", "comparison_feature_sets",
                    "comparison_protocol"):
            if old_config[key] != new_config[key]:
                raise ValueError("Registry experimental fields changed.")


def provenance(root: Path) -> dict[str, Any]:
    """Verify historical outputs and exact reviewed revisions, without changing guards."""
    bridge = read_json(safe_path(root, BRIDGE))["historical_source_drift"]
    hashes: dict[str, str] = {}
    revisions = []
    versions = {}
    output_count = 0
    for script in ("baseline_ablation_v2", "calibrate_v2", "evaluate_threshold_policy_v2"):
        module = runpy.run_path(str(root / "scripts" / f"{script}.py"),
                               run_name="fairness_readonly_provenance")
        outputs = module["OUTPUTS"]
        trace_path = outputs["trace"]
        trace = read_json(trace_path)
        hashes[trace_path.relative_to(root).as_posix()] = sha256_file(trace_path)
        versions[script] = trace["versions"]
        for rel, historical in trace["input_sha256"].items():
            path = safe_path(root, rel)
            actual = sha256_file(path)
            hashes[rel] = actual
            if actual == historical:
                continue
            verify_revision(root, rel, historical, actual, bridge, script)
            revisions.append({"trace": script, "path": rel, "historical_sha256": historical,
                              "current_sha256": actual})
        for key, expected in trace["output_sha256"].items():
            path = outputs[key]
            safe_path(root, path.relative_to(root).as_posix())
            actual = sha256_file(path)
            if actual != expected:
                raise ValueError(f"Historical output changed: {path.name}")
            hashes[path.relative_to(root).as_posix()] = actual
            output_count += 1
        if "preflight_sha256" in trace:
            if sha256_file(outputs["preflight"]) != trace["preflight_sha256"]:
                raise ValueError("Historical preflight changed.")
            hashes[outputs["preflight"].relative_to(root).as_posix()] = trace["preflight_sha256"]
    for rel in (*INPUTS.values(), PROTOCOL, BRIDGE, "pyproject.toml",
                "requirements/quality-py312.txt", "configs/evaluation/fairness_protocol_v1.yaml",
                "artifacts/reports/fairness_audit_v1.md",
                "src/eris_ml/evaluation/fairness_v2.py",
                "src/eris_ml/evaluation/fairness_v2_reporting.py",
                "src/eris_ml/evaluation/fairness.py",
                "src/eris_ml/data/fairness_recovery.py", "scripts/audit_fairness_v2.py"):
        hashes[rel] = sha256_file(safe_path(root, rel))
    old_audit = read_json(root / "docs/quality/environment_audit_20261001_after.json")
    if hashes[INPUTS["audit"]] != old_audit["after_sha256"][INPUTS["audit"]]:
        raise ValueError("Development-only audit recovery file fingerprint changed.")
    return {"input_sha256": hashes, "reviewed_source_revisions": revisions,
            "historical_versions": versions, "verified_historical_output_entries": output_count}


def write_json(path: Path, value: dict[str, Any]) -> None:
    payload = json.dumps(value, indent=2, allow_nan=False) + "\n"
    with path.open("x", encoding="utf-8") as stream:
        stream.write(payload)


def run(root: Path = ROOT) -> None:
    for relative in OUTPUTS.values():
        path = safe_path(root, relative)
        if path.exists() or path.is_symlink():
            raise ValueError(f"Refusing existing fairness output: {relative}")
    protocol = yaml.safe_load((root / PROTOCOL).read_text(encoding="utf-8"))
    validate_protocol(protocol)
    v1 = yaml.safe_load((root / "configs/evaluation/fairness_protocol_v1.yaml").read_text())
    if (protocol["minimum_support"] != v1["minimum_support"]
        or protocol["screening_thresholds"] != v1["screening_thresholds"]):
        raise ValueError("Unexpected V1 support/screening rule drift.")
    trace = provenance(root)
    frames = {key: pd.read_csv(safe_path(root, path)) for key, path in INPUTS.items()}
    raw, policy = validate_inputs(**frames)
    for rel, digest in trace["input_sha256"].items():
        if sha256_file(safe_path(root, rel)) != digest:
            raise ValueError("Input changed during validation.")
    trace.update({
        "started_at_utc": datetime.now(UTC).isoformat(), "protocol": protocol,
        "branch": subprocess.check_output(["git", "branch", "--show-current"], cwd=root,
                                          text=True).strip(),
        "head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "git_status": subprocess.check_output(["git", "status", "--short"], cwd=root,
                                              text=True).splitlines(),
        "versions": {"python": platform.python_version(), **{
            name: version(name) for name in ("numpy", "pandas", "scikit-learn", "scipy", "mypy")
        }},
        "development_rows": 1176, "target_distribution": {"0": 986, "1": 190},
        "raw_candidates": 2, "saved_top_k_candidates": 6, "rows_each": 1176,
        "alignment": "exact IDs/target/fold; probabilities allclose rtol=1e-7 atol=1e-9",
        "top_k_verified": "per existing fold, floor budget, stable source_row tie order",
        "final_test_accessed": False, "model_fitted": False, "policy_selected": False,
        "employee_level_output_written": False, "business_approved": False,
        "production_approved": False, "python_311_runtime_verified": False,
    })
    for relative in OUTPUTS.values():
        (root / relative).parent.mkdir(parents=True, exist_ok=True)
    write_json(root / OUTPUTS["preflight"], trace)
    preflight_hash = sha256_file(root / OUTPUTS["preflight"])
    print("Preflight frozen: 1176 aligned development rows; existing predictions only.", flush=True)
    tables = summarize(raw, policy, protocol)
    for rel, digest in trace["input_sha256"].items():
        if sha256_file(safe_path(root, rel)) != digest:
            raise ValueError("Input changed during audit.")
    if sha256_file(root / OUTPUTS["preflight"]) != preflight_hash:
        raise ValueError("Preflight changed during audit.")
    for name, table in tables.items():
        with (root / OUTPUTS[name]).open("x", encoding="utf-8", newline="") as stream:
            table.to_csv(stream, index=False, na_rep="undefined")
    trace.update({
        "completed_at_utc": datetime.now(UTC).isoformat(), "preflight_sha256": preflight_hash,
        "output_sha256": {key: sha256_file(root / OUTPUTS[key]) for key in tables},
        "output_rows": {key: len(table) for key, table in tables.items()},
    })
    with (root / OUTPUTS["report"]).open("x", encoding="utf-8") as stream:
        stream.write(build_report(tables, trace))
    trace["output_sha256"]["report"] = sha256_file(root / OUTPUTS["report"])
    write_json(root / OUTPUTS["trace"], trace)
    print(tables["policy"].query("fold == 0 and attribute == 'Overall'")[
        ["feature_set", "capacity_percent", "alerts", "tp", "fp", "fn"]
    ].to_string(index=False))
    print("Aggregate-only outputs complete; no fit, new threshold or final-test access.")


if __name__ == "__main__":
    run()
