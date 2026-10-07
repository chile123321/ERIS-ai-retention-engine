"""Final development fit of the fixed raw Logistic research candidate, not serving."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import warnings
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from eris_ml.data.splitting import assert_development_path
from eris_ml.evaluation.feature_ablation import validate_definitions, validate_fold_assignment
from eris_ml.features.definitions import FeatureDefinition, load_feature_definition
from eris_ml.features.preprocessing import build_preprocessor
from eris_ml.models.bundle_v2 import (
    CANDIDATE,
    FEATURE_ORDER,
    FEATURE_SET,
    FORBIDDEN,
    LIMITATIONS,
    LogisticResearchBundle,
    input_schema,
    load_bundle,
    preprocessing_signature,
    runtime_versions,
    save_bundle,
    validate_input,
    validate_parameters,
    write_json,
)
from eris_ml.models.factory import create_estimator
from eris_ml.models.persistence import sidecar_paths
from eris_ml.models.training import build_model_pipeline
from eris_ml.utils.hashing import sha256_file

PREFIX = "artifacts/reports/logistic_general16_v2_r1"
BUNDLE = "artifacts/models/eris_logistic_general16_v2_r1.joblib"
PREFLIGHT = f"{PREFIX}_preflight.json"
TRACE = f"{PREFIX}_trace.json"
INITIAL = f"{PREFIX}_initial_state.json"
DEVELOPMENT = "data/processed/development_raw_v1.csv"
REGISTRY = "configs/features/feature_set_v2_practical.yaml"
CONFIG = "configs/models/logistic_regression.yaml"
BASELINE = "artifacts/reports/baseline_feature_ablation_v2_trace.json"
BRIDGE = "docs/quality/environment_audit_20261001_readonly.json"
DOC_UPDATES = {"README.md", "docs/v2_status.md", "docs/feature_strategy_v2.md",
               "docs/hr_policy_decision_package_v2.md"}
RTOL, ATOL = 1e-12, 1e-12


def safe_path(root: Path, relative: str) -> Path:
    rel = Path(relative)
    assert_development_path(rel)  # Before exists/stat/hash/read.
    if rel.anchor or rel.drive or ".." in rel.parts:
        raise ValueError("Expected a repository-relative path.")
    path = root / rel
    current = path
    while current != root:
        if current.is_symlink():
            raise ValueError("Symlink input/output rejected.")
        current = current.parent
    return path


def read_mapping(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Expected JSON/YAML mapping.")
    return value


def verify_hashes(root: Path, hashes: dict[str, str]) -> None:
    for rel, digest in hashes.items():
        if sha256_file(safe_path(root, rel)) != digest:
            raise ValueError(f"Fingerprint mismatch: {rel}")


def verify_baseline(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Use exact reviewed old/current hashes and historical bytes, not a blanket bypass."""
    trace = read_mapping(safe_path(root, BASELINE))
    bridge = read_mapping(safe_path(root, BRIDGE))["historical_source_drift"]
    hashes, revisions = {}, []
    for rel, historical in trace["input_sha256"].items():
        current = sha256_file(safe_path(root, rel))
        hashes[rel] = current
        if current == historical:
            continue
        entry = bridge.get(rel, {})
        if (entry.get("current") != current or entry.get("historical") != historical
            or "baseline_ablation_v2" not in entry.get("traces", [])):
            raise ValueError(f"Unreviewed historical source drift: {rel}")
        old = subprocess.check_output(["git", "show", f"HEAD:{rel}"], cwd=root)
        if hashlib.sha256(old).hexdigest() != historical:
            raise ValueError(f"HEAD does not prove historical source: {rel}")
        if rel == REGISTRY:
            previous, now = yaml.safe_load(old), read_mapping(safe_path(root, rel))
            for key in ("version", "preferred_feature_set", "comparison_feature_sets",
                        "comparison_protocol"):
                if previous[key] != now[key]:
                    raise ValueError("Registry experiment fields changed.")
        revisions.append({"path": rel, "historical_sha256": historical,
                          "current_sha256": current, "bridge": BRIDGE})
    for key, digest in trace["output_sha256"].items():
        relative = trace["outputs"][key]
        verify_hashes(root, {relative: digest})
        hashes[relative] = digest
    for rel in (BASELINE, BRIDGE):
        hashes[rel] = sha256_file(safe_path(root, rel))
    return trace, {"input_sha256": hashes, "reviewed_source_revisions": revisions,
                   "historical_versions": trace["versions"]}


def validate_alignment(frame: pd.DataFrame, assignment: pd.DataFrame,
                       oof: pd.DataFrame, *, rows: int = 1176,
                       counts: dict[int, int] | None = None) -> None:
    ids = ["source_row", "EmployeeNumber", "Attrition"]
    if (frame.columns.has_duplicates or len(frame) != rows
        or frame.Attrition.value_counts().to_dict() != (counts or {0: 986, 1: 190})
        or frame[ids].isna().any().any()
        or not frame.source_row.is_unique or not frame.EmployeeNumber.is_unique):
        raise ValueError("Invalid development rows/labels/identities.")
    if (not assignment[ids].reset_index(drop=True).equals(frame[ids].reset_index(drop=True))
        or not np.array_equal(assignment.development_position, np.arange(rows))):
        raise ValueError("Saved assignment differs from development.")
    validate_fold_assignment(frame.Attrition.to_numpy(), assignment.fold.to_numpy())
    selected = oof.loc[(oof.feature_set == FEATURE_SET) & (oof.model == "logistic_regression")]
    columns = [*ids, "fold"]
    if (len(selected) != rows or not selected.source_row.is_unique
        or not selected.EmployeeNumber.is_unique
        or not selected[columns].sort_values("source_row").reset_index(drop=True).equals(
            assignment[columns].sort_values("source_row").reset_index(drop=True))):
        raise ValueError("Baseline OOF ID/target/fold alignment mismatch.")
    scores = selected.probability.to_numpy(dtype=float)
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("Invalid stored OOF probability.")


def prepare(root: Path) -> tuple[pd.DataFrame, FeatureDefinition, dict[str, Any], dict[str, Any]]:
    historical, trace = verify_baseline(root)
    initial = read_mapping(safe_path(root, INITIAL))
    verify_hashes(root, initial["protected_sha256"])
    verify_hashes(root, {p: h for p, h in initial["worktree_sha256"].items()
                         if p not in DOC_UPDATES})
    registry = read_mapping(safe_path(root, REGISTRY))
    definitions = {name: load_feature_definition(safe_path(root, rel))
                   for name, rel in registry["comparison_feature_sets"].items()}
    validate_definitions(definitions)
    definition = definitions[FEATURE_SET]
    if list(definition.all_features) != FEATURE_ORDER:
        raise ValueError("Expected the locked 16-feature order.")
    config = read_mapping(safe_path(root, CONFIG))
    parameters = create_estimator(config).get_params(deep=False)
    validate_parameters(parameters)
    if (config != historical["logistic_config"]
        or parameters != historical["logistic_parameters"]
        or historical["feature_sets"][FEATURE_SET] != FEATURE_ORDER):
        raise ValueError("Resolved baseline config/parameters/features differ from historical OOF.")
    pins = registry["comparison_protocol"]["immutable_split_fingerprints"]
    verify_hashes(root, {DEVELOPMENT: pins["development_sha256"],
                         "data/processed/split_manifest_v1.csv": pins["split_manifest_sha256"]})
    frame = pd.read_csv(safe_path(root, DEVELOPMENT))
    assignment = pd.read_csv(safe_path(root, historical["outputs"]["assignments"]))
    oof = pd.read_csv(safe_path(root, historical["outputs"]["oof"]))
    validate_alignment(frame, assignment, oof)
    features = validate_input(frame.loc[:, FEATURE_ORDER])
    if frame.shape != (1176, 28) or frame.isna().any().any() or features.shape != (1176, 16):
        raise ValueError("Development shape/null contract mismatch.")
    metrics = pd.read_csv(safe_path(root, historical["outputs"]["summary"]))
    row = metrics.loc[(metrics.feature_set == FEATURE_SET)
                      & (metrics.model == "logistic_regression")]
    if len(row) != 1:
        raise ValueError("Missing/duplicate saved OOF evaluation summary.")
    scores = {m: float(row.iloc[0][m]) for m in ("average_precision", "roc_auc", "brier_score")}
    if [round(scores[m], 6) for m in scores] != [.477584, .775729, .110733]:
        raise ValueError("Stored OOF evaluation differs from the published reference.")
    extra = [INITIAL, "pyproject.toml", "requirements/quality-py312.txt",
             "src/eris_ml/features/definitions.py", "src/eris_ml/models/persistence.py",
             "src/eris_ml/utils/hashing.py", "src/eris_ml/data/splitting.py",
             "src/eris_ml/models/bundle_v2.py", "src/eris_ml/models/bundle_v2_build.py",
             "scripts/build_logistic_bundle_v2.py"]
    trace["input_sha256"].update({p: sha256_file(safe_path(root, p)) for p in extra})
    trace.update({
        "candidate": CANDIDATE, "status": "research_demo_candidate",
        "authorization": "Current user request authorizes IBM research/demo final fit only; "
                         "not HR/policy/enterprise-data-contract/production approval.",
        "head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "branch": subprocess.check_output(["git", "branch", "--show-current"], cwd=root,
                                          text=True).strip(),
        "git_status": subprocess.check_output(["git", "status", "--short"], cwd=root,
                                              text=True).splitlines(),
        "prepared_at_utc": datetime.now(UTC).isoformat(), "library_versions": runtime_versions(),
        "config": config, "feature_config": registry["comparison_feature_sets"][FEATURE_SET],
        "model_parameters": parameters, "effective_regularization": "L2",
        "preprocessing": preprocessing_signature(build_preprocessor(definition)),
        "feature_order": FEATURE_ORDER, "input_schema": input_schema(),
        "development_rows": 1176, "target_distribution": {"0": 986, "1": 190},
        "saved_oof_evaluation": scores, "evaluation_source": historical["outputs"]["summary"],
        "evaluation_kind": "saved_pooled_development_OOF_not_final_fit_training_scores",
        "final_test_accessed": False, "business_approved": False, "production_approved": False,
        "threshold": None, "policy_selected": False, "limitations": LIMITATIONS,
        "verification_plan": {
            "fresh_process": "isolated Python process (-I); same installed environment",
            "sample": "first 32 rows of unchanged development order; no IDs in payload",
            "repeat_fit": "two identical full-development fits; persist first only",
            "rtol": RTOL, "atol": ATOL, "no_training_performance_metrics": True,
        },
        "protected_sha256": initial["protected_sha256"],
    })
    verify_hashes(root, trace["input_sha256"])
    return frame, definition, config, trace


def metadata_from_preflight(trace: dict[str, Any], preflight_hash: str) -> dict[str, Any]:
    return {
        "candidate_version": CANDIDATE, "bundle_version": "probability-bundle-v2-r1",
        "status": "research_demo_candidate", "feature_set": FEATURE_SET,
        "feature_count": 16, "feature_order": FEATURE_ORDER, "input_schema": input_schema(),
        "class_order": [0, 1], "positive_class": 1, "target": "Attrition",
        "class_mapping": {"0": "No", "1": "Yes"}, "model_family": "logistic_regression",
        "model_config": CONFIG, "model_parameters": trace["model_parameters"],
        "preprocessing": trace["preprocessing"], "effective_regularization": "L2",
        "calibration": "none", "probability_type": "raw", "threshold": None,
        "post_score_policy": "top-k 10% is a separate unapproved simulation requiring a complete "
                             "batch; not a model threshold and not executed by this bundle",
        "training_rows": trace["development_rows"],
        "target_distribution": trace["target_distribution"],
        "training_scope": "development_only", "business_approved": False,
        "production_approved": False, "enterprise_data_contract_approved": False,
        "trusted_artifact_only": True, "limitations": LIMITATIONS,
        "deserialization_warning": "Pickle/joblib executes code: trust origin, not just checksums.",
        "library_versions": trace["library_versions"],
        "saved_oof_evaluation": trace["saved_oof_evaluation"],
        "evaluation_kind": trace["evaluation_kind"],
        "provenance": {"preflight_sha256": preflight_hash, "preflight_path": PREFLIGHT,
                       "input_sha256": trace["input_sha256"], "head": trace["head"],
                       "reviewed_source_revisions": trace["reviewed_source_revisions"]},
    }


def assert_equal_probabilities(expected: np.ndarray, actual: np.ndarray) -> float:
    if (expected.shape != actual.shape or not np.isfinite(actual).all()
        or ((actual < 0) | (actual > 1)).any()
        or not np.allclose(expected, actual, rtol=RTOL, atol=ATOL)):
        raise ValueError("Bundle/repeat-fit probability parity failed.")
    return float(np.max(np.abs(expected - actual)))


def fit_and_check(frame: pd.DataFrame, definition: FeatureDefinition, config: dict[str, Any],
                  metadata: dict[str, Any]) -> tuple[LogisticResearchBundle, dict[str, Any]]:
    features = validate_input(frame.loc[:, FEATURE_ORDER])
    first = build_model_pipeline(definition, create_estimator(config))
    repeat = build_model_pipeline(definition, create_estimator(config))
    first.fit(features, frame.Attrition)
    bundle = LogisticResearchBundle(first, metadata)
    before = bundle.predict_proba(features)
    repeat.fit(features, frame.Attrition)
    error = assert_equal_probabilities(before, np.asarray(repeat.predict_proba(features)[:, 1]))
    return bundle, {"repeat_fit_rows": len(features), "repeat_fit_max_absolute_error": error,
                    "class_order": list(map(int, first.classes_)),
                    "fit_feature_count": int(first.n_features_in_),
                    "n_iter": first.named_steps["model"].n_iter_.tolist()}


def fresh_process_check(path: Path, sample: pd.DataFrame, probabilities: np.ndarray
                        ) -> dict[str, Any]:
    payload = {"records": sample.to_dict(orient="records"), "expected": probabilities.tolist()}
    result = subprocess.run(
        [sys.executable, "-I", "-m", "eris_ml.models.bundle_v2_build", "--verify", str(path)],
        input=json.dumps(payload, allow_nan=False), capture_output=True, text=True, check=True,
        timeout=90,
    )
    value = json.loads(result.stdout)
    if not isinstance(value, dict) or value.get("class_order") != [0, 1]:
        raise ValueError("Invalid clean-process verification result.")
    return value


def check_loaded_schema(bundle: LogisticResearchBundle, sample: pd.DataFrame) -> dict[str, Any]:
    """Exercise the actual reloaded object, not just the standalone validator."""
    invalid = {
        "missing_column": sample.drop(columns="Age"),
        "duplicate_column": pd.concat([sample, sample[["Age"]]], axis=1),
        "numeric_string": sample.assign(Age="30"),
        "boolean": sample.assign(Age=True),
        "nonfinite": sample.assign(Age=np.inf),
        "null": sample.assign(Age=np.nan),
        "unknown_OverTime": sample.assign(OverTime="unknown"),
        "invalid_ordinal": sample.assign(Education=99),
        **{f"excluded_{name}": sample.assign(**{name: 0}) for name in sorted(FORBIDDEN)},
    }
    for name, records in invalid.items():
        try:
            bundle.predict_proba(records)
        except ValueError:
            continue
        raise ValueError(f"Loaded bundle failed to reject {name}.")
    error = assert_equal_probabilities(bundle.predict_proba(sample),
                                       bundle.predict_proba(sample.loc[:, FEATURE_ORDER[::-1]]))
    return {"rejected_cases": list(invalid), "column_reordering_max_absolute_error": error}


def build(root: Path) -> dict[str, Any]:
    path = safe_path(root, BUNDLE)
    outputs = [path, *sidecar_paths(path), safe_path(root, PREFLIGHT), safe_path(root, TRACE)]
    for output in outputs:
        if output.exists() or output.is_symlink():
            raise ValueError(f"Output exists: {output.name}; inspect provenance, never overwrite.")
    frame, definition, config, trace = prepare(root)
    write_json(root / PREFLIGHT, trace)  # Mandatory immutable evidence BEFORE either fit.
    preflight_hash = sha256_file(root / PREFLIGHT)
    metadata = metadata_from_preflight(trace, preflight_hash)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        bundle, checks = fit_and_check(frame, definition, config, metadata)
    warning_messages = [f"{w.category.__name__}: {w.message}" for w in caught]
    # A non-converged final fit is not silently published.
    if any(w.category.__name__ == "ConvergenceWarning" for w in caught):
        raise ValueError("Final fit did not converge; preflight retained, no bundle saved.")
    sample = frame.loc[:, FEATURE_ORDER].iloc[:32].copy()
    before = bundle.predict_proba(sample)
    verify_hashes(root, trace["input_sha256"])
    verify_hashes(root, trace["protected_sha256"])
    save_bundle(bundle, path)
    loaded = load_bundle(path)
    checks["same_process_reload_max_absolute_error"] = assert_equal_probabilities(
        before, loaded.predict_proba(sample))
    checks["fresh_process"] = fresh_process_check(path, sample, before)
    verify_hashes(root, trace["input_sha256"])
    verify_hashes(root, trace["protected_sha256"])
    if sha256_file(root / PREFLIGHT) != preflight_hash:
        raise ValueError("Preflight changed during fit.")
    trace.update({
        "completed_at_utc": datetime.now(UTC).isoformat(), "preflight_sha256": preflight_hash,
        "checks": checks, "warnings": warning_messages,
        "output_sha256": {p.relative_to(root).as_posix(): sha256_file(p)
                          for p in (path, *sidecar_paths(path))},
        "protected_hashes_unchanged": len(trace["protected_sha256"]),
        "training_performance_metrics_computed": False,
    })
    write_json(root / TRACE, trace)
    return trace


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] != "--verify":
        raise SystemExit("Use scripts/build_logistic_bundle_v2.py to build; --verify is internal.")
    trusted_bundle = load_bundle(Path(sys.argv[2]))
    request = json.loads(sys.stdin.read())
    probabilities = trusted_bundle.predict_proba(pd.DataFrame(request["records"]))
    deviation = assert_equal_probabilities(
        np.asarray(request["expected"], dtype=float), probabilities)
    print(json.dumps({"rows": len(probabilities), "class_order": [0, 1],
                      "max_absolute_error": deviation, "finite_in_range": True,
                      "schema_checks": check_loaded_schema(
                          trusted_bundle, pd.DataFrame(request["records"])),
                      "interpreter": sys.executable, "library_versions": runtime_versions()}))
