"""One-time, pre-authorized evaluation of the frozen capstone candidate."""

from __future__ import annotations

import hashlib
import io
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    fbeta_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from eris_ml.features.definitions import FeatureDefinition, load_feature_definition
from eris_ml.models.training import build_model_pipeline, normalize_binary_target
from eris_ml.utils.freeze import (
    CANDIDATE_NAME,
    DEVELOPMENT_CSV,
    FEATURE_CONFIG,
    HASH_PATHS,
    MODEL_CONFIG,
    MODEL_PARAMETERS,
    validate_freeze_manifest,
    verify_manifest_hashes,
)

MANIFEST_PATH = "configs/release/candidate_v1.yaml"
AUTHORIZATION_PATH = "configs/release/final_evaluation_authorization_v1.yaml"
LEDGER_PATH = "configs/release/final_evaluation_record_v1.yaml"
FINAL_TEST_PATH = "data/processed/final_test_raw_v1.csv"
REPORT_PATH = "artifacts/reports/final_test_evaluation_v1.md"
METRICS_PATH = "artifacts/reports/final_test_metrics_v1.json"
PREDICTIONS_PATH = "artifacts/predictions/final_test_predictions_v1.csv"
THRESHOLD = 0.345651
GUARDRAILS = {
    "pr_auc_minimum": 0.45,
    "roc_auc_minimum": 0.75,
    "brier_maximum": 0.135462,
    "precision_minimum": 0.45,
    "recall_minimum": 0.45,
    "alert_rate_maximum": 0.20,
}
CI_METRICS = ("pr_auc", "roc_auc", "brier", "precision", "recall", "f1", "f2", "alert_rate")
LEDGER_STATES = {
    "authorized_not_started", "started", "completed", "failed_before_access",
    "failed_after_access",
}


def utc_now() -> str:
    """Return a timezone-aware, unambiguous audit timestamp."""
    return datetime.now(UTC).isoformat(timespec="seconds")


def load_yaml(path: Path) -> dict[str, Any]:
    """Read a YAML mapping, rejecting malformed audit records."""
    with path.open(encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected YAML mapping: {path}")
    return value


def atomic_yaml(path: Path, value: dict[str, Any]) -> None:
    """Replace an audit record atomically within its own directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.",
            suffix=".tmp", delete=False,
        ) as stream:
            temporary = stream.name
            yaml.safe_dump(value, stream, sort_keys=False, allow_unicode=True)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and os.path.exists(temporary):
            os.unlink(temporary)


def sha256_bytes(contents: bytes) -> str:
    """Hash bytes already obtained in a single authorized file read."""
    return hashlib.sha256(contents).hexdigest()


def manifest_sha256(path: Path) -> str:
    """Hash only the frozen manifest, never the locked final test."""
    return sha256_bytes(path.read_bytes())


def authorization_record(manifest_hash: str, timestamp: str) -> dict[str, Any]:
    """Create the exact project-owner authorization without widening the scope."""
    return {
        "authorization_version": "final-evaluation-authorization-v1",
        "authorized": True,
        "authorized_by": "project_owner",
        "authorized_scope": "capstone_final_evaluation",
        "authorized_candidate": CANDIDATE_NAME,
        "maximum_runs": 1,
        "candidate_manifest": {"path": MANIFEST_PATH, "sha256": manifest_hash},
        "final_test": {"path": FINAL_TEST_PATH, "access_allowed_after_preflight": True},
        "pre_registered_guardrails": GUARDRAILS.copy(),
        "restrictions": {
            "tuning_allowed": False, "calibration_allowed": False,
            "threshold_change_allowed": False, "feature_change_allowed": False,
            "candidate_comparison_allowed": False, "production_approval": False,
        },
        "created_at_utc": timestamp,
    }


def validate_authorization(record: dict[str, Any], manifest_hash: str) -> None:
    """Fail closed on any changed authorization, guardrail, or candidate hash."""
    timestamp = record.get("created_at_utc")
    if not isinstance(timestamp, str):
        raise ValueError("Authorization timestamp is missing.")
    try:
        parsed = datetime.fromisoformat(timestamp)
    except ValueError as exc:
        raise ValueError("Authorization timestamp is malformed.") from exc
    if parsed.tzinfo is None or record != authorization_record(manifest_hash, timestamp):
        raise ValueError("Authorization differs from the pre-registered scope or manifest.")


def initial_ledger() -> dict[str, Any]:
    """Return the only permissible initial one-time ledger state."""
    return {
        "evaluation_version": "final-evaluation-v1", "candidate": CANDIDATE_NAME,
        "maximum_runs": 1, "run_count": 0, "status": "authorized_not_started",
    }


def assert_unused_ledger(record: dict[str, Any]) -> None:
    """Reject a consumed, malformed, or non-initial ledger before any final access."""
    if record.get("status") not in LEDGER_STATES:
        raise ValueError("Unknown final-evaluation ledger state.")
    if record.get("run_count") != 0 or record.get("status") != "authorized_not_started":
        raise ValueError("One-time final evaluation is already used or unavailable.")
    if record != initial_ledger():
        raise ValueError("Initial ledger has unexpected fields or candidate identity.")


def mark_started(path: Path, record: dict[str, Any]) -> dict[str, Any]:
    """Consume the single run atomically immediately before accessing final data."""
    assert_unused_ledger(record)
    if load_yaml(path) != record:
        raise ValueError("Ledger changed before the one-time transition.")
    started = {**record, "run_count": 1, "status": "started", "started_at_utc": utc_now()}
    atomic_yaml(path, started)
    return started


def mark_failure(path: Path, record: dict[str, Any], *, after_access: bool) -> None:
    """Record a terminal post-access failure or recoverable pre-access failure."""
    status = record.get("status")
    if after_access:
        if status != "started" or record.get("run_count") != 1:
            raise ValueError("Post-access failure requires a started ledger.")
        new_status = "failed_after_access"
    else:
        assert_unused_ledger(record)
        new_status = "failed_before_access"
    atomic_yaml(path, {**record, "status": new_status, "failed_at_utc": utc_now()})


def prepare(root: Path) -> dict[str, Any]:
    """Write authorization and initial ledger; never inspect the locked file."""
    manifest_path = root / MANIFEST_PATH
    authorization_path = root / AUTHORIZATION_PATH
    ledger_path = root / LEDGER_PATH
    if authorization_path.exists() or ledger_path.exists():
        raise ValueError("Authorization or ledger already exists; refusing to overwrite.")
    manifest = load_yaml(manifest_path)
    validate_freeze_manifest(manifest)
    verify_manifest_hashes(manifest, root)
    timestamp = utc_now()
    if datetime.fromisoformat(timestamp) <= datetime.fromisoformat(manifest["created_at_utc"]):
        raise ValueError("Authorization must occur after the candidate freeze.")
    authorization = authorization_record(manifest_sha256(manifest_path), timestamp)
    validate_authorization(authorization, manifest_sha256(manifest_path))
    atomic_yaml(authorization_path, authorization)
    atomic_yaml(ledger_path, initial_ledger())
    return authorization


def validate_data(
    frame: pd.DataFrame, definition: FeatureDefinition, *, final: bool,
    development: pd.DataFrame | None = None,
) -> pd.Series:
    """Validate approved row counts, IDs, targets and the fixed 25-feature schema."""
    required = set(definition.all_features) | {"source_row", "EmployeeNumber", "Attrition"}
    if frame.columns.has_duplicates or not required.issubset(frame.columns):
        raise ValueError("Missing or duplicate required data columns.")
    expected_rows = 294 if final else 1176
    if len(frame) != expected_rows or frame["source_row"].isna().any() or (
        frame["EmployeeNumber"].isna().any()
    ):
        raise ValueError("Unexpected row count or missing identifiers.")
    if frame["source_row"].duplicated().any() or frame["EmployeeNumber"].duplicated().any():
        raise ValueError("Duplicate employee identifiers or source rows.")
    if frame.duplicated().any():
        raise ValueError("Unexpected duplicate data rows.")
    target = normalize_binary_target(frame["Attrition"])
    counts = target.value_counts().to_dict()
    if counts != ({0: 247, 1: 47} if final else {0: 986, 1: 190}):
        raise ValueError("Target distribution differs from the approved split.")
    if final:
        if development is None:
            raise ValueError("Development identifiers are required for overlap checks.")
        for key in ("source_row", "EmployeeNumber"):
            if not set(frame[key]).isdisjoint(set(development[key])):
                raise ValueError(f"Final/development {key} overlap detected.")
    return target


def preflight(root: Path) -> tuple[dict[str, Any], dict[str, Any], pd.DataFrame,
                                    FeatureDefinition, pd.Series]:
    """Verify frozen evidence, authorization, ledger and development only."""
    manifest_path = root / MANIFEST_PATH
    manifest = load_yaml(manifest_path)
    validate_freeze_manifest(manifest)
    verify_manifest_hashes(manifest, root)
    if len(manifest["artifact_hashes"]) != 11 or (
        any(manifest["artifact_hashes"][name]["status"] != "present" for name in HASH_PATHS)
    ):
        raise ValueError("All 11 frozen artifacts must be present and hash-matched.")
    manifest_hash = manifest_sha256(manifest_path)
    authorization = load_yaml(root / AUTHORIZATION_PATH)
    validate_authorization(authorization, manifest_hash)
    if datetime.fromisoformat(authorization["created_at_utc"]) <= datetime.fromisoformat(
        manifest["created_at_utc"]
    ):
        raise ValueError("Authorization must postdate candidate freeze.")
    ledger = load_yaml(root / LEDGER_PATH)
    assert_unused_ledger(ledger)
    definition = load_feature_definition(root / FEATURE_CONFIG)
    if definition.version != "v1-full" or len(definition.all_features) != 25:
        raise ValueError("Frozen feature set changed.")
    model_config = load_yaml(root / MODEL_CONFIG)
    if (model_config.get("model") != "xgboost"
        or model_config.get("random_seed") != 42
        or model_config.get("parameters") != MODEL_PARAMETERS):
        raise ValueError("Frozen model configuration changed.")
    development = pd.read_csv(root / DEVELOPMENT_CSV)
    target = validate_data(development, definition, final=False)
    return manifest, authorization, development, definition, target


def fit_frozen(
    manifest: dict[str, Any], development: pd.DataFrame,
    definition: FeatureDefinition, target: pd.Series,
) -> Pipeline:
    """Fit the exact raw XGBoost preprocessing pipeline on development only."""
    from xgboost import XGBClassifier

    estimator = XGBClassifier(random_state=42, **manifest["model"]["parameters"])
    pipeline = build_model_pipeline(definition, estimator)
    pipeline.fit(development.loc[:, list(definition.all_features)], target)
    return pipeline


def assert_runtime_candidate(manifest: dict[str, Any]) -> None:
    """Recheck the operational identity immediately before fitting."""
    if (manifest.get("candidate_name") != CANDIDATE_NAME
        or manifest.get("feature_schema", {}).get("feature_count") != 25
        or manifest.get("model", {}).get("family") != "xgboost"
        or manifest.get("model", {}).get("parameters") != MODEL_PARAMETERS
        or manifest.get("calibration", {}).get("method") != "none"
        or manifest.get("calibration", {}).get("probability_type") != "raw"
        or manifest.get("threshold", {}).get("value") != THRESHOLD
        or manifest.get("threshold", {}).get("policy") != "capacity_15_percent"
        or manifest.get("fairness", {}).get("status") != "review_known_limitation"
        or manifest.get("holdout", {}).get("status") != "retained_with_protocol_deviation"
        or manifest.get("approvals", {}).get("business_approved") is not False
        or manifest.get("approvals", {}).get("production_approved") is not False):
        raise ValueError("Runtime candidate differs from the frozen authorization.")


def read_final_once(path: Path) -> tuple[pd.DataFrame, str]:
    """Open the final file once; hash and parse the same bytes in memory."""
    with path.open("rb") as stream:
        contents = stream.read()
    return pd.read_csv(io.BytesIO(contents)), sha256_bytes(contents)


def final_metrics(target: Any, probability: Any, threshold: float = THRESHOLD) -> dict[str, Any]:
    """Compute pre-registered probability and operational metrics safely."""
    y = np.asarray(target)
    p = np.asarray(probability, dtype=float)
    if y.ndim != 1 or p.ndim != 1 or len(y) == 0 or len(y) != len(p):
        raise ValueError("Paired target and probability vectors are required.")
    if set(np.unique(y).tolist()) != {0, 1}:
        raise ValueError("Both binary target classes are required.")
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Probabilities must be finite and in [0,1].")
    if threshold != THRESHOLD:
        raise ValueError("Only the frozen threshold is permitted.")
    predicted = (p >= threshold).astype(int)
    tn, fp, fn, tp = (int(value) for value in
                      confusion_matrix(y, predicted, labels=[0, 1]).ravel())
    alerts = tp + fp
    return {
        "pr_auc": float(average_precision_score(y, p)),
        "roc_auc": float(roc_auc_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, np.clip(p, np.finfo(float).eps,
                                              1 - np.finfo(float).eps))),
        "accuracy": float(accuracy_score(y, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(y, predicted)),
        "precision": float(precision_score(y, predicted, zero_division=0)),
        "recall": float(recall_score(y, predicted, zero_division=0)),
        "f1": float(f1_score(y, predicted, zero_division=0)),
        "f2": float(fbeta_score(y, predicted, beta=2, zero_division=0)),
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "alerts": alerts, "alert_rate": float(alerts / len(y)),
        "fp_per_tp": None if tp == 0 else float(fp / tp),
        "fn_per_tp": None if tp == 0 else float(fn / tp),
    }


def stratified_bootstrap(
    target: Any, probability: Any, *, iterations: int = 2000, seed: int = 42,
) -> dict[str, Any]:
    """Resample paired target/probability rows within both target classes."""
    if iterations < 1:
        raise ValueError("Bootstrap iterations must be positive.")
    y = np.asarray(target)
    p = np.asarray(probability, dtype=float)
    final_metrics(y, p)
    rng = np.random.default_rng(seed)
    negatives = np.flatnonzero(y == 0)
    positives = np.flatnonzero(y == 1)
    samples: dict[str, list[float]] = {metric: [] for metric in CI_METRICS}
    for _ in range(iterations):
        index = np.concatenate((rng.choice(negatives, len(negatives), replace=True),
                                rng.choice(positives, len(positives), replace=True)))
        scores = final_metrics(y[index], p[index])
        for metric in CI_METRICS:
            samples[metric].append(float(scores[metric]))
    return {
        "method": "target_stratified_paired_percentile_bootstrap",
        "confidence_level": 0.95, "iterations_requested": iterations,
        "valid_iterations": len(samples["pr_auc"]), "random_state": seed,
        "intervals": {
            metric: [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]
            for metric, values in samples.items()
        },
    }


def guardrail_decision(metrics: dict[str, Any]) -> tuple[str, dict[str, bool]]:
    """Apply only pre-registered point-estimate guardrails, without adjustment."""
    checks = {
        "pr_auc_minimum": metrics["pr_auc"] >= GUARDRAILS["pr_auc_minimum"],
        "roc_auc_minimum": metrics["roc_auc"] >= GUARDRAILS["roc_auc_minimum"],
        "brier_maximum": metrics["brier"] <= GUARDRAILS["brier_maximum"],
        "precision_minimum": metrics["precision"] >= GUARDRAILS["precision_minimum"],
        "recall_minimum": metrics["recall"] >= GUARDRAILS["recall_minimum"],
        "alert_rate_maximum": metrics["alert_rate"] <= GUARDRAILS["alert_rate_maximum"],
    }
    result = ("PASS_WITH_KNOWN_LIMITATIONS" if all(checks.values())
              else "DOES_NOT_MEET_PREDECLARED_GUARDRAILS")
    return result, checks


def execute_once(root: Path) -> dict[str, Any]:
    """Fit before access, consume the ledger, read once, then seal results."""
    from eris_ml.evaluation.final_reporting import write_outputs

    ledger_path = root / LEDGER_PATH
    record = load_yaml(ledger_path)
    assert_unused_ledger(record)
    try:
        manifest, authorization, development, definition, dev_target = preflight(root)
        assert_runtime_candidate(manifest)
        if any((root / relative).exists() for relative in
               (REPORT_PATH, METRICS_PATH, PREDICTIONS_PATH)):
            raise ValueError("Final-evaluation output already exists; refusing overwrite.")
        pipeline = fit_frozen(manifest, development, definition, dev_target)
    except Exception:
        mark_failure(ledger_path, record, after_access=False)
        raise

    # There must be no final-file metadata, header, hash or content access above this line.
    started = mark_started(ledger_path, record)
    try:
        final, final_hash = read_final_once(root / FINAL_TEST_PATH)
        target = validate_data(final, definition, final=True, development=development)
        probabilities = np.asarray(
            pipeline.predict_proba(final.loc[:, list(definition.all_features)])[:, 1],
            dtype=float,
        )
        metrics = final_metrics(target.to_numpy(), probabilities)
        bootstrap = stratified_bootstrap(target.to_numpy(), probabilities)
        result, checks = guardrail_decision(metrics)
        predictions = final.loc[:, ["source_row", "EmployeeNumber", "Attrition"]].copy()
        predictions["Attrition"] = target.to_numpy()
        predictions["probability"] = probabilities
        predictions["prediction"] = (probabilities >= THRESHOLD).astype(int)
        predictions["threshold"] = THRESHOLD
        predictions["candidate_version"] = CANDIDATE_NAME
        manifest_hash = manifest_sha256(root / MANIFEST_PATH)
        payload = {
            "evaluation_version": "final-evaluation-v1",
            "candidate": CANDIDATE_NAME,
            "authorization": authorization,
            "candidate_manifest_sha256": manifest_hash,
            "final_test_sha256": final_hash,
            "row_count": len(final),
            "target_distribution": {"0": 247, "1": 47},
            "threshold": THRESHOLD,
            "metrics": metrics,
            "bootstrap": bootstrap,
            "guardrails_passed": checks,
            "result": result,
            "business_approved": False,
            "production_approved": False,
            "holdout_status": "retained_with_protocol_deviation",
        }
        write_outputs(root, payload, predictions)
        completed = {
            **started, "status": "completed", "completed_at_utc": utc_now(),
            "final_test_sha256": final_hash, "row_count": len(final),
            "target_distribution": {0: 247, 1: 47},
            "candidate_manifest_sha256": manifest_hash, "threshold": THRESHOLD,
            "result": result, "report_path": REPORT_PATH, "metrics_path": METRICS_PATH,
            "predictions_path": PREDICTIONS_PATH,
        }
        atomic_yaml(ledger_path, completed)
        return payload
    except Exception:
        mark_failure(ledger_path, started, after_access=True)
        raise
