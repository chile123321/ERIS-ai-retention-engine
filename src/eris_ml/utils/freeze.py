"""Validate and freeze the research-only Step 10 XGBoost candidate without model fitting."""

import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from eris_ml.features.definitions import load_feature_definition
from eris_ml.utils.hashing import sha256_file

CANDIDATE_NAME = "eris-xgboost-v1"
FEATURE_CONFIG = "configs/features/feature_set_v1_full.yaml"
MODEL_CONFIG = "configs/models/xgboost_tuned_v1.yaml"
CALIBRATION_CONFIG = "configs/models/xgboost_tuned_calibrated_v1.yaml"
THRESHOLD_CONFIG = "configs/threshold/threshold_candidates_v1.yaml"
FAIRNESS_PROTOCOL = "configs/evaluation/fairness_protocol_v1.yaml"
DEVELOPMENT_CSV = "data/processed/development_raw_v1.csv"
FAIRNESS_CSV = "data/processed/fairness_audit_development_v1.csv"
KNOWN_DEVELOPMENT_SHA256 = "5cb4c6a3226ab5e16b355af64163e5857aac0e1637eb2fb88f91fcc9f1989754"
HASH_PATHS = (
    FEATURE_CONFIG,
    MODEL_CONFIG,
    CALIBRATION_CONFIG,
    THRESHOLD_CONFIG,
    FAIRNESS_PROTOCOL,
    DEVELOPMENT_CSV,
    FAIRNESS_CSV,
    "artifacts/reports/hyperparameter_tuning_v1.md",
    "artifacts/reports/calibration_v1.md",
    "artifacts/reports/threshold_selection_v1.md",
    "artifacts/reports/fairness_audit_v1.md",
)
ALLOWED_USES = (
    "academic_capstone_evaluation", "prototype_demonstration",
    "decision_support_research", "technical_integration_testing",
)
PROHIBITED_USES = (
    "production_hr_deployment", "automated_employment_decisions",
    "employee_termination_decisions", "promotion_salary_or_disciplinary_decisions",
    "claims_of_fairness_for_vietnamese_enterprises",
)
MODEL_PARAMETERS = {
    "objective": "binary:logistic", "eval_metric": "logloss", "tree_method": "hist",
    "scale_pos_weight": 1.0, "n_jobs": 1, "subsample": 0.6,
    "reg_lambda": 10.0, "reg_alpha": 0.5, "n_estimators": 500,
    "min_child_weight": 3, "max_depth": 1, "learning_rate": 0.1,
    "gamma": 0.5, "colsample_bytree": 0.6,
}
INCIDENT_STATEMENT = (
    "The official locked final-test file was not opened and no final-test metric was computed. "
    "A mixed fairness file containing final-test rows was inadvertently loaded during schema "
    "inspection before the fairness audit. The workflow stopped before analysis, the incident "
    "was documented, and no final-test information was used to change the model, features, "
    "calibration or threshold."
)


def assert_freeze_path(path: Path, root: Path, *, allowed: tuple[str, ...]) -> str:
    """Reject forbidden, unapproved or symlinked paths before reading file contents."""
    normalized = path.as_posix().replace("\\", "/").lower()
    if "final_test" in normalized or "/data/quarantine/" in f"/{normalized.strip('/')}/":
        raise ValueError("Final-test or quarantine paths are prohibited in Step 10E.")
    if path.is_absolute():
        try:
            relative = path.relative_to(root).as_posix()
        except ValueError as exc:
            raise ValueError("Freeze input must be inside the repository.") from exc
    else:
        relative = path.as_posix()
    if relative not in allowed or (root / relative).is_symlink():
        raise ValueError("Freeze input path is not on the approved allowlist.")
    return relative


def hash_approved_artifacts(
    root: Path, paths: tuple[str, ...] = HASH_PATHS,
) -> dict[str, dict[str, str | None]]:
    """Hash only allowlisted existing artifacts; mark missing files without regeneration."""
    results: dict[str, dict[str, str | None]] = {}
    for name in paths:
        relative = assert_freeze_path(Path(name), root, allowed=HASH_PATHS)
        path = root / relative
        results[relative] = {"status": "present", "sha256": sha256_file(path)} if (
            path.is_file()
        ) else {"status": "missing", "sha256": None}
    return results


def _candidate_row(threshold_config: dict[str, Any]) -> dict[str, Any]:
    rows = threshold_config.get("candidates")
    if not isinstance(rows, list):
        raise ValueError("Threshold candidate records are missing.")
    matches = [row for row in rows if isinstance(row, dict)
               and row.get("scenario") == "capacity_15_percent"]
    if len(matches) != 1:
        raise ValueError("Exactly one capacity-15 threshold candidate is required.")
    return matches[0]


def validate_sources(
    root: Path, model: dict[str, Any], calibration: dict[str, Any],
    thresholds: dict[str, Any], fairness: dict[str, Any],
) -> dict[str, Any]:
    """Check all frozen values against existing development-only configuration."""
    definition = load_feature_definition(root / FEATURE_CONFIG)
    if definition.version != "v1-full" or len(definition.all_features) != 25:
        raise ValueError("Frozen feature schema must be v1-full with 25 features.")
    if (model.get("model") != "xgboost" or model.get("role") != "tuned_candidate"
        or model.get("candidate_name") != "xgboost_tuned_v1"
        or model.get("random_seed") != 42
        or model.get("parameters") != MODEL_PARAMETERS):
        raise ValueError("Model configuration differs from the tuned XGBoost candidate.")
    if (calibration.get("base_model_config") != MODEL_CONFIG
        or calibration.get("calibration", {}).get("method") != "none"
        or calibration.get("selection", {}).get("production_approved") is not False
        or calibration.get("selection", {}).get("final_test_evaluated") is not False):
        raise ValueError("Calibration must remain raw/none and unapproved.")
    if (thresholds.get("model") != "xgboost_tuned_raw"
        or thresholds.get("business_approved") is not False
        or thresholds.get("production_approved") is not False
        or thresholds.get("final_test_evaluated") is not False
        or thresholds.get("recommended_provisional_candidate") != "capacity_15_percent"):
        raise ValueError("Threshold policy differs from the unapproved capacity-15 candidate.")
    candidate = _candidate_row(thresholds)
    if abs(float(candidate.get("threshold", -1)) - 0.345651) > 5e-7:
        raise ValueError("Threshold differs from frozen 0.345651 working value.")
    if candidate.get("cross_fitted_capacity_met") is not True or (
        candidate.get("feasible_folds") != 5
    ):
        raise ValueError("Capacity-15 cross-fitted policy evidence is incomplete.")
    expected_metrics = {
        "expected_alert_rate": 0.14540816326530612,
        "expected_precision": 0.6549707602339181,
        "expected_recall": 0.5894736842105263,
        "cross_fitted_alert_rate": 0.14625850340136054,
        "cross_fitted_precision": 0.6453488372093024,
        "cross_fitted_recall": 0.5842105263157895,
    }
    if any(abs(float(candidate.get(key, -1)) - value) > 1e-10
           for key, value in expected_metrics.items()):
        raise ValueError("Capacity-15 development evidence differs from Step 10C.")
    from eris_ml.evaluation.fairness import validate_protocol

    validate_protocol(fairness)
    return candidate


def build_freeze_manifest(
    root: Path, model: dict[str, Any], calibration: dict[str, Any],
    thresholds: dict[str, Any], fairness: dict[str, Any],
    library_versions: dict[str, str], *, created_at_utc: str | None = None,
) -> dict[str, Any]:
    """Create a candidate specification from fixed sources; no training or test data access."""
    candidate = validate_sources(root, model, calibration, thresholds, fairness)
    hashes = hash_approved_artifacts(root)
    development_hash = hashes[DEVELOPMENT_CSV]
    if development_hash["status"] != "present" or (
        development_hash["sha256"] != KNOWN_DEVELOPMENT_SHA256
    ):
        raise ValueError("Development CSV hash differs from earlier Step 10 evidence.")
    timestamp = created_at_utc or datetime.now(UTC).isoformat(timespec="seconds")
    manifest: dict[str, Any] = {
        "manifest_version": "candidate-freeze-v1",
        "candidate_name": CANDIDATE_NAME,
        "candidate_version": "v1",
        "created_at_utc": timestamp,
        "step_10_status": "COMPLETE",
        "governance_decision": "accepted_with_known_limitations",
        "scope": {"capstone_research_prototype": True, "capstone_evaluation": True,
                  "production": False, "allowed_uses": list(ALLOWED_USES),
                  "prohibited_uses": list(PROHIBITED_USES)},
        "dataset": {"name": "IBM HR Attrition public/synthetic benchmark",
                    "version": "ibm-hr-attrition-v1",
                    "raw_data_sha256": None,
                    "raw_data_hash_status": "missing_from_approved_evidence",
                    "development_split_version": "existing-v1",
                    "development_rows": 1176,
                    "development_target_distribution": {0: 986, 1: 190}},
        "feature_schema": {"config": FEATURE_CONFIG, "version": "v1-full",
                           "feature_count": 25},
        "preprocessing": {
            "version": "preprocessing-v1",
            "implementation": "src/eris_ml/features/preprocessing.py",
            "nominal": {"imputer": "most_frequent",
                        "encoder": "one_hot_handle_unknown_ignore"},
            "ordinal": {"imputer": "median", "scaler": "standard_scaler"},
            "numeric": {"imputer": "median", "scaler": "standard_scaler"},
        },
        "model": {"family": "xgboost", "config": MODEL_CONFIG,
                  "parameters": MODEL_PARAMETERS.copy()},
        "calibration": {"config": CALIBRATION_CONFIG, "method": "none",
                        "probability_type": "raw"},
        "threshold": {
            "value": 0.345651, "policy": "capacity_15_percent",
            "config": THRESHOLD_CONFIG,
            "full_development_oof_candidate": {
                "exact_threshold": candidate["threshold"],
                "expected_alert_rate": candidate["expected_alert_rate"],
                "expected_precision": candidate["expected_precision"],
                "expected_recall": candidate["expected_recall"],
            },
            "cross_fitted_policy_evaluation": {
                "alert_rate": candidate["cross_fitted_alert_rate"],
                "precision": candidate["cross_fitted_precision"],
                "recall": candidate["cross_fitted_recall"],
                "feasible_folds": candidate["feasible_folds"],
            },
        },
        "random_seeds": {"outer_cv": 42, "inner_cv": 43},
        "library_versions": library_versions,
        "development_evidence": {"pr_auc": 0.666912, "roc_auc": 0.848004,
                                 "brier": 0.088031},
        "fairness": {
            "protocol": FAIRNESS_PROTOCOL, "status": "review_known_limitation",
            "production_fairness_approved": False,
            "gender": {"status": "INSUFFICIENT_EVIDENCE", "tpr_gap": 0.079,
                       "fpr_gap": 0.003, "selection_rate_ratio": 0.999,
                       "tpr_gap_ci_95": [0.005, 0.220]},
            "age_group": {"status": "REVIEW", "tpr_gap": 0.517,
                          "fpr_gap": 0.112, "equalized_odds_gap": 0.517,
                          "tpr_gap_ci_95": [0.330, 0.758],
                          "tpr_18_29": 0.784, "tpr_50_plus": 0.267},
            "marital_status": {"status": "REVIEW_EXPLORATORY", "tpr_gap": 0.328},
            "intersectional": {"status": "INSUFFICIENT_EVIDENCE",
                               "female_50_plus_positive_count": 2},
        },
        "holdout": {"status": "retained_with_protocol_deviation",
                    "final_test_file_opened": False,
                    "final_test_metrics_computed": False,
                    "final_test_used_for_model_selection": False,
                    "mixed_fairness_file_loaded_during_schema_inspection": True,
                    "incident_documented": True,
                    "incident_summary": INCIDENT_STATEMENT},
        "approvals": {"capstone_evaluation_approved": True,
                      "business_approved": False, "production_approved": False,
                      "final_test_evaluation_authorized": False},
        "artifact_hashes": hashes,
        "missing_artifacts": [name for name, item in hashes.items()
                              if item["status"] == "missing"],
    }
    validate_freeze_manifest(manifest)
    return manifest


def validate_freeze_manifest(manifest: dict[str, Any]) -> None:
    """Fail closed if frozen scope, evidence, approvals, or incident flags are changed."""
    if manifest.get("manifest_version") != "candidate-freeze-v1" or (
        manifest.get("candidate_name") != CANDIDATE_NAME
        or manifest.get("candidate_version") != "v1"
        or manifest.get("step_10_status") != "COMPLETE"
        or manifest.get("governance_decision") != "accepted_with_known_limitations"
    ):
        raise ValueError("Candidate identity or governance decision changed.")
    timestamp = manifest.get("created_at_utc")
    if not isinstance(timestamp, str) or datetime.fromisoformat(timestamp).tzinfo is None:
        raise ValueError("Manifest timestamp must include a timezone.")
    scope = manifest.get("scope", {})
    if scope != {"capstone_research_prototype": True, "capstone_evaluation": True,
                 "production": False, "allowed_uses": list(ALLOWED_USES),
                 "prohibited_uses": list(PROHIBITED_USES)}:
        raise ValueError("Candidate scope cannot be broadened.")
    dataset = manifest.get("dataset", {})
    if (dataset.get("development_split_version") != "existing-v1"
        or dataset.get("development_rows") != 1176
        or dataset.get("development_target_distribution") != {0: 986, 1: 190}
        or dataset.get("raw_data_hash_status") != "missing_from_approved_evidence"
        or dataset.get("raw_data_sha256") is not None):
        raise ValueError("Dataset provenance or explicit raw-hash gap changed.")
    if manifest.get("feature_schema") != {"config": FEATURE_CONFIG, "version": "v1-full",
                                           "feature_count": 25}:
        raise ValueError("Feature schema differs from frozen 25-feature v1.")
    preprocessing = manifest.get("preprocessing", {})
    if preprocessing != {
        "version": "preprocessing-v1", "implementation": "src/eris_ml/features/preprocessing.py",
        "nominal": {"imputer": "most_frequent", "encoder": "one_hot_handle_unknown_ignore"},
        "ordinal": {"imputer": "median", "scaler": "standard_scaler"},
        "numeric": {"imputer": "median", "scaler": "standard_scaler"},
    }:
        raise ValueError("Preprocessing specification differs from frozen pipeline.")
    if manifest.get("model") != {"family": "xgboost", "config": MODEL_CONFIG,
                                 "parameters": MODEL_PARAMETERS}:
        raise ValueError("Model configuration differs from frozen tuned XGBoost.")
    if manifest.get("calibration") != {"config": CALIBRATION_CONFIG, "method": "none",
                                           "probability_type": "raw"}:
        raise ValueError("Candidate must use uncalibrated raw probabilities.")
    threshold = manifest.get("threshold", {})
    if (threshold.get("value") != 0.345651
        or threshold.get("policy") != "capacity_15_percent"
        or threshold.get("config") != THRESHOLD_CONFIG
        or threshold.get("full_development_oof_candidate") != {
            "exact_threshold": 0.3456506431102752,
            "expected_alert_rate": 0.14540816326530612,
            "expected_precision": 0.6549707602339181,
            "expected_recall": 0.5894736842105263,
        }
        or threshold.get("cross_fitted_policy_evaluation") != {
            "alert_rate": 0.14625850340136054,
            "precision": 0.6453488372093024,
            "recall": 0.5842105263157895,
            "feasible_folds": 5,
        }):
        raise ValueError("Threshold differs from frozen capacity-15 working policy.")
    if manifest.get("random_seeds") != {"outer_cv": 42, "inner_cv": 43}:
        raise ValueError("Cross-validation seeds differ from development evidence.")
    if manifest.get("development_evidence") != {
        "pr_auc": 0.666912, "roc_auc": 0.848004, "brier": 0.088031,
    }:
        raise ValueError("Development discrimination/calibration evidence changed.")
    versions = manifest.get("library_versions")
    if not isinstance(versions, dict) or not {
        "python", "pandas", "scikit-learn", "xgboost", "numpy", "matplotlib"
    }.issubset(versions) or not all(isinstance(value, str) and value for value in
                                    versions.values()):
        raise ValueError("Library version record is incomplete.")
    fairness = manifest.get("fairness", {})
    if (fairness.get("protocol") != FAIRNESS_PROTOCOL
        or fairness.get("status") != "review_known_limitation"
        or fairness.get("production_fairness_approved") is not False
        or fairness.get("gender", {}).get("status") != "INSUFFICIENT_EVIDENCE"
        or fairness.get("age_group", {}).get("status") != "REVIEW"
        or fairness.get("age_group", {}).get("tpr_gap") != 0.517
        or fairness.get("marital_status", {}).get("status") != "REVIEW_EXPLORATORY"
        or fairness.get("intersectional", {}).get("status") != "INSUFFICIENT_EVIDENCE"):
        raise ValueError("Known fairness limitations cannot be removed or marked PASS.")
    holdout = manifest.get("holdout", {})
    if holdout != {
        "status": "retained_with_protocol_deviation", "final_test_file_opened": False,
        "final_test_metrics_computed": False, "final_test_used_for_model_selection": False,
        "mixed_fairness_file_loaded_during_schema_inspection": True,
        "incident_documented": True, "incident_summary": INCIDENT_STATEMENT,
    }:
        raise ValueError("Holdout protocol deviation cannot be erased.")
    if manifest.get("approvals") != {
        "capstone_evaluation_approved": True, "business_approved": False,
        "production_approved": False, "final_test_evaluation_authorized": False,
    }:
        raise ValueError("Business, production, or final-test authorization is prohibited.")
    hashes = manifest.get("artifact_hashes", {})
    if not isinstance(hashes, dict) or set(hashes) != set(HASH_PATHS):
        raise ValueError("Artifact hash allowlist is incomplete or altered.")
    for name, item in hashes.items():
        if not isinstance(item, dict) or set(item) != {"status", "sha256"}:
            raise ValueError(f"Malformed hash record: {name}")
        if item["status"] == "present":
            if not isinstance(item["sha256"], str) or not re.fullmatch(
                r"[0-9a-f]{64}", item["sha256"]
            ):
                raise ValueError(f"Invalid SHA-256 for {name}.")
        elif item != {"status": "missing", "sha256": None}:
            raise ValueError(f"Invalid missing-artifact record: {name}.")
    if manifest.get("missing_artifacts") != [name for name in HASH_PATHS
                                             if hashes[name]["status"] == "missing"]:
        raise ValueError("Missing-artifact list disagrees with hash records.")


def verify_manifest_hashes(manifest: dict[str, Any], root: Path) -> None:
    """Rehash only approved inputs and reject drift after manifest construction."""
    validate_freeze_manifest(manifest)
    if hash_approved_artifacts(root) != manifest["artifact_hashes"]:
        raise ValueError("A frozen artifact changed or disappeared after hashing.")
