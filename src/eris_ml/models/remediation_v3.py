"""Post-holdout development-only fixed-candidate study; no final artifacts are inputs."""

from __future__ import annotations

import copy
import re
import subprocess
import time
import warnings
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits

from eris_ml.evaluation.feature_ablation import validate_fold_assignment
from eris_ml.evaluation.threshold_policy_v2 import top_k
from eris_ml.features.schema_v3 import build_pipeline, validate_features
from eris_ml.models import baseline_v3 as baseline
from eris_ml.models import calibration_v3 as calibration
from eris_ml.models import candidate_v3 as candidate
from eris_ml.models import tuning_v3 as tuning
from eris_ml.models.bundle_v3 import versions
from eris_ml.utils.hashing import sha256_file

PRIMARY = [
    "rf_balanced_raw",
    "rf_balanced_sigmoid",
    "rf_unweighted_raw",
    "rf_unweighted_sigmoid",
    "logistic_baseline_raw",
]
SENSITIVITY = "rf_unweighted_no_age_raw"
FAMILIES = ["rf_balanced", "rf_unweighted", "logistic_baseline", "rf_unweighted_no_age"]
SCENARIOS = calibration.SCENARIOS
MODEL_SHA = "ac76d9a5df2e9d87aa20015fdf93cbee9ec4e9f222c79de5fec347148dc72039"
POLICY: dict[str, Any] = {
    "study": "post_final_development_remediation_v3_v1",
    "main_candidates": PRIMARY,
    "sensitivity_only": SENSITIVITY,
    "outer_folds": "saved_V2_assignments_used_by_V3_unchanged",
    "inner_cv": {"n_splits": 4, "shuffle": True, "random_state": 43},
    "calibration": {"method": "sigmoid", "ensemble": False, "n_jobs": 1},
    "rf_base": {
        "n_estimators": 600,
        "max_depth": 4,
        "min_samples_leaf": 12,
        "min_samples_split": 10,
        "max_features": "sqrt",
        "class_weight": "balanced",
        "bootstrap": True,
        "random_state": 42,
        "n_jobs": 4,
    },
    "dummy": "saved_r1_outer_train_prior_OOF",
    "guardrails": {
        "brier_below_dummy": True,
        "log_loss_below_dummy": True,
        "maximum_brier_worse_folds": 2,
        "maximum_ap_drop": 0.02,
        "maximum_roc_drop": 0.02,
        "maximum_absolute_mean_bias": 0.05,
        "raw_counterpart_for_raw": "self; absolute dummy/bias rules still apply",
    },
    "ranking_order": [
        "AP_desc",
        "ROC_desc",
        "sum_TP_all_three_capacities_desc",
        "AP_fold_sample_SD_asc",
        "candidate_order",
    ],
    "probability_order": [
        "guardrails_pass_only",
        "Brier_asc",
        "LogLoss_asc",
        "absolute_mean_bias_asc",
        "candidate_order",
    ],
    "joint_recommendation": {
        "eligible_probability_candidate": True,
        "max_AP_gap_from_ranking_winner": 0.02,
        "max_ROC_gap_from_ranking_winner": 0.02,
        "top_k_and_stability": "reported_context_not_new_optimization",
        "otherwise": "ranking_only_research_output_not_probability",
    },
    "top_k": {
        "capacities_percent": [5, 10, 15],
        "batch": "each_saved_outer_validation_fold",
        "rounding": "floor",
        "ties": "source_row_ascending_without_labels",
    },
    "optional_missing": SCENARIOS,
    "stress_scope": "all_candidates_no_retraining",
    "age_groups": ["<30", "30-39", "40-49", ">=50"],
    "support": {"n": 50, "positives": 10, "negatives": 10},
    "reliability": {"count": 5, "strategy": "quantile", "duplicate_edges": "collapse"},
    "calibration_intercept_slope": "not_estimated_no_validated_existing_implementation",
    "reference_threshold": 0.5,
    "expected_base_fits": 70,
    "expected_calibrator_fits": 10,
    "execution": {"outer_n_jobs": 1, "native_threads": 1, "rf_n_jobs": 4},
    "selection_data": "development_only_after_final_observation_not_independent_validation",
    "final_holdout_access": False,
    "f1_predictions_access": False,
    "production_approved": False,
    "business_approved": False,
    "bundle_creation": False,
}


def study_path(root: Path, relative: str) -> Path:
    normalized = "/" + relative.replace("\\", "/").lower().strip("/") + "/"
    if "/f1/" in normalized or "final_evaluation" in normalized:
        raise ValueError("Final evaluation artifacts are not remediation inputs.")
    return baseline.safe_path(root, relative)


def sensitivity_config(config: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(config)
    result["version"] = "v3_sensitivity_no_age_14"
    result["status"] = "sensitivity_only_not_candidate_replacement"
    result["features"] = [f for f in result["features"] if f["name"] != "Age"]
    result["non_features"]["audit_only"].append("Age")
    if len(result["features"]) != 14 or any(f["name"] == "Age" for f in result["features"]):
        raise ValueError("No-Age sensitivity schema must contain exactly fourteen inputs.")
    return result


def pipeline_for(config: dict[str, Any], rf_params: dict[str, Any], family: str) -> Any:
    if family not in FAMILIES:
        raise ValueError("Unknown fixed remediation candidate.")
    if family == "logistic_baseline":
        return build_pipeline(config, "logistic_regression")
    definition = sensitivity_config(config) if family == "rf_unweighted_no_age" else config
    pipeline = build_pipeline(definition, "random_forest")
    params = dict(rf_params)
    if family != "rf_balanced":
        params["class_weight"] = None
    pipeline.set_params(**{f"model__{key}": value for key, value in params.items()})
    return pipeline


def prepare(
    root: Path,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    # This established path verifies r1/t1/c1 development provenance, never reads f1.
    config, frame, _, _, inherited = candidate.prepare(root)
    assignments = pd.read_csv(study_path(root, config["data"]["assignments"]))
    protected = dict(inherited["protected_sha256"])
    b1 = study_path(root, "artifacts/reports/v3_final_15/b1")
    preflight = tuning.read_json(b1 / "preflight.json")
    meta = tuning.read_json(b1 / "metadata.json")
    if tuning.digest(preflight["provenance"]) != preflight["identity_sha256"]:
        raise ValueError("b1 preflight identity mismatch.")
    tuning.verify_hashes(root, preflight["provenance"]["protected_sha256"])
    tuning.verify_hashes(root, meta["output_sha256"])
    protected.update(meta["output_sha256"])
    protected[(b1 / "metadata.json").relative_to(root).as_posix()] = sha256_file(
        b1 / "metadata.json"
    )
    bundle_dir = study_path(root, "artifacts/models/v3_final_15/b1")
    manifest = tuning.read_json(bundle_dir / "manifest.json")
    if (
        manifest["pipeline.joblib"] != MODEL_SHA
        or sha256_file(bundle_dir / "pipeline.joblib") != MODEL_SHA
    ):
        raise ValueError("Frozen b1 model hash mismatch.")
    metadata = tuning.read_json(bundle_dir / "metadata.json")
    params = metadata["resolved_model_parameters"]
    if any(params.get(k) != v for k, v in POLICY["rf_base"].items()):
        raise ValueError("b1 RF configuration differs from requested fixed intervention.")
    reference = pd.read_csv(study_path(root, "artifacts/predictions/v3_final_15/r1/oof.csv"))
    reference = tuning.aligned_oof(reference, assignments, list(config["models"]))
    for relative in (
        "src/eris_ml/models/remediation_v3.py",
        "src/eris_ml/evaluation/remediation_v3_reporting.py",
        "scripts/remediate_v3.py",
        "tests/integration/test_remediation_v3.py",
    ):
        protected[relative] = sha256_file(study_path(root, relative))
    identity = {
        "policy": POLICY,
        "feature_config": config,
        "sensitivity_feature_config": sensitivity_config(config),
        "rf_parameters": params,
        "resolved_pipelines": {
            family: baseline.json_safe(pipeline_for(config, params, family)) for family in FAMILIES
        },
        "protected_sha256": protected,
        "versions": versions(),
        "data_provenance": inherited["data_provenance"],
        "historical_source_bridges": "unchanged established V2/r1/t1/c1 lineage",
        "no_f1_artifact_read": True,
    }
    return config, frame, assignments, reference, identity


def fit_fold(
    frame: pd.DataFrame,
    assignments: pd.DataFrame,
    config: dict[str, Any],
    params: dict[str, Any],
    family: str,
    fold: int,
) -> dict[str, Any]:
    if (
        not frame[baseline.IDS]
        .reset_index(drop=True)
        .equals(assignments[baseline.IDS].reset_index(drop=True))
    ):
        raise ValueError("Development IDs/labels are not aligned before fit.")
    target, folds = frame.Attrition.to_numpy(), assignments.fold.to_numpy()
    validate_fold_assignment(target, folds)
    train, valid = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
    if not len(valid):
        raise ValueError("Outer validation fold missing.")
    definition = sensitivity_config(config) if family == "rf_unweighted_no_age" else config
    names = [f["name"] for f in definition["features"]]
    X = validate_features(frame[names], definition)
    original = X.copy(deep=True)
    template = pipeline_for(config, params, family)
    inner = tuning.inner_splits(target[train])
    models: dict[str, Any] = {}
    diagnostics: dict[str, Any] = {}
    started = time.monotonic()
    with warnings.catch_warnings(record=True) as captured, threadpool_limits(limits=1):
        warnings.simplefilter("always")
        raw = clone(template).fit(X.iloc[train], target[train])
        models[f"{family}_raw"] = raw
        if family in ("rf_balanced", "rf_unweighted"):
            calibrated: Any = CalibratedClassifierCV(
                estimator=clone(template),
                method="sigmoid",
                ensemble=False,
                n_jobs=1,
                cv=StratifiedKFold(**POLICY["inner_cv"]),
            )
            calibrated.fit(X.iloc[train], target[train])
            models[f"{family}_sigmoid"] = calibrated
            if len(calibrated.calibrated_classifiers_) != 1:
                raise ValueError("Expected one ensemble=False refitted base model.")
        rows = []
        optional = [f["name"] for f in definition["features"] if not f["required"]]
        for fitted in models.values():
            base = fitted if fitted is raw else fitted.calibrated_classifiers_[0].estimator
            if list(base.feature_names_in_) != names or not np.array_equal(base.classes_, [0, 1]):
                raise ValueError("Feature order or positive class mismatch.")
            imputer = (
                base.named_steps["preprocessor"]
                .named_transformers_["optional_numeric"]
                .named_steps["imputer"]
            )
            np.testing.assert_allclose(imputer.statistics_, X.iloc[train][optional].median())
        for scenario, missing in SCENARIOS.items():
            masked = X.iloc[valid].copy(deep=True)
            if missing:
                masked.loc[:, missing] = np.nan
            predictions: dict[str, np.ndarray] = {}
            for name, fitted in models.items():
                p = np.asarray(fitted.predict_proba(masked)[:, list(fitted.classes_).index(1)])
                predictions[name] = p
                block = assignments.iloc[valid][["source_row", "EmployeeNumber", "fold"]].copy()
                block["y_true"] = target[valid]
                block["feature_set"] = definition["version"]
                block["model"], block["scenario"], block["probability"] = name, scenario, p
                for percent in (5, 10, 15):
                    block[f"top_{percent}"] = top_k(p, block.source_row.to_numpy(), percent)
                rows.extend(block.to_dict("records"))
            if len(models) == 2:
                fitted = models[f"{family}_sigmoid"].calibrated_classifiers_[0]
                raw_p = predictions[f"{family}_raw"]
                base_p = fitted.estimator.predict_proba(masked)[:, 1]
                np.testing.assert_allclose(raw_p, base_p, atol=1e-12, rtol=1e-12)
                diagnostics[scenario] = calibration.rank_diagnostics(
                    raw_p,
                    predictions[f"{family}_sigmoid"],
                    assignments.iloc[valid].source_row.to_numpy(),
                    float(fitted.calibrators[0].a_),
                    1e-12,
                )
        pd.testing.assert_frame_equal(X, original)
    return cast(
        dict[str, Any],
        baseline.json_safe(
            {
                "family": family,
                "fold": fold,
                "base_fits": 6 if len(models) == 2 else 1,
                "calibrator_fits": 1 if len(models) == 2 else 0,
                "parameters": raw.named_steps["model"].get_params(),
                "feature_order": names,
                "train_source_rows": assignments.iloc[train].source_row.tolist(),
                "inner_validation_source_rows": [
                    assignments.iloc[train[v]].source_row.tolist() for _, v in inner
                ],
                "optional_training_medians": X.iloc[train][optional].median().to_dict(),
                "rank_diagnostics": diagnostics,
                "predictions": rows,
                "elapsed_seconds": time.monotonic() - started,
                "warnings": dict(Counter(f"{w.category.__name__}: {w.message}" for w in captured)),
            }
        ),
    )


def validate_oof(oof: pd.DataFrame, assignments: pd.DataFrame) -> pd.DataFrame:
    if set(oof.model) != set(PRIMARY + [SENSITIVITY]) or set(oof.scenario) != set(SCENARIOS):
        raise ValueError("Missing remediation candidates/scenarios.")
    blocks = []
    for model in PRIMARY + [SENSITIVITY]:
        feature_set = "v3_sensitivity_no_age_14" if model == SENSITIVITY else "v3_final_15"
        for scenario in SCENARIOS:
            block = oof.loc[(oof.model == model) & (oof.scenario == scenario)].copy()
            if set(block.feature_set) != {feature_set}:
                raise ValueError("Wrong feature schema in OOF.")
            # Reuse alignment validation; restore the independently checked feature tag.
            checked = tuning.aligned_oof(
                block.assign(feature_set="v3_final_15"), assignments, [model]
            )
            checked["feature_set"] = feature_set
            for _, rows in checked.groupby("fold"):
                for percent in (5, 10, 15):
                    np.testing.assert_array_equal(
                        rows[f"top_{percent}"],
                        top_k(rows.probability.to_numpy(), rows.source_row.to_numpy(), percent),
                    )
            blocks.append(checked)
    return pd.concat(blocks, ignore_index=True)


def run(root: Path, run_id: str, resume: bool = False) -> None:
    from eris_ml.evaluation.remediation_v3_reporting import analyze, report_results

    if not re.fullmatch(r"devfix[1-9][0-9]*", run_id):
        raise ValueError("Use a separate devfixN run ID.")
    report = study_path(root, f"artifacts/reports/v3_final_15/{run_id}")
    pred = study_path(root, f"artifacts/predictions/v3_final_15/{run_id}")
    if (report.exists() or pred.exists()) and not resume:
        raise FileExistsError("Study outputs exist; use verified --resume; no overwrite.")
    if resume and not (report / "preflight.json").exists():
        raise ValueError("Cannot resume without preflight.")
    config, frame, assignments, reference, identity = prepare(root)
    digest = tuning.digest(identity)
    if resume:
        prior = tuning.read_json(report / "preflight.json")
        if prior["identity"] != baseline.json_safe(identity) or prior["identity_sha256"] != digest:
            raise ValueError("Study provenance/source/config/environment changed.")
        if (report / "manifest.json").exists():
            manifest = tuning.read_json(report / "manifest.json")
            tuning.verify_hashes(root, manifest["output_sha256"])
            tuning.verify_hashes(root, identity["protected_sha256"])
            print("Verified completed development study; no refit or recomputation.")
            return
    else:
        baseline.write_json(report / "protocol.json", POLICY)
        baseline.write_json(
            report / "preflight.json",
            {
                "created_at_utc": datetime.now(UTC).isoformat(),
                "identity": identity,
                "identity_sha256": digest,
                "git": {
                    k: subprocess.check_output(["git", *v], cwd=root, text=True).strip()
                    for k, v in {
                        "head": ["rev-parse", "HEAD"],
                        "branch": ["branch", "--show-current"],
                        "status": ["status", "--short"],
                    }.items()
                },
            },
        )
    print(
        "Protocol locked before results: 1176 development rows, 70 base fits, 10 sigmoid fits.",
        flush=True,
    )
    checkpoints = []
    for family in FAMILIES:
        for fold in range(1, 6):
            path = report / "checkpoints" / f"{family}_fold_{fold}.json"
            if path.exists():
                record = tuning.load_checkpoint(path, digest)
                if record["family"] != family or record["fold"] != fold:
                    raise ValueError("Checkpoint identity mismatch.")
            else:
                print(f"Fitting {family}, fold {fold}/5", flush=True)
                record = fit_fold(
                    frame, assignments, config, identity["rf_parameters"], family, fold
                )
                tuning.save_checkpoint(path, digest, record)
                print(
                    f"Saved {record['elapsed_seconds']:.1f}s; warnings={record['warnings']}",
                    flush=True,
                )
            checkpoints.append(record)
    oof = validate_oof(
        pd.DataFrame([row for c in checkpoints for row in c["predictions"]]), assignments
    )
    # Baseline reference is a regression check, never a new tuned Logistic.
    logistic = oof.loc[(oof.model == "logistic_baseline_raw") & (oof.scenario == "original")]
    previous = reference.loc[reference.model == "logistic_regression"].assign(
        model="logistic_baseline_raw"
    )
    difference = calibration.check_raw(
        logistic, previous, assignments, {"atol": 1e-9, "rtol": 1e-7}
    )
    analysis, bins, groups = analyze(oof, reference, frame)
    pred.mkdir(parents=True, exist_ok=True)
    for name, block in (
        ("oof", oof.loc[oof.scenario == "original"]),
        ("optional_missing_oof", oof.loc[oof.scenario != "original"]),
    ):
        path = pred / f"{name}.csv"
        if not path.exists():
            block.to_csv(path, index=False, mode="x")
        pd.testing.assert_frame_equal(
            pd.read_csv(path),
            block.reset_index(drop=True),
            check_exact=False,
            rtol=1e-12,
            atol=1e-14,
        )
    report_results(report, analysis, bins, groups)
    tuning.verify_hashes(root, identity["protected_sha256"])
    files = [p for directory in (report, pred) for p in directory.rglob("*") if p.is_file()]
    baseline.write_json(
        report / "manifest.json",
        {
            "output_sha256": {p.relative_to(root).as_posix(): sha256_file(p) for p in files},
            "identity_sha256": digest,
            "base_fits": sum(c["base_fits"] for c in checkpoints),
            "calibrator_fits": sum(c["calibrator_fits"] for c in checkpoints),
            "logistic_r1_max_absolute_difference": difference,
            "protected_hashes_unchanged": True,
            "final_holdout_accessed": False,
            "f1_predictions_accessed": False,
            "model_serialized": False,
            "warnings": [c["warnings"] for c in checkpoints],
        },
    )
    print(f"Completed {run_id}; development-only; no b2 bundle or API changes.", flush=True)
