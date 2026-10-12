"""V3 cross-fitted sigmoid plus inference-only missingness stress; no new search."""

from __future__ import annotations

import re
import subprocess
import time
import warnings
from collections import Counter
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import yaml
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits

from eris_ml.evaluation.feature_ablation import validate_fold_assignment
from eris_ml.evaluation.threshold_policy_v2 import top_k
from eris_ml.features.schema_v3 import build_pipeline, validate_features
from eris_ml.models import baseline_v3 as baseline
from eris_ml.models import tuning_v3 as tuning
from eris_ml.utils.hashing import sha256_file

PROTOCOL = "configs/calibration/calibration_protocol_v3.yaml"
MODELS = ("rf_tuned", "logistic_baseline")
CANDIDATES = [f"{m}_{method}" for m in MODELS for method in ("raw", "sigmoid")]
REFERENCE = "rf_baseline_raw"
SCENARIOS = {
    "original": [],
    "missing_training": ["TrainingTimesLastYear"],
    "missing_manager": ["YearsWithCurrManager"],
    "missing_both": ["TrainingTimesLastYear", "YearsWithCurrManager"],
}


def load_policy(root: Path) -> dict[str, Any]:
    policy: dict[str, Any] = yaml.safe_load(baseline.safe_path(root, PROTOCOL).read_text("utf-8"))
    expected = {
        "calibration": {
            "method": "sigmoid",
            "ensemble": False,
            "cv": {"n_splits": 4, "shuffle": True, "random_state": 43},
        },
        "models": list(MODELS),
        "methods": ["raw", "sigmoid"],
        "raw_regression": {"rtol": 1e-7, "atol": 1e-9},
        "rank_numerical_tolerance": 1e-12,
        "bins": {"count": 5, "strategy": "quantile", "duplicate_edges": "collapse"},
        "recommendation": {
            "require_pooled_brier_reduction": True,
            "require_pooled_log_loss_reduction": True,
            "minimum_brier_improved_folds": 3,
            "maximum_ap_drop": 0.005,
            "maximum_roc_auc_drop": 0.005,
            "small_brier_improvement": 0.001,
        },
        "top_k": {
            "capacity_percent": [5, 10, 15],
            "batch": "each_outer_validation_fold",
            "rounding": "floor_integer_percent_times_rows",
            "order": ["probability_descending", "source_row_ascending"],
            "business_approved": False,
        },
        "optional_missing": SCENARIOS,
        "execution": {"outer_n_jobs": 1, "calibration_n_jobs": 1, "native_threads": 1},
        "base_estimator_fits": 50,
        "final_test_accessed": False,
        "production_approved": False,
        "overlap_definition": "intersection_size_divided_by_original_top_10_count",
    }
    if policy.get("protocol_version") != "v3-final-15-calibration-v1" or any(
        policy.get(key) != value for key, value in expected.items()
    ):
        raise ValueError("V3 calibration protocol differs from the locked experiment.")
    return policy


def pipeline_for(config: dict[str, Any], model: str, parameters: dict[str, Any]) -> Any:
    if model == "logistic_baseline":
        if parameters:
            raise ValueError("Logistic baseline cannot receive tuned parameters.")
        return build_pipeline(config, "logistic_regression")
    if model != "rf_tuned":
        raise ValueError("Only fixed Logistic baseline and fold-specific tuned RF are allowed.")
    pipeline = build_pipeline(config, "random_forest")
    pipeline.set_params(**parameters)
    return pipeline


def check_raw(
    current: pd.DataFrame,
    reference: pd.DataFrame,
    assignments: pd.DataFrame,
    tolerance: dict[str, float],
) -> float:
    models = sorted(set(current.model))
    left = tuning.aligned_oof(current, assignments, models)
    right = tuning.aligned_oof(reference, assignments, models)
    difference = float(np.max(np.abs(left.probability.to_numpy() - right.probability.to_numpy())))
    if not np.allclose(
        left.probability, right.probability, rtol=tolerance["rtol"], atol=tolerance["atol"]
    ):
        raise ValueError(
            f"Raw regression failed by row ID: max absolute difference={difference:.17g}"
        )
    return difference


def rank_diagnostics(
    raw: np.ndarray, calibrated: np.ndarray, ids: np.ndarray, slope: float, tolerance: float
) -> dict[str, Any]:
    """Fail on meaningful reversals; retain exact rank/tie differences for inspection."""
    raw_gap = raw[:, None] - raw[None, :]
    calibrated_gap = calibrated[:, None] - calibrated[None, :]
    inversions = int(np.sum((raw_gap > tolerance) & (calibrated_gap < -tolerance)))
    changes = {}
    for percent in (5, 10, 15):
        left, right = top_k(raw, ids, percent), top_k(calibrated, ids, percent)
        lost, gained = raw[(left == 1) & (right == 0)], raw[(left == 0) & (right == 1)]
        boundary_gap = float(np.max(lost) - np.min(gained)) if len(lost) else 0.0
        changes[str(percent)] = {
            "selected_membership_changes": int(np.sum(left != right)),
            "max_swapped_raw_gap": boundary_gap,
        }
        if boundary_gap > tolerance:
            raise ValueError("Material within-fold top-k change; investigate scores/sigmoid/ties.")
    if not np.isfinite(slope) or slope >= 0 or inversions:
        raise ValueError(
            "Sigmoid is not increasing or ranking reversed; investigate before reporting."
        )
    return {
        "sigmoid_a_negative_means_increasing": slope,
        "material_pairwise_inversions": inversions,
        "exact_rank_position_changes": int(
            np.sum(np.lexsort((ids, -raw)) != np.lexsort((ids, -calibrated)))
        ),
        "raw_duplicate_scores": len(raw) - len(np.unique(raw)),
        "sigmoid_duplicate_scores": len(raw) - len(np.unique(calibrated)),
        "top_k_changes": changes,
        "numerical_tolerance": tolerance,
    }


def fit_outer(
    frame: pd.DataFrame,
    assignments: pd.DataFrame,
    config: dict[str, Any],
    policy: dict[str, Any],
    model: str,
    fold: int,
    parameters: dict[str, Any],
    raw_reference: pd.DataFrame,
) -> dict[str, Any]:
    if (
        not frame[baseline.IDS]
        .reset_index(drop=True)
        .equals(assignments[baseline.IDS].reset_index(drop=True))
    ):
        raise ValueError("Development/assignment mismatch before calibration fit.")
    target, folds = frame.Attrition.to_numpy(), assignments.fold.to_numpy()
    validate_fold_assignment(target, folds)
    train, valid = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
    if not len(valid):
        raise ValueError("Missing outer fold.")
    X = validate_features(frame[[f["name"] for f in config["features"]]], config)
    original = X.copy(deep=True)
    cv = StratifiedKFold(**policy["calibration"]["cv"])
    splits = list(cv.split(X.iloc[train], target[train]))
    expected_splits = tuning.inner_splits(target[train])
    for (it, iv), (et, ev) in zip(splits, expected_splits, strict=True):
        if not np.array_equal(it, et) or not np.array_equal(iv, ev):
            raise ValueError("Calibration split protocol mismatch.")
    template = pipeline_for(config, model, parameters)
    calibrated: Any = CalibratedClassifierCV(
        estimator=clone(template),
        method="sigmoid",
        cv=cv,
        ensemble=False,
        n_jobs=1,
    )
    started = time.monotonic()
    predictions, diagnostics = [], {}
    max_difference = 0.0
    with warnings.catch_warnings(record=True) as captured, threadpool_limits(limits=1):
        warnings.simplefilter("always")
        calibrated.fit(X.iloc[train], target[train])
        if len(calibrated.calibrated_classifiers_) != 1 or not np.array_equal(
            calibrated.classes_, [0, 1]
        ):
            raise ValueError("Expected one ensemble=False classifier with classes [0,1].")
        component = calibrated.calibrated_classifiers_[0]
        base = component.estimator
        if not np.array_equal(base.classes_, [0, 1]) or len(component.calibrators) != 1:
            raise ValueError("Wrong base classes or sigmoid count.")
        sigmoid = component.calibrators[0]
        optional = [f["name"] for f in config["features"] if not f["required"]]
        imputer = (
            base.named_steps["preprocessor"]
            .named_transformers_["optional_numeric"]
            .named_steps["imputer"]
        )
        median = X.iloc[train][optional].median().to_numpy()
        if not np.allclose(imputer.statistics_, median):
            raise ValueError("Optional imputer is not fitted on outer-training medians.")
        for scenario, missing in policy["optional_missing"].items():
            masked = X.iloc[valid].copy(deep=True)
            if missing:
                masked.loc[:, missing] = np.nan
            raw = np.asarray(base.predict_proba(masked)[:, 1], dtype=float)
            probability = np.asarray(calibrated.predict_proba(masked)[:, 1], dtype=float)
            scores = base.decision_function(masked) if hasattr(base, "decision_function") else raw
            # sklearn prefers decision_function for Logistic, predict_proba for RF.
            if not np.allclose(sigmoid.predict(scores), probability, rtol=1e-12, atol=1e-14):
                raise ValueError("Sigmoid output differs from the documented base response.")
            for method, values in (("raw", raw), ("sigmoid", probability)):
                block = assignments.iloc[valid][["source_row", "EmployeeNumber", "fold"]].copy()
                block["y_true"] = target[valid]
                block["feature_set"] = "v3_final_15"
                block["model"] = f"{model}_{method}"
                block["probability"] = values
                block["scenario"] = scenario
                block = tuning.aligned_oof(block, assignments.iloc[valid], [f"{model}_{method}"])
                if scenario == "original" and method == "raw":
                    ref = raw_reference.loc[
                        (raw_reference.model == f"{model}_raw") & (raw_reference.fold == fold)
                    ].drop(columns="scenario", errors="ignore")
                    max_difference = check_raw(
                        block.drop(columns="scenario"),
                        ref,
                        assignments.iloc[valid],
                        policy["raw_regression"],
                    )
                for percent in policy["top_k"]["capacity_percent"]:
                    block[f"top_{percent}"] = top_k(values, block.source_row.to_numpy(), percent)
                predictions.extend(block.to_dict("records"))
            diagnostics[scenario] = rank_diagnostics(
                raw,
                probability,
                assignments.iloc[valid].source_row.to_numpy(),
                float(sigmoid.a_),
                policy["rank_numerical_tolerance"],
            )
        pd.testing.assert_frame_equal(X, original)
        if not np.array_equal(imputer.statistics_, median):
            raise ValueError("Stress predictions changed imputation state.")
    return cast(
        dict[str, Any],
        baseline.json_safe(
            {
                "model": model,
                "fold": fold,
                "ensemble": False,
                "base_estimator_fits": 5,
                "calibrator_fits": 1,
                "parameters": base.named_steps["model"].get_params(),
                "raw_max_absolute_difference": max_difference,
                "response_method": "decision_function"
                if hasattr(base, "decision_function")
                else "predict_proba",
                "sigmoid_coefficients": {"a": float(sigmoid.a_), "b": float(sigmoid.b_)},
                "optional_training_medians": dict(zip(optional, median.tolist(), strict=True)),
                "outer_training_source_rows": assignments.iloc[train].source_row.tolist(),
                "calibration_validation_source_rows": [
                    assignments.iloc[train[iv]].source_row.tolist() for _, iv in splits
                ],
                "rank_diagnostics": diagnostics,
                "predictions": predictions,
                "elapsed_seconds": time.monotonic() - started,
                "warnings": dict(Counter(f"{w.category.__name__}: {w.message}" for w in captured)),
            }
        ),
    )


def prepare(
    root: Path, policy: dict[str, Any]
) -> tuple[
    dict[str, Any],
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    dict[int, dict[str, Any]],
    dict[str, Any],
]:
    search_policy = tuning.load_search_policy(root)
    config, frame, assignments, old_oof, prior_identity = tuning.prepare_run(root, search_policy)
    if policy["feature_config"] != search_policy["feature_config"] or (
        policy["baseline_metadata"] != search_policy["baseline_metadata"]
    ):
        raise ValueError("Calibration source differs from V3 baseline.")
    meta_path = baseline.safe_path(root, policy["tuning_metadata"])
    metadata = tuning.read_json(meta_path)
    pre_path = meta_path.parent / "preflight.json"
    if sha256_file(pre_path) != metadata["preflight_sha256"]:
        raise ValueError("Tuning preflight hash mismatch.")
    prior = tuning.read_json(pre_path)
    if (
        prior["identity"] != prior_identity
        or tuning.digest(prior_identity) != metadata["identity_sha256"]
    ):
        raise ValueError("Historical t1 config/data/source/environment differs; no silent bridge.")
    protected = dict(prior_identity["protected_sha256"])
    for key in ("checkpoint_sha256", "output_sha256"):
        tuning.verify_hashes(root, metadata[key])
        protected.update(metadata[key])
    for path in (meta_path, pre_path):
        protected[path.relative_to(root).as_posix()] = sha256_file(path)
    oof_paths = [p for p in metadata["output_sha256"] if p.endswith("/oof.csv")]
    if len(oof_paths) != 1:
        raise ValueError("Expected exactly one t1 OOF artifact.")
    tuned_oof = tuning.aligned_oof(
        pd.read_csv(baseline.safe_path(root, oof_paths[0])), assignments, list(tuning.MODELS)
    )
    parameters: dict[int, dict[str, Any]] = {}
    for relative in metadata["checkpoint_sha256"]:
        record = tuning.load_checkpoint(
            baseline.safe_path(root, relative), metadata["identity_sha256"]
        )
        if record["model"] != "random_forest":
            continue
        fold = record["fold"]
        if type(fold) is not int or fold in parameters or fold not in range(1, 6):
            raise ValueError("Duplicate or invalid RF outer fold.")
        params = record["best_params"]
        allowed = prior["identity"]["forest_candidates"]
        if (
            params not in allowed
            or params != record["inner_cv_results"][record["best_index"]]["params"]
        ):
            raise ValueError("RF best parameters are not from the recorded t1 search.")
        for name, values in search_policy["random_forest"]["distributions"].items():
            if params.get(f"model__{name}") not in values:
                raise ValueError("RF parameter is outside the t1 allowlist.")
        if set(params) != {
            f"model__{name}" for name in search_policy["random_forest"]["distributions"]
        }:
            raise ValueError("Unexpected RF parameter name.")
        parameters[fold] = {**params, "model__n_jobs": search_policy["execution"]["forest_n_jobs"]}
        resolved = (
            pipeline_for(config, "rf_tuned", parameters[fold]).named_steps["model"].get_params()
        )
        if baseline.json_safe(resolved) != record["resolved_best_model_params"]:
            raise ValueError("Resolved RF pipeline parameters differ from t1.")
        fold_assignments = assignments.loc[assignments.fold == fold]
        check_raw(
            pd.DataFrame(record["predictions"]),
            tuned_oof.loc[(tuned_oof.model == "random_forest") & (tuned_oof.fold == fold)],
            fold_assignments,
            {"rtol": 1e-12, "atol": 1e-14},
        )
        if set(record["outer_train_source_rows"]) != set(
            assignments.loc[assignments.fold != fold, "source_row"]
        ):
            raise ValueError("t1 outer-training membership mismatch.")
    if set(parameters) != set(range(1, 6)):
        raise ValueError("Missing t1 RF fold parameters.")
    reference = pd.concat(
        [
            old_oof.loc[old_oof.model == "logistic_regression"].assign(
                model="logistic_baseline_raw"
            ),
            old_oof.loc[old_oof.model == "random_forest"].assign(model=REFERENCE),
            tuned_oof.loc[tuned_oof.model == "random_forest"].assign(model="rf_tuned_raw"),
        ],
        ignore_index=True,
    )
    sources = [
        PROTOCOL,
        "src/eris_ml/models/calibration_v3.py",
        "src/eris_ml/evaluation/calibration_v3_reporting.py",
        "scripts/evaluate_calibration_v3.py",
        "src/eris_ml/evaluation/calibration_reporting.py",
        "src/eris_ml/evaluation/threshold_policy_v2.py",
        "src/eris_ml/evaluation/thresholding.py",
    ]
    for relative in sources:
        protected[relative] = sha256_file(baseline.safe_path(root, relative))
    identity = baseline.json_safe(
        {
            "policy": policy,
            "feature_config": config,
            "protected_sha256": protected,
            "python": prior_identity["python"],
            "library_versions": {
                **prior_identity["library_versions"],
                "matplotlib": version("matplotlib"),
                "threadpoolctl": version("threadpoolctl"),
            },
            "data_provenance": prior_identity["data_provenance"],
            "t1_identity_sha256": metadata["identity_sha256"],
            "resolved_pipelines": {
                str(f): {
                    model: baseline.json_safe(
                        pipeline_for(config, model, parameters[f] if model == "rf_tuned" else {})
                    )
                    for model in MODELS
                }
                for f in range(1, 6)
            },
            "calibration_scope": (
                "t1 RF parameters fixed inside outer train; "
                "not an independent inner tuning evaluation"
            ),
        }
    )
    return config, frame, assignments, reference, parameters, identity


def validate_results(oof: pd.DataFrame, assignments: pd.DataFrame) -> pd.DataFrame:
    if set(oof.scenario) != set(SCENARIOS):
        raise ValueError("Incomplete optional-missing scenarios.")
    blocks = []
    for scenario in SCENARIOS:
        block = tuning.aligned_oof(oof.loc[oof.scenario == scenario], assignments, CANDIDATES)
        for _, rows in block.groupby(["model", "fold"]):
            for percent in (5, 10, 15):
                expected = top_k(rows.probability.to_numpy(), rows.source_row.to_numpy(), percent)
                if not np.array_equal(rows[f"top_{percent}"].to_numpy(), expected):
                    raise ValueError("Saved top-k prediction differs from locked batch/tie rule.")
        blocks.append(block)
    return pd.concat(blocks, ignore_index=True)


def run(root: Path, run_id: str, resume: bool = False) -> None:
    from eris_ml.evaluation.calibration_v3_reporting import analyze, append_summary, plot_bins

    if not re.fullmatch(r"c[0-9][a-zA-Z0-9_-]*", run_id):
        raise ValueError("Calibration run ID must begin with c followed by a digit.")
    report = baseline.safe_path(root, f"artifacts/reports/v3_final_15/{run_id}")
    predictions = baseline.safe_path(root, f"artifacts/predictions/v3_final_15/{run_id}")
    if (report.exists() or predictions.exists()) and not resume:
        raise FileExistsError("Run outputs exist; use verified --resume or a new run ID.")
    if resume and not (report / "preflight.json").exists():
        raise ValueError("Cannot resume without a complete preflight.")
    policy = load_policy(root)
    config, frame, assignments, reference, parameters, identity = prepare(root, policy)
    identity_hash = tuning.digest(identity)
    summary = baseline.safe_path(root, "artifacts/reports/v3_final_15/summary.md")
    preflight_path = report / "preflight.json"
    if resume:
        preflight = tuning.read_json(preflight_path)
        if preflight["identity"] != identity or preflight["identity_sha256"] != identity_hash:
            raise ValueError("Resume identity changed: data/folds/parameters/source/environment.")
    else:
        preflight = {
            "created_at_utc": datetime.now(UTC).isoformat(),
            "run_id": run_id,
            "identity_sha256": identity_hash,
            "identity": identity,
            "summary_before": summary.read_text(encoding="utf-8"),
            "summary_before_sha256": sha256_file(summary),
            "summary_before_bytes": summary.stat().st_size,
            "git": {
                key: subprocess.check_output(["git", *args], cwd=root, text=True).strip()
                for key, args in {
                    "head": ["rev-parse", "HEAD"],
                    "branch": ["branch", "--show-current"],
                    "status": ["status", "--short"],
                }.items()
            },
        }
        baseline.write_json(preflight_path, preflight)
    print("Preflight passed: 1176 development rows; 50 base fits, no search.", flush=True)
    checkpoints, hashes = [], {}
    for model in MODELS:
        for fold in range(1, 6):
            path = report / "checkpoints" / f"{model}_fold_{fold}.json"
            if path.exists():
                record = tuning.load_checkpoint(path, identity_hash)
                if record["model"] != model or record["fold"] != fold:
                    raise ValueError("Calibration checkpoint model/fold mismatch.")
                print(f"Reused {model} fold {fold}", flush=True)
            else:
                print(f"Calibrating {model} fold {fold}/5", flush=True)
                record = fit_outer(
                    frame,
                    assignments,
                    config,
                    policy,
                    model,
                    fold,
                    parameters[fold] if model == "rf_tuned" else {},
                    reference,
                )
                tuning.save_checkpoint(path, identity_hash, record)
                print(
                    f"Saved: {record['elapsed_seconds']:.1f}s; raw max diff "
                    f"{record['raw_max_absolute_difference']:.3g}; "
                    f"warnings={sum(record['warnings'].values())}",
                    flush=True,
                )
            checkpoints.append(record)
            hashes[path.relative_to(root).as_posix()] = sha256_file(path)
    oof = validate_results(
        pd.DataFrame([p for c in checkpoints for p in c["predictions"]]), assignments
    )
    analysis, bins = analyze(oof, reference, policy)
    tuning.verify_hashes(root, identity["protected_sha256"])
    report_paths = []
    for key in ("calibration", "top_k", "optional_missing"):
        path = report / f"{key}.json"
        tuning.preserve_json(path, analysis[key], resume)
        report_paths.append(path)
    predictions.mkdir(parents=True, exist_ok=True)
    for name, block in (
        ("oof", oof.loc[oof.scenario == "original"]),
        ("optional_missing_oof", oof.loc[oof.scenario != "original"]),
        ("reliability_bins", bins),
    ):
        path = (report if name == "reliability_bins" else predictions) / f"{name}.csv"
        if not path.exists():
            block.to_csv(path, index=False, mode="x")
        pd.testing.assert_frame_equal(
            pd.read_csv(path),
            block.reset_index(drop=True),
            check_exact=False,
            rtol=1e-12,
            atol=1e-14,
        )
        report_paths.append(path)
    plot = report / "reliability.png"
    if not plot.exists():
        plot_bins(bins, plot)
    elif not (report / "metadata.json").exists():
        raise ValueError("Existing unverified plot; refusing to replace it.")
    report_paths.append(plot)
    completed = {
        "identity_sha256": identity_hash,
        "preflight_sha256": sha256_file(preflight_path),
        "checkpoint_sha256": hashes,
        "output_sha256": {p.relative_to(root).as_posix(): sha256_file(p) for p in report_paths},
        "base_estimator_fits": sum(c["base_estimator_fits"] for c in checkpoints),
        "calibrator_fits": len(checkpoints),
        "stress_extra_fits": 0,
        "raw_max_absolute_difference": max(c["raw_max_absolute_difference"] for c in checkpoints),
        "warnings": [c["warnings"] for c in checkpoints],
        "protected_hashes_unchanged": True,
        "rows_per_candidate_and_scenario": 1176,
        "oof_csv_roundtrip": "passed",
        "final_test_accessed": False,
        "model_serialized": False,
        "full_development_fit": False,
        "business_approved": False,
        "production_approved": False,
    }
    if (report / "metadata.json").exists():
        tuning.verify_hashes(root, tuning.read_json(report / "metadata.json")["output_sha256"])
    append_summary(summary, preflight, run_id, analysis, checkpoints)
    tuning.preserve_json(report / "metadata.json", completed, resume)
    print(
        f"Completed {run_id}: 4 x 1176 original OOF; 3 missing scenarios; 50 base fits.", flush=True
    )
