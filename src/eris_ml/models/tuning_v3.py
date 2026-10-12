"""Checkpointed nested V3 tuning; reuse baseline pipelines, metrics and saved folds."""

from __future__ import annotations

import hashlib
import json
import platform
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
from sklearn.model_selection import (
    GridSearchCV,
    ParameterGrid,
    ParameterSampler,
    RandomizedSearchCV,
    StratifiedKFold,
)
from threadpoolctl import threadpool_limits

from eris_ml.evaluation.feature_ablation import validate_fold_assignment
from eris_ml.features.schema_v3 import build_pipeline, load_protocol, validate_features
from eris_ml.models import baseline_v3 as baseline
from eris_ml.utils.hashing import sha256_file

MODELS = ("logistic_regression", "random_forest")
PROTOCOL = "configs/tuning/tuning_protocol_v3.yaml"


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(baseline.json_safe(value), sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def search_spaces(policy: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], str]:
    """Preserve pure L1/L2 on old and new sklearn APIs without changing the environment."""
    major, minor = (int(part) for part in version("scikit-learn").split(".")[:2])
    modern = (major, minor) >= (1, 8)
    logistic = policy["logistic"]
    grid = {
        "model__C": logistic["C"],
        "model__class_weight": logistic["class_weight"],
        ("model__l1_ratio" if modern else "model__penalty"): (
            [1.0, 0.0] if modern else ["l1", "l2"]
        ),
    }
    forest = {f"model__{k}": v for k, v in policy["random_forest"]["distributions"].items()}
    return grid, forest, "l1_ratio=1/0 maps to L1/L2" if modern else "penalty=l1/l2"


def load_search_policy(root: Path) -> dict[str, Any]:
    policy: dict[str, Any] = yaml.safe_load(baseline.safe_path(root, PROTOCOL).read_text("utf-8"))
    expected_rf = {
        "n_estimators": [300, 600],
        "max_depth": [None, 4, 6, 10],
        "min_samples_leaf": [1, 2, 4, 8, 12],
        "min_samples_split": [2, 5, 10],
        "max_features": ["sqrt", 0.5, 1.0],
        "class_weight": [None, "balanced", "balanced_subsample"],
        "bootstrap": [True],
    }
    if (
        policy["inner_cv"] != {"n_splits": 4, "shuffle": True, "random_state": 43}
        or policy["scoring"] != "average_precision"
        or policy["tie_breaker"] != "first_candidate_in_deterministic_search_order_exact_score_tie"
        or policy["estimator_seed"] != 42
        or policy["search_seed"] != 43
        or policy["logistic"]
        != {
            "C": [0.01, 0.1, 1, 10, 100],
            "regularization": ["L1", "L2"],
            "class_weight": [None, "balanced"],
            "solver": "liblinear",
            "max_iter": 10000,
        }
        or policy["random_forest"] != {"n_iter": 40, "distributions": expected_rf}
        or policy["execution"]
        != {
            "outer_n_jobs": 1,
            "search_n_jobs": 1,
            "forest_n_jobs": 4,
            "native_threads": 1,
        }
        or policy["threshold_reference_only"] != 0.5
        or policy["full_development_fit"]
        or policy["final_test_accessed"]
    ):
        raise ValueError("Production V3 search budget/protocol has changed.")
    return policy


def aligned_oof(
    oof: pd.DataFrame, assignments: pd.DataFrame, expected_models: list[str]
) -> pd.DataFrame:
    """Join by stable ID, not row position. Reject missing/duplicate/foreign identities."""
    columns = {
        "source_row",
        "EmployeeNumber",
        "y_true",
        "fold",
        "feature_set",
        "model",
        "probability",
    }
    if (
        oof.columns.has_duplicates
        or not columns.issubset(oof.columns)
        or oof[list(columns)].isna().any().any()
        or set(oof.model) != set(expected_models)
        or set(oof.feature_set) != {"v3_final_15"}
    ):
        raise ValueError("Invalid OOF schema/models/features/missing values.")
    if any(not assignments[c].is_unique for c in baseline.IDS[:2]):
        raise ValueError("Duplicate assignment IDs.")
    reference = assignments.rename(columns={"Attrition": "y_true"})
    reference = reference[["source_row", "EmployeeNumber", "y_true", "fold"]]
    blocks = []
    for model in expected_models:
        block = oof.loc[oof.model == model].copy()
        if (
            len(block) != len(reference)
            or any(not block[c].is_unique for c in baseline.IDS[:2])
            or set(block.source_row) != set(reference.source_row)
        ):
            raise ValueError("OOF IDs are missing, duplicated or foreign.")
        joined = reference.merge(
            block, on="source_row", validate="one_to_one", suffixes=("_ref", "")
        )
        for name in ("EmployeeNumber", "y_true", "fold"):
            if not np.array_equal(joined[name], joined[f"{name}_ref"]):
                raise ValueError(f"OOF {name} mismatch by source_row.")
        probability = joined.probability.to_numpy(dtype=float)
        if not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
            raise ValueError("OOF probabilities must be finite and in [0,1].")
        blocks.append(joined[list(oof.columns)])
    return pd.concat(blocks, ignore_index=True)


def inner_splits(target: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    if set(target) != {0, 1} or min(Counter(target).values()) < 4:
        raise ValueError("Inner CV requires at least four rows in each class.")
    splits = list(
        StratifiedKFold(4, shuffle=True, random_state=43).split(np.zeros(len(target)), target)
    )
    seen = np.zeros(len(target), dtype=int)
    for train, validation in splits:
        if np.intersect1d(train, validation).size or len(train) + len(validation) != len(target):
            raise ValueError("Inner fold isolation failed.")
        seen[validation] += 1
    if not np.all(seen == 1):
        raise ValueError("Inner validation coverage failed.")
    return splits


def search_pipeline(config: dict[str, Any], policy: dict[str, Any], model: str) -> Any:
    pipeline = build_pipeline(config, model)
    if model == "logistic_regression":
        pipeline.set_params(
            model__solver=policy["logistic"]["solver"],
            model__max_iter=policy["logistic"]["max_iter"],
        )
    elif model == "random_forest":
        pipeline.set_params(model__n_jobs=policy["execution"]["forest_n_jobs"])
    else:
        raise ValueError("Only Logistic and Random Forest may be tuned.")
    return pipeline


def fit_outer(
    frame: pd.DataFrame,
    assignments: pd.DataFrame,
    config: dict[str, Any],
    policy: dict[str, Any],
    model: str,
    fold: int,
) -> dict[str, Any]:
    """No full-development fit. Search and refit see only this outer-training subset."""
    if (
        not frame[baseline.IDS]
        .reset_index(drop=True)
        .equals(assignments[baseline.IDS].reset_index(drop=True))
    ):
        raise ValueError("Development/assignment alignment failed before fit.")
    y, folds = frame.Attrition.to_numpy(), assignments.fold.to_numpy()
    validate_fold_assignment(y, folds)
    train, validation = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
    if not len(validation):
        raise ValueError("Missing outer validation fold.")
    names = [f["name"] for f in config["features"]]
    X = validate_features(frame[names], config)
    splits = inner_splits(y[train])
    grid, forest, api = search_spaces(policy)
    pipeline = search_pipeline(config, policy, model)
    options = dict(
        estimator=pipeline,
        scoring="average_precision",
        refit=True,
        cv=splits,
        n_jobs=1,
        error_score="raise",
        return_train_score=False,
    )
    search: Any = (
        GridSearchCV(param_grid=grid, **options)
        if model == "logistic_regression"
        else RandomizedSearchCV(
            param_distributions=forest,
            n_iter=policy["random_forest"]["n_iter"],
            random_state=43,
            **options,
        )
    )
    started = time.monotonic()
    with warnings.catch_warnings(record=True) as captured, threadpool_limits(limits=1):
        warnings.simplefilter("always")
        search.fit(X.iloc[train], y[train])
        best = search.best_estimator_
        if not np.array_equal(best.classes_, [0, 1]):
            raise ValueError("Unexpected positive class ordering.")
        probability = best.predict_proba(X.iloc[validation])[:, 1]
        optional = [f["name"] for f in config["features"] if not f["required"]]
        imputer = (
            best.named_steps["preprocessor"]
            .named_transformers_["optional_numeric"]
            .named_steps["imputer"]
        )
        if not np.allclose(imputer.statistics_, X.iloc[train][optional].median().to_numpy()):
            raise ValueError("Refit imputer statistics differ from outer-training medians.")
        for missing in ([optional[0]], [optional[1]], optional):
            smoke = best.predict_proba(X.iloc[validation[:2]].drop(columns=missing))[:, 1]
            if not np.isfinite(smoke).all() or ((smoke < 0) | (smoke > 1)).any():
                raise ValueError("Optional missing prediction failed after refit.")
    result = search.cv_results_
    candidates = []
    for index, parameters in enumerate(result["params"]):
        scores = [float(result[f"split{i}_test_score"][index]) for i in range(4)]
        if not np.isfinite(scores).all():
            raise ValueError("Nonfinite inner scores.")
        candidates.append(
            {
                "candidate_index": index,
                "params": parameters,
                "split_test_scores": scores,
                "mean_test_score": float(result["mean_test_score"][index]),
                "std_test_score": float(result["std_test_score"][index]),
                "rank_test_score": int(result["rank_test_score"][index]),
                "mean_fit_time": float(result["mean_fit_time"][index]),
                "std_fit_time": float(result["std_fit_time"][index]),
                "mean_score_time": float(result["mean_score_time"][index]),
                "std_score_time": float(result["std_score_time"][index]),
            }
        )
    winner = int(np.argmax([c["mean_test_score"] for c in candidates]))
    if search.best_index_ != winner:
        raise ValueError("Search tie-breaking did not select first maximum score.")
    predictions = assignments.iloc[validation][["source_row", "EmployeeNumber", "fold"]].copy()
    predictions["y_true"] = y[validation]
    predictions["feature_set"] = "v3_final_15"
    predictions["model"] = model
    predictions["probability"] = probability
    metric = baseline.metrics(y[validation], probability, 0.5)
    return cast(
        dict[str, Any],
        baseline.json_safe(
            {
                "model": model,
                "fold": fold,
                "candidate_count": len(candidates),
                "inner_fits": 4 * len(candidates),
                "refit_count": 1,
                "elapsed_seconds": time.monotonic() - started,
                "parameter_api": api,
                "best_params": search.best_params_,
                "best_index": winner,
                "best_inner_score_not_independent_evaluation": search.best_score_,
                "resolved_best_model_params": best.named_steps["model"].get_params(),
                "inner_cv_results": candidates,
                "outer_metrics": metric,
                "outer_train_source_rows": assignments.iloc[train].source_row.tolist(),
                "inner_validation_source_rows": [
                    assignments.iloc[train[iv]].source_row.tolist() for _, iv in splits
                ],
                "predictions": predictions.to_dict("records"),
                "positive_class": 1,
                "optional_missing_smoke": "passed",
                "outer_train_median_check": "passed",
                "warnings": dict(Counter(f"{w.category.__name__}: {w.message}" for w in captured)),
            }
        ),
    )


def save_checkpoint(path: Path, identity: str, payload: dict[str, Any]) -> None:
    baseline.write_json(
        path, {"identity": identity, "payload_sha256": digest(payload), "payload": payload}
    )


def load_checkpoint(path: Path, identity: str) -> dict[str, Any]:
    record = read_json(path)
    if record["identity"] != identity or record["payload_sha256"] != digest(record["payload"]):
        raise ValueError(f"Checkpoint identity/integrity mismatch: {path.name}")
    payload: dict[str, Any] = record["payload"]
    return payload


def summarize(oof: pd.DataFrame) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for model, block in oof.groupby("model", sort=True):
        pooled = baseline.metrics(block.y_true.to_numpy(), block.probability.to_numpy(), 0.5)
        folds = [
            {
                "fold": int(str(f)),
                **baseline.metrics(b.y_true.to_numpy(), b.probability.to_numpy(), 0.5),
            }
            for f, b in block.groupby("fold", sort=True)
        ]
        output[str(model)] = {
            "pooled_oof": pooled,
            "folds": folds,
            "fold_mean_std": {
                k: {
                    "mean": float(np.mean([f[k] for f in folds])),
                    "sample_std": float(np.std([f[k] for f in folds], ddof=1)),
                }
                for k in pooled
            },
        }
    return output


def compare(
    baseline_oof: pd.DataFrame, tuned: pd.DataFrame, assignments: pd.DataFrame
) -> dict[str, Any]:
    tuned = aligned_oof(tuned, assignments, list(MODELS))
    reference = aligned_oof(baseline_oof, assignments, sorted(set(baseline_oof.model)))
    old, new = summarize(reference), summarize(tuned)
    comparisons = {}
    for model in MODELS:
        delta = {
            k: new[model]["pooled_oof"][k] - old[model]["pooled_oof"][k]
            for k in old[model]["pooled_oof"]
        }
        folds = [
            {"fold": a["fold"], **{k: b[k] - a[k] for k in a if k != "fold"}}
            for a, b in zip(old[model]["folds"], new[model]["folds"], strict=True)
        ]
        comparisons[model] = {
            "pooled_delta_tuned_minus_baseline": delta,
            "fold_deltas": folds,
            "ap_improved_folds": sum(d["average_precision"] > 0 for d in folds),
            "small_positive_ap_change": 0 < delta["average_precision"] < 0.005,
        }
    return {
        "baseline": old,
        "tuned": new,
        "comparisons": comparisons,
        "threshold_reference_only": 0.5,
        "positive_rate": float(assignments.Attrition.mean()),
        "statistical_significance_test_performed": False,
    }


def verify_hashes(root: Path, hashes: dict[str, str]) -> None:
    for relative, expected in hashes.items():
        if sha256_file(baseline.safe_path(root, relative)) != expected:
            raise ValueError(f"Protected input/artifact hash mismatch: {relative}")


def prepare_run(
    root: Path, policy: dict[str, Any]
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    config = load_protocol(baseline.safe_path(root, policy["feature_config"]))
    frame, assignments, provenance = baseline.prepare(root, config)
    metadata_path = baseline.safe_path(root, policy["baseline_metadata"])
    metadata = read_json(metadata_path)
    outputs = metadata["outputs"]
    preflight_path = baseline.safe_path(root, outputs["preflight"])
    if sha256_file(preflight_path) != metadata["preflight_sha256"]:
        raise ValueError("Baseline preflight hash mismatch.")
    original = read_json(preflight_path)
    # V3 baseline source/config have not changed; V2 historical bridge remains untouched.
    verify_hashes(root, original["input_sha256"])
    hashes = dict(original["input_sha256"])
    for key, expected in metadata["outputs_sha256"].items():
        path = baseline.safe_path(root, outputs[key])
        if sha256_file(path) != expected:
            raise ValueError(f"Baseline {key} hash mismatch.")
        hashes[outputs[key]] = expected
    hashes[outputs["preflight"]] = sha256_file(preflight_path)
    hashes[policy["baseline_metadata"]] = sha256_file(metadata_path)
    oof = pd.read_csv(baseline.safe_path(root, outputs["oof"]))
    oof = aligned_oof(oof, assignments, list(config["models"]))
    recalculated = summarize(oof)
    recorded = read_json(baseline.safe_path(root, outputs["metrics"]))
    for record in recorded["models"]:
        actual = recalculated[record["model"]]
        for key, value in record["pooled_oof"].items():
            if not np.isclose(actual["pooled_oof"][key], value, rtol=1e-12, atol=1e-14):
                raise ValueError("Baseline OOF metric regression failed.")
    sources = [
        PROTOCOL,
        "src/eris_ml/models/tuning_v3.py",
        "scripts/tune_v3.py",
        "src/eris_ml/evaluation/tuning_v3_reporting.py",
    ]
    for relative in sources:
        hashes[relative] = sha256_file(baseline.safe_path(root, relative))
    versions = {p: version(p) for p in ("numpy", "pandas", "scipy", "scikit-learn", "joblib")}
    for package, installed in original["library_versions"].items():
        if package in versions and installed != versions[package]:
            raise ValueError(f"Baseline library version mismatch: {package}")
    grid, forest, api = search_spaces(policy)
    identity = {
        "policy": policy,
        "feature_config": config,
        "protected_sha256": hashes,
        "python": platform.python_version(),
        "library_versions": versions,
        "logistic_candidates": list(ParameterGrid(grid)),
        "forest_candidates": list(ParameterSampler(forest, n_iter=40, random_state=43)),
        "parameter_api": api,
        "resolved_base_pipelines": {
            m: baseline.json_safe(search_pipeline(config, policy, m)) for m in MODELS
        },
        "data_provenance": provenance,
        "final_test_accessed": False,
    }
    return config, frame, assignments, oof, baseline.json_safe(identity)


def preserve_json(path: Path, payload: Any, resume: bool) -> None:
    if path.exists():
        if not resume or read_json(path) != baseline.json_safe(payload):
            raise ValueError(f"Refusing to overwrite existing artifact: {path}")
    else:
        baseline.write_json(path, payload)


def run(root: Path, run_id: str, resume: bool = False) -> None:
    from eris_ml.evaluation.tuning_v3_reporting import append_summary

    if not re.fullmatch(r"t[0-9][a-zA-Z0-9_-]*", run_id):
        raise ValueError("Tuning run ID must start with t followed by a digit (e.g. t1).")
    report_dir = baseline.safe_path(root, f"artifacts/reports/v3_final_15/{run_id}")
    prediction_path = baseline.safe_path(
        root, f"artifacts/predictions/v3_final_15/{run_id}/oof.csv"
    )
    preflight_path = report_dir / "preflight.json"
    if report_dir.exists() and not resume:
        raise FileExistsError("Run exists. Use --resume after identity checks, or a new run ID.")
    policy = load_search_policy(root)
    config, frame, assignments, reference, identity = prepare_run(root, policy)
    identity_hash = digest(identity)
    summary_path = baseline.safe_path(root, "artifacts/reports/v3_final_15/summary.md")
    if preflight_path.exists():
        preflight = read_json(preflight_path)
        if preflight["identity_sha256"] != identity_hash or preflight["identity"] != identity:
            raise ValueError("Resume rejected: config/data/folds/source/environment changed.")
    else:
        preflight = {
            "created_at_utc": datetime.now(UTC).isoformat(),
            "run_id": run_id,
            "identity_sha256": identity_hash,
            "identity": identity,
            "summary_before": summary_path.read_text(encoding="utf-8"),
            "summary_before_sha256": sha256_file(summary_path),
            "git": {
                name: subprocess.check_output(["git", *args], cwd=root, text=True).strip()
                for name, args in {
                    "head": ["rev-parse", "HEAD"],
                    "branch": ["branch", "--show-current"],
                    "status": ["status", "--short"],
                }.items()
            },
        }
        baseline.write_json(preflight_path, preflight)
    print(
        f"Preflight passed: {len(frame)} rows; 20 LR / 40 RF candidates per outer fold.", flush=True
    )
    checkpoints, checkpoint_hashes = [], {}
    for model in MODELS:
        for fold in range(1, 6):
            path = report_dir / "checkpoints" / f"{model}_fold_{fold}.json"
            if path.exists():
                result = load_checkpoint(path, identity_hash)
                if result["model"] != model or result["fold"] != fold:
                    raise ValueError("Checkpoint model/fold mismatch.")
                print(f"Reused verified checkpoint: {model} fold {fold}", flush=True)
            else:
                print(f"Fitting {model} outer fold {fold}/5", flush=True)
                result = fit_outer(frame, assignments, config, policy, model, fold)
                save_checkpoint(path, identity_hash, result)
                print(
                    f"Saved {model} fold {fold}: {result['elapsed_seconds']:.1f}s; "
                    f"warnings={sum(result['warnings'].values())}",
                    flush=True,
                )
            checkpoints.append(result)
            checkpoint_hashes[str(path.relative_to(root)).replace("\\", "/")] = sha256_file(path)
    oof = pd.DataFrame([row for c in checkpoints for row in c["predictions"]])
    oof = aligned_oof(oof, assignments, list(MODELS))
    comparison = compare(reference, oof, assignments)
    verify_hashes(root, identity["protected_sha256"])
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    if prediction_path.exists():
        existing = aligned_oof(pd.read_csv(prediction_path), assignments, list(MODELS))
        pd.testing.assert_frame_equal(existing, oof, check_exact=False, rtol=1e-14, atol=1e-15)
    else:
        oof.to_csv(prediction_path, index=False, mode="x")
    # Always check the saved CSV, not only the in-memory frame.
    roundtrip = aligned_oof(pd.read_csv(prediction_path), assignments, list(MODELS))
    pd.testing.assert_frame_equal(roundtrip, oof, check_exact=False, rtol=1e-14, atol=1e-15)
    preserve_json(report_dir / "metrics_comparison.json", comparison, resume)
    completion = {
        "identity_sha256": identity_hash,
        "preflight_sha256": sha256_file(preflight_path),
        "checkpoint_sha256": checkpoint_hashes,
        "warnings": [c["warnings"] for c in checkpoints],
        "inner_fits": sum(c["inner_fits"] for c in checkpoints),
        "outer_refits": len(checkpoints),
        "full_development_fits": 0,
        "output_sha256": {
            str(p.relative_to(root)).replace("\\", "/"): sha256_file(p)
            for p in (prediction_path, report_dir / "metrics_comparison.json")
        },
        "rows_per_model": oof.groupby("model").size().to_dict(),
        "protected_hashes_unchanged": True,
        "oof_csv_roundtrip": "passed",
        "final_test_accessed": False,
        "model_serialized": False,
    }
    append_summary(summary_path, preflight["summary_before"], run_id, comparison, checkpoints)
    preserve_json(report_dir / "metadata.json", completion, resume)
    print(
        f"Completed {run_id}: {completion['inner_fits']} inner fits + 10 outer refits.", flush=True
    )
