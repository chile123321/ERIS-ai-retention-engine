"""Development-only, non-destructive pre-API audit of the frozen V3 b2 bundle.

This command deliberately never opens the final holdout or f1 prediction files.
It creates one new audit directory and refuses to reuse an existing directory.
"""

from __future__ import annotations

import argparse
import copy
import importlib.metadata
import json
import math
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time
import tracemalloc
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from threadpoolctl import threadpool_limits

from eris_ml.models import remediation_v3 as study
from eris_ml.models.bundle_v3_b2 import (
    FIXED_PARAMETERS,
    MODEL_VERSION,
    explain_one,
    load_bundle,
    predict_batch,
    predict_one,
)
from eris_ml.utils.hashing import sha256_file

ROOT = Path(__file__).resolve().parents[1]
BASE = Path("artifacts/reports/v3_final_15")
BUNDLE = ROOT / "artifacts/models/v3_final_15/b2"
FORBIDDEN = ("final_test_raw_v1.csv", "/f1/predictions", "/f1/targets")


def safe(relative: str) -> Path:
    normalized = "/" + relative.replace("\\", "/").lower().strip("/")
    if (
        any(part in normalized for part in FORBIDDEN)
        or "/predictions/v3_final_15/f1/" in normalized
    ):
        raise ValueError("Final-holdout and f1 employee-level inputs are forbidden.")
    path = (ROOT / relative).resolve()
    if not path.is_relative_to(ROOT.resolve()):
        raise ValueError("Path escapes repository.")
    return path


def read_json(relative: str) -> dict[str, Any]:
    return json.loads(safe(relative).read_text(encoding="utf-8"))


def digest(relative: str) -> str:
    return sha256_file(safe(relative))


def write_json(directory: Path, name: str, value: Any) -> None:
    (directory / name).write_text(json.dumps(value, indent=2, default=str) + "\n", encoding="utf-8")


def run_command(argv: list[str], timeout: int = 600) -> dict[str, Any]:
    start = time.perf_counter()
    result = subprocess.run(
        argv, cwd=ROOT, capture_output=True, text=True, timeout=timeout, check=False
    )
    return {
        "command": argv,
        "exit_code": result.returncode,
        "seconds": round(time.perf_counter() - start, 3),
        "stdout_tail": result.stdout[-9000:],
        "stderr_tail": result.stderr[-3000:],
    }


def lineage() -> dict[str, Any]:
    records: dict[str, Any] = {}
    for stage in ("r1", "t1", "c1", "b1", "devfix1", "b2"):
        meta_name = "manifest.json" if stage == "devfix1" else "metadata.json"
        meta_path = f"{BASE.as_posix()}/{stage}/{meta_name}"
        meta = read_json(meta_path)
        hashes = meta.get("output_sha256", {})
        if stage == "r1":
            hashes = {meta["outputs"][key]: value for key, value in meta["outputs_sha256"].items()}
        mismatch = [path for path, value in hashes.items() if digest(path) != value]
        if mismatch:
            raise ValueError(f"{stage} historical output hash mismatch: {mismatch}")
        records[stage] = {
            "metadata": meta_path,
            "metadata_sha256": digest(meta_path),
            "verified_output_count": len(hashes),
            "output_hashes_match": True,
        }
    b2pre = read_json(f"{BASE.as_posix()}/b2/preflight.json")
    mismatch = [p for p, h in b2pre["protected_sha256"].items() if digest(p) != h]
    if mismatch:
        raise ValueError(f"B2 protected input changed: {mismatch}")
    records["b2"]["protected_input_count"] = len(b2pre["protected_sha256"])
    ledger = read_json(f"{BASE.as_posix()}/final_evaluation_consumed.json")
    f1status = read_json(f"{BASE.as_posix()}/f1/final_evaluation_status.json")
    f1freeze = read_json(f"{BASE.as_posix()}/f1/pre_evaluation_freeze.json")
    f1manifest = read_json(f"{BASE.as_posix()}/f1/output_manifest.json")
    if not ledger.get("run_consumed") or ledger.get("model_version") == MODEL_VERSION:
        raise ValueError("f1 consumed ledger does not preserve b1/b2 distinction.")
    records["f1_metadata_only"] = {
        "ledger": str(BASE / "final_evaluation_consumed.json"),
        "ledger_sha256": digest(f"{BASE.as_posix()}/final_evaluation_consumed.json"),
        "model_version": ledger.get("model_version"),
        "run_consumed": ledger.get("run_consumed"),
        "status_keys": sorted(f1status),
        "freeze_keys": sorted(f1freeze),
        "manifest_keys": sorted(f1manifest),
        "employee_level_f1_outputs_opened": False,
        "f1_output_integrity_rechecked": False,
    }
    return records


def metrics(block: pd.DataFrame) -> dict[str, float]:
    y = block.y_true.to_numpy(dtype=int)
    p = block.probability.to_numpy(dtype=float)
    return {
        "average_precision": float(average_precision_score(y, p)),
        "roc_auc": float(roc_auc_score(y, p)),
        "brier_score": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "mean_probability": float(np.mean(p)),
        "prevalence": float(np.mean(y)),
    }


def development_checks(frame: pd.DataFrame, assignments: pd.DataFrame) -> dict[str, Any]:
    if len(frame) != 1176 or frame.Attrition.value_counts().to_dict() != {0: 986, 1: 190}:
        raise ValueError("Development population differs from locked trace.")
    if frame.source_row.duplicated().any() or frame.EmployeeNumber.duplicated().any():
        raise ValueError("Development IDs duplicate.")
    if (
        not frame[["source_row", "EmployeeNumber", "Attrition"]]
        .reset_index(drop=True)
        .equals(assignments[["source_row", "EmployeeNumber", "Attrition"]].reset_index(drop=True))
    ):
        raise ValueError("Development/assignment ID or target alignment failed.")
    if set(assignments.fold) != {1, 2, 3, 4, 5}:
        raise ValueError("Saved fold IDs differ.")
    for fold in range(1, 6):
        train = set(assignments.loc[assignments.fold != fold, "source_row"])
        valid = set(assignments.loc[assignments.fold == fold, "source_row"])
        if train & valid or not valid:
            raise ValueError("Outer train/validation overlap.")
    return {
        "rows": len(frame),
        "target_distribution": {"0": 986, "1": 190},
        "unique_source_row": True,
        "unique_employee_number": True,
        "fold_sizes": {
            str(k): int(v) for k, v in assignments.fold.value_counts().sort_index().items()
        },
        "outer_train_validation_disjoint": True,
    }


def recompute_devfix(assignments: pd.DataFrame) -> dict[str, Any]:
    prefix = "artifacts/predictions/v3_final_15/devfix1/"
    original = pd.read_csv(safe(prefix + "oof.csv"))
    stress = pd.read_csv(safe(prefix + "optional_missing_oof.csv"))
    all_oof = study.validate_oof(pd.concat([original, stress], ignore_index=True), assignments)
    historical_metrics = read_json(f"{BASE.as_posix()}/devfix1/metrics.json")
    historical_top = read_json(f"{BASE.as_posix()}/devfix1/top_k.json")
    historical_optional = read_json(f"{BASE.as_posix()}/devfix1/optional_missing.json")
    maximum = 0.0
    checked = 0
    summaries: dict[str, Any] = {}
    for model, group in all_oof.groupby("model", sort=True):
        base = group.loc[group.scenario == "original"]
        scores = metrics(base)
        for key in ("average_precision", "roc_auc", "brier_score", "log_loss"):
            maximum = max(maximum, abs(scores[key] - historical_metrics[model]["pooled_oof"][key]))
            checked += 1
        for fold, rows in base.groupby("fold"):
            calculated = metrics(rows)
            old = next(x for x in historical_metrics[model]["folds"] if x["fold"] == fold)
            for key in ("average_precision", "roc_auc", "brier_score", "log_loss"):
                maximum = max(maximum, abs(calculated[key] - old[key]))
                checked += 1
        capacities: dict[str, Any] = {}
        for percent in (5, 10, 15):
            flag = base[f"top_{percent}"].to_numpy(dtype=bool)
            y = base.y_true.to_numpy(dtype=int)
            for _, fold_rows in base.groupby("fold"):
                saved = fold_rows[f"top_{percent}"].to_numpy(dtype=bool)
                ranked = np.lexsort(
                    (fold_rows.source_row.to_numpy(), -fold_rows.probability.to_numpy())
                )
                independently_selected = np.zeros(len(fold_rows), dtype=bool)
                independently_selected[ranked[: int(len(fold_rows) * percent / 100)]] = True
                if not np.array_equal(saved, independently_selected):
                    raise ValueError(f"Top-k rounding/tie policy mismatch: {model} {percent}")
            tp, fp = int(np.sum(flag & (y == 1))), int(np.sum(flag & (y == 0)))
            old = next(x for x in historical_top[model] if x["capacity_percent"] == percent)[
                "pooled"
            ]
            if (tp, fp, int(sum(flag))) != (old["tp"], old["fp"], old["alerts"]):
                raise ValueError(f"Top-k count mismatch: {model} {percent}")
            capacities[str(percent)] = {"alerts": int(sum(flag)), "tp": tp, "fp": fp}
            checked += 3
        for scenario, rows in group.groupby("scenario"):
            if scenario == "original":
                continue
            current = metrics(rows)
            old = historical_optional[model][scenario]["pooled"]["probability_metrics"]
            for key in ("average_precision", "roc_auc", "brier_score", "log_loss"):
                maximum = max(maximum, abs(current[key] - old[key]))
                checked += 1
        summaries[model] = {"pooled_oof": scores, "top_k": capacities}
    if maximum > 1e-12:
        raise ValueError(f"Development metric regression exceeds tolerance: {maximum}")
    return {
        "oof_rows": len(all_oof),
        "candidates": len(summaries),
        "scenarios": sorted(all_oof.scenario.unique().tolist()),
        "metric_values_compared": checked,
        "maximum_abs_metric_difference": maximum,
        "summaries": summaries,
    }


def fairness(frame: pd.DataFrame, assignments: pd.DataFrame) -> dict[str, Any]:
    oof = pd.read_csv(safe("artifacts/predictions/v3_final_15/devfix1/oof.csv"))
    base = oof.loc[oof.model.isin(["rf_unweighted_raw", "rf_unweighted_no_age_raw"])].copy()
    base = base.merge(frame[["source_row", "Age"]], on="source_row", validate="many_to_one")
    bins = [-1, 29, 39, 49, np.inf]
    base["AgeGroup"] = pd.cut(base.Age, bins=bins, labels=["<30", "30-39", "40-49", ">=50"])
    historical = pd.read_csv(safe(f"{BASE.as_posix()}/devfix1/age_subgroups.csv"))
    result: dict[str, Any] = {}
    for model, model_rows in base.groupby("model"):
        groups: dict[str, Any] = {}
        for age_group, rows in model_rows.groupby("AgeGroup", observed=True):
            flag = rows.top_10.to_numpy(dtype=bool)
            y = rows.y_true.to_numpy(dtype=int)
            tp, fp = int(np.sum(flag & (y == 1))), int(np.sum(flag & (y == 0)))
            pos, neg = int(sum(y)), int(sum(y == 0))
            old = historical.loc[
                (historical.model == model)
                & (historical.attribute == "AgeGroup")
                & (historical.group == str(age_group))
                & (historical.fold == 0)
            ]
            if len(old) != 1 or (len(rows), pos, tp, fp) != tuple(
                int(old.iloc[0][key]) for key in ("n", "positives", "tp", "fp")
            ):
                raise ValueError(f"Age group aggregate mismatch: {model} {age_group}")
            groups[str(age_group)] = {
                "n": len(rows),
                "positives": pos,
                "negatives": neg,
                "alerts": int(sum(flag)),
                "tp": tp,
                "fp": fp,
                "fn": pos - tp,
                "tn": neg - fp,
                "tpr": tp / pos if pos else None,
                "fpr": fp / neg if neg else None,
                "precision": tp / int(sum(flag)) if sum(flag) else None,
                "pooled_support_sufficient": len(rows) >= 50 and pos >= 10 and neg >= 10,
                "folds_with_minimum_support": int(
                    sum(
                        len(part) >= 50
                        and sum(part.y_true == 1) >= 10
                        and sum(part.y_true == 0) >= 10
                        for _, part in rows.groupby("fold")
                    )
                ),
            }
        if sum(x["n"] for x in groups.values()) != len(assignments):
            raise ValueError("Age groups do not cover development.")
        result[model] = groups
    return {
        "age_groups": result,
        "protocol_support": {"n": 50, "positive": 10, "negative": 10},
        "age_sensitivity_trace_sha256": digest(f"{BASE.as_posix()}/devfix1/age_sensitivity.json"),
    }


def tamper_tests() -> dict[str, Any]:
    checked: dict[str, str] = {}
    for name in ("pipeline.joblib", "metadata.json", "background.npy", "feature_mapping.json"):
        with tempfile.TemporaryDirectory(prefix="eris_audit_tamper_") as tmp:
            location = Path(tmp) / "b2"
            shutil.copytree(BUNDLE, location)
            with (location / name).open("ab") as file:
                file.write(b"audit-tamper")
            try:
                load_bundle(location)
            except ValueError as exc:
                checked[name] = type(exc).__name__
            else:
                raise ValueError(f"Tampered {name} was accepted.")
    for mutation in ("missing_manifest", "extra_file"):
        with tempfile.TemporaryDirectory(prefix="eris_audit_tamper_") as tmp:
            location = Path(tmp) / "b2"
            shutil.copytree(BUNDLE, location)
            if mutation == "missing_manifest":
                (location / "manifest.json").unlink()
            else:
                (location / "unexpected.txt").write_text("x", encoding="utf-8")
            try:
                load_bundle(location)
            except (ValueError, FileNotFoundError, KeyError) as exc:
                checked[mutation] = type(exc).__name__
            else:
                raise ValueError(f"{mutation} was accepted.")
    return {
        "tampered_copies_rejected": checked,
        "actual_bundle_mutated": False,
        "warning": (
            "SHA-256 manifest gives integrity, not publisher authentication; joblib requires trust."
        ),
    }


def inference(bundle: Any, frame: pd.DataFrame) -> dict[str, Any]:
    names = bundle.metadata["feature_order"]
    if len(names) != 15 or set(names) & {
        "source_row",
        "EmployeeNumber",
        "Attrition",
        "Gender",
        "MaritalStatus",
        "JobRole",
    }:
        raise ValueError("B2 feature schema differs or contains identifier/target/audit field.")
    rng = np.random.default_rng(1209)
    positions = rng.choice(len(frame), size=500, replace=False)
    records = frame.iloc[positions][names].to_dict("records")
    original = copy.deepcopy(records)
    with threadpool_limits(limits=1):
        scores = np.array([x["risk_score"] for x in predict_batch(bundle, records)])
    if (
        len(scores) != 500
        or not np.isfinite(scores).all()
        or not np.all((0 <= scores) & (scores <= 1))
    ):
        raise ValueError("Valid-schema fuzz output invalid.")
    if records != original:
        raise ValueError("Inference mutated input records.")
    mutations: list[dict[str, Any]] = []
    seed = records[0]
    for source in records[:3]:
        for name in names:
            if name in bundle.metadata["required_features"]:
                mutations.append({k: v for k, v in source.items() if k != name})
            # NaN is intentionally accepted as missing for optional numeric inputs.
            bad_values = (float("inf"), "invalid")
            if name in bundle.metadata["required_features"]:
                bad_values += (float("nan"),)
            mutations.extend({**source, name: bad} for bad in bad_values)
    mutations.extend(
        [
            {**seed, "source_row": 1},
            {**seed, "Age": True},
            {**seed, "Age": 10**400},
            {**seed, "Department": "Unknown"},
        ]
    )
    outcomes: dict[str, int] = {}
    for mutation in mutations:
        try:
            predict_one(bundle, mutation)
        except Exception as exc:
            outcomes[type(exc).__name__] = outcomes.get(type(exc).__name__, 0) + 1
        else:
            raise ValueError("Invalid schema mutation accepted.")
    optional = bundle.metadata["optional_features"]
    optional_cases = [
        {k: v for k, v in seed.items() if k != optional[0]},
        {**seed, optional[0]: None},
        {k: v for k, v in seed.items() if k not in optional},
    ]
    for item in optional_cases:
        result = predict_one(bundle, item)
        if not math.isfinite(result["risk_score"]):
            raise ValueError("Optional missing input failed.")
    reversed_score = predict_one(bundle, dict(reversed(list(seed.items()))))["risk_score"]
    if abs(reversed_score - scores[0]) > 1e-12:
        raise ValueError("Key order changes prediction.")
    for size in (1, 10, 100):
        batch = np.array([p["risk_score"] for p in predict_batch(bundle, records[:size])])
        if not np.allclose(batch, scores[:size], atol=1e-12, rtol=1e-12):
            raise ValueError("Batch/single prediction inconsistency.")
    reordered = frame.iloc[positions[:10]][list(reversed(names))]
    reordered_scores = np.array([p["risk_score"] for p in predict_batch(bundle, reordered)])
    if not np.allclose(reordered_scores, scores[:10], atol=1e-12, rtol=1e-12):
        raise ValueError("DataFrame column order changes probability.")
    special_cases = {
        "empty_batch": [],
        "duplicate_dataframe_columns": pd.DataFrame(
            [[seed[n] for n in names] + [seed[names[0]]]], columns=names + [names[0]]
        ),
        "category_whitespace": {**seed, "Department": "Sales "},
        "category_unicode": {**seed, "OverTime": "Ｎｏ"},
        "numeric_string": {**seed, "Age": str(seed["Age"])},
        "negative_numeric": {**seed, "Age": -1},
        "employee_metadata": {**seed, "employee_id": "synthetic"},
    }
    rejected_special: dict[str, str] = {}
    for label, payload in special_cases.items():
        try:
            predict_batch(bundle, payload)
        except Exception as exc:
            rejected_special[label] = type(exc).__name__
        else:
            raise ValueError(f"Invalid special case accepted: {label}")
    if predict_one(bundle, seed)["risk_score"] != predict_one(bundle, seed)["risk_score"]:
        raise ValueError("Repeated inference differs.")
    return {
        "feature_order": names,
        "valid_fuzz_records": 500,
        "invalid_mutations": len(mutations),
        "invalid_rejection_exceptions": outcomes,
        "optional_missing_cases": 3,
        "batch_sizes": [1, 10, 100, 500],
        "key_order_invariant": True,
        "dataframe_order_invariant": True,
        "special_cases_rejected": rejected_special,
        "input_not_mutated": True,
        "probability_range": [float(scores.min()), float(scores.max())],
        "sample_source": "fixed-seed development rows, not holdout",
    }


def explain_checks(bundle: Any, frame: pd.DataFrame) -> dict[str, Any]:
    rng = np.random.default_rng(2718)
    names = bundle.metadata["feature_order"]
    records = frame.iloc[rng.choice(len(frame), 20, replace=False)][names].to_dict("records")
    pool = frame.iloc[:100][names]
    ranks = np.argsort(bundle.pipeline.predict_proba(pool)[:, 1])
    records += pool.iloc[ranks[[0, len(ranks) // 2, -1]]].to_dict("records")
    optional = bundle.metadata["optional_features"]
    records += [{k: v for k, v in records[0].items() if k != feature} for feature in optional]
    records += [{k: v for k, v in records[0].items() if k not in optional}]
    records += [{**records[0], optional[0]: None, optional[1]: None}]
    errors = []
    for record in records:
        result = explain_one(bundle, record)
        contributions = result["contributions"]
        if set(contributions) != set(names):
            raise ValueError("SHAP contribution names differ from original feature order.")
        error = abs(result["base_value"] + sum(contributions.values()) - result["risk_score"])
        if error > 1e-6:
            raise ValueError("SHAP local additivity failed.")
        errors.append(error)
    return {
        "cases": len(records),
        "score_extremes": "low/middle/high from fixed 100 development rows",
        "max_absolute_additivity_error": max(errors),
        "positive_class": bundle.metadata["positive_class"],
        "interpretation": "model behavior only; not causal or HR advice",
    }


def reproducibility(bundle: Any, config: dict[str, Any], frame: pd.DataFrame) -> dict[str, Any]:
    names = bundle.metadata["feature_order"]
    X = frame[names]
    params = bundle.metadata["resolved_model_parameters"]
    if any(params.get(k) != value for k, value in FIXED_PARAMETERS.items()):
        raise ValueError("Saved RF fixed parameters differ.")
    with (
        tempfile.TemporaryDirectory(prefix="eris_audit_refit_") as tmp,
        threadpool_limits(limits=1),
    ):
        fresh = study.pipeline_for(config, params, "rf_unweighted")
        fresh.fit(X, frame.Attrition.to_numpy())
        probe = X.iloc[[0, 1, 2, 35, 120, 500, 1175]]
        old = bundle.pipeline.predict_proba(probe)[:, 1]
        new = fresh.predict_proba(probe)[:, 1]
        difference = float(np.max(np.abs(old - new)))
        if difference > 1e-12:
            raise ValueError(f"Fixed refit prediction regression: {difference}")
        old_imputer = (
            bundle.pipeline.named_steps["preprocessor"]
            .named_transformers_["optional_numeric"]
            .named_steps["imputer"]
        )
        new_imputer = (
            fresh.named_steps["preprocessor"]
            .named_transformers_["optional_numeric"]
            .named_steps["imputer"]
        )
        np.testing.assert_array_equal(old_imputer.statistics_, new_imputer.statistics_)
        old_encoder = bundle.pipeline.named_steps["preprocessor"].named_transformers_["categorical"]
        new_encoder = fresh.named_steps["preprocessor"].named_transformers_["categorical"]
        for old_categories, new_categories in zip(
            old_encoder.categories_, new_encoder.categories_, strict=True
        ):
            np.testing.assert_array_equal(old_categories, new_categories)
        path = Path(tmp) / "pipeline.joblib"
        joblib.dump(fresh, path)
        temporary_sha = sha256_file(path)
    return {
        "temporary_refit_only": True,
        "fit_rows": len(X),
        "probe_rows": len(probe),
        "max_abs_probability_difference": difference,
        "optional_medians_equal": True,
        "encoder_categories_equal": True,
        "temporary_binary_sha256": temporary_sha,
        "original_binary_sha256": sha256_file(BUNDLE / "pipeline.joblib"),
        "temporary_directory_removed": True,
        "official_bundle_overwritten": False,
        "in_sample_performance_reported": False,
    }


def performance(bundle: Any, frame: pd.DataFrame) -> dict[str, Any]:
    names = bundle.metadata["feature_order"]
    records = frame.iloc[:100][names].to_dict("records")
    operations = {
        "load_bundle": lambda: load_bundle(BUNDLE),
        "predict_one": lambda: predict_one(bundle, records[0]),
        "predict_batch_10": lambda: predict_batch(bundle, records[:10]),
        "predict_batch_100": lambda: predict_batch(bundle, records),
        "explain_one": lambda: explain_one(bundle, records[0]),
    }
    output: dict[str, Any] = {}
    with threadpool_limits(limits=1):
        for name, action in operations.items():
            action()
            times = []
            for _ in range(30):
                start = time.perf_counter()
                action()
                times.append(time.perf_counter() - start)
            tracemalloc.start()
            action()
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            output[name] = {
                "iterations": 30,
                "median_seconds": float(np.median(times)),
                "p95_seconds": float(np.percentile(times, 95)),
                "minimum_seconds": min(times),
                "maximum_seconds": max(times),
                "python_tracemalloc_peak_bytes": peak,
            }
        expected = [predict_one(bundle, row)["risk_score"] for row in records[:4]]
        start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=4) as pool:
            actual = list(pool.map(lambda row: predict_one(bundle, row)["risk_score"], records[:4]))
        if not np.allclose(actual, expected, atol=1e-12, rtol=1e-12):
            raise ValueError("Concurrent one-row inference changed output.")
        output["concurrent_four_threads"] = {
            "cases": 4,
            "seconds": time.perf_counter() - start,
            "same_as_sequential": True,
        }
        starts = (0, 10, 20, 30)
        expected_batch = [
            [item["risk_score"] for item in predict_batch(bundle, records[i : i + 10])]
            for i in starts
        ]
        start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=4) as pool:
            actual_batch = list(
                pool.map(
                    lambda i: [
                        item["risk_score"] for item in predict_batch(bundle, records[i : i + 10])
                    ],
                    starts,
                )
            )
        if not np.allclose(actual_batch, expected_batch, atol=1e-12, rtol=1e-12):
            raise ValueError("Concurrent batch inference changed output.")
        output["concurrent_four_batches"] = {
            "batches": 4,
            "size_each": 10,
            "seconds": time.perf_counter() - start,
            "same_as_sequential": True,
        }
    previous = read_json(f"{BASE.as_posix()}/b2/performance.json")["measurements"]
    comparisons = {}
    for key in ("predict_one", "predict_batch_100", "explain_one"):
        old_median = previous[key]["median_seconds"]
        ratio = output[key]["median_seconds"] / old_median
        comparisons[key] = {
            "b2_median_seconds": old_median,
            "ratio": ratio,
            "over_twice": ratio > 2,
        }
    return {
        "measurements": output,
        "comparison_to_b2_report": comparisons,
        "SLA_approved": False,
        "cpu_count": os.cpu_count(),
        "rf_n_jobs": FIXED_PARAMETERS["n_jobs"],
        "memory_note": "tracemalloc measures Python allocations, not full process RSS",
        "cpu_oversubscription_note": "threadpool_limits(1); RF n_jobs=4 remains in saved model",
    }


def documents() -> dict[str, Any]:
    paths = [
        "README.md",
        "docs/v2_status.md",
        f"{BASE.as_posix()}/b2/model_card.md",
        f"{BASE.as_posix()}/summary.md",
    ]
    found = {p: safe(p).exists() for p in paths}
    readme = safe("README.md").read_text(encoding="utf-8")
    card = safe(f"{BASE.as_posix()}/b2/model_card.md").read_text(encoding="utf-8")
    return {
        "files": found,
        "readme_mentions_b2": "b2" in readme.lower(),
        "readme_stale_v3_undefined": "V3 remains undefined" in readme,
        "model_card_version": MODEL_VERSION in card,
        "model_card_no_production_approval": "No production or business approval" in card,
        "f1_applies_to_b1_only": "f1 evaluation applies only to b1" in card,
    }


def security() -> dict[str, Any]:
    scan = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True)
    tracked = scan.stdout.splitlines()
    sensitive_paths = [
        p
        for p in tracked
        if p.startswith("artifacts/predictions/") or p.endswith("final_test_raw_v1.csv")
    ]
    ignored = run_command(
        [
            "git",
            "check-ignore",
            "artifacts/predictions/v3_final_15/devfix1/oof.csv",
            "artifacts/models/v3_final_15/b2/pipeline.joblib",
        ],
        30,
    )
    path_scan = run_command(
        [
            "rg",
            "-l",
            "-i",
            "-g",
            "*.py",
            "-g",
            "*.md",
            "-g",
            "*.yaml",
            "api[_-]?key|password|token|database[_-]?url|C:\\\\Users\\\\|OneDrive",
            "src",
            "scripts",
            "tests",
            "docs",
            "configs",
            "README.md",
        ],
        60,
    )
    return {
        "tracked_employee_predictions_or_final_csv_paths": sensitive_paths,
        "artifact_ignore_check": ignored,
        "possible_sensitive_source_paths_only": path_scan["stdout_tail"].splitlines(),
        "source_scan_values_redacted": True,
        "joblib_trusted_local_only": True,
        "manifest_integrity_not_authentication": True,
        "f1_predictions_or_holdout_opened": False,
        "partial_write_risk": (
            "save_bundle writes files before manifest; loader rejects incomplete directory"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="audit_b2_1")
    args = parser.parse_args()
    if not re.fullmatch(r"audit_b2_[1-9][0-9]*", args.run_id):
        raise ValueError("Use a versioned audit_b2_N run ID.")
    output = safe(f"{BASE.as_posix()}/{args.run_id}")
    if output.exists():
        raise FileExistsError(f"Audit output already exists: {output}")
    print("Preflight lineage and development provenance", flush=True)
    git = {
        name: run_command(["git", *argv], 30)["stdout_tail"].strip()
        for name, argv in {
            "branch": ["branch", "--show-current"],
            "head": ["rev-parse", "HEAD"],
            "status_before": ["status", "--short"],
        }.items()
    }
    lineage_result = lineage()
    config, frame, assignments, _, identity = study.prepare(ROOT)
    population = development_checks(frame, assignments)
    initial_sha = {
        p: digest(p)
        for p in (
            "data/processed/development_raw_v1.csv",
            "artifacts/predictions/v3_final_15/devfix1/oof.csv",
            "artifacts/predictions/v3_final_15/devfix1/optional_missing_oof.csv",
            "artifacts/models/v3_final_15/b2/pipeline.joblib",
        )
    }
    print("Recomputing saved development evidence", flush=True)
    evidence = recompute_devfix(assignments)
    age = fairness(frame, assignments)
    bundle = load_bundle(BUNDLE)
    print("Inference, manifest tamper and SHAP audit", flush=True)
    tamper = tamper_tests()
    fuzz = inference(bundle, frame)
    shap = explain_checks(bundle, frame)
    print("Temporary reproducibility refit (outside repository)", flush=True)
    replay = reproducibility(bundle, config, frame)
    print("30-iteration performance and concurrency audit", flush=True)
    perf = performance(bundle, frame)
    if {p: digest(p) for p in initial_sha} != initial_sha:
        raise ValueError("Protected data/output hash changed during audit.")
    print("Quality gates", flush=True)
    gates = {
        "pytest_full": run_command([sys.executable, "-m", "pytest", "-q"], 900),
        "mypy_strict_src": run_command([sys.executable, "-m", "mypy", "src"], 300),
        "ruff_repo": run_command([sys.executable, "-m", "ruff", "check", "."], 180),
        "pip_check": run_command([sys.executable, "-m", "pip", "check"], 180),
        "git_diff_check": run_command(["git", "diff", "--check"], 30),
        "compileall": run_command(
            [sys.executable, "-m", "compileall", "-q", "src/eris_ml", "scripts"], 180
        ),
    }
    cli_scripts = [
        "baseline_v3.py",
        "tune_v3.py",
        "evaluate_calibration_v3.py",
        "build_candidate_v3.py",
        "evaluate_final_v3.py",
        "remediate_v3.py",
        "build_candidate_v3_b2.py",
        "audit_v3_pre_api.py",
    ]
    gates["v3_cli_help"] = {
        name: run_command([sys.executable, f"scripts/{name}", "--help"], 60) for name in cli_scripts
    }
    if any(item["exit_code"] for item in gates["v3_cli_help"].values()):
        raise ValueError("One or more V3 CLI --help checks failed.")
    if {p: digest(p) for p in initial_sha} != initial_sha:
        raise ValueError("Protected data/output hash changed after quality gates.")
    environment = {
        "python": sys.version,
        "platform": platform.platform(),
        "executable": sys.executable,
        "packages": {
            name: (
                importlib.metadata.version(name)
                if importlib.metadata.packages_distributions()
                else None
            )
            for name in (
                "numpy",
                "pandas",
                "scikit-learn",
                "scipy",
                "shap",
                "joblib",
                "xgboost",
                "mypy",
                "ruff",
                "pytest",
            )
        },
        "python_311_runtime_tested": False,
        "source_identity_sha256": identity.get("identity_sha256"),
        "git": git,
    }
    findings = [
        {
            "id": "AUD-001",
            "severity": "P2",
            "status": "open",
            "summary": (
                "Extreme integer input raises OverflowError rather than a schema ValueError."
            ),
            "evidence": fuzz["invalid_rejection_exceptions"],
            "recommendation": (
                "Normalize validator error at future versioned interface; "
                "preserve b2 protected source now."
            ),
        },
        {
            "id": "AUD-002",
            "severity": "P2",
            "status": "open_environment",
            "summary": (
                "Optional Streamlit development dependency absent; "
                "strict mypy src reports import-not-found."
            ),
            "evidence": gates["mypy_strict_src"]["stdout_tail"],
            "recommendation": (
                "Install declared development extra in clean environment and rerun gate; "
                "do not suppress typing."
            ),
        },
        {
            "id": "AUD-003",
            "severity": "P2",
            "status": "fixed",
            "summary": "README V3 status was stale; updated to distinguish b2 from consumed b1 f1.",
        },
        {
            "id": "AUD-004",
            "severity": "P2",
            "status": "fixed",
            "summary": (
                "Typing boundary test required only arg-type diagnostic; "
                "accepts call-overload rejection."
            ),
        },
        {
            "id": "AUD-005",
            "severity": "P3",
            "status": "limitation",
            "summary": (
                "Manifest proves local integrity, not authenticity; "
                "joblib deserialization requires trusted origin."
            ),
        },
        {
            "id": "AUD-006",
            "severity": "P2",
            "status": "limitation",
            "summary": (
                "b2 lacks independent final or external validation; "
                "f1 evaluation belongs only to b1."
            ),
        },
    ]
    finding_details = {
        "AUD-001": (
            ["src/eris_ml/features/schema_v3.py", "src/eris_ml/models/bundle_v3_b2.py"],
            "Very large integers are rejected but may surface an unhelpful exception to callers.",
            "Run audit_v3_pre_api.py; inspect security_review.json inference exception counts.",
        ),
        "AUD-002": (
            ["pyproject.toml", "src/eris_ml/manual_ui/v2_streamlit.py"],
            "The full-source typing gate fails in this local environment; "
            "core b2 inference passes.",
            "Run .venv\\Scripts\\python.exe -m mypy src.",
        ),
        "AUD-003": (
            ["README.md"],
            "Readers could mistake V3 as undefined or apply b1 final evidence to b2.",
            "Read README.md and compare to b2 metadata and f1 consumed ledger.",
        ),
        "AUD-004": (
            ["tests/unit/test_typing_boundary.py"],
            "A diagnostic-code wording change failed a valid typing rejection test.",
            "Run .venv\\Scripts\\python.exe -m pytest tests/unit/test_typing_boundary.py -q.",
        ),
        "AUD-005": (
            ["src/eris_ml/models/bundle_v3_b2.py", "artifacts/models/v3_final_15/b2/manifest.json"],
            "A correctly hashed but untrusted joblib can still execute code when loaded.",
            "Inspect load_bundle and compare a tampered copy using the audit script.",
        ),
        "AUD-006": (
            ["artifacts/reports/v3_final_15/b2/model_card.md"],
            "Development post-selection evidence cannot establish new-employer performance.",
            "Read b2 model_card.md and f1 final_evaluation_status.json metadata only.",
        ),
    }
    for finding in findings:
        affected, impact, command = finding_details[finding["id"]]
        finding.update(affected_files=affected, user_impact=impact, reproduction=command)
    exit_status = (
        "READY_FOR_API_WITH_LIMITATIONS"
        if gates["pytest_full"]["exit_code"] == 0 and gates["ruff_repo"]["exit_code"] == 0
        else "BLOCKED"
    )
    report = [
        "# ERIS V3 b2 comprehensive pre-API audit",
        "",
        f"Run: `{args.run_id}` · UTC: {datetime.now(UTC).isoformat()} · Verdict: **{exit_status}**",
        "",
        "## Boundary and lineage",
        "",
        "Development only: 1,176 rows, 190 positives, five saved outer folds. "
        "No final-test CSV or f1 employee-level predictions/targets were opened or hashed. "
        "f1 metadata documents a consumed b1 evaluation; not independent evidence for b2.",
        "",
        "Historical output hashes verified across r1/t1/c1/b1/devfix1/b2; "
        f"b2 protected inputs: {lineage_result['b2']['protected_input_count']}. "
        "Original bundle SHA-256: "
        f"`{initial_sha['artifacts/models/v3_final_15/b2/pipeline.joblib']}`. "
        "Hashes were unchanged at exit.",
        "",
        "## Development evidence independently recomputed",
        "",
        f"{evidence['metric_values_compared']} saved metric values compared; largest "
        f"absolute difference {evidence['maximum_abs_metric_difference']:.3g}. "
        "RF unweighted raw OOF: "
        f"AP {evidence['summaries']['rf_unweighted_raw']['pooled_oof']['average_precision']:.6f}, "
        f"ROC-AUC {evidence['summaries']['rf_unweighted_raw']['pooled_oof']['roc_auc']:.6f}, "
        f"Brier {evidence['summaries']['rf_unweighted_raw']['pooled_oof']['brier_score']:.6f}. "
        "Post-selection development results, not new independent validation.",
        "",
        "## Inference, explanation and robustness",
        "",
        f"{fuzz['valid_fuzz_records']} valid records and {fuzz['invalid_mutations']} "
        "invalid mutations tested; all invalid inputs rejected. "
        f"Exception mix: {fuzz['invalid_rejection_exceptions']}. "
        f"Six tamper scenarios rejected on copies. {shap['cases']} SHAP cases passed; "
        f"max additivity error {shap['max_absolute_additivity_error']:.3g}.",
        "",
        f"Temporary fixed refit matched saved bundle on {replay['probe_rows']} fixed "
        f"development rows; max probability difference "
        f"{replay['max_abs_probability_difference']:.3g}. No in-sample performance claim. "
        "Temporary artifact was removed. Performance diagnostics are not an SLA.",
        "",
        "## Fairness and policy limits",
        "",
        "AgeGroup aggregates and top-k 10% were checked against saved development artifacts. "
        "Small older-age subgroups have weak per-fold support. No fairness certification, "
        "HR policy approval, transfer validation, or 3–6 month forecast claim follows "
        "from IBM data.",
        "",
        "## Quality gate",
        "",
    ]
    for name, result in gates.items():
        if name == "v3_cli_help":
            report.append(f"- `{name}`: {len(result)} scripts exited 0")
        else:
            report.append(f"- `{name}` exit {result['exit_code']}: `{' '.join(result['command'])}`")
    report.extend(["", "## Findings", ""])
    for finding in findings:
        report.append(
            f"- {finding['id']} {finding['severity']} ({finding['status']}): {finding['summary']}"
        )
    report.extend(
        [
            "",
            "## Decision",
            "",
            "Research/demo review may proceed with limitations; API/backend design must "
            "wait for explicit owner coordination, HR/data contract decisions and "
            "validation plan. No production approval.",
            "",
        ]
    )
    output.mkdir(parents=True, exist_ok=False)
    write_json(
        output,
        "artifact_lineage.json",
        {
            "lineage": lineage_result,
            "population": population,
            "protected_sha256_before_after": initial_sha,
        },
    )
    write_json(output, "environment.json", environment)
    write_json(output, "test_results.json", gates)
    write_json(output, "performance.json", perf)
    write_json(output, "reproducibility.json", replay)
    write_json(
        output,
        "security_review.json",
        {"tamper": tamper, "inference": fuzz, "explanations": shap, "repository": security()},
    )
    write_json(output, "documentation_consistency.json", documents())
    write_json(output, "findings.json", {"verdict": exit_status, "findings": findings})
    write_json(output, "development_evidence.json", {"metrics": evidence, "age_audit": age})
    (output / "audit_report.md").write_text("\n".join(report), encoding="utf-8")
    files = [p for p in output.iterdir() if p.is_file()]
    write_json(
        output,
        "audit_manifest.json",
        {
            "run_id": args.run_id,
            "verdict": exit_status,
            "created_utc": datetime.now(UTC).isoformat(),
            "output_sha256": {p.name: sha256_file(p) for p in files},
            "self_hash_excluded": True,
            "final_test_opened_or_hashed": False,
            "historical_protected_hashes_unchanged": True,
        },
    )
    print(f"Audit complete: {output} ({exit_status})", flush=True)


if __name__ == "__main__":
    main()
