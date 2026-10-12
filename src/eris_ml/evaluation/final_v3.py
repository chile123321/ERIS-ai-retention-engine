"""Explicitly authorized, one-time frozen V3 evaluation; no fitting or model changes."""

from __future__ import annotations

import hashlib
import io
import json
import subprocess
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

from eris_ml.evaluation.calibration_reporting import reliability_bins, validate_probabilities
from eris_ml.evaluation.threshold_policy_v2 import top_k
from eris_ml.models import baseline_v3 as baseline
from eris_ml.models import tuning_v3 as tuning
from eris_ml.models.bundle_v2 import write_exclusive
from eris_ml.models.bundle_v3 import CandidateBundle, load_bundle, predict_batch, versions
from eris_ml.models.training import normalize_binary_target
from eris_ml.utils.hashing import sha256_file

MODEL_VERSION = "rf-v3-final15-b1"
MODEL_SHA = "ac76d9a5df2e9d87aa20015fdf93cbee9ec4e9f222c79de5fec347148dc72039"
BUNDLE = "artifacts/models/v3_final_15/b1"
REPORT = "artifacts/reports/v3_final_15/f1"
PREDICTIONS = "artifacts/predictions/v3_final_15/f1/holdout_predictions.csv"
HOLDOUT = "data/processed/final_test_raw_v1.csv"
RAW = "data/raw/WA_Fn-UseC_-HR-Employee-Attrition.csv"
SUMMARY = "artifacts/reports/v3_final_15/summary.md"
CLAIM = "artifacts/reports/v3_final_15/final_evaluation_consumed.json"
PROBABILITY_METRICS = ("average_precision", "roc_auc", "brier_score", "log_loss")
AGE_GROUPS = ["<30", "30-39", "40-49", ">=50"]
PROTOCOL: dict[str, Any] = {
    "model_version": MODEL_VERSION,
    "expected_model_sha256": MODEL_SHA,
    "expected_rows": 294,
    "expected_target": {"0": 247, "1": 47},
    "bootstrap": {
        "iterations": 2000,
        "random_state": 44,
        "confidence": 0.95,
        "method": "stratified_positive_then_negative_percentile",
        "conditions_on": "fixed_model_fixed_class_counts_not_training_uncertainty",
    },
    "top_k": {
        "percent": [5, 10, 15],
        "batch": "entire_294_row_holdout",
        "rounding": "floor",
        "order": ["probability_desc", "source_row_asc"],
    },
    "threshold_reference": 0.5,
    "threshold_equality": "positive_if_probability_ge_threshold",
    "dummy_prior": "190/1176_development_prevalence_not_fitted_on_holdout",
    "reliability": {"bins": 5, "strategy": "quantile", "duplicates": "collapse"},
    "subgroup": {
        "age_groups": AGE_GROUPS,
        "minimum_n": 50,
        "minimum_positive": 10,
        "minimum_negative": 10,
        "unsupported_ap_brier": "null",
        "missing_audit_values": "explicit_Missing_group",
        "quota_changes": False,
    },
    "notable_degradation": {
        "absolute": 0.05,
        "higher_is_better": ["average_precision", "roc_auc"],
        "lower_is_better": ["brier_score", "log_loss"],
    },
    "evidence_interpretation": {
        "strong": "ranking lower CIs above prevalence/0.5, losses below dummy, all top-k lift>1",
        "weak": "ranking not above random, either loss not below dummy, or all top-k lift<=1",
        "moderate": "remaining mixed/uncertain cases; no production pass threshold",
        "limitation": "Descriptive rubric; previously used V1 holdout, not virgin V3 evidence",
    },
    "production_approved": False,
    "business_approved": False,
}


def now() -> str:
    return datetime.now(UTC).isoformat()


def path_at(root: Path, relative: str) -> Path:
    """Evaluation-only path boundary; development guards are deliberately unchanged."""
    relative_path = Path(relative)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise ValueError("Expected repository-relative evaluation path.")
    current = root
    for part in relative_path.parts:
        current /= part
        if current.is_symlink():
            raise ValueError("Symlink evaluation input/output rejected.")
    return current


def preflight(root: Path) -> tuple[CandidateBundle, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Do not stat, hash or read HOLDOUT here. Smoke uses the saved development probe."""
    bundle_path = path_at(root, BUNDLE)
    if sha256_file(bundle_path / "pipeline.joblib") != MODEL_SHA:
        raise ValueError("Frozen model SHA mismatch before final-target access.")
    bundle = load_bundle(bundle_path)
    if (
        bundle.metadata["model_version"] != MODEL_VERSION
        or bundle.metadata["evaluation_status"] != "development_only"
        or bundle.metadata["final_holdout_evaluated"] is not False
    ):
        raise ValueError("Wrong candidate identity/evaluation state.")
    b1 = path_at(root, "artifacts/reports/v3_final_15/b1")
    pre = tuning.read_json(b1 / "preflight.json")
    meta = tuning.read_json(b1 / "metadata.json")
    if (
        sha256_file(b1 / "preflight.json") != bundle.metadata["provenance"]["preflight_sha256"]
        or tuning.digest(pre["provenance"]) != pre["identity_sha256"]
    ):
        raise ValueError("b1 preflight identity mismatch.")
    protected = dict(pre["provenance"]["protected_sha256"])
    if len(protected) != 66:
        raise ValueError("Expected all 66 b1 protected hashes.")
    tuning.verify_hashes(root, protected)
    tuning.verify_hashes(root, meta["output_sha256"])
    protected.update(meta["output_sha256"])
    protected[(b1 / "metadata.json").relative_to(root).as_posix()] = sha256_file(
        b1 / "metadata.json"
    )
    probe = tuning.read_json(b1 / "reload_probe.json")
    actual = predict_batch(bundle, probe["records"])
    if not np.allclose(
        [r["probability"] for r in actual],
        [r["probability"] for r in probe["expected"]],
        atol=1e-12,
        rtol=1e-12,
    ) or any(
        a["imputed_features"] != b["imputed_features"]
        for a, b in zip(actual, probe["expected"], strict=True)
    ):
        raise ValueError("Saved development smoke regression failed.")
    config = bundle.metadata["feature_config"]
    development = pd.read_csv(path_at(root, config["data"]["development"]), usecols=baseline.IDS)
    development["Attrition"] = normalize_binary_target(development.Attrition)
    if (
        len(development) != 1176
        or development.Attrition.value_counts().to_dict() != {0: 986, 1: 190}
        or development[baseline.IDS[:2]].isna().any().any()
        or any(not development[k].is_unique for k in baseline.IDS[:2])
    ):
        raise ValueError("Development membership/prior mismatch.")
    manifest = pd.read_csv(path_at(root, config["data"]["manifest"]))
    if (
        len(manifest) != 1470
        or manifest[baseline.IDS[:2] + ["split"]].isna().any().any()
        or any(not manifest[k].is_unique for k in baseline.IDS[:2])
        or set(manifest.split) != {"development", "final_test"}
    ):
        raise ValueError("Invalid locked membership manifest.")
    historical = "artifacts/reports/logistic_general16_v2_r1_trace.json"
    raw_hash = tuning.read_json(path_at(root, historical))["protected_sha256"][RAW]
    if sha256_file(path_at(root, RAW)) != raw_hash:
        raise ValueError("Raw dataset hash differs from historical V2 trace.")
    ledger_path = "configs/release/final_evaluation_record_v1.yaml"
    ledger = yaml.safe_load(path_at(root, ledger_path).read_text("utf-8"))
    if ledger["run_count"] != 1 or ledger["status"] != "completed":
        raise ValueError("Historical V1 use ledger differs from established provenance.")
    for relative in (
        historical,
        RAW,
        ledger_path,
        "src/eris_ml/evaluation/final_v3.py",
        "src/eris_ml/evaluation/final_v3_reporting.py",
        "scripts/evaluate_final_v3.py",
        "tests/integration/test_final_v3.py",
        "src/eris_ml/models/training.py",
    ):
        protected[relative] = sha256_file(path_at(root, relative))
    summary = path_at(root, SUMMARY).read_bytes()
    freeze = {
        "created_at_utc": now(),
        "protocol": PROTOCOL,
        "authorization": "Explicit project-owner task: V3 step5 frozen b1 evaluation once",
        "library_versions": {
            **versions(),
            "matplotlib": version("matplotlib"),
            "PyYAML": version("PyYAML"),
        },
        "protected_sha256": protected,
        "verified_b1_protected_count": 66,
        "smoke": "passed",
        "bundle_manifest_sha256": sha256_file(bundle_path / "manifest.json"),
        "holdout_expected_sha256": ledger["final_test_sha256"],
        "raw_expected_sha256": raw_hash,
        "prior_holdout_use": {
            "candidate": ledger["candidate"],
            "completed_at_utc": ledger["completed_at_utc"],
            "not_virgin_independent_v3_holdout": True,
        },
        "summary_before_bytes": len(summary),
        "summary_before_sha256": hashlib.sha256(summary).hexdigest(),
        "git": {
            k: subprocess.check_output(["git", *v], cwd=root, text=True).strip()
            for k, v in {
                "head": ["rev-parse", "HEAD"],
                "branch": ["branch", "--show-current"],
                "status": ["status", "--short"],
            }.items()
        },
    }
    return bundle, development, manifest, freeze


def validate_holdout(
    frame: pd.DataFrame, development: pd.DataFrame, manifest: pd.DataFrame
) -> pd.DataFrame:
    if (
        frame.columns.has_duplicates
        or not set(baseline.IDS).issubset(frame.columns)
        or len(frame) != 294
        or frame[baseline.IDS].isna().any().any()
        or any(not frame[k].is_unique for k in baseline.IDS[:2])
    ):
        raise ValueError("Invalid final holdout rows/IDs/target.")
    result = frame.copy()
    result["Attrition"] = normalize_binary_target(frame.Attrition)
    if result.Attrition.value_counts().to_dict() != {0: 247, 1: 47}:
        raise ValueError("Final holdout class distribution mismatch.")
    for key in baseline.IDS[:2]:
        if not set(frame[key]).isdisjoint(development[key]):
            raise ValueError("Development/final ID overlap.")
    for split, block in (("development", development), ("final_test", result)):
        expected = manifest.loc[manifest.split == split, baseline.IDS[:2]].sort_values("source_row")
        actual = block[baseline.IDS[:2]].sort_values("source_row")
        if not actual.reset_index(drop=True).equals(expected.reset_index(drop=True)):
            raise ValueError("IDs do not match locked split membership.")
    return result


def probability_metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    y, p = validate_probabilities(y, p)
    if set(y) != {0, 1}:
        raise ValueError("Both classes required for ranking metrics.")
    return {
        "average_precision": float(average_precision_score(y, p)),
        "roc_auc": float(roc_auc_score(y, p)),
        "brier_score": float(brier_score_loss(y, p)),
        "log_loss": float(
            log_loss(y, np.clip(p, np.finfo(float).eps, 1 - np.finfo(float).eps), labels=[0, 1])
        ),
    }


def bootstrap(y: np.ndarray, p: np.ndarray, iterations: int = 2000) -> dict[str, Any]:
    rng = np.random.default_rng(44)
    positive, negative = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    if not len(positive) or not len(negative) or iterations < 1:
        raise ValueError("Stratified bootstrap requires both classes and iterations.")
    samples: dict[str, list[float]] = {key: [] for key in PROBABILITY_METRICS}
    for _ in range(iterations):
        indices = np.concatenate(
            [
                rng.choice(positive, len(positive), replace=True),
                rng.choice(negative, len(negative), replace=True),
            ]
        )
        for key, value in probability_metrics(y[indices], p[indices]).items():
            samples[key].append(value)
    return {
        "iterations": iterations,
        "random_state": 44,
        "method": PROTOCOL["bootstrap"],
        "intervals": {
            k: {"lower": float(np.quantile(v, 0.025)), "upper": float(np.quantile(v, 0.975))}
            for k, v in samples.items()
        },
    }


def operational(y: np.ndarray, selected: np.ndarray) -> dict[str, Any]:
    if len(y) != len(selected) or not set(y).issubset({0, 1}) or not set(selected).issubset({0, 1}):
        raise ValueError("Binary labels/policy required.")
    n, positives = len(y), int(sum(y))
    tp = int(sum((y == 1) & (selected == 1)))
    fp = int(sum((y == 0) & (selected == 1)))
    fn, tn, alerts = positives - tp, n - positives - fp, tp + fp
    return {
        "n": n,
        "positives": positives,
        "negatives": n - positives,
        "prevalence": positives / n if n else None,
        "alerts": alerts,
        "alert_rate": alerts / n if n else None,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": tp / alerts if alerts else None,
        "recall": tp / positives if positives else None,
        "fpr": fp / (n - positives) if n - positives else None,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
        "f2": 5 * tp / (5 * tp + fp + 4 * fn) if 5 * tp + fp + 4 * fn else None,
        "balanced_accuracy": 0.5 * (tp / positives + tn / (n - positives))
        if positives and n - positives
        else None,
    }


def summarize_predictions(predictions: pd.DataFrame, development_prior: float) -> dict[str, Any]:
    y, p = validate_probabilities(predictions.y_true.to_numpy(), predictions.probability.to_numpy())
    dummy = probability_metrics(y, np.full(len(y), development_prior))
    return {
        "model_version": MODEL_VERSION,
        "bundle_sha256": MODEL_SHA,
        "probability_metrics": probability_metrics(y, p),
        "prevalence": float(y.mean()),
        "mean_probability": float(p.mean()),
        "dummy_prior": development_prior,
        "dummy_metrics": dummy,
        "threshold_0_5_reference": operational(y, (p >= 0.5).astype(int)),
    }


def subgroup_table(predictions: pd.DataFrame) -> pd.DataFrame:
    frame = predictions.copy()
    frame["AgeGroup"] = pd.cut(
        frame.Age, [-np.inf, 30, 40, 50, np.inf], labels=AGE_GROUPS, right=False
    )
    records = []
    for attribute in ["AgeGroup", *[k for k in ("Gender", "MaritalStatus") if k in frame]]:
        groups = frame[attribute].astype("string").fillna("Missing")
        domain = AGE_GROUPS if attribute == "AgeGroup" else sorted(groups.unique())
        for group in domain:
            block = frame.loc[groups == group]
            y, p = block.y_true.to_numpy(), block.probability.to_numpy()
            metrics = operational(y, block.top_10.to_numpy())
            supported = len(y) >= 50 and sum(y) >= 10 and sum(y == 0) >= 10
            records.append(
                {
                    "attribute": attribute,
                    "group": group,
                    **metrics,
                    "tpr": metrics["recall"],
                    "low_support": not supported,
                    "average_precision": float(average_precision_score(y, p))
                    if supported
                    else None,
                    "brier_score": float(brier_score_loss(y, p)) if supported else None,
                }
            )
        block_records = records[-len(domain) :]
        if sum(r["n"] for r in block_records) != len(frame) or sum(
            r["alerts"] for r in block_records
        ) != int(frame.top_10.sum()):
            raise ValueError("Subgroup totals do not match entire holdout.")
    return pd.DataFrame(records)


def resume(root: Path) -> None:
    """Verify immutable outputs only. No holdout read, model load, prediction or scoring."""
    report = path_at(root, REPORT)
    manifest = tuning.read_json(report / "output_manifest.json")
    for relative, expected in manifest["sha256"].items():
        if sha256_file(path_at(root, relative)) != expected:
            raise ValueError(f"Existing evaluation output changed: {relative}")
    status = tuning.read_json(report / "final_evaluation_status.json")
    if (
        status["evaluation_run_id"] != "f1"
        or not status["final_holdout_evaluated"]
        or status["evaluated_bundle_sha256"] != MODEL_SHA
        or status["evaluated_model_version"] != MODEL_VERSION
    ):
        raise ValueError("Consumed evaluation status mismatch.")
    freeze = tuning.read_json(report / "pre_evaluation_freeze.json")
    tuning.verify_hashes(root, freeze["protected_sha256"])
    summary = path_at(root, SUMMARY).read_bytes()
    if (
        hashlib.sha256(summary[: manifest["summary_after_bytes"]]).hexdigest()
        != (manifest["summary_after_sha256"])
    ):
        raise ValueError("Final evaluation summary prefix changed.")
    print("Verified existing f1 bytes only; no inference, metric evaluation or holdout access.")


def run(root: Path, run_id: str = "f1", verify_only: bool = False) -> None:
    if run_id != "f1":
        raise ValueError("Only the explicitly authorized one-time f1 evaluation is allowed.")
    if verify_only:
        resume(root)
        return
    report, pred_path, claim = (
        path_at(root, REPORT),
        path_at(root, PREDICTIONS),
        path_at(root, CLAIM),
    )
    if report.exists() or pred_path.parent.exists() or claim.exists():
        raise FileExistsError("f1 outputs/consumption record exist; --resume is verification only.")
    bundle, development, membership, freeze = preflight(root)
    baseline.write_json(report / "pre_evaluation_freeze.json", freeze)
    # Exclusive claim precedes the only holdout read. A failure cannot silently start another run.
    baseline.write_json(
        claim,
        {
            "evaluation_run_id": "f1",
            "model_version": MODEL_VERSION,
            "model_sha256": MODEL_SHA,
            "access_started_at_utc": now(),
            "run_consumed": True,
            "freeze_sha256": sha256_file(report / "pre_evaluation_freeze.json"),
        },
    )
    print("Freeze verified and written; opening authorized locked holdout once.", flush=True)
    stage = "holdout_read"
    try:
        contents = path_at(root, HOLDOUT).read_bytes()
        if hashlib.sha256(contents).hexdigest() != freeze["holdout_expected_sha256"]:
            raise ValueError("Locked holdout bytes differ from historical V1 hash.")
        holdout = validate_holdout(pd.read_csv(io.BytesIO(contents)), development, membership)
        stage = "inference"
        names = bundle.metadata["feature_order"]
        X = bundle.inputs(holdout[names])
        # Key ordering is validated without a second holdout prediction pass.
        pd.testing.assert_frame_equal(X, bundle.inputs(holdout[list(reversed(names))]))
        results = predict_batch(bundle, X)
        predictions = holdout[["source_row", "EmployeeNumber", "Age"]].copy()
        for column in ("Gender", "MaritalStatus"):
            if column in holdout:
                predictions[column] = holdout[column]
        predictions["y_true"] = holdout.Attrition.to_numpy()
        predictions["probability"] = [row["probability"] for row in results]
        predictions["imputed_features"] = [json.dumps(row["imputed_features"]) for row in results]
        predictions["model_version"] = MODEL_VERSION
        predictions["bundle_sha256"] = MODEL_SHA
        validate_probabilities(predictions.y_true.to_numpy(), predictions.probability.to_numpy())
        if len(results) != 294 or any(r["model_version"] != MODEL_VERSION for r in results):
            raise ValueError("Prediction length/model version mismatch.")
        for percent in (5, 10, 15):
            predictions[f"top_{percent}"] = top_k(
                predictions.probability.to_numpy(), predictions.source_row.to_numpy(), percent
            )
        write_exclusive(pred_path, predictions.to_csv(index=False).encode("utf-8"))
        status = {
            "final_holdout_evaluated": True,
            "evaluation_status": "final_holdout_evaluated",
            "evaluated_model_version": MODEL_VERSION,
            "evaluated_bundle_sha256": MODEL_SHA,
            "evaluated_row_count": 294,
            "evaluated_positive_count": 47,
            "evaluation_run_id": "f1",
            "timestamp": now(),
            "production_approved": False,
            "predictions_sha256": sha256_file(pred_path),
            "prior_v1_holdout_use_disclosed": True,
        }
        baseline.write_json(report / "final_evaluation_status.json", status)
        stage = "metrics_reporting"
        reloaded = pd.read_csv(pred_path, float_precision="round_trip")
        np.testing.assert_array_equal(
            reloaded.probability.to_numpy(), predictions.probability.to_numpy()
        )
        metrics = summarize_predictions(reloaded, float(development.Attrition.mean()))
        if metrics != summarize_predictions(predictions, float(development.Attrition.mean())):
            raise ValueError("CSV and JSON metric calculation mismatch.")
        intervals = bootstrap(reloaded.y_true.to_numpy(), reloaded.probability.to_numpy())
        top_rows = []
        for percent in (5, 10, 15):
            row = operational(reloaded.y_true.to_numpy(), reloaded[f"top_{percent}"].to_numpy())
            top_rows.append(
                {
                    "capacity_percent": percent,
                    **row,
                    "lift": row["precision"] / metrics["prevalence"],
                }
            )
        top = pd.DataFrame(top_rows)
        bins = pd.DataFrame(reliability_bins(reloaded.y_true, reloaded.probability, count=5))
        bins["positives"] = np.rint(bins["count"] * bins.observed_positive_rate).astype(int)
        groups = subgroup_table(reloaded)
        baseline.write_json(report / "final_metrics.json", metrics)
        baseline.write_json(report / "bootstrap_intervals.json", intervals)
        for name, table in (("top_k", top), ("calibration_bins", bins), ("subgroups", groups)):
            write_exclusive(report / f"{name}.csv", table.to_csv(index=False, na_rep="NA").encode())
        from eris_ml.evaluation.final_v3_reporting import render

        render(root, report, metrics, intervals, top, bins, groups, freeze)
        tuning.verify_hashes(root, freeze["protected_sha256"])
        outputs = [p for p in report.iterdir() if p.is_file()] + [pred_path, claim]
        summary = path_at(root, SUMMARY).read_bytes()
        baseline.write_json(
            report / "output_manifest.json",
            {
                "sha256": {p.relative_to(root).as_posix(): sha256_file(p) for p in outputs},
                "summary_after_bytes": len(summary),
                "summary_after_sha256": hashlib.sha256(summary).hexdigest(),
                "protected_hashes_unchanged": True,
                "fitted_model_unchanged": True,
                "holdout_read_count": 1,
                "holdout_predict_batch_calls": 1,
                "fit_calls": 0,
            },
        )
        print(
            "f1 complete; holdout consumed, bundle unchanged. Resume verifies outputs only.",
            flush=True,
        )
    except Exception as exc:
        baseline.write_json(
            report / "failure.json",
            {
                "timestamp": now(),
                "stage": stage,
                "holdout_access_attempted": True,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "predictions_sha256": sha256_file(pred_path) if pred_path.exists() else None,
                "action": "Stop; no new evaluation. Mechanical recovery must preserve predictions.",
            },
        )
        raise
