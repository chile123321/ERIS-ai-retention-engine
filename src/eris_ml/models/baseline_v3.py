"""Four fixed V3 baselines on the pinned development membership and saved V2 folds."""

from __future__ import annotations

import json
import platform
import re
import subprocess
import warnings
from collections import Counter
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.base import BaseEstimator
from sklearn.metrics import log_loss

from eris_ml.data.splitting import assert_development_path
from eris_ml.evaluation.feature_ablation import validate_fold_assignment
from eris_ml.evaluation.metrics import calculate_metrics
from eris_ml.features.schema_v3 import build_pipeline, load_protocol, validate_features
from eris_ml.utils.hashing import sha256_file

IDS = ["source_row", "EmployeeNumber", "Attrition"]
SUMMARY_TITLE = "# V3 baseline — v3_final_15"


def safe_path(root: Path, relative: str) -> Path:
    """Guard before exists/stat/hash/read; refuse path escapes and symlink components."""
    assert_development_path(Path(relative))
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("Expected a repository-relative path.")
    current = root
    for part in path.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError("Symlink input/output rejected.")
    return current


def json_safe(value: Any) -> Any:
    if isinstance(value, BaseEstimator):
        return {"class": type(value).__name__, "parameters": json_safe(value.get_params(False))}
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return "nonfinite_parameter_sentinel"
    if isinstance(value, type):
        return f"{value.__module__}.{value.__name__}"
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(json_safe(payload), stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def validate_alignment(
    development: pd.DataFrame, assignments: pd.DataFrame, manifest: pd.DataFrame
) -> np.ndarray:
    """No split recreation or order-based guessing; IDs, targets and saved positions agree."""
    for frame in (development, assignments):
        if (
            frame.columns.has_duplicates
            or not set(IDS).issubset(frame.columns)
            or frame[IDS].isna().any().any()
            or any(not frame[name].is_unique for name in IDS[:2])
        ):
            raise ValueError("Invalid development/assignment IDs or target.")
    if (
        manifest[IDS[:2] + ["split"]].isna().any().any()
        or any(not manifest[name].is_unique for name in IDS[:2])
        or set(manifest["split"]) != {"development", "final_test"}
    ):
        raise ValueError("Invalid split manifest membership.")
    subset = manifest.loc[manifest.split == "development", IDS[:2]]
    expected = development[IDS[:2]].sort_values("source_row").reset_index(drop=True)
    if not subset.sort_values("source_row").reset_index(drop=True).equals(expected):
        raise ValueError("Development IDs differ from split manifest.")
    if not assignments[IDS].reset_index(drop=True).equals(
        development[IDS].reset_index(drop=True)
    ) or not np.array_equal(assignments.development_position, np.arange(len(development))):
        raise ValueError("Saved fold IDs/targets/development order mismatch.")
    folds = assignments.fold.to_numpy()
    validate_fold_assignment(development.Attrition.to_numpy(), folds)
    return folds


def prepare(
    root: Path, config: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    data = config["data"]
    paths = {
        key: safe_path(root, data[key])
        for key in ("development", "manifest", "assignments", "provenance_trace", "holdout_ledger")
    }
    trace = json.loads(paths["provenance_trace"].read_text(encoding="utf-8"))
    hashes = {data[key]: sha256_file(path) for key, path in paths.items()}
    for key in ("development", "manifest"):
        if hashes[data[key]] != trace["input_sha256"][data[key]]:
            raise ValueError(f"Immutable {key} hash mismatch.")
    if hashes[data["assignments"]] != trace["output_sha256"]["assignments"]:
        raise ValueError("Saved V2 assignment hash mismatch; refusing a replacement.")
    if trace["cv"] != config["cv"]:
        raise ValueError("Saved fold protocol differs from V3.")
    names = [item["name"] for item in config["features"]]
    frame = pd.read_csv(paths["development"], usecols=lambda name: name in names + IDS)
    target = frame[config["target"]["name"]]
    if target.dtype.kind not in "iu":
        target = target.map(config["target"]["mapping"])
    if target.isna().any() or set(target) != {0, 1}:
        raise ValueError("Attrition must be encoded No=0, Yes=1.")
    frame["Attrition"] = target.astype(int)
    if (
        len(frame) != data["expected_rows"]
        or frame.Attrition.value_counts().to_dict() != data["expected_target"]
    ):
        raise ValueError("Development size or target distribution mismatch.")
    assignments = pd.read_csv(paths["assignments"])
    # IDs/split labels only, not employee attributes or holdout outcomes.
    manifest = pd.read_csv(paths["manifest"], usecols=IDS[:2] + ["split"])
    if len(manifest) != data["expected_manifest_rows"]:
        raise ValueError("Split manifest row count mismatch.")
    validate_alignment(frame, assignments, manifest)
    features = validate_features(frame.drop(columns=IDS), config)
    observations = {}
    for item in config["features"]:
        name = item["name"]
        column = features[name]
        observations[name] = (
            {
                "min": float(column.min()),
                "max": float(column.max()),
                "missing": int(column.isna().sum()),
            }
            if item["type"] == "integer"
            else {"categories": sorted(column.unique().tolist()), "missing": 0}
        )
    ledger = yaml.safe_load(paths["holdout_ledger"].read_text(encoding="utf-8"))
    provenance = {
        "input_sha256": hashes,
        "development_rows": len(frame),
        "target_distribution": frame.Attrition.value_counts().to_dict(),
        "fold_source": data["assignments"],
        "folds_recreated": False,
        "manifest_rows": len(manifest),
        "holdout_membership_rows": len(manifest) - len(frame),
        "holdout_status_from_ledger_only": {
            "run_count": ledger["run_count"],
            "status": ledger["status"],
            "completed_at_utc": ledger["completed_at_utc"],
            "not_a_fresh_independent_V3_holdout": True,
        },
        "observed_development_ranges_not_business_bounds": observations,
        "historical_source_fingerprints": (
            "Old V2 trace preserved; only development/manifest/assignment hashes reused. "
            "V3 executes new source files recorded here, not historical V2 training code."
        ),
    }
    return pd.concat([frame[IDS], features], axis=1), assignments, provenance


def metrics(target: np.ndarray, probability: np.ndarray, threshold: float) -> dict[str, Any]:
    base = calculate_metrics(target, probability, threshold)
    keys = (
        "average_precision",
        "roc_auc",
        "brier_score",
        "precision",
        "recall",
        "f1",
        "f2",
        "balanced_accuracy",
        "accuracy",
        "alert_rate",
    )
    return {
        **{name: base[name] for name in keys},
        **base["confusion_matrix"],
        "alerts": base["predicted_positive_count"],
        "log_loss": float(
            log_loss(
                target,
                np.clip(probability, np.finfo(float).eps, 1 - np.finfo(float).eps),
                labels=[0, 1],
            )
        ),
    }


def evaluate(
    development: pd.DataFrame, assignments: pd.DataFrame, config: dict[str, Any]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Fit 4 x 5 pipelines; never export a fitted estimator or search any parameter."""
    names = [item["name"] for item in config["features"]]
    X = validate_features(development.loc[:, [n for n in names if n in development]], config)
    y, folds = development.Attrition.to_numpy(), assignments.fold.to_numpy()
    if not development[IDS].reset_index(drop=True).equals(assignments[IDS].reset_index(drop=True)):
        raise ValueError("Evaluation IDs/target mismatch.")
    validate_fold_assignment(y, folds)
    all_oof, records = [], []
    for model in config["models"]:
        scores, seen = np.full(len(X), np.nan), np.zeros(len(X), dtype=int)
        fold_records: list[dict[str, Any]] = []
        for fold in range(1, config["cv"]["n_splits"] + 1):
            train, validation = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
            pipeline = build_pipeline(config, model)
            pipeline.fit(X.iloc[train], y[train])
            if not np.array_equal(pipeline.classes_, [0, 1]):
                raise ValueError("Expected class order [0, 1].")
            prediction = np.asarray(pipeline.predict_proba(X.iloc[validation])[:, 1], dtype=float)
            single = float(pipeline.predict_proba(X.iloc[validation[:1]])[0, 1])
            if not np.isclose(single, prediction[0], rtol=1e-10, atol=1e-12):
                raise ValueError("Single versus batch prediction mismatch.")
            scores[validation] = prediction
            seen[validation] += 1
            imputer = (
                pipeline.named_steps["preprocessor"]
                .named_transformers_["optional_numeric"]
                .named_steps["imputer"]
            )
            optional = [item["name"] for item in config["features"] if not item["required"]]
            expected = np.nanmedian(X.iloc[train][optional].to_numpy(), axis=0)
            if not np.allclose(imputer.statistics_, expected):
                raise ValueError("Optional median differs from train-only median.")
            fold_records.append(
                {
                    "fold": fold,
                    "train_rows": len(train),
                    "validation_rows": len(validation),
                    "validation_positives": int(y[validation].sum()),
                    "optional_training_medians": dict(zip(optional, expected, strict=True)),
                    "metrics": metrics(y[validation], prediction, config["threshold_reference"]),
                }
            )
            print(f"{model}: fold {fold}/5 complete", flush=True)
        if not np.all(seen == 1):
            raise ValueError("Every row must receive exactly one OOF probability.")
        pooled = metrics(y, scores, config["threshold_reference"])
        dispersion = {
            key: {
                "mean": float(np.mean([r["metrics"][key] for r in fold_records])),
                "sample_std": float(np.std([r["metrics"][key] for r in fold_records], ddof=1)),
            }
            for key in pooled
        }
        records.append(
            {
                "model": model,
                "pooled_oof": pooled,
                "fold_mean_std": dispersion,
                "folds": fold_records,
                "single_batch_check": "passed_all_five_folds",
            }
        )
        rows = assignments[IDS + ["fold"]].rename(columns={"Attrition": "y_true"}).copy()
        rows["feature_set"], rows["model"], rows["probability"] = config["version"], model, scores
        all_oof.append(rows)
    oof = pd.concat(all_oof, ignore_index=True)
    validate_oof(oof, assignments, config)
    records.sort(key=lambda record: record["pooled_oof"]["average_precision"], reverse=True)
    return oof, {
        "primary_metric": "average_precision",
        "threshold_reference": 0.5,
        "positive_rate": float(np.mean(y)),
        "models": records,
    }


def validate_oof(oof: pd.DataFrame, assignments: pd.DataFrame, config: dict[str, Any]) -> None:
    if (
        len(oof) != len(assignments) * len(config["models"])
        or oof.isna().any().any()
        or set(oof.model) != set(config["models"])
        or set(oof.feature_set) != {config["version"]}
    ):
        raise ValueError("Incomplete OOF model/row coverage.")
    reference = assignments[IDS + ["fold"]].rename(columns={"Attrition": "y_true"})
    for _, rows in oof.groupby("model", sort=False):
        if any(not rows[name].is_unique for name in IDS[:2]) or not rows[
            reference.columns
        ].reset_index(drop=True).equals(reference.reset_index(drop=True)):
            raise ValueError("OOF ID/target/fold alignment failure.")
        metrics(rows.y_true.to_numpy(), rows.probability.to_numpy(), config["threshold_reference"])


def summary_text(
    result: dict[str, Any], config: dict[str, Any], run_id: str, warning_counts: dict[str, int]
) -> str:
    features = ", ".join(f"`{item['name']}`" for item in config["features"])
    introduction = f"""{SUMMARY_TITLE}

Run `{run_id}`; development only, 1176 rows (986 No, 190 Yes).
Saved V2 Stratified 5-fold assignment, shuffle=True, seed=42; no split/fold recreation.

15 ordered features: {features}

13 required; optional TrainingTimesLastYear and YearsWithCurrManager use train-only
median imputation. Missing required/unknown categories are rejected. No missing indicators.
Logistic scales numeric/ordinal values; RF/XGBoost do not. Fixed IBM one-hot vocabulary;
all validation/imputation/encoding/scaling is inside each fold pipeline. No target/IDs in X.

## Actual pooled OOF results

AP is average_precision_score (not trapezoidal PR area). SD is sample SD over five folds.
Threshold 0.5 is a baseline reference, not an HR policy. Alert rule: probability >= 0.5.
"""
    headers = [
        "Model",
        "AP",
        "Fold AP mean ± SD",
        "ROC-AUC",
        "Brier",
        "Log loss",
        "Precision",
        "Recall",
        "F1",
        "F2",
        "Balanced accuracy",
        "TP",
        "FP",
        "TN",
        "FN",
        "Alerts",
    ]
    lines = [
        introduction,
        "| " + " | ".join(headers) + " |",
        "|---|" + "---:|" * (len(headers) - 1),
    ]
    for record in result["models"]:
        m, d = record["pooled_oof"], record["fold_mean_std"]["average_precision"]
        values = [
            record["model"],
            f"{m['average_precision']:.6f}",
            f"{d['mean']:.6f} ± {d['sample_std']:.6f}",
        ]
        values += [
            f"{m[k]:.6f}"
            for k in (
                "roc_auc",
                "brier_score",
                "log_loss",
                "precision",
                "recall",
                "f1",
                "f2",
                "balanced_accuracy",
            )
        ]
        values += [str(m[k]) for k in ("tp", "fp", "tn", "fn", "alerts")]
        lines.append("| " + " | ".join(values) + " |")
    ranked = [record for record in result["models"] if record["model"] != "dummy_prior"]
    lines.append(f"""
## Interpretation and next candidates

Development positive rate: {result["positive_rate"]:.6f}; compare AP to this prevalence.
Proposed tuning candidates by pooled AP: **{ranked[0]["model"]}** and **{ranked[1]["model"]}**.
This is a development research shortlist, not production selection.
Fold variability and Brier must be considered alongside AP; small differences do not
establish a significant winner. No statistical significance test was performed.

All pooled metrics, individual folds and means/sample SDs: `{run_id}/metrics.json`.
Resolved estimator/preprocessing configs, versions and hashes: `{run_id}/preflight.json`
(written before fit) and `{run_id}/metadata.json`. No fitted model is saved.
Missing optional inputs are supported at prediction, but robustness under real missingness
has not been evaluated. IBM observed ranges are not business validity bounds.
Income currency, distance unit, employer mapping and time horizon remain unconfirmed.
Final test V1 was already evaluated once per ledger; it was not opened or hashed here.
IBM cross-validation does not certify new-employer performance or a 3–6 month horizon.
No search, calibration, early stopping, SHAP, feature selection, bundle or API work.

Recorded training warnings: {json.dumps(warning_counts, ensure_ascii=False)}
""")
    return "\n".join(lines)


def run(root: Path, config_path: Path, run_id: str = "r1") -> dict[str, Any]:
    if not re.fullmatch(r"[A-Za-z0-9_]+", run_id):
        raise ValueError("Run ID must contain only letters, digits or underscore.")
    assert_development_path(config_path)
    config = load_protocol(config_path)
    outputs = {
        key: safe_path(root, value.format(run_id=run_id))
        for key, value in config["outputs"].items()
    }
    if len(set(outputs.values())) != len(outputs):
        raise ValueError("Output paths must be distinct.")
    for key, path in outputs.items():
        allowed = (
            root / "artifacts" / ("predictions" if key == "oof" else "reports") / "v3_final_15"
        )
        if not path.is_relative_to(allowed):
            raise ValueError("Only dedicated V3 output directories are allowed.")
        if path.exists():
            if key != "summary" or not path.read_text(encoding="utf-8").startswith(SUMMARY_TITLE):
                raise ValueError(f"Refusing existing output {path}; use a new --run-id.")
    frame, assignments, preflight = prepare(root, config)
    sources = [
        config_path,
        Path(__file__),
        root / "scripts/baseline_v3.py",
        root / "src/eris_ml/features/schema_v3.py",
        root / "src/eris_ml/models/factory.py",
        root / "src/eris_ml/evaluation/metrics.py",
        root / "src/eris_ml/evaluation/feature_ablation.py",
    ]
    preflight["input_sha256"].update(
        {p.relative_to(root).as_posix(): sha256_file(p) for p in sources}
    )
    preflight.update(
        {
            "created_at_utc": datetime.now(UTC).isoformat(),
            "run_id": run_id,
            "config": config,
            "git": {
                name: subprocess.check_output(["git", *args], cwd=root, text=True).strip()
                for name, args in {
                    "branch": ["branch", "--show-current"],
                    "head": ["rev-parse", "HEAD"],
                    "worktree": ["status", "--short"],
                }.items()
            },
            "library_versions": {
                "python": platform.python_version(),
                **{
                    name: version(name)
                    for name in ("numpy", "pandas", "scipy", "scikit-learn", "xgboost")
                },
            },
            "resolved_pipelines": {
                model: json_safe(build_pipeline(config, model)) for model in config["models"]
            },
            "final_test_accessed": False,
            "model_serialized": False,
            "business_approved": False,
            "production_approved": False,
        }
    )
    write_json(outputs["preflight"], preflight)
    print("Preflight passed and persisted; beginning 20 train-only baseline fits.", flush=True)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        oof, result = evaluate(frame, assignments, config)
    warning_counts = dict(Counter(f"{w.category.__name__}: {w.message}" for w in caught))
    for rel, digest in preflight["input_sha256"].items():
        if sha256_file(safe_path(root, rel)) != digest:
            raise ValueError(f"Input changed during run: {rel}")
    outputs["oof"].parent.mkdir(parents=True, exist_ok=True)
    oof.to_csv(outputs["oof"], index=False, mode="x")
    reloaded = pd.read_csv(outputs["oof"])
    validate_oof(reloaded, assignments, config)
    if not np.allclose(oof.probability, reloaded.probability, rtol=1e-12, atol=1e-15):
        raise ValueError("OOF CSV roundtrip differs.")
    write_json(outputs["metrics"], result)
    write_json(
        outputs["metadata"],
        {
            "completed_at_utc": datetime.now(UTC).isoformat(),
            "run_id": run_id,
            "preflight_sha256": sha256_file(outputs["preflight"]),
            "outputs_sha256": {key: sha256_file(outputs[key]) for key in ("oof", "metrics")},
            "outputs": {key: path.relative_to(root).as_posix() for key, path in outputs.items()},
            "warnings": warning_counts,
            "input_hashes_unchanged": True,
            "rows_per_model": oof.groupby("model").size().to_dict(),
            "oof_csv_roundtrip": "passed",
            "single_batch_consistency": "passed_20_fits",
            "final_test_accessed": False,
            "model_serialized": False,
        },
    )
    # The sole mutable summary belongs to V3; per-run technical records are immutable.
    outputs["summary"].parent.mkdir(parents=True, exist_ok=True)
    outputs["summary"].write_text(
        summary_text(result, config, run_id, warning_counts), encoding="utf-8"
    )
    print(f"Complete. Summary: {outputs['summary']}; warnings: {warning_counts}", flush=True)
    return result
