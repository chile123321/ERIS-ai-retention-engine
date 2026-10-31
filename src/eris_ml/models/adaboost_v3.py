"""Isolated development-only AdaBoost study; no bundle, API or holdout access."""

from __future__ import annotations

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
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from threadpoolctl import threadpool_limits

from eris_ml.evaluation.feature_ablation import validate_fold_assignment
from eris_ml.evaluation.threshold_policy_v2 import top_k
from eris_ml.evaluation.thresholding import confusion_metrics
from eris_ml.features.schema_v3 import build_pipeline, load_protocol, validate_features
from eris_ml.models import baseline_v3 as baseline
from eris_ml.models.tuning_v3 import aligned_oof, inner_splits
from eris_ml.utils.hashing import sha256_file

GRID = {
    "model__estimator__max_depth": [1, 2],
    "model__n_estimators": [50, 100, 200],
    "model__learning_rate": [0.05, 0.1, 0.5, 1.0],
}
MODELS = [
    f"adaboost_{family}_{method}"
    for family in ("baseline", "tuned")
    for method in ("raw", "sigmoid")
]
RF = "rf_unweighted_raw"
METRICS = ("average_precision", "roc_auc", "brier_score", "log_loss")
POLICY = {
    "version": "adaboost-v3-study-v1",
    "outer_cv": "unchanged saved V2 assignment; 5 folds",
    "inner_cv": {"n_splits": 4, "shuffle": True, "random_state": 43},
    "baseline": {"max_depth": 1, "n_estimators": 50, "learning_rate": 1.0},
    "estimator_and_tree_seed": 42,
    "grid": GRID,
    "selection_metric": "average_precision",
    "tie_break": "AP within absolute 1e-12 of maximum: depth, estimators, learning_rate ascending",
    "calibration": {"method": "sigmoid", "ensemble": False, "n_jobs": 1},
    "calibration_decision": {
        "status": "exploratory proposal only; not an independently evaluated selection policy",
        "maximum_ap_or_roc_drop": 0.005,
        "brier": "strictly improve pooled Brier and improve at least 3/5 folds",
        "log_loss": "pooled log loss must not worsen",
    },
    "top_k": {
        "percent": [5, 10, 15],
        "rounding": "floor per outer-validation fold",
        "ties": "probability descending, source_row ascending, no labels",
    },
    "execution": {"search_jobs": 1, "native_threads": 1},
    "resampling": False,
    "additional_class_balancing": False,
    "final_test_accessed": False,
    "model_serialized": False,
    "production_approved": False,
}


def study_path(root: Path, relative: str) -> Path:
    """Guard before any read/hash/stat; also refuse consumed f1 artifacts."""
    normalized = relative.replace("\\", "/").lower()
    if "/f1/" in f"/{normalized}/" or "final_evaluation_consumed" in normalized:
        raise ValueError("Consumed final evaluation is not an experiment input.")
    return baseline.safe_path(root, relative)


def pipeline_for(config: dict[str, Any]) -> Pipeline:
    # Reuse precisely the unfitted RF preprocessing, not a fitted bundle transformer.
    pipeline = build_pipeline(config, "random_forest")
    pipeline.set_params(
        model=AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=1, random_state=42),
            n_estimators=50,
            learning_rate=1.0,
            random_state=42,
        )
    )
    return pipeline


def choose_best(results: dict[str, Any]) -> int:
    scores = np.asarray(results["mean_test_score"], dtype=float)
    if not np.isfinite(scores).all():
        raise ValueError("Nonfinite inner-CV search score; do not select a candidate.")
    tied = np.flatnonzero(scores >= scores.max() - 1e-12)
    return int(
        min(
            tied,
            key=lambda i: (
                results["params"][i]["model__estimator__max_depth"],
                results["params"][i]["model__n_estimators"],
                results["params"][i]["model__learning_rate"],
            ),
        )
    )


def positive_probability(estimator: Any, X: pd.DataFrame) -> np.ndarray:
    if not np.array_equal(estimator.classes_, [0, 1]):
        raise ValueError("Expected classes [0,1].")
    probability = np.asarray(estimator.predict_proba(X)[:, 1], dtype=float)
    if (
        probability.shape != (len(X),)
        or not np.isfinite(probability).all()
        or ((probability < 0) | (probability > 1)).any()
    ):
        raise ValueError("Invalid probability shape/range/finiteness.")
    return probability


def fit_fold(
    frame: pd.DataFrame,
    assignments: pd.DataFrame,
    config: dict[str, Any],
    fold: int,
    grid: dict[str, list[Any]] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Search/refit/calibrate exclusively on outer-train; evaluate four fixed procedures."""
    if (
        not frame[baseline.IDS]
        .reset_index(drop=True)
        .equals(assignments[baseline.IDS].reset_index(drop=True))
    ):
        raise ValueError("Development/assignment IDs or targets mismatch before fitting.")
    y, folds = frame.Attrition.to_numpy(), assignments.fold.to_numpy()
    validate_fold_assignment(y, folds)
    if fold not in range(1, 6):
        raise ValueError("Invalid outer fold.")
    train, valid = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
    X = validate_features(frame[[f["name"] for f in config["features"]]], config)
    splits = inner_splits(y[train])
    for it, iv in splits:
        if np.intersect1d(train[it], valid).size or np.intersect1d(train[iv], valid).size:
            raise ValueError("Inner/outer fold leakage.")
    started = time.monotonic()
    search: Any = GridSearchCV(
        pipeline_for(config),
        GRID if grid is None else grid,
        scoring="average_precision",
        cv=splits,
        refit=choose_best,
        n_jobs=1,
        error_score="raise",
        return_train_score=False,
    )
    search.fit(X.iloc[train], y[train])
    params = search.cv_results_["params"]
    search_rows = [
        {
            "parameters": param,
            "mean_inner_ap": float(search.cv_results_["mean_test_score"][i]),
            "inner_fold_ap": [
                float(search.cv_results_[f"split{k}_test_score"][i]) for k in range(4)
            ],
        }
        for i, param in enumerate(params)
    ]
    raw_baseline = pipeline_for(config).fit(X.iloc[train], y[train])
    blocks, diagnostics = [], {}
    for family, raw in (("baseline", raw_baseline), ("tuned", search.best_estimator_)):
        raw_probability = positive_probability(raw, X.iloc[valid])
        calibrated: Any = CalibratedClassifierCV(
            estimator=clone(raw),
            method="sigmoid",
            cv=splits,
            ensemble=False,
            n_jobs=1,
        )
        calibrated.fit(X.iloc[train], y[train])
        base = calibrated.calibrated_classifiers_[0].estimator
        repeated = positive_probability(base, X.iloc[valid])
        if not np.allclose(raw_probability, repeated, rtol=1e-10, atol=1e-12):
            raise ValueError("Calibration's refitted raw base differs from matching raw model.")
        optional = [f["name"] for f in config["features"] if not f["required"]]
        expected_medians = X.iloc[train][optional].median().to_numpy()
        for fitted in (raw, base):
            actual = (
                fitted.named_steps["preprocessor"]
                .named_transformers_["optional_numeric"]
                .named_steps["imputer"]
                .statistics_
            )
            if not np.allclose(actual, expected_medians):
                raise ValueError("Imputation does not match outer-training medians.")
        diagnostics[family] = {
            "effective_estimators": len(raw.named_steps["model"].estimators_),
            "resolved_model": baseline.json_safe(raw.named_steps["model"]),
            "raw_refit_max_difference": float(np.max(np.abs(raw_probability - repeated))),
            "optional_train_medians": dict(zip(optional, expected_medians, strict=True)),
        }
        for method, probabilities in (
            ("raw", raw_probability),
            ("sigmoid", positive_probability(calibrated, X.iloc[valid])),
        ):
            block = (
                assignments.iloc[valid][baseline.IDS + ["fold"]]
                .rename(columns={"Attrition": "y_true"})
                .copy()
            )
            block["feature_set"] = config["version"]
            block["model"] = f"adaboost_{family}_{method}"
            block["probability"] = probabilities
            blocks.append(block)
    trace = {
        "outer_fold": fold,
        "train_rows": len(train),
        "validation_rows": len(valid),
        "train_source_rows": frame.iloc[train].source_row.tolist(),
        "validation_source_rows": frame.iloc[valid].source_row.tolist(),
        "inner_splits_outer_train_positions": [
            {"train": t.tolist(), "validation": v.tolist()} for t, v in splits
        ],
        "best_parameters": search.best_params_,
        "search_results": search_rows,
        "diagnostics": diagnostics,
        "seconds": time.monotonic() - started,
    }
    return pd.concat(blocks, ignore_index=True), trace


def calibration_proposal(raw: dict[str, Any], sigmoid: dict[str, Any]) -> dict[str, Any]:
    r, s = raw["pooled"], sigmoid["pooled"]
    improved = sum(
        b["brier_score"] < a["brier_score"]
        for a, b in zip(raw["folds"], sigmoid["folds"], strict=True)
    )
    checks = {
        "ap_guardrail": s["average_precision"] >= r["average_precision"] - 0.005,
        "roc_guardrail": s["roc_auc"] >= r["roc_auc"] - 0.005,
        "brier_improves": s["brier_score"] < r["brier_score"],
        "log_loss_not_worse": s["log_loss"] <= r["log_loss"],
        "at_least_three_improved_folds": improved >= 3,
    }
    return {
        "proposed_method": "sigmoid" if all(checks.values()) else "raw",
        "checks": checks,
        "brier_improved_folds": improved,
        "independently_evaluated_selection_policy": False,
    }


def analyze(oof: pd.DataFrame, assignments: pd.DataFrame) -> dict[str, Any]:
    oof = aligned_oof(oof, assignments, MODELS + [RF])
    records: dict[str, Any] = {}
    for model in MODELS + [RF]:
        rows = oof[oof.model.eq(model)]
        y, p = rows.y_true.to_numpy(), rows.probability.to_numpy()
        folds = []
        policies = []
        for fold in range(1, 6):
            block = rows[rows.fold.eq(fold)]
            scores = baseline.metrics(block.y_true.to_numpy(), block.probability.to_numpy(), 0.5)
            folds.append({"fold": fold, **{m: scores[m] for m in METRICS}})
        for percent in (5, 10, 15):
            selected = np.zeros(len(rows), dtype=int)
            fold_policy = []
            for fold in range(1, 6):
                positions = np.flatnonzero(rows.fold.to_numpy() == fold)
                flags = top_k(p[positions], rows.source_row.to_numpy()[positions], percent)
                selected[positions] = flags
                fold_policy.append({"fold": fold, **policy_metrics(y[positions], flags)})
            policies.append(
                {
                    "capacity_percent": percent,
                    "pooled": policy_metrics(y, selected),
                    "folds": fold_policy,
                }
            )
        scores = baseline.metrics(y, p, 0.5)
        records[model] = {
            "rows": len(rows),
            "pooled": {m: scores[m] for m in METRICS},
            "folds": folds,
            "fold_mean_sd": {
                m: {
                    "mean": float(np.mean([f[m] for f in folds])),
                    "sample_sd": float(np.std([f[m] for f in folds], ddof=1)),
                }
                for m in METRICS
            },
            "mean_probability": float(p.mean()),
            "observed_positive_rate": float(y.mean()),
            "top_k": policies,
        }
    comparisons = {}
    for left, right in [(m, RF) for m in MODELS] + [
        ("adaboost_tuned_raw", "adaboost_baseline_raw"),
        ("adaboost_baseline_sigmoid", "adaboost_baseline_raw"),
        ("adaboost_tuned_sigmoid", "adaboost_tuned_raw"),
    ]:
        a, b = records[left], records[right]
        comparisons[f"{left}_minus_{right}"] = {
            "pooled": {m: a["pooled"][m] - b["pooled"][m] for m in METRICS},
            "folds": [
                {"fold": f["fold"], **{m: f[m] - g[m] for m in METRICS}}
                for f, g in zip(a["folds"], b["folds"], strict=True)
            ],
        }
    return {
        "models": records,
        "paired_deltas": comparisons,
        "exploratory_calibration_proposals": {
            family: calibration_proposal(
                records[f"adaboost_{family}_raw"], records[f"adaboost_{family}_sigmoid"]
            )
            for family in ("baseline", "tuned")
        },
    }


def policy_metrics(y: np.ndarray, selected: np.ndarray) -> dict[str, Any]:
    return confusion_metrics(
        int(np.sum((y == 0) & (selected == 0))),
        int(np.sum((y == 0) & (selected == 1))),
        int(np.sum((y == 1) & (selected == 0))),
        int(np.sum((y == 1) & (selected == 1))),
    )


def report_text(result: dict[str, Any]) -> str:
    lines = [
        "# AdaBoost V3 development study",
        "",
        "1176 rows, 190 positives. Same saved V2 outer folds and v3_final_15 features.",
        "24-grid inner 4-fold search; seeds 42/43. No balancing or resampling.",
        "All transformations fit in training pipelines. No final test, bundle or API changes.",
        "",
        "## Pooled OOF and fold mean +/- sample SD",
        "",
        "| Candidate | AP | ROC-AUC | Brier | Log loss |",
        "|---|---|---|---|---|",
    ]
    for name, record in result["models"].items():
        values = [
            f"{record['pooled'][m]:.6f} ({record['fold_mean_sd'][m]['mean']:.6f}"
            f" +/- {record['fold_mean_sd'][m]['sample_sd']:.6f})"
            for m in METRICS
        ]
        lines.append("| " + " | ".join([name, *values]) + " |")
    lines += [
        "",
        "## Top-k (floor within each validation fold; source_row ascending ties)",
        "",
        "| Candidate | Capacity | Alerts | TP | FP | FN | Precision | Recall |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, record in result["models"].items():
        for row in record["top_k"]:
            m = row["pooled"]
            lines.append(
                f"| {name} | {row['capacity_percent']}% | {m['alerts']} | {m['tp']} | "
                f"{m['fp']} | {m['fn']} | {m['precision']:.6f} | {m['recall']:.6f} |"
            )
    lines += ["", "## Calibration proposals (exploratory, not independent validation)", ""]
    for name, decision in result["exploratory_calibration_proposals"].items():
        lines.append(
            f"- {name}: {decision['proposed_method']}; "
            f"Brier improves in {decision['brier_improved_folds']}/5 folds; "
            f"checks: {json.dumps(decision['checks'])}."
        )
    lines += [
        "",
        "## Scope and interpretation",
        "",
        "Raw and sigmoid are separate, pre-specified evaluation procedures. Sigmoid uses",
        "4-fold OOF scores within outer-training (ensemble=False), then refits its base",
        "on all outer-training. Each sigmoid refitted base matches its raw counterpart.",
        "Tuned parameters are selected exclusively in outer-training. Its calibration",
        "uses those fixed parameters; they are not reselected within calibration folds.",
        "The final calibration proposal uses pooled OOF and is EXPLORATORY: no score here",
        "is an independent evaluation of an adaptively selected raw/sigmoid policy.",
        "RF B2 parameters were previously selected using development experiments; this",
        "paired comparison is not an independent or symmetric algorithm-selection test.",
        "No significance claim. Outer folds are technical batches, not HR months.",
        "IBM cross-sectional labels do not validate new-employer performance, intervention",
        "effects, fairness or a 3-6 month horizon. No business/production approval.",
        "",
        "## Reproduction and detailed evidence",
        "",
        "Run scripts/train_adaboost_v3.py with a fresh --run-id (existing outputs refused).",
        "preflight.json freezes policy/config/versions/Git/input hashes before fitting.",
        "fold_1.json through fold_5.json record all inner scores, selected parameters,",
        "split indices, medians and raw-refit checks. metrics.json includes individual",
        "fold metrics, paired deltas, top-k counts and calibration guardrails.",
        "metadata.json records warnings, output hashes and protected-input checks.",
        "No estimator is serialized. SHAP and packaging require a subsequent decision.",
        "",
    ]
    return "\n".join(lines)


def run(root: Path, run_id: str) -> dict[str, Any]:
    if not re.fullmatch(r"[A-Za-z0-9_]+", run_id):
        raise ValueError("Invalid run ID.")
    report_dir = study_path(root, f"artifacts/reports/adaboost_v3/{run_id}")
    prediction_dir = study_path(root, f"artifacts/predictions/adaboost_v3/{run_id}")
    if report_dir.exists() or prediction_dir.exists():
        raise ValueError("Run outputs already exist; use a fresh run ID. Nothing overwritten.")
    config_path = study_path(root, "configs/features/feature_set_v3_final_15.yaml")
    config = load_protocol(config_path)
    frame, assignments, provenance = baseline.prepare(root, config)
    old_preflight = study_path(root, "artifacts/reports/v3_final_15/devfix1/preflight.json")
    historical = json.loads(old_preflight.read_text(encoding="utf-8"))["identity"]
    if baseline.json_safe(config) != historical["feature_config"]:
        raise ValueError("Feature/config differs from RF comparison study.")
    protected = dict(provenance["input_sha256"])
    for directory, manifest_name, key in (
        ("artifacts/reports/v3_final_15/devfix1", "manifest.json", "output_sha256"),
        ("artifacts/models/v3_final_15/b2", "manifest.json", None),
    ):
        manifest_path = study_path(root, f"{directory}/{manifest_name}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        protected[f"{directory}/{manifest_name}"] = sha256_file(manifest_path)
        for name, digest in (manifest[key] if key else manifest).items():
            relative = name if key else f"{directory}/{name}"
            if key and not (
                relative.startswith("artifacts/reports/v3_final_15/devfix1/")
                or relative.startswith("artifacts/predictions/v3_final_15/devfix1/")
            ):
                raise ValueError("Unexpected manifest path.")
            if sha256_file(study_path(root, relative)) != digest:
                raise ValueError(f"Protected artifact hash mismatch: {relative}")
            protected[relative] = digest
    model_path = "artifacts/models/v3_final_15/b2/pipeline.joblib"
    if protected[model_path] != "0c72841a8505a910d31c2a687fbcbf5259e3afb60f9eddcb7586801519b5628f":
        raise ValueError("B2 model hash mismatch.")
    rf_path = study_path(root, "artifacts/predictions/v3_final_15/devfix1/oof.csv")
    rf = pd.read_csv(rf_path)
    rf = aligned_oof(rf.loc[rf.model.eq(RF) & rf.scenario.eq("original")], assignments, [RF])
    rf = rf.drop(
        columns=[
            c
            for c in rf
            if c
            not in {
                "source_row",
                "EmployeeNumber",
                "y_true",
                "fold",
                "feature_set",
                "model",
                "probability",
            }
        ]
    )
    sources = [config_path, Path(__file__), root / "scripts/train_adaboost_v3.py", old_preflight]
    sources += [
        root / "src/eris_ml" / p
        for p in (
            "features/schema_v3.py",
            "models/baseline_v3.py",
            "models/tuning_v3.py",
            "models/factory.py",
            "evaluation/metrics.py",
            "evaluation/thresholding.py",
            "evaluation/feature_ablation.py",
            "evaluation/threshold_policy_v2.py",
        )
    ]
    protected.update({p.relative_to(root).as_posix(): sha256_file(p) for p in sources})
    preflight = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "run_id": run_id,
        "policy": POLICY,
        "feature_config": config,
        "data_provenance": provenance,
        "resolved_baseline_pipeline": baseline.json_safe(pipeline_for(config)),
        "protected_sha256": protected,
        "historical_source_bridges": historical["historical_source_bridges"],
        "historical_trace_modified": False,
        "versions": {
            "python": platform.python_version(),
            **{
                p: version(p)
                for p in ("numpy", "pandas", "scipy", "scikit-learn", "joblib", "threadpoolctl")
            },
        },
        "git": {
            key: subprocess.check_output(["git", *args], cwd=root, text=True).strip()
            for key, args in {
                "branch": ["branch", "--show-current"],
                "head": ["rev-parse", "HEAD"],
                "worktree": ["status", "--short"],
            }.items()
        },
    }
    baseline.write_json(report_dir / "preflight.json", preflight)
    print("Preflight locked: alignment/hashes passed; 24-grid nested search begins.", flush=True)
    blocks = [rf]
    started = time.monotonic()
    warning_counts: Counter[str] = Counter()
    with threadpool_limits(limits=1):
        for fold in range(1, 6):
            with warnings.catch_warnings(record=True) as captured:
                warnings.simplefilter("always")
                block, trace = fit_fold(frame, assignments, config, fold)
            warning_counts.update(f"{w.category.__name__}: {w.message}" for w in captured)
            blocks.append(block)
            baseline.write_json(report_dir / f"fold_{fold}.json", trace)
            print(
                f"Fold {fold}/5 complete in {trace['seconds']:.1f}s: {trace['best_parameters']}",
                flush=True,
            )
    oof = aligned_oof(pd.concat(blocks, ignore_index=True), assignments, MODELS + [RF])
    result = analyze(oof, assignments)
    for relative, digest in protected.items():
        if sha256_file(study_path(root, relative)) != digest:
            raise ValueError(f"Protected input changed during run: {relative}")
    prediction_dir.mkdir(parents=True, exist_ok=False)
    oof.to_csv(prediction_dir / "oof.csv", index=False, mode="x")
    reloaded = aligned_oof(pd.read_csv(prediction_dir / "oof.csv"), assignments, MODELS + [RF])
    if not np.allclose(oof.probability, reloaded.probability, rtol=1e-12, atol=1e-15):
        raise ValueError("OOF CSV roundtrip failure.")
    baseline.write_json(report_dir / "metrics.json", result)
    with (report_dir / "report.md").open("x", encoding="utf-8") as stream:
        stream.write(report_text(result))
    outputs = [*report_dir.iterdir(), prediction_dir / "oof.csv"]
    baseline.write_json(
        report_dir / "metadata.json",
        {
            "completed_at_utc": datetime.now(UTC).isoformat(),
            "seconds": time.monotonic() - started,
            "warnings": dict(warning_counts),
            "protected_hashes_unchanged": True,
            "rows_per_model": oof.groupby("model").size().to_dict(),
            "oof_roundtrip_passed": True,
            "output_sha256": {p.relative_to(root).as_posix(): sha256_file(p) for p in outputs},
            "final_test_accessed": False,
            "model_serialized": False,
        },
    )
    print(f"Completed: {report_dir / 'report.md'}", flush=True)
    return result
