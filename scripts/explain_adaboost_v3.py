"""Reproduce a1 baseline sigmoid, validate permutation SHAP, then fit a separate demo bundle."""

from __future__ import annotations

import json
import subprocess
import sys
import warnings
from collections import Counter
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

from eris_ml.explainability import adaboost_shap as explain
from eris_ml.features.schema_v3 import load_protocol
from eris_ml.models import adaboost_v3 as a1
from eris_ml.models import baseline_v3 as baseline
from eris_ml.models.tuning_v3 import aligned_oof
from eris_ml.utils.hashing import sha256_file

ROOT = Path(__file__).resolve().parents[1]


def plots(directory: Path, records: list[dict[str, Any]]) -> None:
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    primary = [r for r in records if r["scenario"] == "original"]
    names = primary[0]["high"]["feature_order"]
    importance = np.mean(np.abs([r["high"]["contributions"] for r in primary]), axis=0) * 100
    order = np.argsort(importance)
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(np.asarray(names)[order], importance[order], color="#227c9d")
    ax.set_xlabel("Mean |SHAP| (percentage points of calibrated probability)")
    ax.set_title("15 fixed validation profiles across 5 folds — not population importance")
    fig.tight_layout()
    fig.savefig(directory / "global_shap.png", dpi=150)
    plt.close(fig)
    example = primary[0]["high"]
    values = np.asarray(example["contributions_percentage_points"])
    order = np.argsort(np.abs(values))
    labels = [f"{names[i]} = {example['feature_values'][names[i]]}" for i in order]
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(labels, values[order], color=np.where(values[order] >= 0, "#c43b4d", "#227c9d"))
    ax.axvline(0, color="black", linewidth=0.7)
    ax.set_xlabel("SHAP contribution (percentage points); red increases / blue decreases")
    ax.set_title(
        f"Fixed validation example: base {example['base_value']:.2%} "
        f"→ P(Attrition=Yes) {example['estimated_probability']:.2%}"
    )
    fig.tight_layout()
    fig.savefig(directory / "individual_shap.png", dpi=150)
    plt.close(fig)


def inspect_record(explainer: explain.Explainer, X: pd.DataFrame) -> dict[str, Any]:
    low = explainer.explain(X, cycles=16, seed=42)
    high = explainer.explain(X, cycles=64, seed=42)
    other = explainer.explain(X, cycles=64, seed=43)
    return {
        "low": low,
        "high": high,
        "other_seed": other,
        "seed_stability": explain.stability(high, other),
        "budget_stability": explain.stability(low, high),
    }


def gate(records: list[dict[str, Any]]) -> dict[str, Any]:
    p = explain.PROTOCOL
    seed = [r["seed_stability"] for r in records]
    budget = [r["budget_stability"] for r in records]
    all_stability = seed + budget
    summary = {
        "profiles_including_missing": len(records),
        "max_additivity_error": max(
            r[k]["additivity_error"] for r in records for k in ("low", "high", "other_seed")
        ),
        "max_seed_difference": max(s["max_abs_difference"] for s in seed),
        "max_budget_difference": max(s["max_abs_difference"] for s in budget),
        "min_material_sign_agreement": min(s["material_sign_agreement"] for s in all_stability),
        "mean_top5_overlap": float(np.mean([s["top5_overlap"] for s in all_stability])),
        "min_top5_overlap": min(s["top5_overlap"] for s in all_stability),
        "high_budget_seconds_median": float(np.median([r["high"]["seconds"] for r in records])),
        "high_budget_seconds_p95": float(
            np.quantile([r["high"]["seconds"] for r in records], 0.95)
        ),
        "masked_rows_checked": sum(
            r[k]["masked_rows_checked"] for r in records for k in ("low", "high", "other_seed")
        ),
    }
    checks = {
        "additivity": summary["max_additivity_error"] <= p["additivity_atol"],
        "seed_stability": summary["max_seed_difference"] <= p["max_seed_difference"],
        "budget_stability": summary["max_budget_difference"] <= p["max_budget_difference"],
        "sign_stability": summary["min_material_sign_agreement"]
        >= p["minimum_material_sign_agreement"],
        "mean_top5": summary["mean_top5_overlap"] >= p["minimum_mean_top5_overlap"],
        "min_top5": summary["min_top5_overlap"] >= p["minimum_top5_overlap"],
    }
    return {"passed": all(checks.values()), "checks": checks, "summary": summary}


def run(root: Path) -> None:
    report = a1.study_path(root, "artifacts/reports/adaboost_v3/s1")
    bundle_path = a1.study_path(root, "artifacts/models/adaboost_v3/s1")
    if report.exists() or bundle_path.exists():
        raise ValueError("s1 output exists; refusing overwrite. Inspect it before a new run.")
    parent = a1.study_path(root, "artifacts/reports/adaboost_v3/a1")
    pre = json.loads((parent / "preflight.json").read_text(encoding="utf-8"))
    meta = json.loads((parent / "metadata.json").read_text(encoding="utf-8"))
    protected = {
        **pre["protected_sha256"],
        **meta["output_sha256"],
        "artifacts/reports/adaboost_v3/a1/metadata.json": sha256_file(parent / "metadata.json"),
    }
    for relative, digest in protected.items():
        if sha256_file(a1.study_path(root, relative)) != digest:
            raise ValueError(f"a1 provenance mismatch: {relative}")
    current_versions = explain.versions()
    for name, expected in pre["versions"].items():
        actual = current_versions["python"] if name == "python" else version(name)
        if expected != actual:
            raise ValueError(f"a1 environment mismatch: {name}")
    config = load_protocol(a1.study_path(root, "configs/features/feature_set_v3_final_15.yaml"))
    if baseline.json_safe(config) != pre["feature_config"]:
        raise ValueError("a1 feature schema changed.")
    frame, assignment, provenance = baseline.prepare(root, config)
    names = [f["name"] for f in config["features"]]
    X, y = frame[names], frame.Attrition.to_numpy()
    oof = pd.read_csv(a1.study_path(root, "artifacts/predictions/adaboost_v3/a1/oof.csv"))
    oof = aligned_oof(
        oof[oof.model.eq("adaboost_baseline_sigmoid")], assignment, ["adaboost_baseline_sigmoid"]
    )
    plan = []
    for fold in range(1, 6):
        train = np.flatnonzero(assignment.fold.to_numpy() != fold)
        valid = np.flatnonzero(assignment.fold.to_numpy() == fold)
        chosen = valid[np.argsort(frame.iloc[valid].source_row.to_numpy())[:3]]
        background = np.random.default_rng(42).choice(train, size=32, replace=False)
        if set(background) & set(valid):
            raise ValueError("Background/validation overlap.")
        plan.append(
            {
                "fold": fold,
                "train": train.tolist(),
                "valid": valid.tolist(),
                "selected": chosen.tolist(),
                "background": background.tolist(),
            }
        )
    sources = [
        Path(__file__),
        root / "src/eris_ml/explainability/adaboost_shap.py",
        root / "scripts/manual_adaboost_v3.py",
    ]
    protected.update({p.relative_to(root).as_posix(): sha256_file(p) for p in sources})
    baseline.write_json(
        report / "preflight.json",
        {
            "created_at_utc": datetime.now(UTC).isoformat(),
            "protocol": explain.PROTOCOL,
            "fold_plan": plan,
            "versions": current_versions,
            "protected_sha256": protected,
            "data_provenance": provenance,
            "a1_policy": pre["policy"],
            "git": {
                k: subprocess.check_output(["git", *args], cwd=root, text=True).strip()
                for k, args in {
                    "branch": ["branch", "--show-current"],
                    "head": ["rev-parse", "HEAD"],
                    "status": ["status", "--short"],
                }.items()
            },
        },
    )
    print("s1 preflight locked; a1 hashes/alignment verified. No tuning.", flush=True)
    records, fold_traces = [], []
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        for item in plan:
            train, valid = item["train"], item["valid"]
            model = explain.fit_candidate(X.iloc[train], y[train], config)
            probability = a1.positive_probability(model, X.iloc[valid])
            expected = oof.iloc[valid].probability.to_numpy()
            if not np.allclose(probability, expected, rtol=1e-9, atol=1e-12):
                raise ValueError(f"Fold {item['fold']}: reproduced probability differs from a1.")
            explainer = explain.Explainer(model, config, X.iloc[item["background"]])
            explainer.explain(X.iloc[item["selected"][:1]], cycles=1)  # JIT/cache warm-up excluded.
            fold_records = []
            for index in item["selected"]:
                result = inspect_record(explainer, X.iloc[[index]])
                fold_records.append(
                    {
                        "fold": item["fold"],
                        "scenario": "original",
                        "development_position": index,
                        **result,
                    }
                )
                print(f"Fold {item['fold']} validation profile complete", flush=True)
            for missing in (
                ["TrainingTimesLastYear"],
                ["YearsWithCurrManager"],
                ["TrainingTimesLastYear", "YearsWithCurrManager"],
            ):
                masked = X.iloc[item["selected"][:1]].copy()
                masked[missing] = np.nan
                result = inspect_record(explainer, masked)
                fold_records.append(
                    {
                        "fold": item["fold"],
                        "scenario": "+".join(missing),
                        "development_position": item["selected"][0],
                        **result,
                    }
                )
            bad = X.iloc[item["selected"][:1]].copy()
            bad["Department"] = "unrecognized department"
            try:
                explainer.explain(bad, cycles=1)
            except ValueError:
                pass
            else:
                raise ValueError("Unknown categorical input was not rejected.")
            trace = {
                "fold": item["fold"],
                "raw_a1_max_difference": float(np.max(np.abs(probability - expected))),
                "invalid_input_rejected": True,
                "records": fold_records,
                "gate": gate(fold_records),
            }
            baseline.write_json(report / f"fold_{item['fold']}.json", trace)
            records.extend(fold_records)
            fold_traces.append(trace)
            print(f"Fold {item['fold']}: {trace['gate']}", flush=True)
        outcome = gate(records)
        plots(report, records)
        final_smoke: dict[str, Any] = {"bundle_created": False}
        if outcome["passed"]:
            print("SHAP gates passed; fitting frozen candidate on full development.", flush=True)
            final_model = explain.fit_candidate(X, y, config)
            bg_indices = np.random.default_rng(42).choice(len(X), size=32, replace=False)
            sample = X.iloc[[int(np.argmin(frame.source_row.to_numpy()))]]
            metadata = {
                "model_id": explain.MODEL_ID,
                "status": "research_demo_candidate",
                "feature_config": config,
                "feature_order": names,
                "class_order": [0, 1],
                "positive_class": 1,
                "threshold": None,
                "versions": current_versions,
                "business_approved": False,
                "production_approved": False,
                "fit_rows": len(X),
                "target_distribution": frame.Attrition.value_counts().to_dict(),
                "calibration": pre["policy"]["calibration"],
                "resolved_model": baseline.json_safe(final_model),
                "shap_protocol": explain.PROTOCOL,
                "background_development_positions": bg_indices.tolist(),
                "manual_sample": sample.iloc[0].to_dict(),
                "provenance": protected,
                "shap_validation_gate": outcome,
                "evaluation": {
                    "source": "a1 development OOF; exploratory model selection",
                    "not_full_fit_performance": True,
                    "metrics": json.loads((parent / "metrics.json").read_text())["models"][
                        "adaboost_baseline_sigmoid"
                    ]["pooled"],
                },
                "limitations": [
                    "IBM coding only; employer mapping not validated",
                    "No causal, intervention, fairness or 3-6 month claim",
                    "Consumed historical holdout is not independent V3 validation",
                    "Marginal SHAP hybrids may violate real-world feature dependencies",
                    "Small fixed SHAP audit sample; no general stability guarantee",
                ],
            }
            explain.save_bundle(bundle_path, final_model, X.iloc[bg_indices], metadata)
            loaded = explain.load_bundle(bundle_path)
            before = a1.positive_probability(final_model, sample)
            after = a1.positive_probability(loaded.model, sample)
            if not np.allclose(before, after, rtol=1e-12, atol=1e-15):
                raise ValueError("Bundle reload probability mismatch.")
            command = [
                sys.executable,
                "-B",
                str(root / "scripts/manual_adaboost_v3.py"),
                "--bundle",
                str(bundle_path),
                "--sample",
                "--json",
            ]
            smoke = json.loads(subprocess.check_output(command, cwd=root, text=True))
            if not np.isclose(smoke["estimated_probability"], before[0], rtol=1e-12, atol=1e-15):
                raise ValueError("Clean-process bundle probability differs.")
            final_smoke = {
                "bundle_created": True,
                "clean_process": True,
                "reload_max_difference": float(np.max(np.abs(before - after))),
                "example": smoke,
                "bundle_sha256": sha256_file(bundle_path / "pipeline.joblib"),
            }
    for relative, digest in protected.items():
        if sha256_file(a1.study_path(root, relative)) != digest:
            raise ValueError(f"Protected file changed: {relative}")
    baseline.write_json(
        report / "result.json",
        {
            "outcome": outcome,
            "bundle_smoke": final_smoke,
            "protected_hashes_unchanged": True,
            "final_test_accessed": False,
            "warnings": dict(Counter(f"{w.category.__name__}: {w.message}" for w in captured)),
            "max_a1_oof_difference": max(t["raw_a1_max_difference"] for t in fold_traces),
        },
    )
    lines = [
        "# AdaBoost baseline + sigmoid: permutation SHAP and demo bundle",
        "",
        f"SHAP gate passed: {outcome['passed']}; bundle created: {final_smoke['bundle_created']}.",
        "15 fixed original validation profiles (3/fold), plus 15 optional-missing variants.",
        "Models/backgrounds come only from each outer-training partition; all 1176 recreated",
        "probabilities checked against a1. Same baseline parameters, inner 4-fold seed 43.",
        "No tuning. Marginal permutation SHAP explains calibrated P(Attrition=Yes), not votes.",
        "Category integer codes and optional -1 are lossless masker transport only; decoded",
        "and domain-checked on every model call. They are NOT new model features/imputation.",
        "",
        "## Checks and warm latency",
        "",
        "```json",
        json.dumps(outcome, indent=2),
        "```",
        "",
        "## Overall and individual explanations",
        "",
        "![Fixed validation sample importance](global_shap.png)",
        "",
        "![Anonymous validation example](individual_shap.png)",
        "",
        "Individual example is outer-validation evidence; bundle smoke is a different",
        "full-development fitted model and is NOT a performance estimate. Base values are",
        "mean calibrated predictions over each 32-row background, not population prevalence.",
        "Contributions are percentage points; top-five alone need not sum to the probability.",
        "",
        "## Limits and use",
        "",
        "Selection of this candidate follows exploratory a1 results. No independent claim.",
        "Stability thresholds assess only this fixed small sample and fixed backgrounds.",
        "Correlated tenure/age/income hybrids can be implausible despite schema validity.",
        "Do not interpret this as causal/conditional SHAP, fairness approval or HR advice.",
        "IBM coding, future-employer mapping and time horizon remain unvalidated.",
        "No threshold, API, B2 change, final-test access or production approval.",
        "",
        "## Artifacts",
        "",
        "preflight.json locks sample/background rules, gates, versions and provenance.",
        "fold_*.json contain validation/stress contributions, values and paired stability.",
        "result.json contains clean-process smoke and bundle checksum.",
        "Offline console: .\\.venv\\Scripts\\python.exe scripts/manual_adaboost_v3.py",
        "",
    ]
    with (report / "report.md").open("x", encoding="utf-8") as stream:
        stream.write("\n".join(lines))
    outputs = list(report.iterdir()) + (list(bundle_path.iterdir()) if bundle_path.exists() else [])
    baseline.write_json(
        report / "manifest.json", {p.relative_to(root).as_posix(): sha256_file(p) for p in outputs}
    )
    print(f"Complete: {report / 'report.md'}", flush=True)


if __name__ == "__main__":
    run(ROOT)
