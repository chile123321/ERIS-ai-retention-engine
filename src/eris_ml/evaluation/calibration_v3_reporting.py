"""Aggregate V3 calibration, per-fold batch policies and inference-only missingness."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from eris_ml.evaluation.calibration_reporting import reliability_bins
from eris_ml.evaluation.threshold_policy_v2 import budget, operational, top_k
from eris_ml.models import baseline_v3 as baseline
from eris_ml.models import tuning_v3 as tuning
from eris_ml.models.calibration_v3 import CANDIDATES, MODELS, REFERENCE, SCENARIOS


def select_method(
    raw: dict[str, Any], sigmoid: dict[str, Any], rule: dict[str, Any]
) -> dict[str, Any]:
    deltas = {
        name: sigmoid["pooled_oof"][name] - raw["pooled_oof"][name]
        for name in ("average_precision", "roc_auc", "brier_score", "log_loss")
    }
    fold_deltas = [
        {"fold": a["fold"], **{name: b[name] - a[name] for name in deltas}}
        for a, b in zip(raw["folds"], sigmoid["folds"], strict=True)
    ]
    brier_folds = sum(d["brier_score"] < 0 for d in fold_deltas)
    loss_folds = sum(d["log_loss"] < 0 for d in fold_deltas)
    checks = {
        "pooled_brier_reduced": deltas["brier_score"] < 0,
        "pooled_log_loss_reduced": deltas["log_loss"] < 0,
        "brier_fold_support": brier_folds >= rule["minimum_brier_improved_folds"],
        "ap_guardrail": deltas["average_precision"] >= -rule["maximum_ap_drop"],
        "roc_guardrail": deltas["roc_auc"] >= -rule["maximum_roc_auc_drop"],
    }
    return {
        "recommended_method": "sigmoid" if all(checks.values()) else "raw",
        "checks": checks,
        "pooled_delta_sigmoid_minus_raw": deltas,
        "fold_deltas": fold_deltas,
        "brier_improved_folds": brier_folds,
        "log_loss_improved_folds": loss_folds,
        "small_positive_brier_improvement": 0
        < -deltas["brier_score"]
        < rule["small_brier_improvement"],
        "statistical_significance_test": False,
        "production_approved": False,
    }


def policy_metrics(frame: pd.DataFrame, percent: int) -> dict[str, Any]:
    """Micro aggregate predictions selected separately in each saved validation fold."""
    block = frame.reset_index(drop=True)
    selected = np.zeros(len(block), dtype=int)
    records = []
    for fold, rows in block.groupby("fold", sort=True):
        pred = top_k(rows.probability.to_numpy(), rows.source_row.to_numpy(), percent)
        selected[rows.index.to_numpy()] = pred
        record = operational(rows.y_true.to_numpy(), pred)
        expected = budget(len(rows), percent)
        if record["alerts"] != expected:
            raise ValueError("Top-k fold budget mismatch.")
        records.append(
            {
                "fold": int(str(fold)),
                "n": len(rows),
                "positives": int(rows.y_true.sum()),
                "k": expected,
                **record,
            }
        )
    pooled = {
        "n": len(block),
        "positives": int(block.y_true.sum()),
        **operational(block.y_true.to_numpy(), selected),
    }
    for name in ("alerts", "tp", "fp", "fn", "tn"):
        if pooled[name] != sum(record[name] for record in records):
            raise ValueError("Pooled top-k counts differ from fold sums.")
    return {"capacity_percent": percent, "pooled": pooled, "folds": records}


def stress_metrics(block: pd.DataFrame, original: pd.DataFrame) -> dict[str, Any]:
    paired = block.merge(
        original[["source_row", "y_true", "fold", "probability", "top_10"]],
        on="source_row",
        suffixes=("", "_original"),
        validate="one_to_one",
    )
    if len(paired) != len(original) or any(
        not np.array_equal(paired[k], paired[f"{k}_original"]) for k in ("y_true", "fold")
    ):
        raise ValueError("Stress/original alignment mismatch.")

    def describe(rows: pd.DataFrame) -> dict[str, Any]:
        metric = baseline.metrics(rows.y_true.to_numpy(), rows.probability.to_numpy(), 0.5)
        old_metric = baseline.metrics(
            rows.y_true.to_numpy(), rows.probability_original.to_numpy(), 0.5
        )
        overlap = int(((rows.top_10 == 1) & (rows.top_10_original == 1)).sum())
        denominator = int(rows.top_10_original.sum())
        return {
            "n": len(rows),
            "probability_metrics": metric,
            "delta_vs_original": {
                key: metric[key] - old_metric[key]
                for key in ("average_precision", "brier_score", "log_loss")
            },
            "top_10": operational(rows.y_true.to_numpy(), rows.top_10.to_numpy()),
            "mean_absolute_probability_change": float(
                np.mean(np.abs(rows.probability.to_numpy() - rows.probability_original.to_numpy()))
            ),
            "top_10_overlap_count": overlap,
            "top_10_original_count": denominator,
            "top_10_overlap_fraction": overlap / denominator if denominator else None,
        }

    return {
        "pooled": describe(paired),
        "folds": [
            {"fold": int(str(fold)), **describe(rows)}
            for fold, rows in paired.groupby("fold", sort=True)
        ],
    }


def analyze(
    oof: pd.DataFrame, reference: pd.DataFrame, policy: dict[str, Any]
) -> tuple[dict[str, Any], pd.DataFrame]:
    original = oof.loc[oof.scenario == "original"]
    metrics = tuning.summarize(original)
    decisions = {
        m: select_method(metrics[f"{m}_raw"], metrics[f"{m}_sigmoid"], policy["recommendation"])
        for m in MODELS
    }
    bins = []
    for candidate, rows in original.groupby("model", sort=True):
        metric = metrics[str(candidate)]
        metric["mean_probability"] = float(rows.probability.mean())
        metric["observed_positive_rate"] = float(rows.y_true.mean())
        for b in reliability_bins(rows.y_true, rows.probability, count=policy["bins"]["count"]):
            bins.append(
                {
                    "model": candidate,
                    **b,
                    "positives": int(round(b["observed_positive_rate"] * b["count"])),
                }
            )
    policies = {}
    for candidate in CANDIDATES + [REFERENCE]:
        rows = (
            reference.loc[reference.model == candidate]
            if candidate == REFERENCE
            else original.loc[original.model == candidate]
        )
        policies[candidate] = [policy_metrics(rows, p) for p in policy["top_k"]["capacity_percent"]]
    for index in range(3):
        expected = policies[CANDIDATES[0]][index]
        for candidate_policies in policies.values():
            if candidate_policies[index]["pooled"]["alerts"] != expected["pooled"]["alerts"] or [
                f["k"] for f in candidate_policies[index]["folds"]
            ] != [f["k"] for f in expected["folds"]]:
                raise ValueError("Unequal top-k budgets across candidates.")
    missing: dict[str, Any] = {}
    for candidate in CANDIDATES:
        missing[candidate] = {}
        for scenario in SCENARIOS:
            block = oof.loc[(oof.model == candidate) & (oof.scenario == scenario)]
            missing[candidate][scenario] = stress_metrics(
                block, original.loc[original.model == candidate]
            )
    return {
        "calibration": {"metrics": metrics, "decisions": decisions},
        "top_k": policies,
        "optional_missing": missing,
    }, pd.DataFrame(bins)


def plot_bins(bins: pd.DataFrame, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if path.exists():
        raise FileExistsError("Refusing to replace a reliability plot.")
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharex=True, sharey=True)
    for ax, model in zip(axes, MODELS, strict=True):
        ax.plot([0, 1], [0, 1], "--", color="gray", label="Perfect calibration")
        for method, marker in (("raw", "o"), ("sigmoid", "s")):
            rows = bins.loc[bins.model == f"{model}_{method}"]
            ax.plot(
                rows.mean_predicted_probability,
                rows.observed_positive_rate,
                marker=marker,
                label=method,
            )
        ax.set(
            title=model.replace("_", " "),
            xlim=(0, 1),
            ylim=(0, 1),
            xlabel="Mean predicted probability",
            ylabel="Observed IBM attrition rate",
        )
        ax.grid(alpha=0.2)
        ax.legend(loc="upper left")
    figure.suptitle("Development OOF diagnostics — 5 quantile bins; counts in reliability_bins.csv")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def report_section(run_id: str, analysis: dict[str, Any], checkpoints: list[dict[str, Any]]) -> str:
    calibration = analysis["calibration"]
    lines = [
        f"\n\n## Calibration, top-k and optional missing {run_id} — development only\n",
        "1176 development rows (986 negative, 190 positive), the same 15 ordered features "
        "and saved V2/V3 outer folds. Logistic baseline r1 and fold-specific RF tuned t1 only. "
        "CalibratedClassifierCV: sigmoid, ensemble=False, stratified 4-fold seed 43 inside "
        "outer training. Fifty base-estimator fits + ten sigmoid fits, zero extra stress fits. "
        "Preprocessing and median imputation remain inside the pipeline. Raw probabilities "
        "come from the calibrator's base estimator refitted on the entire outer-training fold.\n",
        "RF hyperparameters were selected inside outer training in t1 and are now fixed. "
        "Inner calibration scores are not an independent assessment of the whole tuning process; "
        "outer validation is excluded from search, preprocessing fit and calibration.\n",
        "Recommendation locked before fitting: both pooled Brier and log loss must decrease, "
        "Brier must improve in at least 3/5 folds, and neither AP nor ROC-AUC may drop by more "
        "than 0.005. A Brier improvement below 0.001 is small. These are project heuristics, "
        "not a significance test.\n",
        "| Candidate | Pooled AP | Fold AP mean ± sample SD "
        "| ROC-AUC | Brier | Log loss | Mean p |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for candidate in CANDIDATES:
        row = calibration["metrics"][candidate]
        p, ap = row["pooled_oof"], row["fold_mean_std"]["average_precision"]
        lines.append(
            f"| {candidate} | {p['average_precision']:.6f} | {ap['mean']:.6f} ± "
            f"{ap['sample_std']:.6f} | {p['roc_auc']:.6f} | {p['brier_score']:.6f} | "
            f"{p['log_loss']:.6f} | {row['mean_probability']:.6f} |"
        )
    lines += [
        "\nObserved positive rate: 190/1176 = 0.161565.\n",
        "| Model | Fold | ΔAP | ΔROC | ΔBrier | ΔLog loss (sigmoid − raw) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for model, decision in calibration["decisions"].items():
        for d in decision["fold_deltas"]:
            lines.append(
                f"| {model} | {d['fold']} | {d['average_precision']:+.6f} | "
                f"{d['roc_auc']:+.6f} | {d['brier_score']:+.6f} | {d['log_loss']:+.6f} |"
            )
    for model, decision in calibration["decisions"].items():
        lines.append(
            f"\n{model}: recommend **{decision['recommended_method']}**. "
            f"Brier improved {decision['brier_improved_folds']}/5 folds; log loss improved "
            f"{decision['log_loss_improved_folds']}/5. Checks: {decision['checks']}. "
            f"Small Brier improvement: {decision['small_positive_brier_improvement']}.\n"
        )
    lines += [
        "\n### Reliability diagnostics\n",
        f"![Development reliability]({run_id}/reliability.png)\n",
        f"Five quantile bins and sample support: `{run_id}/reliability_bins.csv`. "
        "Ties stay together; duplicate edges collapse and empty bins are omitted. These plots "
        "are development diagnostics: few bins or a lower Brier score do not establish fully "
        "reliable probabilities. Increasing sigmoid can retain within-fold ranks while changing "
        "cross-fold ranks; pooled AP is not required to stay identical.\n",
        "### Hypothetical top-k, selected separately in each validation fold\n",
        "k=floor(percent*n/100), descending probability, ascending source_row for exact ties; "
        "labels never break ties. Pooled recall=sum(TP)/190 and precision=sum(TP)/sum(alerts). "
        "These technical CV folds are not real HR batches. Top-k requires a whole batch and "
        "is not a fixed probability threshold. No capacity or HR policy is approved.\n",
        "| Candidate | Capacity | Alerts | TP | FP | FN | Precision | Recall | F2 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for candidate, policies in analysis["top_k"].items():
        for policy in policies:
            p = policy["pooled"]
            lines.append(
                f"| {candidate} | {policy['capacity_percent']}% | {p['alerts']} | "
                f"{p['tp']} | {p['fp']} | {p['fn']} | {p['precision']:.6f} | "
                f"{p['recall']:.6f} | {p['f2']:.6f} |"
            )
    lines += [
        "\n### Inference-only optional-missing stress\n",
        "Only copies of outer validation are masked. No retraining, recalibration, imputer refit "
        "or raw file modification. Overlap=intersection/original top-10% list size; pooled overlap "
        "sums the five intersections and divides by total original alerts. This is not Jaccard. "
        "It is a missing-input stress test, not an estimate of real company missingness. "
        "Technical ability to accept missing inputs does not imply adequate prediction quality.\n",
        "| Candidate | Missing | AP (Δ) | Brier | Log loss | Top10 TP "
        "| Precision | Recall | Mean abs Δp | Overlap |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for candidate, scenarios in analysis["optional_missing"].items():
        for scenario, record in scenarios.items():
            p = record["pooled"]
            m, top = p["probability_metrics"], p["top_10"]
            lines.append(
                f"| {candidate} | {scenario} | {m['average_precision']:.6f} "
                f"({p['delta_vs_original']['average_precision']:+.6f}) | {m['brier_score']:.6f} | "
                f"{m['log_loss']:.6f} | {top['tp']} | {top['precision']:.6f} | "
                f"{top['recall']:.6f} | {p['mean_absolute_probability_change']:.6f} | "
                f"{p['top_10_overlap_fraction']:.2%} |"
            )
    for candidate, scenarios in analysis["optional_missing"].items():
        worst = min(
            (s for s in scenarios if s != "original"),
            key=lambda s: scenarios[s]["pooled"]["delta_vs_original"]["average_precision"],
        )
        p = scenarios[worst]["pooled"]
        lines.append(
            f"\n{candidate}: lowest stress AP occurs for {worst}, "
            f"ΔAP={p['delta_vs_original']['average_precision']:+.6f}, "
            f"ΔBrier={p['delta_vs_original']['brier_score']:+.6f}, "
            f"top10 TP={p['top_10']['tp']} vs original "
            f"{scenarios['original']['pooled']['top_10']['tp']}. "
            "Missing policy remains unchanged.\n"
        )
    maximum = max(c["raw_max_absolute_difference"] for c in checkpoints)
    rank_changes = sum(
        d["exact_rank_position_changes"]
        for c in checkpoints
        for d in c["rank_diagnostics"].values()
    )
    lines += [
        "\n### Provenance, checks and next candidate\n",
        f"Raw max absolute difference from r1/t1 by stable ID: {maximum:.17g} "
        "(rtol=1e-7, atol=1e-9). Classes [0,1]; all candidates/scenarios cover development IDs. "
        f"Exact rank-position changes across all 40 fold/scenario checks: {rank_changes}. "
        "Checkpoints record sigmoid a/b, response method, ties, material inversions and "
        "top-k membership changes; material reversals stop the run rather than being ignored. "
        f"Captured warnings: {sum(sum(c['warnings'].values()) for c in checkpoints)}.\n",
        f"Preflight/metadata and ten fold checkpoints: `{run_id}/`. Complete per-fold calibration, "
        f"top-k and stress metrics: `{run_id}/calibration.json`, `{run_id}/top_k.json`, "
        f"`{run_id}/optional_missing.json`. Employee-level OOF stays in ignored artifacts.\n",
        "For the next research/demo packaging and SHAP review, consider RF tuned with its "
        f"recommended method ({calibration['decisions']['rf_tuned']['recommended_method']}), "
        "retaining Logistic baseline with its recommended method "
        f"({calibration['decisions']['logistic_baseline']['recommended_method']}) as a comparator. "
        "This is a proposal, not a final production model: fold-specific RF parameters are not "
        "one full-development fit configuration. Review the missingness trade-off before locking "
        "a packaging candidate; no new parameter search or missing policy has been performed.\n",
        "No full-development final fit, holdout access, threshold search, SHAP, bundle or API. "
        "V1 final test was used once and is not an independent new V3 holdout. IBM cross-sectional "
        "results do not validate performance/fairness in another company, a 3–6 month horizon "
        "or intervention benefits. No HR or production approval.\n",
    ]
    return "\n".join(lines)


def append_summary(
    path: Path,
    preflight: dict[str, Any],
    run_id: str,
    analysis: dict[str, Any],
    checkpoints: list[dict[str, Any]],
) -> None:
    data = path.read_bytes()
    original_size = preflight["summary_before_bytes"]
    if hashlib.sha256(data[:original_size]).hexdigest() != preflight["summary_before_sha256"]:
        raise ValueError("Baseline/tuning summary prefix changed; refusing to append.")
    addition = report_section(run_id, analysis, checkpoints)
    current = path.read_text(encoding="utf-8")
    if current == preflight["summary_before"] + addition:
        return
    if current != preflight["summary_before"]:
        raise ValueError("Summary changed after preflight; refusing to overwrite it.")
    with path.open("a", encoding="utf-8") as stream:
        stream.write(addition)
