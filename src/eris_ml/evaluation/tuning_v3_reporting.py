"""Append one nested-tuning section to the existing V3 summary; preserve baseline text."""

from pathlib import Path
from typing import Any


def section(run_id: str, result: dict[str, Any], checkpoints: list[dict[str, Any]]) -> str:
    lines = [
        f"\n\n## Nested tuning {run_id} — development only\n",
        "Saved V2 outer folds; inner stratified 4-fold seed 43; average_precision refit. "
        "Exact ties choose the first candidate in deterministic search order. "
        "Logistic: 20 candidates/fold, liblinear, max_iter=10000, seed 42; "
        "L1/L2 API mapping recorded in preflight. RF: 40 sampled candidates/fold, "
        "search seed 43, estimator seed 42. Search/outer jobs=1, RF jobs=4, native threads=1. "
        "1,200 inner fits + 10 outer-training refits; no full-development fit.\n",
        f"Positive rate: {result['positive_rate']:.6f}. "
        "Pooled AP and mean fold AP are different aggregation estimands.\n",
        "| Model | Version | Pooled AP | Fold AP mean ± sample SD | ROC-AUC | Brier | Log loss |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for model in ("logistic_regression", "random_forest", "xgboost", "dummy_prior"):
        for variant in ("baseline", "tuned"):
            if model not in result[variant]:
                continue
            row = result[variant][model]
            p, f = row["pooled_oof"], row["fold_mean_std"]["average_precision"]
            lines.append(
                f"| {model} | {variant} | {p['average_precision']:.6f} | "
                f"{f['mean']:.6f} ± {f['sample_std']:.6f} | {p['roc_auc']:.6f} | "
                f"{p['brier_score']:.6f} | {p['log_loss']:.6f} |"
            )
    lines += [
        "\n### Paired deltas (tuned − baseline)\n",
        "| Model | Pooled ΔAP | ΔROC | ΔBrier | ΔLog loss | Improved AP folds |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    details = []
    for model, comparison in result["comparisons"].items():
        d = comparison["pooled_delta_tuned_minus_baseline"]
        lines.append(
            f"| {model} | {d['average_precision']:+.6f} | {d['roc_auc']:+.6f} | "
            f"{d['brier_score']:+.6f} | {d['log_loss']:+.6f} | "
            f"{comparison['ap_improved_folds']}/5 |"
        )
        deltas = ", ".join(
            f"fold {f['fold']}: {f['average_precision']:+.6f}" for f in comparison["fold_deltas"]
        )
        details.append(
            f"\n{model}: {deltas}. "
            f"Small positive AP change (<0.005): {comparison['small_positive_ap_change']}.\n"
        )
    lines += details
    lines += [
        "\n### Outer-fold tuned metrics and selections\n",
        "| Model | Fold | AP | ROC | Brier | Log loss | Selected parameters |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for c in checkpoints:
        p = c["outer_metrics"]
        parameters = "; ".join(
            f"{k.removeprefix('model__')}={v}" for k, v in c["best_params"].items()
        )
        lines.append(
            f"| {c['model']} | {c['fold']} | {p['average_precision']:.6f} | "
            f"{p['roc_auc']:.6f} | {p['brier_score']:.6f} | {p['log_loss']:.6f} | "
            f"{parameters} |"
        )
    lines += [
        "\n### Operational reference only: threshold 0.5\n",
        "| Model | Version | Precision | Recall | F1 | F2 | Balanced accuracy "
        "| TP | FP | TN | FN | Alerts |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model in ("logistic_regression", "random_forest"):
        for variant in ("baseline", "tuned"):
            p = result[variant][model]["pooled_oof"]
            cells = [
                f"{p[k]:.6f}" for k in ("precision", "recall", "f1", "f2", "balanced_accuracy")
            ]
            cells += [str(p[k]) for k in ("tp", "fp", "tn", "fn", "alerts")]
            lines.append(f"| {model} | {variant} | " + " | ".join(cells) + " |")
    lines += [
        "\nAll metrics and fold sample SDs: `" + run_id + "/metrics_comparison.json`. "
        "Preflight includes exact candidates, pipelines, data/source/config hashes, "
        "versions and Git state. Checkpoints include all inner scores, selected parameters, "
        "outer predictions and warnings; inner best scores are NOT independent evaluation.\n",
        "Selections differ by outer fold; they are not one final parameter configuration. "
        "No significance test was performed. Brier/log-loss increases indicate poorer raw "
        "probability quality, even when AP increases. A <0.005 AP gain is only a small change. "
        "Threshold 0.5 is not an HR policy. No calibration, threshold search or deployment. "
        "No final-test CSV access; V1's test is not a fresh V3 holdout. IBM cross-validation "
        "does not establish performance at another company or a 3–6 month prediction horizon.\n",
        f"Warnings captured: {sum(sum(c['warnings'].values()) for c in checkpoints)}. "
        "Exact messages/counts retained in fold checkpoints.\n",
    ]
    lines += ["\n### Candidates to consider for calibration (not executed)\n"]
    for model, comparison in result["comparisons"].items():
        delta = comparison["pooled_delta_tuned_minus_baseline"]
        improved = comparison["ap_improved_folds"]
        keep_tuned = delta["average_precision"] >= 0.005 and improved >= 3
        proposed = "tuned alongside its baseline control" if keep_tuned else "baseline"
        lines.append(
            f"{model}: carry {proposed} forward for consideration. "
            f"Pooled ΔAP={delta['average_precision']:+.6f}, {improved}/5 AP folds improved; "
            f"ΔBrier={delta['brier_score']:+.6f}, ΔLog loss={delta['log_loss']:+.6f}. "
            "This is a development recommendation, not a final model or production approval.\n"
        )
    return "\n".join(lines)


def append_summary(
    path: Path,
    original: str,
    run_id: str,
    result: dict[str, Any],
    checkpoints: list[dict[str, Any]],
) -> None:
    addition = section(run_id, result, checkpoints)
    current = path.read_text(encoding="utf-8")
    if current == original + addition:
        return
    if current != original:
        raise ValueError("Summary changed after preflight; refusing to overwrite existing text.")
    with path.open("a", encoding="utf-8") as stream:
        stream.write(addition)
