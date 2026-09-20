"""Calibration metrics, reliability visualization, and one development-only report."""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from eris_ml.evaluation.metrics import calculate_metrics
from eris_ml.models.calibration import METHODS, MODELS, OOF_COLUMNS

MODEL_LABELS = {"logistic": "Logistic baseline", "xgboost": "XGBoost tuned"}


def validate_probabilities(target: Any, probability: Any) -> tuple[np.ndarray, np.ndarray]:
    """Require nonempty aligned binary labels and finite probabilities in [0, 1]."""
    labels = np.asarray(target)
    scores = np.asarray(probability, dtype=float)
    if labels.ndim != 1 or scores.ndim != 1 or len(labels) != len(scores) or not len(labels):
        raise ValueError("Labels and probabilities must be aligned nonempty vectors.")
    if set(np.unique(labels).tolist()) - {0, 1}:
        raise ValueError("Labels must be binary 0/1.")
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("Probabilities must be finite and in [0, 1].")
    return labels.astype(int), scores


def reliability_bins(target: Any, probability: Any, count: int = 10) -> list[dict[str, Any]]:
    """Aggregate occupied quantile bins, keeping tied probabilities together."""
    labels, scores = validate_probabilities(target, probability)
    if type(count) is not int or count < 1:
        raise ValueError("Bin count must be a positive integer.")
    edges = np.unique(np.quantile(scores, np.linspace(0, 1, count + 1)))
    # Duplicate quantile edges collapse rather than splitting identical probabilities.
    indices = np.searchsorted(edges[1:-1], scores, side="right")
    bins = []
    for bin_id, index in enumerate(np.unique(indices), start=1):
        mask = indices == index
        predicted = float(np.mean(scores[mask]))
        observed = float(np.mean(labels[mask]))
        bins.append(
            {
                "bin_id": bin_id,
                "count": int(np.sum(mask)),
                "mean_predicted_probability": predicted,
                "observed_positive_rate": observed,
                "absolute_gap": abs(predicted - observed),
            }
        )
    return bins


def calibration_metrics(target: Any, probability: Any, bins: int = 10) -> dict[str, Any]:
    """Compute calibration, discrimination, and fixed-0.5 operational context."""
    labels, scores = validate_probabilities(target, probability)
    rows = reliability_bins(labels, scores, bins)
    operational = calculate_metrics(labels, scores, threshold=0.5)
    clipped = np.clip(scores, np.finfo(float).eps, 1 - np.finfo(float).eps)
    return {
        **operational,
        "log_loss": float(log_loss(labels, clipped, labels=[0, 1])),
        "expected_calibration_error": float(
            sum(row["count"] / len(scores) * row["absolute_gap"] for row in rows)
        ),
        "maximum_calibration_error": float(max(row["absolute_gap"] for row in rows)),
        "mean_probability": float(np.mean(scores)),
        "observed_positive_rate": float(np.mean(labels)),
        "reliability_bins": rows,
    }


def select_method(
    metrics: dict[str, dict[str, Any]],
    fold_metrics: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    """Prefer stable Brier gains under calibration/discrimination guardrails."""
    raw = metrics["raw"]
    improved = {
        method: sum(
            baseline["brier_score"] > candidate["brier_score"]
            for baseline, candidate in zip(fold_metrics["raw"], fold_metrics[method], strict=True)
        )
        for method in ("sigmoid", "isotonic")
    }
    fold_count = len(fold_metrics["raw"])
    reasons: dict[str, list[str]] = {}
    for method in ("sigmoid", "isotonic"):
        candidate = metrics[method]
        failures = []
        if candidate["brier_score"] >= raw["brier_score"]:
            failures.append("Brier not improved")
        if candidate["log_loss"] > raw["log_loss"] + 0.005:
            failures.append("Log Loss guardrail")
        if candidate["expected_calibration_error"] > raw["expected_calibration_error"] + 0.01:
            failures.append("ECE guardrail")
        if candidate["average_precision"] < raw["average_precision"] - 0.005:
            failures.append("PR-AUC guardrail")
        if candidate["roc_auc"] < raw["roc_auc"] - 0.005:
            failures.append("ROC-AUC guardrail")
        if improved[method] < (fold_count // 2 + 1):
            failures.append("fold stability")
        if method == "isotonic" and improved[method] < max(4, fold_count - 1):
            failures.append("isotonic stability")
        reasons[method] = failures
    eligible = {method: not reasons[method] for method in ("sigmoid", "isotonic")}
    selected = "raw"
    if eligible["sigmoid"]:
        selected = "sigmoid"
    if eligible["isotonic"]:
        iso = metrics["isotonic"]
        sig = metrics["sigmoid"]
        # Small positive sample: isotonic needs a clear, stable advantage over sigmoid.
        if (selected == "raw" and improved["isotonic"] >= max(4, fold_count - 1)) or (
            iso["brier_score"] <= sig["brier_score"] - 0.002
            and iso["expected_calibration_error"] <= sig["expected_calibration_error"] - 0.005
            and improved["isotonic"] >= max(4, fold_count - 1)
        ):
            selected = "isotonic"
    return {
        "method": selected, "eligible": eligible,
        "improved_folds": improved, "rejection_reasons": reasons,
    }


def summarize_calibration(oof: pd.DataFrame, bins: int = 10) -> dict[str, Any]:
    """Pooled OOF and outer-fold metrics, plus separate decisions per model."""
    if set(OOF_COLUMNS) - set(oof.columns) or "outer_fold" not in oof:
        raise ValueError("Calibration OOF is missing a candidate or outer-fold column.")
    if oof["source_row"].isna().any() or not oof["source_row"].is_unique:
        raise ValueError("Calibration OOF source_row must be complete and unique.")
    if oof["EmployeeNumber"].isna().any() or not oof["EmployeeNumber"].is_unique:
        raise ValueError("Calibration OOF EmployeeNumber must be complete and unique.")
    labels = oof["Attrition"].to_numpy()
    pooled: dict[str, dict[str, Any]] = {}
    folds: dict[str, list[dict[str, Any]]] = {}
    for column in OOF_COLUMNS:
        pooled[column] = calibration_metrics(labels, oof[column].to_numpy(), bins)
        folds[column] = [
            calibration_metrics(
                oof.loc[oof["outer_fold"] == fold, "Attrition"].to_numpy(),
                oof.loc[oof["outer_fold"] == fold, column].to_numpy(), bins,
            )
            for fold in sorted(oof["outer_fold"].unique())
        ]
    decisions = {
        model: select_method(
            {method: pooled[f"{model}_{method}_probability"] for method in METHODS},
            {method: folds[f"{model}_{method}_probability"] for method in METHODS},
        )
        for model in MODELS
    }
    return {"pooled": pooled, "folds": folds, "decisions": decisions}


def candidate_configuration(model: str, method: str) -> dict[str, Any]:
    """An unfitted candidate recipe; 5-fold calibration is for later full-dev fitting."""
    if model not in MODELS or method not in METHODS:
        raise ValueError("Unknown model or calibration method.")
    name = (
        "xgboost_tuned_calibrated_v1" if model == "xgboost"
        else "logistic_calibrated_reference_v1"
    )
    base = (
        "configs/models/xgboost_tuned_v1.yaml" if model == "xgboost"
        else "configs/models/logistic_regression.yaml"
    )
    return {
        "model_version": name.replace("_", "-"),
        "base_model_config": base,
        "calibration": {
            "method": "none" if method == "raw" else method,
            "cv_folds": 5,
            "shuffle": True,
            "random_state": 42,
            "ensemble": True,
        },
        "selection": {
            "development_only": True,
            "final_test_evaluated": False,
            "production_approved": False,
        },
    }


def plot_reliability(summary: dict[str, Any], destination: Path) -> None:
    """Render the six reliability curves on matched axes with an explicit diagonal."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
    colors = {"raw": "#1f77b4", "sigmoid": "#ff7f0e", "isotonic": "#2ca02c"}
    for axis, model in zip(axes, MODELS, strict=True):
        axis.plot([0, 1], [0, 1], linestyle="--", color="black", label="Perfect calibration")
        for method in METHODS:
            rows = summary["pooled"][f"{model}_{method}_probability"]["reliability_bins"]
            axis.plot(
                [row["mean_predicted_probability"] for row in rows],
                [row["observed_positive_rate"] for row in rows],
                marker="o", linewidth=1.5, markersize=4, color=colors[method], label=method,
            )
        axis.set(xlim=(0, 1), ylim=(0, 1), title=MODEL_LABELS[model],
                 xlabel="Mean predicted probability", ylabel="Observed attrition rate")
        axis.grid(alpha=0.25)
        axis.legend(loc="upper left")
    figure.suptitle("Development OOF reliability · 10 quantile bins (ties retained)")
    figure.tight_layout()
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=160)
    plt.close(figure)


def _n(value: float) -> str:
    return f"{value:.6f}"


def write_calibration_report(
    summary: dict[str, Any], metadata: dict[str, Any], destination: Path,
    plot_path: Path,
) -> None:
    """Write the only main Markdown report for the six cross-fitted candidates."""
    pooled, folds, decisions = summary["pooled"], summary["folds"], summary["decisions"]
    lines = [
        "# Cross-fitted probability calibration v1", "", "## A. Setup", "",
        f"- Development rows: {metadata['development_rows']}.",
        f"- Target distribution: 0={metadata['negative_count']}, 1={metadata['positive_count']}.",
        f"- Feature set: {metadata['feature_set_version']} ({metadata['feature_count']} features).",
        f"- Development CSV SHA-256: `{metadata['development_sha256']}`.",
        (
            f"- Outer CV: StratifiedKFold({metadata['outer_splits']}, shuffle=True, "
            f"random_state={metadata['outer_seed']})."
        ),
        (
            f"- Calibration CV: StratifiedKFold({metadata['calibration_splits']}, "
            f"shuffle=True, random_state={metadata['calibration_seed']}), entirely "
            "within each outer-training fold; ensemble=True."
        ),
        "- Methods: raw, sigmoid, isotonic; no hyperparameter or threshold search.",
        f"- Reliability bins: {metadata['bins']} quantile bins; duplicate edges collapsed.",
        "- Locked final test accessed: No.",
        (
            f"- Libraries: Python {metadata['python_version']}, pandas "
            f"{metadata['pandas_version']}, scikit-learn {metadata['sklearn_version']}, "
            f"XGBoost {metadata['xgboost_version']}, matplotlib "
            f"{metadata['matplotlib_version']}."
        ),
        (
            "- XGBoost outer parameters: exactly one rank-one record per fold from "
            "Step 10A search CSV; the artifact has no explicit model column, so its "
            "XGBoost filename, config model, and parameter allowlist establish provenance."
        ),
        (
        "- Step 10A outer folds and row alignment verified; both regenerated raw "
            "OOF vectors passed np.allclose(rtol=1e-7, atol=1e-9)."
        ),
        f"- Step 10A OOF CSV SHA-256: `{metadata['step10a_oof_sha256']}`.",
        f"- Step 10A XGBoost search CSV SHA-256: `{metadata['step10a_search_sha256']}`.",
        f"- Logistic baseline config SHA-256: `{metadata['logistic_config_sha256']}`.",
        f"- XGBoost candidate config SHA-256: `{metadata['xgboost_config_sha256']}`.",
        f"- Fold-selected XGBoost parameters: `{metadata['xgboost_fold_parameters']}`.",
        "", "## B. Calibration comparison", "",
        (
            "| Model | Method | Brier | Log Loss | ECE | MCE | "
            "Mean predicted probability | Observed positive rate |"
        ),
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for model in MODELS:
        for method in METHODS:
            row = pooled[f"{model}_{method}_probability"]
            lines.append(
                f"| {MODEL_LABELS[model]} | {method} | {_n(row['brier_score'])} | "
                f"{_n(row['log_loss'])} | {_n(row['expected_calibration_error'])} | "
                f"{_n(row['maximum_calibration_error'])} | {_n(row['mean_probability'])} | "
                f"{_n(row['observed_positive_rate'])} |"
            )
    lines += [
        "", "## C. Discrimination guardrail", "",
        "| Model | Method | PR-AUC | ROC-AUC | PR-AUC Δ vs raw | ROC-AUC Δ vs raw |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for model in MODELS:
        raw = pooled[f"{model}_raw_probability"]
        for method in METHODS:
            row = pooled[f"{model}_{method}_probability"]
            lines.append(
                f"| {MODEL_LABELS[model]} | {method} | {_n(row['average_precision'])} | "
                f"{_n(row['roc_auc'])} | "
                f"{_n(row['average_precision'] - raw['average_precision'])} | "
                f"{_n(row['roc_auc'] - raw['roc_auc'])} |"
            )
    lines += [
        "", "## D. Operational context at threshold 0.5", "",
        (
            "Threshold 0.5 is illustrative and has not been selected by business "
            "criteria. These values did not determine the calibration method."
        ),
        "",
        (
            "| Model | Method | Accuracy | Balanced accuracy | Precision | Recall | "
            "F1 | F2 | TN | FP | FN | TP | Alerts | Alert rate | FP/TP |"
        ),
        (
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | "
            "---: | ---: | ---: | ---: | ---: | ---: | ---: |"
        ),
    ]
    for model in MODELS:
        for method in METHODS:
            row = pooled[f"{model}_{method}_probability"]
            cm = row["confusion_matrix"]
            ratio = row["false_positives_per_true_positive"]
            lines.append(
                f"| {MODEL_LABELS[model]} | {method} | {_n(row['accuracy'])} | "
                f"{_n(row['balanced_accuracy'])} | {_n(row['precision'])} | {_n(row['recall'])} | "
                f"{_n(row['f1'])} | {_n(row['f2'])} | {cm['tn']} | {cm['fp']} | {cm['fn']} | "
                f"{cm['tp']} | {row['predicted_positive_count']} | {_n(row['alert_rate'])} | "
                f"{'N/A' if ratio is None else _n(ratio)} |"
            )
    lines += [
        "", "## E. Fold stability", "",
        (
            "| Model | Method | Mean Brier | Brier std | Mean Log Loss | Mean ECE | "
            "Mean PR-AUC | PR-AUC std | Brier improved folds |"
        ),
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for model in MODELS:
        baseline = folds[f"{model}_raw_probability"]
        for method in METHODS:
            records = folds[f"{model}_{method}_probability"]
            averages = {
                key: float(np.mean([item[key] for item in records]))
                for key in (
                    "brier_score", "log_loss", "expected_calibration_error", "average_precision"
                )
            }
            deviations = {
                key: float(np.std([item[key] for item in records], ddof=1))
                for key in ("brier_score", "average_precision")
            }
            improved = sum(
                old["brier_score"] > new["brier_score"]
                for old, new in zip(baseline, records, strict=True)
            )
            lines.append(
                f"| {MODEL_LABELS[model]} | {method} | {_n(averages['brier_score'])} | "
                f"{_n(deviations['brier_score'])} | {_n(averages['log_loss'])} | "
                f"{_n(averages['expected_calibration_error'])} | "
                f"{_n(averages['average_precision'])} | "
                f"{_n(deviations['average_precision'])} | {improved}/{len(records)} |"
            )
    lines += [
        "", "### Per-fold Brier improvement (raw minus calibrated)", "",
        "| Model | Method | "
        + " | ".join(f"Fold {fold}" for fold in range(1, len(baseline) + 1)) + " |",
        "| --- | --- | " + " | ".join("---:" for _ in baseline) + " |",
    ]
    for model in MODELS:
        raw_folds = folds[f"{model}_raw_probability"]
        for method in ("sigmoid", "isotonic"):
            calibrated_folds = folds[f"{model}_{method}_probability"]
            gains = [
                old["brier_score"] - new["brier_score"]
                for old, new in zip(raw_folds, calibrated_folds, strict=True)
            ]
            lines.append(
                f"| {MODEL_LABELS[model]} | {method} | "
                + " | ".join(_n(gain) for gain in gains) + " |"
            )
    lines += [
        "", "## F. Reliability diagram", "",
        f"![OOF reliability curves]({plot_path.name})", "",
        "Reliability curves are descriptive, not causal evidence.",
        "", "### Reliability bin values", "",
        (
            "| Model | Method | Bin ID | Count | Mean predicted probability | "
            "Observed positive rate | Absolute gap |"
        ),
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for model in MODELS:
        for method in METHODS:
            for row in pooled[f"{model}_{method}_probability"]["reliability_bins"]:
                lines.append(
                    f"| {MODEL_LABELS[model]} | {method} | {row['bin_id']} | "
                    f"{row['count']} | {_n(row['mean_predicted_probability'])} | "
                    f"{_n(row['observed_positive_rate'])} | {_n(row['absolute_gap'])} |"
                )
    lines += [
        "", "## G. Decision", "",
        (
            "A calibrated method must improve pooled Brier, stay within +0.005 Log "
            "Loss and +0.01 ECE, lose no more than 0.005 PR-AUC/ROC-AUC, and improve "
            "Brier in a majority of folds. Isotonic also requires at least 4 improved "
            "folds; over eligible sigmoid it needs at least 0.002 additional Brier "
            "gain and 0.005 ECE gain. These are descriptive guardrails, not "
            "significance tests."
        ),
        "",
    ]
    for model in MODELS:
        choice = decisions[model]["method"]
        raw = pooled[f"{model}_raw_probability"]
        selected = pooled[f"{model}_{choice}_probability"]
        gain = raw["brier_score"] - selected["brier_score"]
        log_delta = selected["log_loss"] - raw["log_loss"]
        ece_delta = selected["expected_calibration_error"] - raw["expected_calibration_error"]
        pr_delta = selected["average_precision"] - raw["average_precision"]
        roc_delta = selected["roc_auc"] - raw["roc_auc"]
        wins = decisions[model]["improved_folds"].get(choice, 0)
        lines.append(
            f"- {MODEL_LABELS[model]}: **{choice}**; Brier improvement {_n(gain)}, "
            f"Log Loss Δ {_n(log_delta)}, ECE Δ {_n(ece_delta)}, PR-AUC Δ {_n(pr_delta)}, "
            f"ROC-AUC Δ {_n(roc_delta)}, Brier improved "
            f"{wins}/{len(folds[f'{model}_raw_probability'])} folds."
        )
        for method in ("sigmoid", "isotonic"):
            candidate = pooled[f"{model}_{method}_probability"]
            rejection = decisions[model]["rejection_reasons"][method]
            status = "eligible" if not rejection else "rejected: " + ", ".join(rejection)
            candidate_ece_delta = candidate["expected_calibration_error"] - raw[
                "expected_calibration_error"
            ]
            lines.append(
                f"  - {method}: Brier gain "
                f"{_n(raw['brier_score'] - candidate['brier_score'])}; "
                f"Log Loss Δ {_n(candidate['log_loss'] - raw['log_loss'])}; "
                f"ECE Δ {_n(candidate_ece_delta)}; "
                f"PR-AUC Δ {_n(candidate['average_precision'] - raw['average_precision'])}; "
                f"ROC-AUC Δ {_n(candidate['roc_auc'] - raw['roc_auc'])}; "
                f"Brier improved {decisions[model]['improved_folds'][method]}/"
                f"{len(folds[f'{model}_raw_probability'])} folds; {status}."
            )
    lines += [
        (
            "- XGBoost tuned remains the development champion for the next "
            "business-aware threshold study; Logistic baseline remains the stable "
            "reference. Calibration does not select a production model."
        ),
        "- Final test remains locked; conditions to open it are not yet met.",
        (
            "- The 5-fold calibration setting in candidate YAMLs is an unfitted "
            "future full-development recipe, distinct from the 4-fold calibration "
            "CV nested inside this 5-fold evaluation."
        ),
        (
            "- Next: choose a threshold using HR alert capacity, false-negative "
            "and false-positive costs, minimum recall, and maximum alert rate—not "
            "maximum F1 alone."
        ),
        "",
    ]
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines), encoding="utf-8")
