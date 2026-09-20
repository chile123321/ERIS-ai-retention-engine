"""Single Markdown report for development-only nested tuning."""

import json
from pathlib import Path
from typing import Any

from eris_ml.models.tuning import RESULT_NAMES, parameter_frequencies

LABELS = {
    "logistic_baseline": "Logistic baseline",
    "logistic_tuned": "Logistic tuned",
    "xgboost_baseline": "XGBoost baseline",
    "xgboost_tuned": "XGBoost tuned",
}
METRICS = (
    "accuracy", "balanced_accuracy", "average_precision", "roc_auc", "precision",
    "recall", "f1", "f2", "brier_score",
)


def _number(value: float) -> str:
    return f"{value:.6f}"


def decide_candidates(results: dict[str, Any]) -> dict[str, Any]:
    """Apply the stated PR-AUC, Brier, and overfitting decision rule."""
    metrics = results["oof_metrics"]
    summary = results["fold_summary"]
    logistic_best = (
        "logistic_tuned"
        if metrics["logistic_tuned"]["average_precision"]
        > metrics["logistic_baseline"]["average_precision"]
        else "logistic_baseline"
    )
    xgboost_improved = (
        metrics["xgboost_tuned"]["average_precision"]
        > metrics["xgboost_baseline"]["average_precision"]
    )
    xgboost_best = "xgboost_tuned" if xgboost_improved else "xgboost_baseline"
    xgboost_fold_wins = sum(
        tuned["validation_average_precision"] > baseline["validation_average_precision"]
        for baseline, tuned in zip(
            results["fold_records"]["xgboost_baseline"],
            results["fold_records"]["xgboost_tuned"],
            strict=True,
        )
    )
    xgboost_champion = (
        xgboost_improved
        and xgboost_fold_wins >= 4
        and metrics[xgboost_best]["average_precision"]
        > metrics[logistic_best]["average_precision"] + 0.01
        and summary[xgboost_best]["overfitting_gap"]["mean"] < 0.347023
        and metrics[xgboost_best]["brier_score"]
        <= metrics["xgboost_baseline"]["brier_score"] + 0.01
    )
    champion = xgboost_best if xgboost_champion else logistic_best
    challenger = logistic_best if xgboost_champion else (xgboost_best if xgboost_improved else None)
    return {
        "champion": champion,
        "challenger": challenger,
        "xgboost_improved": xgboost_improved,
        "xgboost_fold_wins": xgboost_fold_wins,
        "logistic_improved": logistic_best == "logistic_tuned",
    }


def write_tuning_report(
    results: dict[str, Any],
    full_search: dict[str, Any],
    search_configs: dict[str, dict[str, Any]],
    metadata: dict[str, Any],
    destination: Path,
) -> dict[str, Any]:
    """Write the only main tuning report and return its champion decision."""
    decision = decide_candidates(results)
    metrics = results["oof_metrics"]
    summaries = results["fold_summary"]
    lines = [
        "# Nested hyperparameter tuning v1",
        "",
        "## A. Experiment setup",
        "",
        f"- Development rows: {metadata['development_rows']}",
        f"- Target distribution: 0={metadata['negative_count']}, 1={metadata['positive_count']}",
        f"- Feature set: {metadata['feature_set_version']} ({metadata['feature_count']} features)",
        (
            f"- Outer CV: StratifiedKFold({metadata['outer_splits']} folds, shuffle=True, "
            f"seed={metadata['outer_seed']})"
        ),
        (
            f"- Inner CV: StratifiedKFold({metadata['inner_splits']} folds, shuffle=True, "
            f"seed={metadata['inner_seed']})"
        ),
        "- Primary metric: average_precision (PR-AUC)",
        f"- Operational threshold: {metadata['threshold']}",
        "- Locked final test accessed: No.",
        f"- Development CSV SHA-256: `{metadata['development_sha256']}`",
        (
            f"- Libraries: Python {metadata['python_version']}, pandas "
            f"{metadata['pandas_version']}, scikit-learn {metadata['sklearn_version']}, "
            f"XGBoost {metadata['xgboost_version']}."
        ),
        "",
        "## B. Search spaces",
        "",
        (
            "- Logistic GridSearchCV: "
            f"{metadata['logistic_combinations']} combinations × {metadata['inner_splits']} "
            "inner folds per outer fold; penalty and C only."
        ),
        (
            "- XGBoost RandomizedSearchCV: "
            f"{search_configs['xgboost']['n_iter']} sampled combinations × "
            f"{metadata['inner_splits']} inner folds per outer fold."
        ),
        "- Preprocessing, class weights, scale_pos_weight, and threshold were not tuned.",
        "- No resampling, early stopping, calibration, or SHAP was used.",
        "",
        "## C. Nested OOF comparison",
        "",
        (
            "| Model | Accuracy | Balanced Accuracy | PR-AUC | ROC-AUC | Precision | "
            "Recall | F1 | F2 | Brier |"
        ),
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name in RESULT_NAMES:
        values = " | ".join(_number(metrics[name][metric]) for metric in METRICS)
        lines.append(f"| {LABELS[name]} | {values} |")

    lines.extend(
        [
            "",
            "## D. Operational impact",
            "",
            "| Model | Alerts | Alert rate | TN | FP | FN | TP | FP/TP |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for name in RESULT_NAMES:
        metric = metrics[name]
        matrix = metric["confusion_matrix"]
        ratio = metric["false_positives_per_true_positive"]
        lines.append(
            f"| {LABELS[name]} | {metric['predicted_positive_count']} | "
            f"{_number(metric['alert_rate'])} | {matrix['tn']} | {matrix['fp']} | "
            f"{matrix['fn']} | {matrix['tp']} | "
            f"{'N/A' if ratio is None else _number(ratio)} |"
        )

    lines.extend(
        [
            "",
            "## E. Fold stability and overfitting",
            "",
            (
                "| Model | Mean train PR-AUC | Mean validation PR-AUC | Validation PR-AUC std | "
                "Overfitting gap | Inner best PR-AUC | Search time (s) | Refit time (s) | "
                "Prediction time (s) |"
            ),
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for name in RESULT_NAMES:
        summary = summaries[name]
        best_scores = [
            row["inner_best_average_precision"]
            for row in results["fold_records"][name]
            if row["inner_best_average_precision"] is not None
        ]
        inner_mean = "N/A" if not best_scores else _number(sum(best_scores) / len(best_scores))
        lines.append(
            f"| {LABELS[name]} | "
            f"{_number(summary['train_average_precision']['mean'])} | "
            f"{_number(summary['validation_average_precision']['mean'])} | "
            f"{_number(summary['validation_average_precision']['standard_deviation'])} | "
            f"{_number(summary['overfitting_gap']['mean'])} | {inner_mean} | "
            f"{_number(summary['search_time_seconds']['mean'])} | "
            f"{_number(summary['refit_time_seconds']['mean'])} | "
            f"{_number(summary['prediction_time_seconds']['mean'])} |"
        )

    logistic_delta = (
        metrics["logistic_tuned"]["average_precision"]
        - metrics["logistic_baseline"]["average_precision"]
    )
    xgboost_delta = (
        metrics["xgboost_tuned"]["average_precision"]
        - metrics["xgboost_baseline"]["average_precision"]
    )
    cross_delta = (
        metrics["xgboost_tuned"]["average_precision"]
        - metrics["logistic_tuned"]["average_precision"]
    )
    lines.extend(
        [
            "",
            "## F. Tuning improvement",
            "",
            f"- Logistic tuned − baseline OOF PR-AUC: {_number(logistic_delta)}.",
            f"- XGBoost tuned − baseline OOF PR-AUC: {_number(xgboost_delta)}.",
            f"- XGBoost tuned − Logistic tuned OOF PR-AUC: {_number(cross_delta)}.",
            "- These point differences have not undergone a statistical significance test.",
            (
                "- XGBoost tuned beat its baseline on validation PR-AUC in "
                f"{decision['xgboost_fold_wins']} of "
                f"{len(results['fold_records']['xgboost_tuned'])} outer folds."
            ),
            "",
            "## G. Best parameters",
            "",
        ]
    )
    for family, name in (("logistic_regression", "logistic_tuned"), ("xgboost", "xgboost_tuned")):
        lines.extend([f"### {LABELS[name]}", ""])
        for row in results["fold_records"][name]:
            params = json.dumps(row["best_parameters"], sort_keys=True)
            lines.append(
                f"- Outer fold {row['outer_fold']}: `{params}`; "
                f"inner best PR-AUC {_number(row['inner_best_average_precision'])}."
            )
        frequencies = parameter_frequencies(results["fold_records"][name])
        lines.append(f"- Selection frequencies: `{json.dumps(frequencies, sort_keys=True)}`.")
        lines.append(
            "- Full-development best parameters: "
            f"`{json.dumps(full_search[family]['best_parameters'], sort_keys=True)}`."
        )
        stable = all(max(counts.values()) >= 4 for counts in frequencies.values())
        lines.append(
            "- Parameter selections are mostly stable across folds."
            if stable else "- Parameter selections vary across outer folds."
        )
        lines.append("")

    champion = decision["champion"]
    challenger = decision["challenger"]
    xgboost_extra_fn = (
        metrics["xgboost_tuned"]["confusion_matrix"]["fn"]
        - metrics["logistic_baseline"]["confusion_matrix"]["fn"]
    )
    tuned_pr_auc_std = summaries["xgboost_tuned"]["validation_average_precision"][
        "standard_deviation"
    ]
    baseline_pr_auc_std = summaries["xgboost_baseline"]["validation_average_precision"][
        "standard_deviation"
    ]
    lines.extend(
        [
            "Full-development search generated the candidate configurations for the next step. "
            "Its CV score is not an unbiased final performance estimate.",
            "",
            "## H. Decision",
            "",
            f"- Champion for the next development stage: {LABELS[champion]}.",
            (
                f"- Challenger: {LABELS[challenger]}."
                if challenger is not None else "- Challenger: none retained."
            ),
            (
                "- A PR-AUC difference within 0.01 favors the simpler candidate with "
                "better Brier Score, lower overfitting gap, and acceptable fold stability."
            ),
            (
                "- XGBoost is not advanced if tuning failed to improve its baseline "
                "OOF PR-AUC."
            ),
            (
                "- Operational caveat at threshold 0.5: XGBoost tuned has "
                f"{xgboost_extra_fn:+d} false negatives versus Logistic baseline; "
                "any alert-volume decision requires a later approved threshold study."
            ),
            (
                "- Fold PR-AUC variability for XGBoost tuned is "
                f"{_number(tuned_pr_auc_std)} versus {_number(baseline_pr_auc_std)} "
                "for its baseline; evaluate this uncertainty before deployment."
            ),
            (
                "- Cross-fitted calibration may be the next development experiment; "
                "it is not done here."
            ),
            "- No production model has been selected; final-test evaluation remains locked.",
            "- IBM HR is a fictional/public benchmark, not evidence of causal relationships.",
            "",
        ]
    )
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines), encoding="utf-8")
    return decision
