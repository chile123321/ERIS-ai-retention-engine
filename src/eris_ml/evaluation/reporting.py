"""Evaluation report interfaces."""

from pathlib import Path
from typing import Any

from eris_ml.evaluation.cross_validation import SCALAR_METRICS

DISPLAY_NAMES = {
    "dummy": "Dummy",
    "dummy_prior": "Dummy",
    "logistic_regression": "Logistic Regression",
    "logistic_regression_balanced": "Logistic balanced",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
}
METRIC_LABELS = {
    "accuracy": "Accuracy",
    "balanced_accuracy": "Balanced Accuracy",
    "average_precision": "PR-AUC",
    "roc_auc": "ROC-AUC",
    "precision": "Precision",
    "recall": "Recall",
    "f1": "F1",
    "f2": "F2",
    "brier_score": "Brier Score",
}


def _score(value: Any) -> str:
    return f"{float(value):.6f}"


def write_evaluation_report(
    results: dict[str, dict[str, Any]],
    destination: Path,
    metadata: dict[str, Any],
) -> None:
    """Write the single traceable baseline evaluation report."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Baseline cross-validation report v1",
        "",
        "## Scope",
        "",
        "- Development-only evaluation.",
        "- The locked final test was not used.",
        "- No tuning, calibration, class balancing, or threshold optimization was performed.",
        "",
        "## Traceability",
        "",
        f"- Development rows: {metadata['development_rows']}",
        f"- Positive count: {metadata['positive_count']}",
        f"- Positive rate: {_score(metadata['positive_rate'])}",
        f"- Development CSV SHA-256: `{metadata['development_sha256']}`",
        f"- Feature-set version: `{metadata['feature_set_version']}`",
        f"- Feature count: {metadata['feature_count']}",
        (
            "- CV: "
            f"StratifiedKFold(n_splits={metadata['n_splits']}, "
            f"shuffle={metadata['shuffle']}, random_state={metadata['random_seed']})"
        ),
        f"- Classification threshold: {metadata['threshold']}",
        f"- Python: `{metadata['python_version']}`",
        f"- pandas: `{metadata['pandas_version']}`",
        f"- scikit-learn: `{metadata['sklearn_version']}`",
        "",
        "## OOF results",
        "",
        (
            "| Model | Accuracy | Balanced Accuracy | PR-AUC | ROC-AUC | Precision | "
            "Recall | F1 | F2 | Brier Score |"
        ),
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    order = (
        "accuracy",
        "balanced_accuracy",
        "average_precision",
        "roc_auc",
        "precision",
        "recall",
        "f1",
        "f2",
        "brier_score",
    )
    for model_name in ("dummy", "logistic_regression"):
        metrics = results[model_name]["oof_metrics"]
        values = " | ".join(_score(metrics[name]) for name in order)
        lines.append(f"| {DISPLAY_NAMES[model_name]} | {values} |")

    lines.extend(["", "## Fold statistics", ""])
    for model_name in ("dummy", "logistic_regression"):
        lines.extend(
            [
                f"### {DISPLAY_NAMES[model_name]}",
                "",
                "| Metric | Mean | Standard deviation |",
                "| --- | ---: | ---: |",
            ]
        )
        summary = results[model_name]["fold_summary"]
        for metric in SCALAR_METRICS:
            lines.append(
                f"| {METRIC_LABELS[metric]} | {_score(summary[metric]['mean'])} | "
                f"{_score(summary[metric]['standard_deviation'])} |"
            )
        lines.append("")

    lines.extend(["## Confusion matrices", ""])
    for model_name in ("dummy", "logistic_regression"):
        matrix = results[model_name]["oof_metrics"]["confusion_matrix"]
        lines.append(
            f"- {DISPLAY_NAMES[model_name]} at threshold {metadata['threshold']}: "
            f"TN={matrix['tn']}, FP={matrix['fp']}, FN={matrix['fn']}, TP={matrix['tp']}"
        )

    dummy_ap = results["dummy"]["oof_metrics"]["average_precision"]
    logistic_ap = results["logistic_regression"]["oof_metrics"]["average_precision"]
    improvement = logistic_ap - dummy_ap
    comparison = (
        "Logistic Regression achieved a higher OOF PR-AUC than Dummy."
        if improvement > 0
        else "Logistic Regression did not achieve a higher OOF PR-AUC than Dummy."
    )
    lines.extend(
        [
            "",
            "## Comparison",
            "",
            f"- Dummy PR-AUC: {_score(dummy_ap)}",
            f"- Logistic Regression PR-AUC: {_score(logistic_ap)}",
            f"- Absolute PR-AUC improvement: {_score(improvement)}",
            f"- Conclusion: {comparison}",
            "",
            "## Limitations",
            "",
            "- IBM HR is a fictional/public benchmark dataset.",
            "- There is no snapshot date or exit date.",
            "- A 3–6 month prediction horizon cannot be verified.",
            "- The dataset is not representative of Vietnamese enterprises.",
            "- Baseline cross-validation does not demonstrate production readiness.",
            "- EDA and correlations do not demonstrate causal relationships.",
            "",
        ]
    )
    destination.write_text("\n".join(lines), encoding="utf-8")


def write_class_weight_report(
    results: dict[str, dict[str, Any]],
    destination: Path,
    metadata: dict[str, Any],
) -> None:
    """Write the development-only class-weight comparison report."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    unweighted = results["logistic_regression"]["oof_metrics"]
    balanced = results["logistic_regression_balanced"]["oof_metrics"]

    lines = [
        "# Logistic Regression class-weight experiment v1",
        "",
        "## Scope",
        "",
        "- Development-only evaluation; the locked final test was not used.",
        "- All candidates use the same 25 features, preprocessing, CV folds, and threshold.",
        "- No hyperparameter tuning, calibration, or threshold optimization was performed.",
        "- These results do not establish production readiness.",
        "",
        "## Traceability",
        "",
        f"- Development rows: {metadata['development_rows']}",
        f"- Positive count: {metadata['positive_count']}",
        f"- Positive rate: {_score(metadata['positive_rate'])}",
        f"- Development CSV SHA-256: `{metadata['development_sha256']}`",
        f"- Feature-set version: `{metadata['feature_set_version']}`",
        f"- Feature count: {metadata['feature_count']}",
        (
            "- CV: "
            f"StratifiedKFold(n_splits={metadata['n_splits']}, "
            f"shuffle={metadata['shuffle']}, random_state={metadata['random_seed']})"
        ),
        f"- Classification threshold: {metadata['threshold']}",
        f"- Python: `{metadata['python_version']}`",
        f"- pandas: `{metadata['pandas_version']}`",
        f"- scikit-learn: `{metadata['sklearn_version']}`",
        "",
        "## Model comparison",
        "",
        (
            "| Model | Accuracy | Balanced Acc. | PR-AUC | ROC-AUC | Precision | "
            "Recall | F1 | F2 | Brier |"
        ),
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    metric_order = (
        "accuracy",
        "balanced_accuracy",
        "average_precision",
        "roc_auc",
        "precision",
        "recall",
        "f1",
        "f2",
        "brier_score",
    )
    for candidate_name in (
        "dummy_prior",
        "logistic_regression",
        "logistic_regression_balanced",
    ):
        metrics = results[candidate_name]["oof_metrics"]
        values = " | ".join(_score(metrics[name]) for name in metric_order)
        label = {
            "dummy_prior": "Dummy",
            "logistic_regression": "Logistic unweighted",
            "logistic_regression_balanced": "Logistic balanced",
        }[candidate_name]
        lines.append(f"| {label} | {values} |")

    lines.extend(
        [
            "",
            "## Operational impact",
            "",
            "| Model | Predicted positive | Alert rate | FP | FN | TP | FP/TP |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for candidate_name, label in (
        ("logistic_regression", "Logistic unweighted"),
        ("logistic_regression_balanced", "Logistic balanced"),
    ):
        metrics = results[candidate_name]["oof_metrics"]
        matrix = metrics["confusion_matrix"]
        ratio = metrics["false_positives_per_true_positive"]
        ratio_text = "N/A" if ratio is None else _score(ratio)
        lines.append(
            f"| {label} | {metrics['predicted_positive_count']} | "
            f"{_score(metrics['alert_rate'])} | {matrix['fp']} | {matrix['fn']} | "
            f"{matrix['tp']} | {ratio_text} |"
        )

    unweighted_matrix = unweighted["confusion_matrix"]
    balanced_matrix = balanced["confusion_matrix"]
    recall_change = balanced["recall"] - unweighted["recall"]
    precision_change = balanced["precision"] - unweighted["precision"]
    fn_reduction = unweighted_matrix["fn"] - balanced_matrix["fn"]
    fp_increase = balanced_matrix["fp"] - unweighted_matrix["fp"]
    alert_increase = (
        balanced["predicted_positive_count"] - unweighted["predicted_positive_count"]
    )
    pr_auc_change = balanced["average_precision"] - unweighted["average_precision"]
    brier_change = balanced["brier_score"] - unweighted["brier_score"]
    advances = recall_change > 0 and fn_reduction > 0
    lines.extend(
        [
            "",
            "## Conclusions",
            "",
            f"1. Recall change (balanced − unweighted): {_score(recall_change)}.",
            f"2. False negatives reduced: {fn_reduction}.",
            f"3. Additional false positives: {fp_increase}.",
            f"4. Precision change (balanced − unweighted): {_score(precision_change)}.",
            (
                "5. PR-AUC change (balanced − unweighted): "
                f"{_score(pr_auc_change)}. Statistical significance was not tested."
            ),
            (
                "6. Brier Score change (balanced − unweighted): "
                f"{_score(brier_change)}; "
                + ("it worsened." if brier_change > 0 else "it did not worsen.")
            ),
            f"7. Additional HR alerts at threshold {metadata['threshold']}: {alert_increase}.",
            (
                "8. Recommendation: retain the balanced candidate for the next comparison "
                "because it improved recall and reduced false negatives, while explicitly "
                "carrying forward its precision, false-positive, alert-volume, and Brier "
                "trade-offs."
                if advances
                else "8. Recommendation: do not advance the balanced candidate based on this "
                "experiment because it did not improve recall and reduce false negatives."
            ),
            "",
            "## Limitations",
            "",
            "- IBM HR is a fictional/public benchmark dataset.",
            "- No snapshot date, exit date, or verifiable 3–6 month prediction horizon exists.",
            "- The dataset is not representative of Vietnamese enterprises.",
            "- Point-estimate CV comparisons do not prove production readiness or causality.",
            "",
        ]
    )
    destination.write_text("\n".join(lines), encoding="utf-8")


TREE_CANDIDATES = (
    "dummy_prior",
    "logistic_regression",
    "logistic_regression_balanced",
    "random_forest",
    "xgboost",
)


def _candidate_label(candidate_name: str) -> str:
    return {
        "dummy_prior": "Dummy",
        "logistic_regression": "Logistic unweighted",
        "logistic_regression_balanced": "Logistic balanced",
        "random_forest": "Random Forest",
        "xgboost": "XGBoost",
    }[candidate_name]


def _select_tuning_candidates(results: dict[str, dict[str, Any]]) -> list[str]:
    """Select at most two candidates using PR-AUC tiers and secondary diagnostics."""
    remaining = [name for name in TREE_CANDIDATES if name != "dummy_prior"]
    selected: list[str] = []
    while remaining and len(selected) < 2:
        best_ap = max(results[name]["oof_metrics"]["average_precision"] for name in remaining)
        tier = [
            name
            for name in remaining
            if best_ap - results[name]["oof_metrics"]["average_precision"] <= 0.02
        ]
        choice = min(
            tier,
            key=lambda name: (
                results[name]["fold_summary"]["average_precision"][
                    "standard_deviation"
                ],
                results[name]["oof_metrics"]["brier_score"],
                results[name]["fold_summary"]["overfitting_gap"]["mean"],
                -results[name]["oof_metrics"]["average_precision"],
            ),
        )
        selected.append(choice)
        remaining.remove(choice)
    return selected


def write_tree_baseline_report(
    results: dict[str, dict[str, Any]],
    destination: Path,
    metadata: dict[str, Any],
    candidate_configs: dict[str, dict[str, Any]],
) -> None:
    """Write the five-candidate tree baseline comparison report."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Tree-based baseline comparison v1",
        "",
        "## A. Experiment setup",
        "",
        f"- Development rows: {metadata['development_rows']}",
        (
            "- Target distribution: "
            f"0={metadata['negative_count']}, 1={metadata['positive_count']} "
            f"(positive rate {_score(metadata['positive_rate'])})"
        ),
        f"- Feature count: {metadata['feature_count']} (`{metadata['feature_set_version']}`)",
        (
            "- CV: StratifiedKFold("
            f"n_splits={metadata['n_splits']}, shuffle={metadata['shuffle']}, "
            f"random_state={metadata['random_seed']})"
        ),
        f"- Classification threshold: {metadata['threshold']}",
        "- Locked final test accessed: No.",
        f"- Development CSV SHA-256: `{metadata['development_sha256']}`",
        (
            f"- Libraries: Python {metadata['python_version']}, pandas "
            f"{metadata['pandas_version']}, scikit-learn {metadata['sklearn_version']}, "
            f"XGBoost {metadata['xgboost_version']}."
        ),
        "",
        "## B. Candidate configurations",
        "",
    ]
    for candidate_name in TREE_CANDIDATES:
        config = candidate_configs[candidate_name]
        parameters = ", ".join(
            f"{key}={value!r}" for key, value in config.get("parameters", {}).items()
        )
        lines.append(
            f"- {_candidate_label(candidate_name)}: `{config['model']}` ({parameters})."
        )

    lines.extend(
        [
            "",
            "## C. OOF model comparison",
            "",
            (
                "| Model | Accuracy | Balanced Accuracy | PR-AUC | ROC-AUC | Precision | "
                "Recall | F1 | F2 | Brier |"
            ),
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    metric_order = (
        "accuracy",
        "balanced_accuracy",
        "average_precision",
        "roc_auc",
        "precision",
        "recall",
        "f1",
        "f2",
        "brier_score",
    )
    for candidate_name in TREE_CANDIDATES:
        metrics = results[candidate_name]["oof_metrics"]
        values = " | ".join(_score(metrics[name]) for name in metric_order)
        lines.append(f"| {_candidate_label(candidate_name)} | {values} |")

    lines.extend(
        [
            "",
            "## D. Confusion matrix and operational impact",
            "",
            "| Model | TN | FP | FN | TP | Alerts | Alert rate | FP/TP |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for candidate_name in TREE_CANDIDATES:
        metrics = results[candidate_name]["oof_metrics"]
        matrix = metrics["confusion_matrix"]
        ratio = metrics["false_positives_per_true_positive"]
        ratio_text = "N/A" if ratio is None else _score(ratio)
        lines.append(
            f"| {_candidate_label(candidate_name)} | {matrix['tn']} | {matrix['fp']} | "
            f"{matrix['fn']} | {matrix['tp']} | {metrics['predicted_positive_count']} | "
            f"{_score(metrics['alert_rate'])} | {ratio_text} |"
        )

    lines.extend(
        [
            "",
            "## E. Stability, overfitting, and runtime",
            "",
            (
                "| Model | Mean train PR-AUC | Mean validation PR-AUC | PR-AUC std | "
                "Overfitting gap | Mean fit time (s) | Mean predict time (s) |"
            ),
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for candidate_name in TREE_CANDIDATES:
        summary = results[candidate_name]["fold_summary"]
        lines.append(
            f"| {_candidate_label(candidate_name)} | "
            f"{_score(summary['train_average_precision']['mean'])} | "
            f"{_score(summary['validation_average_precision']['mean'])} | "
            f"{_score(summary['average_precision']['standard_deviation'])} | "
            f"{_score(summary['overfitting_gap']['mean'])} | "
            f"{_score(summary['fit_time_seconds']['mean'])} | "
            f"{_score(summary['prediction_time_seconds']['mean'])} |"
        )

    logistic_ap = results["logistic_regression"]["oof_metrics"]["average_precision"]
    tree_changes = {
        name: results[name]["oof_metrics"]["average_precision"] - logistic_ap
        for name in ("random_forest", "xgboost")
    }
    overfit_models = [
        name
        for name in (
            "logistic_regression",
            "logistic_regression_balanced",
            "random_forest",
            "xgboost",
        )
        if results[name]["fold_summary"]["train_average_precision"]["mean"] >= 0.95
        and results[name]["fold_summary"]["overfitting_gap"]["mean"] >= 0.10
    ]
    selected = _select_tuning_candidates(results)
    overfit_text = (
        ", ".join(_candidate_label(name) for name in overfit_models)
        if overfit_models
        else (
            "No candidate met the report's overfitting flag "
            "(train PR-AUC ≥ 0.95 and gap ≥ 0.10)"
        )
    )
    lines.extend(
        [
            "",
            "## F. Interpretation",
            "",
            (
                "- Random Forest PR-AUC change versus Logistic unweighted: "
                f"{_score(tree_changes['random_forest'])}."
            ),
            (
                "- XGBoost PR-AUC change versus Logistic unweighted: "
                f"{_score(tree_changes['xgboost'])}."
            ),
            f"- Overfitting warning (near-perfect train PR-AUC with a large gap): {overfit_text}.",
            (
                "- Precision, recall, false negatives, false positives, and alert volume must "
                "be assessed together at threshold 0.5; accuracy is not the selection metric."
            ),
            (
                "- Tree models add implementation and explanation complexity. PR-AUC gains "
                "below 0.02 are treated as a comparison tier and resolved using fold stability, "
                "Brier Score, and overfitting gap."
            ),
            "- No statistical significance test was performed.",
            "- No feature relationship or future explanation should be interpreted causally.",
            "",
            "## G. Decision",
            "",
            (
                "- Candidates retained for the tuning stage: "
                + ", ".join(_candidate_label(name) for name in selected)
                + "."
            ),
            (
                "- Selection used a 0.02 PR-AUC comparison tier, then fold variability, "
                "Brier Score, and overfitting gap; this is not a production-model decision."
            ),
            "- No production model has been selected.",
            "",
        ]
    )
    destination.write_text("\n".join(lines), encoding="utf-8")
