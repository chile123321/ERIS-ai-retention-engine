"""Business-scenario reporting without claiming business approval or causality."""

from pathlib import Path
from typing import Any

import pandas as pd

from eris_ml.evaluation.thresholding import (
    CAPACITIES,
    MODELS,
    metrics_at_threshold,
    per_thousand,
)

LABELS = {"xgboost": "XGBoost tuned raw", "logistic": "Logistic baseline raw"}


def _number(value: Any, digits: int = 6) -> str:
    return "N/A" if value is None or pd.isna(value) else f"{float(value):.{digits}f}"


def recommend_candidates(results: dict[str, Any]) -> dict[str, Any]:
    """Shortlist at most three full-development thresholds; provisionally select only if safe."""
    summary = results["summary"]
    full = results["full_selected"]
    labels = results["oof"]["Attrition"].to_numpy()
    raw_scores = results["oof"]["xgboost_raw_probability"].to_numpy()
    reference_recall = metrics_at_threshold(labels, raw_scores, 0.5)["recall"]
    candidates: list[dict[str, Any]] = []
    # Display a conservative-to-moderate range; 25% remains in scenario analysis.
    for percentage in (10, 15, 20):
        scenario = f"capacity_{percentage}_percent"
        selected = full.loc[(full["model"] == "xgboost") & (full["scenario"] == scenario)]
        evaluated = summary.loc[
            (summary["model"] == "xgboost") & (summary["scenario"] == scenario)
        ]
        if len(selected) != 1 or len(evaluated) != 1 or not bool(selected.iloc[0]["feasible"]):
            continue
        if int(evaluated.iloc[0]["feasible_folds"]) != int(evaluated.iloc[0]["outer_folds"]):
            continue
        source, assessed = selected.iloc[0], evaluated.iloc[0]
        candidates.append({
            "scenario": scenario,
            "threshold": float(source["threshold"]),
            "expected_alert_rate": float(source["alert_rate"]),
            "expected_precision": float(source["precision"]),
            "expected_recall": float(source["recall"]),
            "expected_fp": int(source["fp"]),
            "expected_fn": int(source["fn"]),
            "cross_fitted_alert_rate": float(assessed["alert_rate"]),
            "cross_fitted_precision": float(assessed["precision"]),
            "cross_fitted_recall": float(assessed["recall"]),
            "cross_fitted_capacity_met": bool(assessed["validation_constraint_met"]),
            "threshold_std": float(assessed["threshold_std"]),
            "threshold_range": [
                float(assessed["threshold_minimum"]),
                float(assessed["threshold_maximum"]),
            ],
            "feasible_folds": int(assessed["feasible_folds"]),
        })
    provisional = None
    for candidate in candidates:
        scenario = candidate["scenario"]
        assessed = summary.loc[
            (summary["model"] == "xgboost") & (summary["scenario"] == scenario)
        ].iloc[0]
        logistic = summary.loc[
            (summary["model"] == "logistic") & (summary["scenario"] == scenario)
        ].iloc[0]
        limit = float(scenario.split("_")[1]) / 100
        if (
            candidate["feasible_folds"] == 5
            and not bool(assessed["threshold_unstable"])
            and candidate["cross_fitted_recall"] >= reference_recall + 0.10
            and candidate["cross_fitted_precision"] >= 0.40
            and candidate["cross_fitted_alert_rate"] <= limit
            and candidate["cross_fitted_recall"] >= float(logistic["recall"])
        ):
            provisional = candidate["scenario"]
            break
    return {"candidates": candidates, "recommended_provisional_candidate": provisional}


def candidate_configuration(results: dict[str, Any], decision: dict[str, Any]) -> dict[str, Any]:
    """Persist only unapproved scenario candidates, never a production threshold."""
    analysis = []
    for _, row in results["full_selected"].loc[
        results["full_selected"]["model"] == "xgboost"
    ].iterrows():
        analysis.append({
            "scenario": str(row["scenario"]),
            "threshold": None if not row["feasible"] else float(row["threshold"]),
            "expected_alert_rate": None if not row["feasible"] else float(row["alert_rate"]),
            "expected_recall": None if not row["feasible"] else float(row["recall"]),
            "expected_precision": None if not row["feasible"] else float(row["precision"]),
        })
    return {
        "model": "xgboost_tuned_raw",
        "probability_source": "artifacts/predictions/calibration_oof_v1.csv",
        "business_approved": False,
        "production_approved": False,
        "final_test_evaluated": False,
        "status": "capstone_working_assumption",
        "scenario_analysis": analysis,
        "candidates": decision["candidates"],
        "recommended_provisional_candidate": decision["recommended_provisional_candidate"],
    }


def plot_tradeoff(results: dict[str, Any], destination: Path) -> None:
    """Plot full-development descriptive curves; evaluation remains cross-fitted."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    curve = results["full_curve"]
    full = results["full_selected"]
    figure, axes = plt.subplots(2, 2, figsize=(13, 10))
    colors = {"xgboost": "#176b93", "logistic": "#c7652d"}
    for model in MODELS:
        subset = curve.loc[curve["model"] == model].sort_values("threshold")
        color = colors[model]
        label = LABELS[model]
        axes[0, 0].plot(subset["threshold"], subset["recall"], color=color,
                        label=f"{label} recall")
        axes[0, 0].plot(subset["threshold"], subset["precision"], color=color,
                        linestyle="--", label=f"{label} precision")
        axes[0, 1].plot(subset["threshold"], subset["alert_rate"], color=color,
                        label=label)
        axes[1, 0].plot(subset["alert_rate"], subset["recall"], color=color,
                        label=label)
        axes[1, 1].plot(subset["alert_rate"], subset["precision"], color=color,
                        label=label)
        reference = metrics_at_threshold(
            results["oof"]["Attrition"].to_numpy(),
            results["oof"][f"{model}_raw_probability"].to_numpy(), 0.5,
        )
        axes[0, 0].scatter([0.5], [reference["recall"]], marker="x", s=75, color=color)
        axes[0, 1].scatter([0.5], [reference["alert_rate"]], marker="x", s=75,
                           color=color)
        axes[1, 0].scatter([reference["alert_rate"]], [reference["recall"]],
                           marker="x", s=75, color=color)
        axes[1, 1].scatter([reference["alert_rate"]], [reference["precision"]],
                           marker="x", s=75, color=color)
        for capacity in CAPACITIES:
            scenario = f"capacity_{round(capacity * 100)}_percent"
            selected = full.loc[(full["model"] == model) & (full["scenario"] == scenario)]
            if len(selected) == 1 and selected.iloc[0]["feasible"]:
                point = selected.iloc[0]
                axes[0, 1].scatter([point["threshold"]], [point["alert_rate"]],
                                   marker="o", s=28, color=color)
                axes[1, 0].scatter([point["alert_rate"]], [point["recall"]],
                                   marker="o", s=28, color=color)
                axes[1, 1].scatter([point["alert_rate"]], [point["precision"]],
                                   marker="o", s=28, color=color)
    axes[0, 0].axvline(0.5, color="gray", linestyle=":", label="0.5 reference")
    axes[0, 1].axvline(0.5, color="gray", linestyle=":", label="0.5 reference")
    for axis in (axes[1, 0], axes[1, 1]):
        for capacity in CAPACITIES:
            axis.axvline(capacity, color="gray", linestyle=":", alpha=0.4)
    axes[0, 0].set(xlabel="Threshold", ylabel="Rate", title="Threshold vs recall/precision")
    axes[0, 1].set(xlabel="Threshold", ylabel="Alert rate", title="Threshold vs alerts")
    axes[1, 0].set(xlabel="Alert rate", ylabel="Recall", title="Alert rate vs recall")
    axes[1, 1].set(xlabel="Alert rate", ylabel="Precision", title="Alert rate vs precision")
    for axis in axes.flat:
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
    figure.suptitle("Development OOF trade-offs · circles: capacity candidates · ×: 0.5")
    figure.tight_layout()
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=150)
    plt.close(figure)


def write_threshold_report(
    results: dict[str, Any], decision: dict[str, Any], metadata: dict[str, Any],
    destination: Path, plot_path: Path,
) -> None:
    """Write the single principal report; selected OOF metrics are never resubstituted."""
    summary, folds, full = results["summary"], results["fold_records"], results["full_selected"]
    rows = len(results["oof"])
    lines = [
        "# Business-aware threshold scenario analysis v1", "", "## A. Setup", "",
        f"- Development rows: {rows}; target: 0={metadata['negative_count']}, "
        f"1={metadata['positive_count']}; feature set: {metadata['feature_version']} "
        f"({metadata['feature_count']} features).",
        "- Models: XGBoost tuned raw (primary), Logistic baseline raw (reference).",
        "- Outer CV: StratifiedKFold(5, shuffle=True, random_state=42).",
        "- Inner OOF: StratifiedKFold(4, shuffle=True, random_state=43), "
        "recreated inside each outer-training fold using fold-specific models.",
        "- Threshold selection uses inner OOF only; outer-validation evaluates that "
        "selection. Full-development OOF thresholds below are candidates, not "
        "unbiased performance estimates.",
        "- Scenarios: capacity 10/15/20/25%; minimum recall 60/70%; "
        "hypothetical FN:FP cost ratios 2/5/10 with FP cost 1.",
        "- Final test accessed: No. Business approval: No. Production approval: No.",
        f"- Development SHA-256: `{metadata['development_sha256']}`.",
        f"- Calibration OOF SHA-256: `{metadata['calibration_oof_sha256']}`.",
        f"- XGBoost search CSV SHA-256: `{metadata['xgboost_search_sha256']}`.",
        f"- Model configs SHA-256: XGBoost `{metadata['xgboost_config_sha256']}`; "
        f"Logistic `{metadata['logistic_config_sha256']}`.",
        f"- Policy config SHA-256: `{metadata['policy_sha256']}`.",
        f"- Fold-selected XGBoost parameters: `{metadata['fold_parameters']}`.",
        f"- Libraries: Python {metadata['python_version']}; pandas "
        f"{metadata['pandas_version']}; scikit-learn {metadata['sklearn_version']}; "
        f"XGBoost {metadata['xgboost_version']}; matplotlib {metadata['matplotlib_version']}.",
        "- Both regenerated raw outer-validation vectors matched Step 10B with "
        "np.allclose(rtol=1e-7, atol=1e-9).",
        "", "## B. Threshold 0.5 reference", "",
        "| Model | Precision | Recall | F1 | F2 | FP | FN | TP | Alerts | Alert rate |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for model in MODELS:
        row = metrics_at_threshold(
            results["oof"]["Attrition"].to_numpy(),
            results["oof"][f"{model}_raw_probability"].to_numpy(), 0.5,
        )
        lines.append(
            f"| {LABELS[model]} | {_number(row['precision'])} | {_number(row['recall'])} "
            f"| {_number(row['f1'])} | {_number(row['f2'])} | {row['fp']} | {row['fn']} "
            f"| {row['tp']} | {row['alerts']} | {_number(row['alert_rate'])} |"
        )
    def scenario_table(kind: str) -> None:
        lines.extend([
            "| Model | Scenario | Threshold median | Threshold std | Alert rate | Alerts | "
            "Precision | Recall | F1 | F2 | FP | FN | TP | Feasible folds |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | "
            "---: | ---: | ---: | ---: | ---: |",
        ])
        for _, row in summary.loc[summary["kind"] == kind].iterrows():
            if row["feasible_folds"] != row["outer_folds"]:
                lines.append(
                    f"| {LABELS[row['model']]} | {row['scenario']} | No feasible threshold "
                    f"| | | | | | | | | | | {row['feasible_folds']}/{row['outer_folds']} |"
                )
                continue
            lines.append(
                f"| {LABELS[row['model']]} | {row['scenario']} | "
                f"{_number(row['threshold_median'])} | {_number(row['threshold_std'])} | "
                f"{_number(row['alert_rate'])} | {row['alerts']} | "
                f"{_number(row['precision'])} | {_number(row['recall'])} | "
                f"{_number(row['f1'])} | {_number(row['f2'])} | {row['fp']} | "
                f"{row['fn']} | {row['tp']} | {row['feasible_folds']}/{row['outer_folds']} |"
            )
    lines.extend(["", "## C. Capacity scenarios", ""])
    scenario_table("capacity")
    violations = summary.loc[
        (summary["kind"] == "capacity")
        & (summary["validation_constraint_met"] == False)  # noqa: E712
    ]
    if not violations.empty:
        lines.extend([
            "", "Outer-validation capacity warnings: constraints were enforced on "
            "inner-training OOF; the following realized validation rates exceeded "
            "their nominal caps. They are not eligible as provisional candidates "
            "at those caps:", "",
        ])
        for _, row in violations.iterrows():
            cap = float(str(row["scenario"]).split("_")[1]) / 100
            lines.append(
                f"- {LABELS[row['model']]} {row['scenario']}: "
                f"{_number(row['alert_rate'])} realized vs {_number(cap)} cap."
            )
    lines.extend(["", "## D. Minimum-recall scenarios", ""])
    scenario_table("recall")
    lines.extend([
        "", "## E. Same-capacity comparison", "",
        "The two models use their own fold-selected thresholds under the same "
        "nominal training alert-capacity constraint; realized validation alert "
        "counts differ and may exceed the cap. Adjusting thresholds on "
        "outer-validation to force equal counts would leak information.", "",
        "| Model | Capacity | Recall | Precision | F2 | FP | FN | TP | Actual alert rate | "
        "Threshold median | Threshold std |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for _, row in summary.loc[summary["kind"] == "capacity"].iterrows():
        lines.append(
            f"| {LABELS[row['model']]} | {row['scenario']} | {_number(row.get('recall'))} | "
            f"{_number(row.get('precision'))} | {_number(row.get('f2'))} | "
            f"{row.get('fp', 'N/A')} | {row.get('fn', 'N/A')} | {row.get('tp', 'N/A')} | "
            f"{_number(row.get('alert_rate'))} | {_number(row.get('threshold_median'))} | "
            f"{_number(row.get('threshold_std'))} |"
        )
    lines.extend([
        "", "## F. Cost sensitivity", "",
        "FP cost is a hypothetical unit 1; FN cost ratios are assumptions, not "
        "confirmed business costs. Chosen thresholds and measured costs are "
        "cross-fitted across outer folds.", "",
        "| Model | FN:FP ratio | Threshold median | Normalized cost | Recall | "
        "Precision | Alert rate | FP | FN |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for _, row in summary.loc[summary["kind"] == "cost"].iterrows():
        lines.append(
            f"| {LABELS[row['model']]} | {row['scenario']} | "
            f"{_number(row.get('threshold_median'))} | "
            f"{_number(row.get('normalized_cost'))} | {_number(row.get('recall'))} | "
            f"{_number(row.get('precision'))} | {_number(row.get('alert_rate'))} | "
            f"{row.get('fp', 'N/A')} | {row.get('fn', 'N/A')} |"
        )
    lines.extend([
        "", "## G. Threshold stability and fold-level policy evaluation", "",
        "Instability flag is descriptive: threshold std > 0.10 or range > 0.20. "
        "A feasible training constraint does not guarantee the same alert rate or "
        "recall on unseen outer-validation rows.", "",
        "| Model | Scenario | Mean | Median | Min | Max | Std | Unstable |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ])
    for _, row in summary.iterrows():
        lines.append(
            f"| {LABELS[row['model']]} | {row['scenario']} | "
            f"{_number(row.get('threshold_mean'))} | {_number(row.get('threshold_median'))} "
            f"| {_number(row.get('threshold_minimum'))} | "
            f"{_number(row.get('threshold_maximum'))} | "
            f"{_number(row.get('threshold_std'))} | "
            f"{row.get('threshold_unstable', 'N/A')} |"
        )
    lines.extend([
        "", "| Model | Scenario | Fold | Selected threshold | Train alert rate | "
        "Train recall | Validation alert rate | Precision | Recall | F1 | F2 | "
        "TN | FP | FN | TP |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | "
        "---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for _, row in folds.iterrows():
        lines.append(
            f"| {LABELS[row['model']]} | {row['scenario']} | {row['outer_fold']} | "
            f"{_number(row.get('selected_threshold'))} | "
            f"{_number(row.get('training_alert_rate'))} | "
            f"{_number(row.get('training_recall'))} | "
            f"{_number(row.get('validation_alert_rate'))} | "
            f"{_number(row.get('validation_precision'))} | "
            f"{_number(row.get('validation_recall'))} | "
            f"{_number(row.get('validation_f1'))} | "
            f"{_number(row.get('validation_f2'))} | {row.get('tn', 'N/A')} | "
            f"{row.get('fp', 'N/A')} | {row.get('fn', 'N/A')} | "
            f"{row.get('tp', 'N/A')} |"
        )
    lines.extend([
        "", "## H. Per-1,000 employees", "",
        "These are scaled development OOF validation outcomes, not a forecast "
        "of an actual company's attrition prevalence.", "",
        "| Model | Scenario | Alerts | True positives | False positives | "
        "Missed attrition |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ])
    for _, row in summary.loc[summary["kind"] == "capacity"].iterrows():
        if row["feasible_folds"] != row["outer_folds"]:
            continue
        scaled = per_thousand(row.to_dict())
        lines.append(
            f"| {LABELS[row['model']]} | {row['scenario']} | "
            f"{_number(scaled['alerts'], 1)} | {_number(scaled['true_positives'], 1)} "
            f"| {_number(scaled['false_positives'], 1)} | "
            f"{_number(scaled['missed_attrition'], 1)} |"
        )
    lines.extend([
        "", "## I. Recommendation and business questions", "",
        "The following full-development thresholds use all development OOF raw "
        "probabilities only after cross-fitted policy evaluation. Their expected "
        "metrics are descriptive, not a second unbiased evaluation.", "",
        "| Scenario | Candidate threshold | Expected alert rate | Precision | "
        "Recall | FP | FN | Fold threshold std | Validation cap met |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ])
    for candidate in decision["candidates"]:
        lines.append(
            f"| {candidate['scenario']} | {_number(candidate['threshold'])} | "
            f"{_number(candidate['expected_alert_rate'])} | "
            f"{_number(candidate['expected_precision'])} | "
            f"{_number(candidate['expected_recall'])} | {candidate['expected_fp']} | "
            f"{candidate['expected_fn']} | {_number(candidate['threshold_std'])} | "
            f"{candidate['cross_fitted_capacity_met']} |"
        )
    provisional = decision["recommended_provisional_candidate"]
    xgb_15 = summary.loc[
        (summary["model"] == "xgboost")
        & (summary["scenario"] == "capacity_15_percent")
    ].iloc[0]
    logistic_15 = summary.loc[
        (summary["model"] == "logistic")
        & (summary["scenario"] == "capacity_15_percent")
    ].iloc[0]
    xgb_20 = summary.loc[
        (summary["model"] == "xgboost")
        & (summary["scenario"] == "capacity_20_percent")
    ].iloc[0]
    logistic_20 = summary.loc[
        (summary["model"] == "logistic")
        & (summary["scenario"] == "capacity_20_percent")
    ].iloc[0]
    reference = metrics_at_threshold(
        results["oof"]["Attrition"].to_numpy(),
        results["oof"]["xgboost_raw_probability"].to_numpy(), 0.5,
    )
    recall_60 = summary.loc[
        (summary["model"] == "xgboost")
        & (summary["scenario"] == "minimum_recall_60_percent")
    ].iloc[0]
    recall_70 = summary.loc[
        (summary["model"] == "xgboost")
        & (summary["scenario"] == "minimum_recall_70_percent")
    ].iloc[0]
    lines.extend([
        "", f"- Provisional capstone candidate: "
        f"{provisional or 'none (evidence insufficient)'}. ",
        f"- At nominal 15% capacity, cross-fitted XGBoost recall "
        f"{_number(xgb_15['recall'])} versus Logistic {_number(logistic_15['recall'])}; "
        f"actual alerts {xgb_15['alerts']} versus {logistic_15['alerts']}. "
        f"At 20%, recall {_number(xgb_20['recall'])} versus "
        f"{_number(logistic_20['recall'])}, with {xgb_20['alerts']} versus "
        f"{logistic_20['alerts']} alerts.",
        f"- XGBoost threshold 0.5 gives recall {_number(reference['recall'])} "
        f"at {_number(reference['alert_rate'])} alert rate. It is conservative "
        "under a hypothetical 15% capacity, but actual HR capacity is unknown.",
        f"- Minimum-recall policies reached cross-fitted recall "
        f"{_number(recall_60['recall'])} at {_number(recall_60['alert_rate'])} "
        f"alert rate and {_number(recall_70['recall'])} at "
        f"{_number(recall_70['alert_rate'])}; feasibility depends on HR capacity.",
        "- Provisional rule: feasible on all five training folds, pooled validation "
        "alert rate within cap, threshold std <= 0.10 and range <= 0.20, "
        "precision >= 0.40, recall at least 0.10 above the 0.5 reference, "
        "and recall no lower than Logistic at the same nominal capacity.",
        "- This is not a final business threshold. HR must confirm alert-handling "
        "capacity, actual false-negative/false-positive costs, minimum acceptable "
        "recall, maximum alert rate, and intervention workflow.",
        "- Capacity and cost inputs are hypothetical sensitivities. No statistical "
        "significance or causal interpretation is claimed.",
        "- No production model or threshold is approved; final test remains locked.",
        "", "## Trade-off figure", "",
        f"![Development OOF threshold trade-offs]({plot_path.name})", "",
        "The figure is descriptive and does not establish causality.",
        "", "## Full-development scenario thresholds", "",
        "| Model | Scenario | Threshold | OOF alert rate | OOF precision | OOF recall |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ])
    for _, row in full.iterrows():
        lines.append(
            f"| {LABELS[row['model']]} | {row['scenario']} | "
            f"{_number(row.get('threshold'))} | {_number(row.get('alert_rate'))} | "
            f"{_number(row.get('precision'))} | {_number(row.get('recall'))} |"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines), encoding="utf-8")
