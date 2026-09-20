"""One fairness Markdown report and a neutral development-only audit dashboard."""

from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    if isinstance(value, (float, np.floating)):
        return f"{value:.{digits}f}"
    return str(value)


def _table(rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> str:
    header = "| " + " | ".join(label for _, label in columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    body = ["| " + " | ".join(_fmt(row.get(key), 6 if key == "threshold" else 3)
                                for key, _ in columns) + " |"
            for row in rows]
    return "\n".join([header, separator, *body])


def _ci(row: dict[str, Any], metric: str) -> str:
    low, high = row.get(f"{metric}_ci_low"), row.get(f"{metric}_ci_high")
    valid = row.get(f"{metric}_bootstrap_valid", 0)
    if low is None or high is None or pd.isna(low) or pd.isna(high):
        return f"N/A ({valid} valid)"
    return f"[{low:.3f}, {high:.3f}] ({valid} valid)"


def write_fairness_report(
    path: Path, frame: pd.DataFrame, results: dict[str, Any], protocol: dict[str, Any],
    dashboard_name: str,
) -> None:
    """Report cross-fitted primary fairness, descriptive sensitivity and incident caveat."""
    metrics = results["metrics"].to_dict(orient="records")
    disparities = results["disparities"].to_dict(orient="records")
    sensitivity = results["sensitivity"].to_dict(orient="records")
    primary = [row for row in metrics if row["model"] == "xgboost"]
    xgb = results["comparison"]["xgboost"]["overall"]
    logistic = results["comparison"]["logistic"]["overall"]
    support_rows = [{**row, "support": row["support_status"]} for row in primary]
    performance_rows = [{**row, "alert_ci": _ci(row, "alert_rate"),
                         "tpr_ci": _ci(row, "tpr"), "fpr_ci": _ci(row, "fpr"),
                         "precision_ci": _ci(row, "precision"),
                         "brier_ci": _ci(row, "brier")}
                        for row in primary]
    comparison = []
    for model, overall in (("xgboost", xgb), ("logistic", logistic)):
        by_attribute = results["comparison"][model]["disparities"]
        comparison.append({"model": model, "recall": overall["tpr"],
                           "precision": overall["precision"],
                           "alert_rate": overall["alert_rate"],
                           "gender_tpr": by_attribute["Gender"]["tpr_difference"],
                           "gender_fpr": by_attribute["Gender"]["fpr_difference"],
                           "gender_eo": by_attribute["Gender"]["equalized_odds_gap"],
                           "age_tpr": by_attribute["AgeGroup"]["tpr_difference"],
                           "age_fpr": by_attribute["AgeGroup"]["fpr_difference"],
                           "age_eo": by_attribute["AgeGroup"]["equalized_odds_gap"],
                           "insufficient": sum(
                               item["group_count"] - item["supported_groups"]
                               for item in by_attribute.values()
                           )})
    xgb_status = {row["attribute"]: row["status"] for row in disparities
                  if row["model"] == "xgboost"}
    red_flags = [name for name in ("Gender", "AgeGroup") if xgb_status[name] == "REVIEW"]
    red_flag_label = ", ".join(red_flags) if red_flags else "none at point estimate"
    fixed_age = next(row for row in sensitivity
                     if row["attribute"] == "AgeGroup" and row["threshold"] == 0.345651)
    default_age = next(row for row in sensitivity
                       if row["attribute"] == "AgeGroup" and row["threshold"] == 0.5)
    lines = [
        "# Bước 10D — Fairness và subgroup robustness audit",
        "",
        "## A. Setup và holdout status", "",
        f"- Development rows: {len(frame)}; target distribution: "
        f"{frame['Attrition'].value_counts().sort_index().to_dict()}.",
        "- Fairness input: `data/processed/fairness_audit_development_v1.csv`; "
        "strict one-to-one join on `source_row` + `EmployeeNumber`.",
        "- Primary: XGBoost tuned raw, cross-fitted capacity-15 policy predictions. "
        "Logistic baseline raw is the reference at its own capacity-15 policy.",
        f"- Working threshold: {protocol['working_threshold']:.6f}; "
        "business_approved=false; production_approved=false.",
        "- Audit attributes: Gender, fixed AgeGroup (18–29, 30–39, 40–49, 50+); "
        "exploratory MaritalStatus and Gender × AgeGroup.",
        "- Minimum support: n≥50, positives≥10, negatives≥10. "
        "Bootstrap: 2,000 paired group×target-stratified iterations, 95% percentile CI, seed 42.",
        "- Holdout status: `retained_with_protocol_deviation`; "
        "official final-test file opened: false; "
        "final-test metrics computed: false; final rows used for model decision: false.",
        "- Incident: the official final-test file was not opened and no final-test metric was "
        "computed. A mixed fairness file was inadvertently loaded during schema inspection "
        "before Step 10D; the process stopped before analysis, was logged, and a "
        "development-only audit input was recovered. Do not describe the holdout as untouched.",
        "",
        "## B. Overall performance — cross-fitted capacity 15%", "",
        _table([{"model": model, **overall} for model, overall in
                (("XGBoost raw", xgb), ("Logistic raw", logistic))],
               [("model", "Model"), ("alert_rate", "Alert rate"),
                ("precision", "Precision"), ("tpr", "Recall"),
                ("fpr", "FPR"), ("fnr", "FNR"), ("tn", "TN"), ("fp", "FP"),
                ("fn", "FN"), ("tp", "TP")]),
        "",
        "## C. Group support", "",
        _table(support_rows, [("attribute", "Attribute"), ("group", "Group"),
                              ("n", "N"), ("positive_count", "Positive"),
                              ("negative_count", "Negative"), ("base_rate", "Base rate"),
                              ("support", "Support")]),
        "",
        "## D. Subgroup performance — XGBoost capacity 15%", "",
        "CI columns show [lower, upper] and valid bootstrap iterations. "
        "Unsupported groups have descriptive point estimates but no CI or strong conclusion. "
        "ECE is omitted for unsupported groups.", "",
        _table(performance_rows, [("attribute", "Attribute"), ("group", "Group"),
                                  ("alert_rate", "Alert"), ("tpr", "TPR"),
                                  ("fnr", "FNR"), ("fpr", "FPR"),
                                  ("precision", "Precision"), ("npv", "NPV"),
                                  ("balanced_accuracy", "Balanced acc."),
                                  ("f1", "F1"), ("f2", "F2"),
                                  ("brier", "Brier"), ("ece", "ECE"),
                                  ("calibration_gap", "Cal. gap"),
                                  ("alert_ci", "Alert 95% CI"), ("tpr_ci", "TPR 95% CI"),
                                  ("fpr_ci", "FPR 95% CI"),
                                  ("precision_ci", "Precision 95% CI"),
                                  ("brier_ci", "Brier 95% CI")]),
        "",
        "### Subgroup confusion counts", "",
        _table(primary, [("attribute", "Attribute"), ("group", "Group"),
                         ("tn", "TN"), ("fp", "FP"), ("fn", "FN"), ("tp", "TP")]),
        "",
        "## E. Disparity summary", "",
        "Differences are max–min across groups; ratios are min/max when the maximum is nonzero. "
        "No group is designated privileged. Screening statuses are governance heuristics, "
        "not legal findings. Gaps with unsupported groups remain descriptive only.", "",
        _table(disparities, [("model", "Model"), ("attribute", "Attribute"),
                             ("supported_groups", "Supported"), ("group_count", "Total"),
                             ("selection_rate_difference", "Selection gap"),
                             ("selection_rate_ratio", "Selection ratio"),
                             ("tpr_difference", "TPR gap"), ("tpr_ratio", "TPR ratio"),
                             ("fpr_difference", "FPR gap"), ("fpr_ratio", "FPR ratio"),
                             ("precision_difference", "Precision gap"),
                             ("precision_ratio", "Precision ratio"),
                             ("fnr_difference", "FNR gap"), ("brier_difference", "Brier gap"),
                             ("calibration_gap_difference", "Cal. gap diff"),
                             ("equalized_odds_gap", "EO gap"), ("status", "Status")]),
        "",
        "### Disparity bootstrap intervals", "",
    ]
    for row in disparities:
        lines.append(
            f"- {row['model']} / {row['attribute']}: selection-rate gap "
            f"{_ci(row, 'selection_rate_difference')}; TPR gap {_ci(row, 'tpr_difference')}; "
            f"FPR gap {_ci(row, 'fpr_difference')}; equalized-odds gap "
            f"{_ci(row, 'equalized_odds_gap')}."
        )
    lines += [
        "", "## F. Fixed-threshold sensitivity — descriptive only", "",
        "These four predeclared thresholds are applied to XGBoost raw OOF probabilities. "
        "They are not the primary cross-fitted policy estimate and are not used to select "
        "a new threshold or demographic-specific threshold.", "",
        _table(sensitivity, [("threshold", "Threshold"), ("attribute", "Attribute"),
                             ("selection_rate_difference", "Selection gap"),
                             ("tpr_difference", "TPR gap"),
                             ("fpr_difference", "FPR gap"),
                             ("precision_difference", "Precision gap"),
                             ("equalized_odds_gap", "EO gap"),
                             ("supported_groups", "Supported groups")]),
        "", "## G. XGBoost vs Logistic — same capacity-15 policy", "",
        _table(comparison, [("model", "Model"), ("recall", "Recall"),
                            ("precision", "Precision"), ("alert_rate", "Alert rate"),
                            ("gender_tpr", "Gender TPR gap"),
                            ("gender_fpr", "Gender FPR gap"),
                            ("gender_eo", "Gender EO gap"),
                            ("age_tpr", "Age TPR gap"), ("age_fpr", "Age FPR gap"),
                            ("age_eo", "Age EO gap"),
                            ("insufficient", "Insufficient groups")]),
        "", "## H. Intersectional audit", "",
        "Gender × AgeGroup is exploratory. Groups below the support minimum are listed but "
        "must not support a strong fairness conclusion. No groups were merged post hoc.", "",
        "## I. Dashboard and limitations", "",
        f"![Fairness dashboard]({dashboard_name})", "",
        "- IBM HR attrition is a public/synthetic benchmark, not a representative Vietnamese "
        "workforce sample; results do not establish fairness in a real employer.",
        "- Race, disability, nationality and several other protected attributes are absent. "
        "Small subgroup counts widen uncertainty; differing base rates affect precision "
        "and selection rates.",
        "- Bootstrap percentile intervals quantify sampling variability under this dataset "
        "and policy. They are not statistical-significance tests and do not prove causality "
        "or legal discrimination.",
        "- Re-audit on representative real-world data before deployment; investigate any "
        "screening signal with HR and governance stakeholders.",
        "", "## J. Decision", "",
        f"- Primary screening red flags: {red_flag_label}. "
        "Unsupported or wide-CI groups still limit certainty.",
        f"- Gender: {xgb_status['Gender']}; AgeGroup: {xgb_status['AgeGroup']}; "
        f"MaritalStatus (exploratory): {xgb_status['MaritalStatus']}; "
        f"Gender × AgeGroup (exploratory): {xgb_status['Gender_AgeGroup']}.",
        f"- At the descriptive fixed 0.345651 threshold, AgeGroup TPR gap is "
        f"{fixed_age['tpr_difference']:.3f}, versus {default_age['tpr_difference']:.3f} "
        "at fixed 0.5. This reinforces REVIEW but does not authorize threshold switching.",
        "- XGBoost has higher overall recall/precision than Logistic at similar alert rate, "
        "but its AgeGroup TPR gap is larger. Do not replace or freeze the candidate based "
        "on a single subgroup point estimate; require governance review.",
        "- Keep 0.345651 only as the unchanged capstone working threshold pending governance "
        "and business review; do not automatically freeze the candidate or open final test.",
        "- `business_approved=false`; `production_approved=false`. No retraining, calibration, "
        "tuning, subgroup-specific thresholding, or model serialization was performed.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def plot_fairness_dashboard(path: Path, results: dict[str, Any]) -> None:
    """Draw six neutral subgroup panels with support counts and predeclared thresholds."""
    metrics = results["metrics"]
    primary = metrics[(metrics["model"] == "xgboost") &
                      (metrics["attribute"].isin(["Gender", "AgeGroup"]))]
    sensitivity = results["sensitivity"]
    figure, axes = plt.subplots(2, 3, figsize=(18, 10), layout="constrained")
    colors = {"Gender": "#286090", "AgeGroup": "#bc6c25"}
    for axis, column, title in zip(axes.flat[:4],
                                   ("tpr", "fpr", "alert_rate", "precision"),
                                   ("TPR / Recall", "False-positive rate",
                                    "Alert / selection rate", "Precision"), strict=True):
        ticks = []
        labels = []
        for index, row in enumerate(primary.to_dict(orient="records")):
            low, high = row[f"{column}_ci_low"], row[f"{column}_ci_high"]
            value = row[column]
            if value is None or pd.isna(value):
                continue
            xerr = None
            if low is not None and high is not None and not pd.isna(low) and not pd.isna(high):
                xerr = [[max(0.0, value - low)], [max(0.0, high - value)]]
            axis.errorbar(value, index, xerr=xerr, fmt="o", capsize=3,
                          color=colors[row["attribute"]])
            ticks.append(index)
            labels.append(f"{row['attribute']}: {row['group']} (n={row['n']})")
        axis.set_yticks(ticks, labels)
        axis.set_xlim(0, 1)
        axis.set_xlabel("Rate (95% bootstrap CI where supported)")
        axis.set_title(title)
        axis.grid(axis="x", alpha=0.25)
    for axis, column, title in ((axes.flat[4], "tpr_difference", "TPR gap sensitivity"),
                                (axes.flat[5], "fpr_difference", "FPR gap sensitivity")):
        for attribute in ("Gender", "AgeGroup"):
            part = sensitivity[sensitivity["attribute"] == attribute].sort_values("threshold")
            axis.plot(part["threshold"], part[column], marker="o",
                      label=attribute, color=colors[attribute])
        axis.axhline(0.10, linestyle="--", color="#6b7280", linewidth=1,
                     label="Screening heuristic 0.10")
        axis.axvline(0.345651, linestyle=":", color="#374151", linewidth=1)
        axis.set_ylim(bottom=0)
        axis.set_xlabel("Fixed XGBoost threshold (descriptive)")
        axis.set_ylabel("Max–min gap")
        axis.set_title(title)
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    figure.suptitle("Development-only fairness audit — cross-fitted capacity 15% policy",
                    fontsize=15)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160)
    plt.close(figure)
