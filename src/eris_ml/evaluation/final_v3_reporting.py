"""Aggregate-only final report; explicit prior holdout use and no post-test adjustment."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import matplotlib
import pandas as pd

from eris_ml.evaluation.final_v3 import MODEL_SHA, PROBABILITY_METRICS, SUMMARY
from eris_ml.models import baseline_v3 as baseline
from eris_ml.models import tuning_v3 as tuning
from eris_ml.models.bundle_v2 import write_exclusive


def markdown_table(frame: pd.DataFrame) -> str:
    def value(x: Any) -> str:
        if pd.isna(x):
            return "NA"
        return f"{x:.6f}" if isinstance(x, float) else str(x)

    header = "| " + " | ".join(frame.columns) + " |"
    divider = "| " + " | ".join(["---"] * len(frame.columns)) + " |"
    rows = [
        "| " + " | ".join(value(v) for v in row) + " |"
        for row in frame.itertuples(index=False, name=None)
    ]
    return "\n".join([header, divider, *rows])


def evidence(metrics: dict[str, Any], intervals: dict[str, Any], top: pd.DataFrame) -> str:
    m, dummy = metrics["probability_metrics"], metrics["dummy_metrics"]
    if (
        m["average_precision"] <= metrics["prevalence"]
        or m["roc_auc"] <= 0.5
        or m["brier_score"] >= dummy["brier_score"]
        or m["log_loss"] >= dummy["log_loss"]
        or (top.lift <= 1).all()
    ):
        return "weak"
    ci = intervals["intervals"]
    if (
        ci["average_precision"]["lower"] > metrics["prevalence"]
        and ci["roc_auc"]["lower"] > 0.5
        and (top.lift > 1).all()
    ):
        return "strong"
    return "moderate"


def render(
    root: Path,
    report: Path,
    metrics: dict[str, Any],
    intervals: dict[str, Any],
    top: pd.DataFrame,
    bins: pd.DataFrame,
    groups: pd.DataFrame,
    freeze: dict[str, Any],
) -> None:
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Perfect calibration")
    ax.plot(bins.mean_predicted_probability, bins.observed_positive_rate, "o-", label="b1 raw")
    for row in bins.itertuples():
        ax.annotate(
            f"n={row.count}, +={row.positives}",
            (row.mean_predicted_probability, row.observed_positive_rate),
            xytext=(3, 8),
            textcoords="offset points",
            fontsize=8,
        )
    ax.set(
        xlim=(0, 1),
        ylim=(0, 1),
        xlabel="Mean predicted probability",
        ylabel="Observed attrition rate",
        title="Frozen b1: five quantile bins, final cohort",
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(report / "reliability.png", dpi=160)
    plt.close(fig)
    c1 = tuning.read_json(root / "artifacts/reports/v3_final_15/c1/calibration.json")["metrics"][
        "rf_tuned_raw"
    ]["pooled_oof"]
    selection = tuning.read_json(root / "artifacts/reports/v3_final_15/b1/selection.json")
    comparisons = []
    for key in PROBABILITY_METRICS:
        current, old = metrics["probability_metrics"][key], c1[key]
        delta = current - old
        degradation = (-delta if key in ("average_precision", "roc_auc") else delta) > 0.05
        comparisons.append(
            {
                "metric": key,
                "c1_outer_OOF": old,
                "final_b1": current,
                "delta_final_minus_OOF": delta,
                "absolute_delta": abs(delta),
                "notable_degradation_gt_0_05": degradation,
                "CI95_lower": intervals["intervals"][key]["lower"],
                "CI95_upper": intervals["intervals"][key]["upper"],
            }
        )
    assessment = evidence(metrics, intervals, top)
    development = pd.read_csv(root / "artifacts/reports/v3_final_15/b1/subgroups.csv")
    age_old = development.loc[(development.attribute == "AgeGroup") & (development.fold == 0)]
    age_new = groups.loc[groups.attribute == "AgeGroup"]
    age = age_new[["group", "alerts", "tp", "positives", "recall"]].merge(
        age_old[["group", "alerts", "tp", "positives", "tpr"]],
        on="group",
        suffixes=("_final", "_OOF"),
    )
    age["share_of_final_alerts"] = (
        age.alerts_final / top.loc[top.capacity_percent == 10, "alerts"].iloc[0]
    )
    age["share_of_OOF_alerts"] = age.alerts_OOF / age.alerts_OOF.sum()
    baseline.write_json(
        report / "comparison.json",
        {
            "metrics": comparisons,
            "selection_cv_ap": selection["selection_cv_ap"],
            "evidence": assessment,
            "evidence_is_descriptive_not_production_approval": True,
            "age_concentration": age.to_dict("records"),
            "audit_attributes_available": sorted(set(groups.attribute)),
            "prior_holdout_use": freeze["prior_holdout_use"],
        },
    )
    lines = [
        "# V3 f1 — frozen candidate evaluation on previously used holdout",
        "",
        "Model: `rf-v3-final15-b1`, Random Forest raw, 15 inputs "
        "(13 required, 2 median-imputed optional).",
        f"Model SHA-256: `{MODEL_SHA}`.",
        "",
        "**Provenance limitation:** this 294-row IBM holdout was already evaluated for V1 on "
        "2026-09-20 and had an earlier mixed-file inspection incident. This is the authorized "
        "one-time evaluation of frozen V3 b1, not a new untouched/independent V3 holdout. "
        "No tuning, fitting, calibration, schema or policy change. No production approval.",
        "",
        "## Metrics and development comparison",
        "",
        markdown_table(pd.DataFrame(comparisons)),
        "",
        f"selection_cv_ap={selection['selection_cv_ap']:.9f} selected the final configuration. "
        "c1 OOF used fold-specific tuned configurations; final b1 uses the single full-development "
        "selection (including class_weight=balanced). These scores are not equivalent.",
        "",
        "Bootstrap: 2,000 stratified resamples, seed 44; 95% percentile intervals. "
        "Conditional on this fixed model and 47 positive/247 negative cases, not model-selection "
        "or prevalence uncertainty; not a tool for modifying the candidate.",
        "",
        f"Prevalence={metrics['prevalence']:.6f}; "
        f"mean probability={metrics['mean_probability']:.6f}. "
        f"Dummy prior is the development fraction {metrics['dummy_prior']:.6f}, never estimated "
        f"from holdout labels: Brier={metrics['dummy_metrics']['brier_score']:.6f}, "
        f"Log loss={metrics['dummy_metrics']['log_loss']:.6f}. AP is Average Precision, "
        "not accuracy or percentage of attrition cases detected.",
        "",
        "### Threshold 0.5 reference only",
        "",
        markdown_table(pd.DataFrame([metrics["threshold_0_5_reference"]])),
        "",
        "## Fixed capacity scenarios",
        "",
        markdown_table(
            top[
                [
                    "capacity_percent",
                    "alerts",
                    "tp",
                    "fp",
                    "fn",
                    "precision",
                    "recall",
                    "f2",
                    "lift",
                ]
            ]
        ),
        "",
        "One 294-row batch, k=floor(percent*n/100), probability descending/source_row ascending "
        "ties. Labels never break ties. Capacity is hypothetical, not HR-approved; no optimal "
        "or threshold was selected. Different batch sizes mean direct workload comparisons with "
        "the five development folds are limited.",
        "",
        "## Raw calibration diagnostic",
        "",
        markdown_table(bins),
        "",
        "![Raw reliability](reliability.png)",
        "",
        "Quantile bins collapse duplicate edges. Small bins are uncertain. This is diagnosis only; "
        "no calibrator fitted and no bundle changed.",
        "",
        "## Descriptive subgroups",
        "",
        markdown_table(groups),
        "",
        "Support requires n>=50, positives>=10 and negatives>=10, locked before access. "
        "Unsupported AP/Brier and undefined denominators are NA, not zero. Counts and defined "
        "rates remain descriptive; no fairness verdict or demographic quota. Gender/MaritalStatus "
        "are reported only if present in the locked final CSV; absent attributes were not sourced "
        "from raw/quarantine data. Age is a model input; the others are audit-only.",
        "",
        "### Age concentration versus development OOF",
        "",
        markdown_table(age),
        "",
        "Same common top10 list within each evaluation batch, no subgroup re-ranking. Low positive "
        "counts and different batch scopes limit interpretation; Age and policy were not changed.",
        "",
        "## Interpretation and status",
        "",
        f"Evidence category: **{assessment}**, using the descriptive rubric frozen before access. "
        "A ranking signal can coexist with poor raw probability calibration; a loss worse than "
        "the development-prior dummy prevents a strong overall claim. Wide intervals and prior "
        "holdout reuse further limit the evidence. This is not a production pass/fail standard.",
        "",
        "The V3 evaluation run is consumed. Status is external in `final_evaluation_status.json`; "
        "b1 and its metadata remain byte-identical. A changed model requires new external "
        "evaluation data, not reuse of f1 as independent evidence. IBM cross-sectional labels do "
        "not validate 3–6 month prediction, intervention benefit or employer transfer. "
        "No API/UI/deployment/commit/push.",
    ]
    write_exclusive(report / "report.md", ("\n".join(lines) + "\n").encode("utf-8"))
    summary = root / SUMMARY
    before = summary.read_bytes()
    if (
        len(before) != freeze["summary_before_bytes"]
        or hashlib.sha256(before).hexdigest() != freeze["summary_before_sha256"]
    ):
        raise ValueError("Historical summary changed during evaluation; append refused.")
    summary_lines = [
        "",
        "",
        "## f1: frozen b1 final evaluation (previously used V1 holdout)",
        "",
        markdown_table(pd.DataFrame(comparisons)),
        "",
        f"Descriptive evidence: **{assessment}**. Frozen model unchanged; evaluation consumed. "
        "Not a new independent V3 holdout and not production/HR approval. Full diagnostics, "
        "top-k, subgroup support and limitations: [f1/report.md](f1/report.md). "
        "Status is external; bundle b1 has not been rewritten.",
        "",
    ]
    with summary.open("ab") as stream:
        stream.write("\n".join(summary_lines).encode("utf-8"))
