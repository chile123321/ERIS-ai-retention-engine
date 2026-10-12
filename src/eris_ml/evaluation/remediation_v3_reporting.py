"""Development-only guardrails, missingness and no-Age sensitivity; aggregate report."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

from eris_ml.evaluation.calibration_reporting import reliability_bins
from eris_ml.evaluation.calibration_v3_reporting import policy_metrics, stress_metrics
from eris_ml.models import baseline_v3 as baseline
from eris_ml.models import candidate_v3 as candidate
from eris_ml.models import tuning_v3 as tuning
from eris_ml.models.remediation_v3 import POLICY, PRIMARY, SCENARIOS, SENSITIVITY


def guardrails(metrics: dict[str, Any], dummy: dict[str, Any]) -> dict[str, Any]:
    decisions = {}
    rule = POLICY["guardrails"]
    for name in PRIMARY:
        base = name.replace("_sigmoid", "_raw")
        record, raw = metrics[name], metrics[base]
        score, old = record["pooled_oof"], raw["pooled_oof"]
        fold_delta = [
            {
                "fold": a["fold"],
                "brier_delta": a["brier_score"] - b["brier_score"],
                "log_loss_delta": a["log_loss"] - b["log_loss"],
            }
            for a, b in zip(record["folds"], raw["folds"], strict=True)
        ]
        worse = sum(d["brier_delta"] > 0 for d in fold_delta)
        checks = {
            "brier_below_dummy": score["brier_score"] < dummy["pooled_oof"]["brier_score"],
            "log_loss_below_dummy": score["log_loss"] < dummy["pooled_oof"]["log_loss"],
            "brier_worse_folds_le_2": worse <= rule["maximum_brier_worse_folds"],
            "ap_drop_le_0_02": score["average_precision"] - old["average_precision"]
            >= -rule["maximum_ap_drop"],
            "roc_drop_le_0_02": score["roc_auc"] - old["roc_auc"] >= -rule["maximum_roc_drop"],
            "abs_mean_bias_le_0_05": abs(record["mean_bias"]) <= rule["maximum_absolute_mean_bias"],
        }
        decisions[name] = {
            "raw_counterpart": base,
            "checks": checks,
            "eligible": all(checks.values()),
            "brier_worse_folds": worse,
            "fold_deltas": fold_delta,
            "pooled_delta_vs_raw": {
                k: score[k] - old[k]
                for k in ("average_precision", "roc_auc", "brier_score", "log_loss")
            },
        }
    return decisions


def recommendation(
    metrics: dict[str, Any], decisions: dict[str, Any], top: dict[str, Any]
) -> dict[str, Any]:
    ranking = sorted(
        PRIMARY,
        key=lambda m: (
            -metrics[m]["pooled_oof"]["average_precision"],
            -metrics[m]["pooled_oof"]["roc_auc"],
            -sum(t["pooled"]["tp"] for t in top[m]),
            metrics[m]["fold_mean_std"]["average_precision"]["sample_std"],
            PRIMARY.index(m),
        ),
    )[0]
    eligible = [m for m in PRIMARY if decisions[m]["eligible"]]
    probability = (
        min(
            eligible,
            key=lambda m: (
                metrics[m]["pooled_oof"]["brier_score"],
                metrics[m]["pooled_oof"]["log_loss"],
                abs(metrics[m]["mean_bias"]),
                PRIMARY.index(m),
            ),
        )
        if eligible
        else None
    )
    joint = probability is not None and all(
        metrics[ranking]["pooled_oof"][metric] - metrics[probability]["pooled_oof"][metric] <= 0.02
        for metric in ("average_precision", "roc_auc")
    )
    return {
        "best_ranking_candidate": ranking,
        "best_probability_candidate": probability,
        "probability_guardrail_pass": eligible,
        "joint_probability_ranking_candidate": probability if joint else None,
        "proposal": "b2_research_probability_candidate"
        if joint
        else "ranking_only_research_output",
        "proposal_method": ("sigmoid" if probability.endswith("_sigmoid") else "raw")
        if joint and probability
        else None,
        "b2_created": False,
        "api_changed": False,
        "production_approved": False,
        "independent_validation": False,
        "limitations": "Fixed b1 hyperparameters were previously selected on full development; "
        "this is a post-final exploratory study, not nested-unbiased or external validation. "
        "Guardrails are IBM project screens, not an employer probability certificate. "
        "No reuse of consumed f1 to validate a changed model.",
    }


def analyze(
    oof: pd.DataFrame, reference: pd.DataFrame, frame: pd.DataFrame
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    original = oof.loc[oof.scenario == "original"]
    metrics = tuning.summarize(original)
    dummy = tuning.summarize(reference.loc[reference.model == "dummy_prior"])["dummy_prior"]
    bins = []
    for name, rows in original.groupby("model", sort=True):
        key = str(name)
        metrics[key]["mean_probability"] = float(rows.probability.mean())
        metrics[key]["prevalence"] = float(rows.y_true.mean())
        metrics[key]["mean_bias"] = float(rows.probability.mean() - rows.y_true.mean())
        metrics[key]["calibration_intercept"] = None
        metrics[key]["calibration_slope"] = None
        metrics[key]["calibration_coefficients_status"] = POLICY["calibration_intercept_slope"]
        for b in reliability_bins(rows.y_true, rows.probability, count=5):
            bins.append(
                {
                    "model": key,
                    **b,
                    "positives": int(round(b["count"] * b["observed_positive_rate"])),
                }
            )
    top = {
        name: [policy_metrics(original.loc[original.model == name], p) for p in (5, 10, 15)]
        for name in PRIMARY + [SENSITIVITY]
    }
    for blocks in top.values():
        for block in blocks:
            block["pooled"]["lift"] = block["pooled"]["precision"] / float(frame.Attrition.mean())
            for f in block["folds"]:
                f["lift"] = f["precision"] / (f["positives"] / f["n"])
    decisions = guardrails(metrics, dummy)
    proposed = recommendation(metrics, decisions, top)
    missing = {
        name: {
            scenario: stress_metrics(
                oof.loc[(oof.model == name) & (oof.scenario == scenario)],
                original.loc[original.model == name],
            )
            for scenario in SCENARIOS
            if scenario != "original"
        }
        for name in PRIMARY + [SENSITIVITY]
    }
    cohort = frame[[*baseline.IDS, "Age"]]
    groups = pd.concat(
        [
            candidate.subgroup_table(original.loc[original.model == name], cohort).assign(
                model=name
            )
            for name in PRIMARY + [SENSITIVITY]
        ],
        ignore_index=True,
    )
    old = original.loc[original.model == "rf_unweighted_raw"]
    new = original.loc[original.model == SENSITIVITY]
    pairs = new.merge(
        old[["source_row", "fold", "top_10"]],
        on=["source_row", "fold"],
        validate="one_to_one",
        suffixes=("_no_age", "_with_age"),
    )
    overlaps = []
    for fold in (0, 1, 2, 3, 4, 5):
        rows = pairs if fold == 0 else pairs.loc[pairs.fold == fold]
        intersection = int(((rows.top_10_no_age == 1) & (rows.top_10_with_age == 1)).sum())
        denominator = int(rows.top_10_with_age.sum())
        overlaps.append(
            {
                "fold": fold,
                "intersection": intersection,
                "alerts_each": denominator,
                "overlap_fraction": intersection / denominator,
                "symmetric_difference_people": int(
                    (rows.top_10_no_age != rows.top_10_with_age).sum()
                ),
                "replaced_people_each_list": denominator - intersection,
            }
        )
    gaps = []
    for name in ("rf_unweighted_raw", SENSITIVITY):
        for fold in (0, 1, 2, 3, 4, 5):
            rows = groups.loc[(groups.model == name) & (groups.fold == fold)]
            gaps.append(
                {
                    "model": name,
                    "fold": fold,
                    "alert_rate_max_minus_min": float(
                        rows.alert_rate.max() - rows.alert_rate.min()
                    ),
                    "tpr_max_minus_min": float(rows.tpr.max() - rows.tpr.min()),
                    "all_groups_supported": bool(rows.sufficient_support.all()),
                }
            )
    raw_sigmoid_membership = []
    for family in ("rf_balanced", "rf_unweighted"):
        for fold in (1, 2, 3, 4, 5):
            r = original.loc[
                (original.model == family + "_raw") & (original.fold == fold)
            ].sort_values("source_row")
            s = original.loc[
                (original.model == family + "_sigmoid") & (original.fold == fold)
            ].sort_values("source_row")
            for percent in (5, 10, 15):
                raw_sigmoid_membership.append(
                    {
                        "family": family,
                        "fold": fold,
                        "capacity": percent,
                        "membership_changed": int(
                            np.sum(r[f"top_{percent}"].to_numpy() != s[f"top_{percent}"].to_numpy())
                        ),
                    }
                )
    sensitivity = {
        "name": "v3_sensitivity_no_age_14",
        "selection_eligible": False,
        "pooled_delta_no_age_minus_with_age": {
            k: metrics[SENSITIVITY]["pooled_oof"][k] - metrics["rf_unweighted_raw"]["pooled_oof"][k]
            for k in ("average_precision", "roc_auc", "brier_score", "log_loss")
        },
        "top10_overlap": overlaps,
        "descriptive_age_gaps": gaps,
        "not_fairness_certification": True,
    }
    return (
        {
            "metrics": metrics,
            "dummy": dummy,
            "guardrails": decisions,
            "top_k": top,
            "raw_sigmoid_top_k_check": raw_sigmoid_membership,
            "optional_missing": missing,
            "age_sensitivity": sensitivity,
            "recommendation": proposed,
        },
        pd.DataFrame(bins),
        groups,
    )


def table(frame: pd.DataFrame) -> str:
    def fmt(value: Any) -> str:
        if pd.isna(value):
            return "NA"
        return f"{value:.6f}" if isinstance(value, float) else str(value)

    return "\n".join(
        [
            "| " + " | ".join(frame.columns) + " |",
            "| " + " | ".join(["---"] * len(frame.columns)) + " |",
            *[
                "| " + " | ".join(fmt(v) for v in row) + " |"
                for row in frame.itertuples(index=False, name=None)
            ],
        ]
    )


def report_results(
    report: Path, analysis: dict[str, Any], bins: pd.DataFrame, groups: pd.DataFrame
) -> None:
    for name, value in analysis.items():
        tuning.preserve_json(report / f"{name}.json", value, resume=True)
    metrics = analysis["metrics"]
    comparison = pd.DataFrame(
        [
            {
                "model": name,
                **{
                    k: metrics[name]["pooled_oof"][k]
                    for k in ("average_precision", "roc_auc", "brier_score", "log_loss")
                },
                "mean_probability": metrics[name]["mean_probability"],
                "mean_bias": metrics[name]["mean_bias"],
                "AP_fold_mean": metrics[name]["fold_mean_std"]["average_precision"]["mean"],
                "AP_fold_sample_SD": metrics[name]["fold_mean_std"]["average_precision"][
                    "sample_std"
                ],
                "probability_guardrails_pass": analysis["guardrails"][name]["eligible"],
            }
            for name in PRIMARY
        ]
    )
    folds = pd.DataFrame(
        [
            {"model": name, **row}
            for name in PRIMARY + [SENSITIVITY]
            for row in metrics[name]["folds"]
        ]
    )
    for name, frame in (
        ("metrics_comparison", comparison),
        ("fold_metrics", folds),
        ("calibration_bins", bins),
        ("age_subgroups", groups),
    ):
        path = report / f"{name}.csv"
        if not path.exists():
            frame.to_csv(path, index=False, mode="x", na_rep="NA")
        expected = frame.reset_index(drop=True).copy()
        for column in expected:
            if expected[column].isna().all():
                expected[column] = np.nan
        pd.testing.assert_frame_equal(
            pd.read_csv(path),
            expected,
            check_dtype=False,
            check_exact=False,
            atol=1e-12,
            rtol=1e-12,
        )
    plot = report / "reliability.png"
    if not plot.exists():
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 3, figsize=(13, 8), sharex=True, sharey=True)
        for ax, name in zip(axes.flat, PRIMARY + [SENSITIVITY], strict=True):
            rows = bins.loc[bins.model == name]
            ax.plot([0, 1], [0, 1], "--", color="gray")
            ax.plot(rows.mean_predicted_probability, rows.observed_positive_rate, "o-")
            for b in rows.itertuples():
                ax.annotate(
                    f"n={b.count}, +={b.positives}",
                    (b.mean_predicted_probability, b.observed_positive_rate),
                    fontsize=6,
                    xytext=(2, 5),
                    textcoords="offset points",
                )
            ax.set(
                title=name,
                xlim=(0, 1),
                ylim=(0, 1),
                xlabel="Mean probability",
                ylabel="Observed attrition",
            )
        fig.tight_layout()
        fig.savefig(plot, dpi=150)
        plt.close(fig)
    rec = analysis["recommendation"]
    top_rows = pd.DataFrame(
        [
            {
                "model": name,
                "capacity": b["capacity_percent"],
                **{
                    k: b["pooled"][k]
                    for k in ("alerts", "tp", "fp", "fn", "precision", "recall", "f2", "lift")
                },
            }
            for name in PRIMARY
            for b in analysis["top_k"][name]
        ]
    )
    deltas = pd.DataFrame(
        [
            {
                "model": name,
                **analysis["guardrails"][name]["pooled_delta_vs_raw"],
                "Brier_worse_folds": analysis["guardrails"][name]["brier_worse_folds"],
            }
            for name in PRIMARY
            if name.endswith("sigmoid")
        ]
    )
    selected = list(
        dict.fromkeys([rec["best_ranking_candidate"], rec["best_probability_candidate"]])
    )
    missing_rows = pd.DataFrame(
        [
            {
                "model": name,
                "scenario": scenario,
                "AP": block["pooled"]["probability_metrics"]["average_precision"],
                "Brier": block["pooled"]["probability_metrics"]["brier_score"],
                "mean_absolute_probability_change": block["pooled"][
                    "mean_absolute_probability_change"
                ],
                "top10_overlap": block["pooled"]["top_10_overlap_fraction"],
            }
            for name in selected
            if name is not None
            for scenario, block in analysis["optional_missing"][name].items()
        ]
    )
    lines = [
        f"# {report.name}: development-only remediation recommendation",
        "",
        "This exploratory study follows final evaluation. Final CSV/f1 predictions were not read. "
        "Fixed b1 hyperparameters were selected using all development previously; "
        "these OOF scores are not unbiased nested-selection evaluation. "
        "The consumed final cohort cannot independently certify any revised candidate.",
        "",
        "1176 development rows, 190 positive (16.1565%); same saved five outer folds. "
        "RF balanced/unweighted differ only in class_weight. Independent raw refits and sigmoid "
        "ensemble=False inner 4-fold seed43; seed42 models; transformers are train-only. "
        "70 base fits +10 calibrator fits; stress on validation copies without retraining.",
        "",
        "## Five main candidates",
        "",
        table(comparison),
        "",
        f"Dummy outer-train-prior OOF: {analysis['dummy']['pooled_oof']}",
        "",
        "## Sigmoid minus corresponding raw",
        "",
        table(deltas),
        "",
        "Project guardrails were frozen before results: both losses below dummy; Brier worse in at "
        "most 2/5 folds; AP/ROC drop <=0.02 vs raw; absolute mean bias <=0.05. "
        "Raw models compare to themselves but must pass dummy/bias checks. No significance claim.",
        "",
        "## Ranking and probability decisions",
        "",
        f"Best ranking: **{rec['best_ranking_candidate']}**. Best probability: "
        f"**{rec['best_probability_candidate']}**. Eligible: {rec['probability_guardrail_pass']}.",
        "",
        f"Proposal: **{rec['proposal']}**, candidate={rec['joint_probability_ranking_candidate']}, "
        f"method={rec['proposal_method']}. Selection follows the locked ordering in protocol.json; "
        "joint recommendation requires guardrails and AP/ROC within 0.02 of ranking winner. "
        "This is a proposal only: no b2 fit/bundle, API, HR policy or production approval.",
        "",
        "## Top-k, fixed 5/10/15% scenarios",
        "",
        table(top_rows),
        "",
        f"Raw/sigmoid membership changes across 30 family/fold/capacity checks: "
        f"{sum(r['membership_changed'] for r in analysis['raw_sigmoid_top_k_check'])}. "
        "Per-fold floor budgets and source_row ties are label-independent.",
        "",
        "## Reliability",
        "",
        "![Five quantile bins](reliability.png)",
        "",
        "Bin counts/positives are in calibration_bins.csv. A small mean bias alone does not prove "
        "calibration. Intercept/slope are NA: no validated implementation was found; sigmoid "
        "coefficients are not misrepresented as independent calibration diagnostics.",
        "",
        "## Optional-missing stress for selected ranking/probability candidates",
        "",
        table(missing_rows),
        "",
        "All six configurations were stress-tested; full per-fold and per-scenario values are in "
        "optional_missing.json. No imputer update or retraining occurred.",
        "",
        "## No-Age sensitivity — not eligible for automatic replacement",
        "",
        "No-Age minus with-Age metrics: "
        f"{analysis['age_sensitivity']['pooled_delta_no_age_minus_with_age']}",
        "",
        table(pd.DataFrame(analysis["age_sensitivity"]["top10_overlap"])),
        "",
        table(
            groups.loc[
                (groups.fold == 0) & groups.model.isin(["rf_unweighted_raw", SENSITIVITY]),
                [
                    "model",
                    "group",
                    "n",
                    "positives",
                    "alerts",
                    "alert_rate",
                    "tp",
                    "fn",
                    "tpr",
                    "fpr",
                    "sufficient_support",
                ],
            ]
        ),
        "",
        "Per-fold age counts/rates are in age_subgroups.csv. Support flags use "
        "n>=50, positives>=10, negatives>=10; sparse fold rates are not stable estimates. "
        "No-Age has 14 inputs, Age only audit metadata. Tenure/income can encode age-related "
        "information; removing Age alone does not prove fairness or approve a replacement.",
        "",
        "Threshold 0.5 is reference only (full metrics in metrics.json/fold_metrics.csv). "
        "IBM labels are cross-sectional, not proof of a 3–6 month horizon, intervention benefit or "
        "generalization to new employers. A future probability API still needs a data contract and "
        "external/temporal validation; no API changes were made. No commit/push.",
    ]
    path = report / "recommendation.md"
    text = "\n".join(lines) + "\n"
    if path.exists():
        if path.read_text("utf-8") != text:
            raise ValueError("Existing recommendation changed; refusing overwrite.")
    else:
        path.write_text(text, encoding="utf-8")
