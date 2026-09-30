"""Pooled, paired-fold and traceability reporting for V2 baseline ablation."""

from __future__ import annotations

import json
from typing import Any

import pandas as pd

from eris_ml.evaluation.feature_ablation import METRICS, MODELS, REMOVED, AblationResult


def markdown_table(frame: pd.DataFrame) -> str:
    """Render small numeric tables without an optional tabulate dependency."""
    lines = ["| " + " | ".join(map(str, frame.columns)) + " |",
             "| " + " | ".join("---" for _ in frame.columns) + " |"]
    for row in frame.itertuples(index=False, name=None):
        cells = [f"{value:.6f}" if isinstance(value, float) else str(value) for value in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def build_report(result: AblationResult, trace: dict[str, Any]) -> str:
    """Describe IBM evidence without imposing a post-hoc noninferiority threshold."""
    summary, folds = result.summary, result.folds
    logistic = summary.loc[summary.model == "logistic_regression"].set_index("feature_set")
    general = logistic.loc["v2_general_16"]
    direct = []
    for reference in ("v2_no_department_17", "v2_no_jobrole_17"):
        for model in MODELS:
            indexed = summary.loc[summary.model == model].set_index("feature_set")
            direct.append({
                "model": model, "comparison": f"general_16 minus {reference}",
                **{metric: indexed.loc["v2_general_16", metric] - indexed.loc[reference, metric]
                   for metric in METRICS},
            })
    paired = []
    for name in REMOVED:
        if name == "v2_full_18":
            continue
        current = folds.loc[(folds.model == "logistic_regression") & (folds.feature_set == name)]
        reference = folds.loc[(folds.model == "logistic_regression")
                              & (folds.feature_set == "v2_full_18")].set_index("fold")
        for _, row in current.iterrows():
            paired.append({"feature_set": name, "fold": row["fold"], **{
                metric: row[metric] - reference.loc[row["fold"], metric] for metric in METRICS
            }})
    paired_frame = pd.DataFrame(paired)
    notes = []
    for name in REMOVED:
        if name == "v2_full_18":
            continue
        row = logistic.loc[name]
        changes = paired_frame.loc[paired_frame.feature_set == name]
        worst = changes.loc[changes.average_precision.idxmin()]
        notes.append(
            f"- {name}: pooled delta AP {row['average_precision_delta_vs_full']:+.6f}, "
            f"ROC-AUC {row['roc_auc_delta_vs_full']:+.6f}, "
            f"Brier {row['brier_score_delta_vs_full']:+.6f}. "
            f"AP improved in {int((changes.average_precision > 0).sum())}/5 folds; "
            f"Brier improved in {int((changes.brier_score < 0).sum())}/5 folds. "
            f"Largest AP deterioration: fold {int(worst['fold'])}, "
            f"delta {worst['average_precision']:+.6f}."
        )
    full_ap = logistic.loc["v2_full_18", "average_precision"]
    dummy_ap = summary.loc[(summary.feature_set == "v2_general_16")
                           & (summary.model == "dummy_prior"), "average_precision"].iloc[0]
    interpretation = (
        f"General_16 pooled AP is {general['average_precision']:.6f}, versus Dummy "
        f"{dummy_ap:.6f} and full_18 {full_ap:.6f}. "
        "This baseline supports taking general_16 forward as a candidate with the "
        "full_18 reference retained, not declaring it equivalent or superior. "
        "No acceptance margin or statistical test was pre-registered; five dependent "
        "CV folds do not establish significance. The observed losses and fold-specific "
        "variation must remain visible in the next model-family comparison."
    )
    columns = ["feature_set", "model", *METRICS]
    delta_columns = ["feature_set", "model", *(f"{m}_delta_vs_full" for m in METRICS)]
    stability_columns = ["feature_set", "model"]
    for metric in METRICS:
        stability_columns.extend([f"{metric}_mean", f"{metric}_std",
                                  f"{metric}_min", f"{metric}_max"])
    return "\n\n".join([
        "# V2 baseline feature-ablation",
        "## Setup",
        f"Development: {trace['development_rows']} rows; target "
        f"{trace['target_distribution']}; positive rate {trace['positive_rate']:.6%}. "
        "Four feature sets, two fixed baselines each; one shared StratifiedKFold "
        "assignment (5 folds, shuffle=True, random_state=42), generated once in original "
        "development row order. Every row has exactly one validation prediction per candidate.",
        "The 25-column V1 development feature storage is projected to exactly 18/17/17/16 "
        "columns before fitting. IDs/target never enter preprocessing. Each fold clones and "
        "fits the entire pipeline on training only: nominal most-frequent imputation + "
        "OneHotEncoder(handle_unknown='ignore'), ordinal/numeric median imputation + "
        "StandardScaler. Category vocabularies are learned only on training rows; unseen "
        "categories are not coerced to IBM categories. Dummy uses strategy='prior'.",
        "Logistic reuses configs/models/logistic_regression.yaml unchanged. Resolved "
        "constructor parameters (including library defaults):\n\n```json\n"
        + json.dumps(trace["logistic_parameters"], indent=2) + "\n```\n\n"
        "With liblinear and C=1 the effective regularization is L2. Parameters are fixed "
        "for every feature set; no search, calibration or threshold selection was performed.",
        "## Pooled OOF results",
        "PR-AUC is sklearn average_precision_score, not trapezoidal PR area. Higher AP/ROC "
        "is better; lower Brier is better. Prevalence provides the no-skill AP context. "
        "Dummy learns slightly different training priors between folds; therefore pooled "
        "Dummy AP/ROC can differ slightly from prevalence/0.5 despite constant scores within "
        "each fold. Pooled OOF scores are not the average of the five fold scores.",
        markdown_table(summary[columns]),
        "## Deltas versus full_18 (candidate minus full_18, same model)",
        markdown_table(summary[delta_columns]),
        "## Direct general_16 versus each 17-feature variant",
        markdown_table(pd.DataFrame(direct)),
        "## Fold metrics",
        markdown_table(folds),
        "## Fold stability",
        "Standard deviations use ddof=1; min/max describe the observed five-fold range.",
        markdown_table(summary[stability_columns]),
        "## Paired Logistic fold differences versus full_18",
        markdown_table(paired_frame),
        "## Interpretation and next step",
        "\n".join(notes),
        interpretation,
        "Next step: compare fixed Random Forest and XGBoost baselines for the four locked "
        "feature sets on this exact saved development fold assignment, retaining Logistic "
        "as the reference. Do not tune in that comparison.",
        "IBM cross-validation measures this benchmark only. Removing organization-specific "
        "categories is a portability motivation, not evidence of performance at new employers "
        "or of a 3–6 month prediction horizon. Dictionary currency/overtime/survey/snapshot "
        "rules remain unvalidated for enterprise data. No V1 final-test results are used "
        "here; no direct V1 performance comparison is made. The V1 final-test ledger was "
        "already completed once; that set is not an untouched V2 holdout.",
        "## Traceability and checks",
        "Final-test CSV accessed: No. Production model serialized: No. "
        "OOF: 8 candidates x development_rows, unique row/employee IDs within candidate, "
        "aligned targets/folds, finite probabilities in [0,1]. V1 artifacts are preserved. "
        "All output paths use exclusive creation and the V2 namespace.",
        "```json\n" + json.dumps(trace, indent=2, ensure_ascii=False, allow_nan=False) + "\n```",
    ]) + "\n"
