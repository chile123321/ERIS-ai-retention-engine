"""Development-only tree comparison tables and reproducible experiment provenance."""

import json
from typing import Any

from eris_ml.evaluation.feature_ablation import METRICS, AblationResult
from eris_ml.evaluation.feature_ablation_reporting import markdown_table
from eris_ml.evaluation.model_comparison import ALL_MODELS, comparison_deltas


def build_report(result: AblationResult, trace: dict[str, Any]) -> str:
    summary = result.summary
    deltas = comparison_deltas(result)
    notes = []
    for model in ALL_MODELS:
        rows = summary.loc[summary.model == model].set_index("feature_set")
        gap = rows.loc["v2_general_16", "average_precision"] - rows.loc[
            "v2_full_18", "average_precision"]
        job = deltas.loc[(deltas.model == model)
                         & (deltas.reference_model == model)
                         & (deltas.feature_set == "v2_no_jobrole_17")
                         & (deltas.reference_feature == "v2_full_18")
                         & (deltas.metric == "average_precision")].iloc[0]
        notes.append(f"- {model}: general_16 minus full_18 AP {gap:+.6f}; removing JobRole "
                     f"AP {job.pooled_delta:+.6f}, improved in {int(job.improved_folds)}/5 folds.")
    return "\n\n".join([
        "# V2 baseline model comparison",
        "## Protocol",
        "Development only: 1,176 rows, target {0: 986, 1: 190}, positive rate 16.156463%. "
        "Use the saved baseline_feature_ablation_v2_assignments.csv without regeneration "
        "(StratifiedKFold 5, shuffle=True, seed=42). Logistic probabilities are reused, "
        "with IDs, targets, folds and provenance hashes validated before any training. "
        "Four feature sets are not four folds. Each of 12 candidates has 1,176 OOF rows.",
        "RF and XGBoost reuse the unchanged V1 baseline YAMLs, seed 42 and n_jobs=-1. "
        "All defaults, library versions and input fingerprints are recorded below before fitting. "
        "Forty sequential fold fits, no tuning or early stopping. Exact feature projection "
        "occurs before fit. Train-only nominal imputation/one-hot encoding and numeric/ordinal "
        "median imputation/scaling remain inside each fresh pipeline. IDs and excluded columns "
        "never enter preprocessing. Unknown categories use handle_unknown='ignore'.",
        "## Pooled OOF metrics",
        "PR-AUC = average precision, not trapezoidal area. Higher AP/ROC and lower Brier "
        "are better. Metrics below are pooled, not averages of fold scores.",
        markdown_table(summary[["feature_set", "model", *METRICS]]),
        "## Fold mean and sample SD (ddof=1)",
        markdown_table(summary.drop(columns=list(METRICS))),
        "## Individual fold metrics", markdown_table(result.folds),
        "## Paired deltas",
        "Candidate minus reference. Includes removal of each/both fields, general_16 versus "
        "both 17-feature sets, and trees versus Logistic on the same features. Improved-fold "
        "counts use the correct direction for each metric; ties are not improvements.",
        markdown_table(deltas),
        "## Interpretation", "\n".join(notes),
        "Retain general_16 as the portability-motivated candidate, not a demonstrated "
        "cross-company solution. Inspect the paired fold differences above alongside pooled "
        "scores; five dependent folds do not establish statistical significance or equivalence. "
        "The tracked docs/baseline_model_comparison_v2.md records the result-specific decision. "
        "JobRole is predictive context on IBM, not evidence of causality. Organization-specific "
        "taxonomies still lack a validated cross-employer mapping.",
        "IBM CV does not validate Vietnamese employers or a 3–6 month horizon. No V1 final-test "
        "score is used. The V1 final test was evaluated once previously and is not a new "
        "independent V2 holdout. Its CSV was not accessed. No calibration, threshold selection, "
        "model serialization, API changes, commit or push were performed.",
        "## Full parameter and provenance trace",
        "For XGBoost, null constructor values mean library defaults; the booster configuration "
        "captured from a separate synthetic probe before development training resolves those "
        "defaults. The probe is not an experiment or a fitted-model export.",
        "```json\n" + json.dumps(trace, indent=2, ensure_ascii=False, allow_nan=False) + "\n```",
    ]) + "\n"
