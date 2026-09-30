"""Paired nested-CV evidence without production or cross-company claims."""

import json
from typing import Any

from eris_ml.evaluation.feature_ablation_reporting import markdown_table
from eris_ml.models.logistic_nested_v2 import FEATURES, NestedResult


def build_report(result: NestedResult, trace: dict[str, Any]) -> str:
    notes = []
    for feature in FEATURES:
        delta = result.deltas.loc[(result.deltas.feature_set == feature)
                                  & (result.deltas.reference_model == "baseline")]
        metrics = delta.set_index("metric")
        selected = result.selected.loc[result.selected.feature_set == feature]
        frequencies = selected.groupby(["C", "penalty"]).size().to_dict()
        notes.append(
            f"- {feature}: delta AP {metrics.loc['average_precision', 'pooled_delta']:+.6f} "
            f"(improved in {metrics.loc['average_precision', 'improved_folds']}/5 folds); "
            f"delta ROC {metrics.loc['roc_auc', 'pooled_delta']:+.6f}; "
            f"delta Brier {metrics.loc['brier_score', 'pooled_delta']:+.6f}. "
            f"Selected (C, penalty) frequencies: {frequencies}."
        )
    return "\n\n".join([
        "# V2 Logistic nested-CV tuning",
        "## Locked development protocol",
        "1,176 development rows, target {0: 986, 1: 190}; prevalence 16.156463%. "
        "Saved baseline outer assignment reused without regeneration. Inner StratifiedKFold "
        "4, shuffle=True, random_state=43; identical inner memberships across feature sets. "
        "The complete 20-candidate V1 grid and numerical tie rule were written to the immutable "
        "preflight trace before the first fit. Selection maximizes mean inner average precision; "
        "within 1e-12 of maximum, smaller C then L2 wins. No grid changes after results.",
        "Logistic uses liblinear, no class weighting, seed 42, max_iter=3000 (baseline: 2000). "
        "The higher iteration cap is reused from V1 tuning, not selected from V2 results. "
        "All imputation, categories and scaling are fit inside the pipeline separately for "
        "every inner fit and outer-train refit. Exact input columns are projected before fit. "
        "No Department in either variant; no JobRole in general_16. Outer-validation is used "
        "only for prediction/evaluation. No full-development search or fitted model export.",
        "## Pooled and fold-summary metrics",
        "AP is average precision. SD uses ddof=1; pooled metrics are not fold means.",
        markdown_table(result.summary),
        "## Each outer fold", markdown_table(result.folds),
        "## Selected parameters", markdown_table(result.selected),
        "## Paired deltas",
        "Candidate minus reference; lower Brier is better. Rows are matched by IDs and folds.",
        markdown_table(result.deltas),
        "## Stability and interpretation", "\n".join(notes),
        "These are development estimates of the tuning procedure, not the score of one fixed "
        "deployment configuration. No noninferiority margin or statistical significance claim "
        "is made. Selecting a procedure after repeated development comparisons can itself be "
        "optimistic. See docs/logistic_nested_v2.md for the decision and next step.",
        "General_16 avoids organization-specific JobRole mapping, but that motivation does "
        "not demonstrate cross-company generalization. IBM CV does not validate Vietnamese "
        "employers or a 3–6 month prediction horizon. Final test V1 was evaluated once before; "
        "it is not a new independent V2 holdout. Its CSV and scores were not used here.",
        "## Warnings", "```json\n" + json.dumps(trace["warnings"], indent=2) + "\n```",
        "## Provenance", "```json\n" + json.dumps(trace, indent=2, allow_nan=False) + "\n```",
    ]) + "\n"
