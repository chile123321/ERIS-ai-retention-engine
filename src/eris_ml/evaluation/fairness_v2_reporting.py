"""Aggregate technical report; never publish employee identifiers."""

import json
from typing import Any

import pandas as pd

from eris_ml.evaluation.feature_ablation_reporting import markdown_table


def _table(frame: pd.DataFrame, columns: list[str]) -> str:
    values = frame[columns].astype(object).where(frame[columns].notna(), "undefined")
    return markdown_table(values)


def build_report(tables: dict[str, pd.DataFrame], trace: dict[str, Any]) -> str:
    raw = tables["probability"].query("fold == 0")
    policy = tables["policy"].query("fold == 0")
    gaps = tables["disparities"].query("fold == 0")
    support = raw.loc[raw.feature_set == "v2_general_16"]
    sections = [
        "# V2 development-only fairness/subgroup audit",
        "## Scope",
        "1176 development rows, 190 IBM positives, 986 negatives. Two fixed Logistic "
        "baseline raw candidates; saved top-k 5/10/15% predictions only. No fit or new "
        "threshold. Final-test CSV not accessed. Business/production approval: false.",
        "AgeGroup, Gender and MaritalStatus are audit attributes, not newly added model "
        "features. Age itself is already a feature of both models. General_16 excludes "
        "Department/JobRole; no_department_17 retains JobRole.",
        "## Protocol locked before results",
        "Age bins reuse V1: 18-29, 30-39, 40-49, 50+. Missing/unknown inputs reject before "
        "metrics: no dropped rows or imputation. Empty fold groups have explicit zero "
        "counts. Minimum support is n>=50, positives>=10, negatives>=10; precision "
        "additionally requires >=10 alerts. Unsupported AP/ROC/Brier/TPR/FPR/precision "
        "are undefined, never zero. One-class AUC is not computed. Counts, empirical "
        "prevalence, score distribution and alert fraction are retained as observations.",
        "Every operational metric includes a numerator/denominator in policy.csv: "
        "alert_rate=alerts/n; TPR=TP/positives; FPR=FP/negatives; precision=TP/alerts. "
        "Wilson 95% intervals are descriptive conditional intervals, not full "
        "CV/training/batch uncertainty or simultaneous inference. No significance claim.",
        "Top-k is global within each saved outer-validation fold, NOT reranked within "
        "subgroups. Verification uses floor(batch_size*percent/100), descending score "
        "and source_row ascending ties; predictions are not replaced. Fold 0 denotes "
        "pooled OOF, not another fold. Technical folds do not define real HR work batches. "
        "Budgets sum to 55/115/175 per model; subgroup alert rates need not equal capacity.",
        "## Support",
        _table(support, ["attribute", "group", "n", "positives", "negatives",
                         "base_rate", "support"]),
        "## Raw probability metrics, pooled",
        _table(raw, ["feature_set", "attribute", "group", "n", "positives", "base_rate",
                     "mean_probability", "average_precision", "roc_auc", "brier", "support"]),
        "Score min/10/25/50/75/90/max quantiles and sample SD, plus fold metrics, are in "
        "probability.csv. Insufficient sample/class support is explicit.",
        "## Saved top-k outcomes, pooled",
        _table(policy, ["feature_set", "capacity_percent", "attribute", "group", "n",
                        "positives", "negatives", "alerts", "alert_rate", "tp", "fp", "fn", "tn",
                        "tpr", "fpr", "precision", "support", "precision_support"]),
        "## Between-group descriptive review flags",
        "V1 heuristics reused: TPR max-min>0.10, FPR max-min>0.10, selection-rate "
        "min/max<0.80, or absolute mean-score minus prevalence>0.05. They are not legal "
        "thresholds, causal findings or approval criteria. REVIEW requires all groups "
        "supported; otherwise INSUFFICIENT_EVIDENCE. NO_SCREENING_FLAG is not a pass "
        "certificate. Gaps are undefined when required group metrics are undefined.",
        _table(gaps, ["feature_set", "capacity_percent", "attribute", "supported_groups",
                     "groups", "tpr_gap", "fpr_gap", "selection_rate_ratio",
                     "descriptive_flags", "status"]),
        "## Paired 17 minus 16 comparison",
        "Same employees/fold/global batch budget; demographic alert counts may differ. "
        "Common/exclusive TP is computed on aligned saved decisions before aggregation.",
        _table(tables["paired_policy"].query("fold == 0"),
               ["capacity_percent", "attribute", "group", "delta_tp", "delta_tpr", "delta_fpr",
                "delta_precision", "common_tp", "only_16_tp", "only_17_tp"]),
        "## Fold stability",
        _table(tables["disparity_stability"], [
            "feature_set", "capacity_percent", "attribute", "supported_folds",
            "tpr_gap_valid_folds", "tpr_gap_mean", "tpr_gap_sample_sd", "tpr_gap_min",
            "tpr_gap_max", "fpr_gap_mean", "fpr_gap_sample_sd"]),
        "All per-fold counts, denominators and support flags remain in the metric CSVs. "
        "Stability uses sample SD (ddof=1), range and valid-fold count; no averaging "
        "undefined values as zeros. Paired stability records per-group 17-minus-16 "
        "differences and positive-fold counts; descriptive TP differences remain "
        "available even when rate estimates have insufficient support.",
        "## Review, not approval",
        "Bring flagged disparities, sparse positives and sparse alert denominators to HR. "
        "Do not automatically choose a feature set or mitigate by new subgroup budgets. "
        "JobRole requires reliable employer mapping even when IBM metrics improve.",
        "V1 AgeGroup REVIEW motivated this analysis but involved different models and "
        "threshold-based policies, so no V1 conclusion/effect size is transferred. "
        "IBM cross-sectional labels do not prove fairness at Vietnamese/new employers, "
        "intervention effectiveness or 3-6 month prediction. The previously used V1 "
        "final test is not a fresh V2 holdout; no final-test scores were used.",
        "Next: HR confirmation or explicitly confirmed capstone simulation of batch scope, "
        "capacity, policy/feature option, data semantics and intervention workflow. "
        "No decision is confirmed in this audit.",
        "## Provenance",
        "Preflight was written before computing subgroup results. Historical traces were "
        "not edited. Exact historical/current reviewed source revisions and immutable "
        "output hashes are recorded below. Current source is not falsely claimed to have "
        "produced past OOF. Python 3.11 runtime remains unverified.",
        "~~~json\n" + json.dumps(trace, indent=2, allow_nan=False) + "\n~~~",
    ]
    return "\n\n".join(sections) + "\n"

