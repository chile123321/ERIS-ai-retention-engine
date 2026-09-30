"""HR-oriented capacity scenario reporting; no business optimum or production threshold."""

import json
from typing import Any

import pandas as pd

from eris_ml.evaluation.feature_ablation_reporting import markdown_table


def build_report(tables: dict[str, pd.DataFrame], trace: dict[str, Any]) -> str:
    summary, folds = tables["summary"], tables["folds"]
    columns = [
        "feature_set",
        "policy",
        "capacity_percent",
        "tp",
        "fp",
        "fn",
        "tn",
        "alerts",
        "alert_rate",
        "precision",
        "recall",
        "f1",
        "f2",
        "reviews_per_tp",
        "reviews_per_tp_status",
        "exceeded_folds",
        "excess_alerts",
    ]
    sections = [
        "# V2 cross-fitted threshold policy evaluation",
        "## Hypotheses and scope",
        "1,176 IBM development rows, {0:986, 1:190}. Only fixed Logistic baseline raw: "
        "general_16 and no_department_17. Saved outer folds, fingerprints, row IDs/targets, "
        "and baseline/calibration raw probabilities matched before fitting. HR capacities "
        "5%, 10%, 15% are hypothetical, not confirmed business limits. No cost ratios assumed.",
        "## Frozen policy definitions",
        "Threshold: inner OOF probabilities from four stratified folds (shuffle=True, seed=43) "
        "inside each outer-training set. Every inner fit includes preprocessing and unchanged "
        "Logistic (liblinear, C=1, max_iter=2000, no class weighting, seed=42). Candidates: "
        "unique inner probabilities plus 0 and 1, and a no-alert sentinel nextafter(1,+inf). "
        "Predict positive when p>=threshold, retaining all boundary ties. Feasible alerts must "
        "not exceed floor(n*capacity_percent/100). Maximize recall, then precision, then "
        "threshold. "
        "No-alert is a disabled policy, not a deployable probability cutoff. Select once from "
        "inner labels, apply unchanged to stored outer raw probabilities. Do not adjust the "
        "threshold after observing outer labels or alert volume.",
        "Top-k: each outer-validation fold is a separate simulated batch; k=floor(n*pct/100). "
        "Sort probability descending then source_row ascending (a stable, label-independent tie "
        "rule). Select exactly k even across tied scores. This requires the complete batch and "
        "does not define an employee-independent probability threshold. Top-k is not calculated "
        "over pooled OOF. Summing per-batch rounded limits yields 55/115/175 alerts, not rounded "
        "5%/10%/15% of the pooled 1,176 rows. No new model family or outer refit: reuse verified "
        "outer predictions; only 40 fixed inner pipeline fits are performed.",
        "Reference: p>=0.5, no business approval. capacity_percent=0 in artifacts means no "
        "capacity constraint applies to this reference; its budget field is a placeholder "
        "batch size, not an HR limit. Precision/recall/F scores use zero for zero denominators. "
        "Reviews per true positive = alerts/TP; when TP=0 it is undefined (blank with explicit "
        "undefined_no_tp status), never zero. This is descriptive workload per IBM-positive case.",
        "## Pooled OOF outcomes",
        markdown_table(summary[columns]),
        "## Every outer fold",
        markdown_table(folds),
        "## Inner-selected thresholds (not production thresholds)",
        markdown_table(tables["selections"]),
        "## Threshold variability (sample SD)",
        markdown_table(tables["stability"]),
        "## Paired JobRole comparison",
        "17 minus 16 on identical rows/folds and nominal capacity. fold=0 denotes pooled counts. "
        "Top-k compares equal alert counts; threshold policies may differ in actual alert counts, "
        "so their detection difference is not a same-workload comparison. Positive cases unique "
        "to either policy expose changes hidden by net TP differences.",
        markdown_table(tables["paired"]),
        "## Interpretation and next step",
        "Threshold feasibility on inner OOF does not guarantee a future batch capacity limit; "
        "all validation overruns are reported, never corrected after evaluation. Top-k respects "
        "the batch limit by construction but trades away a common probability cutoff. "
        "Use the observed 5/10/15% workload–detection trade-offs to discuss HR capacity, review "
        "process and false-positive/false-negative consequences. No business-optimal threshold "
        "or production policy is chosen. See docs/threshold_policy_v2.md for interpretation.",
        "IBM is cross-sectional: detected cases mean positive IBM labels, not people who can "
        "be reached before departure in one month or validated departures 3–6 months later. "
        "JobRole benefit on IBM does not establish transfer to other employers and requires "
        "mapping IBM titles. Previous V1 final-test scores were not used; that previously "
        "evaluated set is not an independent new V2 holdout.",
        "One next step: review these capacity scenarios with HR and document an approved "
        "batch size, review capacity, policy type and tie handling before freezing any policy.",
        "## Traceability and safety",
        "No final-test access, tuning, calibration, fitted-model serialization, API changes, "
        "commit or push. Prior artifacts preserved; all outputs exclusively created.",
        "```json\n" + json.dumps(trace, indent=2, allow_nan=False) + "\n```",
    ]
    return "\n\n".join(sections) + "\n"
