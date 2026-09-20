"""Development/final reporting for the frozen one-time academic evaluation."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd

from eris_ml.evaluation.final_evaluation import (
    GUARDRAILS,
    METRICS_PATH,
    PREDICTIONS_PATH,
    REPORT_PATH,
    THRESHOLD,
)

HOLDOUT_DISCLOSURE = (
    "The official locked final-test file was not opened before the authorized evaluation "
    "and no final-test metric was previously computed. A mixed fairness file containing "
    "final-test rows was inadvertently loaded during schema inspection. The workflow "
    "stopped before analysis, the incident was documented, and no final-test information "
    "was used to change the model, features, preprocessing, calibration or threshold."
)
LIMITATIONS = (
    "IBM public/synthetic benchmark; not representative of Vietnamese enterprises.",
    "AgeGroup fairness red flag (development audit REVIEW); no final-test fairness claim.",
    "Gender fairness evidence remains insufficient.",
    "Retained holdout evaluated under a documented protocol deviation.",
    "Raw dataset SHA-256 provenance gap in the frozen candidate evidence.",
    "Only 47 positive cases in the final test; confidence intervals remain important.",
)


def _formatted(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _metric_table(metrics: dict[str, Any], intervals: dict[str, list[float]],
                  checks: dict[str, bool]) -> str:
    guardrail_for = {
        "pr_auc": "pr_auc_minimum", "roc_auc": "roc_auc_minimum",
        "brier": "brier_maximum", "precision": "precision_minimum",
        "recall": "recall_minimum", "alert_rate": "alert_rate_maximum",
    }
    rows = ["| Metric | Point estimate | 95% CI | Guardrail | Pass/Fail |",
            "|---|---:|---|---|---|"]
    for metric in ("pr_auc", "roc_auc", "brier", "log_loss", "accuracy",
                   "balanced_accuracy", "precision", "recall", "f1", "f2", "alert_rate"):
        interval = intervals.get(metric)
        ci = "—" if interval is None else f"[{interval[0]:.6f}, {interval[1]:.6f}]"
        key = guardrail_for.get(metric)
        guardrail = "—" if key is None else f"{key} = {GUARDRAILS[key]}"
        status = "—" if key is None else ("PASS" if checks[key] else "FAIL")
        rows.append(f"| {metric} | {_formatted(metrics[metric])} | {ci} | {guardrail} | {status} |")
    return "\n".join(rows)


def _development_table(metrics: dict[str, Any], intervals: dict[str, list[float]]) -> str:
    development = {
        "pr_auc": 0.666912, "roc_auc": 0.848004, "brier": 0.088031,
        "alert_rate": 0.14625850340136054,
        "precision": 0.6453488372093024, "recall": 0.5842105263157895,
    }
    rows = ["| Metric | Development OOF | Final test | Absolute delta | Relative delta | "
            "Final 95% CI |",
            "|---|---:|---:|---:|---:|---|"]
    for name, baseline in development.items():
        current = float(metrics[name])
        change = current - baseline
        relative = change / baseline if baseline else float("nan")
        interval = intervals[name]
        rows.append(
            f"| {name} | {baseline:.6f} | {current:.6f} | {change:+.6f} | "
            f"{relative:+.2%} | [{interval[0]:.6f}, {interval[1]:.6f}] |"
        )
    return "\n".join(rows)


def build_report(payload: dict[str, Any]) -> str:
    """Render one audit-ready Markdown report without employee-level rows."""
    metrics = payload["metrics"]
    bootstrap = payload["bootstrap"]
    intervals = bootstrap["intervals"]
    checks = payload["guardrails_passed"]
    authorization = payload["authorization"]
    operations = "\n".join(f"| {name} | {_formatted(metrics[name])} |" for name in
                           ("tn", "fp", "fn", "tp", "alerts", "alert_rate",
                            "fp_per_tp", "fn_per_tp"))
    limits = "\n".join(f"- {line}" for line in LIMITATIONS)
    return f"""# Final-test evaluation — eris-xgboost-v1

## A. Authorization

- Authorized by: {authorization['authorized_by']}
- Scope: {authorization['authorized_scope']} (academic capstone only)
- Candidate: {authorization['authorized_candidate']}
- Maximum runs: {authorization['maximum_runs']}; actual run count: 1
- Authorization timestamp (UTC): {authorization['created_at_utc']}
- Business approved: No. Production approved: No.

## B. Frozen specification

- Candidate manifest SHA-256: `{payload['candidate_manifest_sha256']}`
- Model: frozen tuned XGBoost; raw probability; no calibration or model comparison.
- Feature schema: v1-full, 25 features; preprocessing fitted inside the frozen pipeline.
- Threshold: {THRESHOLD:.6f} (`capacity_15_percent`), unchanged.
- Final-test SHA-256: `{payload['final_test_sha256']}`
- Final rows: {payload['row_count']}; target counts: {payload['target_distribution']}.

## C. Holdout disclosure

Status: **retained_with_protocol_deviation**. {HOLDOUT_DISCLOSURE}

## D. Final metrics

{_metric_table(metrics, intervals, checks)}

Target-stratified paired percentile bootstrap: {bootstrap['valid_iterations']} valid /
{bootstrap['iterations_requested']} requested iterations; 95% confidence level; random seed 42.
Guardrails use point estimates registered before the final-test file was opened.

## E. Confusion and operational impact

| Metric | Value |
|---|---:|
{operations}

## F. Development versus final

{_development_table(metrics, intervals)}

Development operating metrics above are cross-fitted capacity-15 policy estimates;
the final metric uses the frozen full-development candidate threshold. Deltas are descriptive,
not an invitation to revise the candidate.

## G. Guardrail result

**{payload['result']}**. All six pre-declared checks must pass for
`PASS_WITH_KNOWN_LIMITATIONS`; no guardrail was changed after access.

## H. Limitations

{limits}

## I. Next-step restriction

Final-test results must not be used to revise this candidate.
Any new model version requires a newly defined evaluation dataset.
No production or business approval is implied by this academic result.
"""


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.",
            suffix=".tmp", delete=False, newline="",
        ) as stream:
            temporary = stream.name
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and os.path.exists(temporary):
            os.unlink(temporary)


def write_outputs(root: Path, payload: dict[str, Any], predictions: pd.DataFrame) -> None:
    """Write the only report, machine-readable metrics and ignored row predictions."""
    if list(predictions.columns) != [
        "source_row", "EmployeeNumber", "Attrition", "probability", "prediction",
        "threshold", "candidate_version",
    ]:
        raise ValueError("Prediction CSV schema differs from the authorized output.")
    for relative in (REPORT_PATH, METRICS_PATH, PREDICTIONS_PATH):
        if (root / relative).exists():
            raise ValueError(f"Refusing to overwrite final-evaluation output: {relative}")
    _atomic_text(root / REPORT_PATH, build_report(payload))
    _atomic_text(root / METRICS_PATH, json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    _atomic_text(root / PREDICTIONS_PATH, predictions.to_csv(index=False))
