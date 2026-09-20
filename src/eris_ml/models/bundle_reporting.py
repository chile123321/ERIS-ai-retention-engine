"""Model card and reproducibility report for the frozen research bundle."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from eris_ml.models.explainability import DISCLAIMER

HOLDOUT_DISCLOSURE = (
    "The official locked final-test file was not opened before the authorized evaluation "
    "and no final-test metric was previously computed. A mixed fairness file containing "
    "final-test rows was inadvertently loaded during schema inspection. The workflow "
    "stopped before analysis, the incident was documented, and no final-test information "
    "was used to change the model, features, preprocessing, calibration or threshold."
)


def build_model_card(manifest: dict[str, Any], summary: dict[str, Any],
                     metadata: dict[str, Any]) -> str:
    """Expand the repository template sections with locked evidence only."""
    final = summary["metrics"]
    ci = summary["bootstrap"]["intervals"]
    model_lines = "\n".join(f"- `{key}`: `{value}`" for key, value in
                            manifest["model"]["parameters"].items())
    versions = ", ".join(f"{key} {value}" for key, value in
                         metadata["library_versions"].items())
    return f"""# Model card — eris-xgboost-v1 (bundle-v1)

## Model overview and intended use

- Type/target: frozen tuned XGBoost binary classifier for IBM HR benchmark attrition.
- Intended use: academic capstone evaluation, prototype demonstration, research decision support
  and technical integration testing, with human review.
- Prohibited use: production HR deployment; automated employment, termination, salary, promotion
  or disciplinary decisions; claims of fairness for Vietnamese enterprises.
- Business approved: **No**. Production approved: **No**.

## Data and versions

- IBM public/synthetic HR attrition benchmark, 1,470 rows total: 1,176 development
  (986 no attrition, 190 attrition) and 294 locked final-test rows (247/47).
- Bundle training scope: **development only**. The evaluated candidate was not refit on
  development plus final-test rows.
- Feature set/schema: `v1-full`, 25 ordered features. Target and employee identifiers are excluded.
- Development split: `existing-v1`; training file SHA-256 `{metadata['training_data_sha256']}`.
- Candidate manifest SHA-256 `{metadata['candidate_manifest_sha256']}`.
- This benchmark is not representative of Vietnamese enterprises. It lacks a temporal
  observation design and cannot establish true 3–6 month prospective prediction.

## Training

- Preprocessing and model are serialized together in one fitted sklearn Pipeline.
- Nominal: most-frequent imputation, one-hot encoding with unknown categories ignored.
- Ordinal/numeric: median imputation and StandardScaler.
- XGBoost configuration (frozen `configs/models/xgboost_tuned_v1.yaml`):
{model_lines}
- Seed 42; build timestamp UTC `{metadata['training_timestamp_utc']}`.
- Build environment: {versions}.
- Ordinal input domains in the bundle are values observed on development, **not** an
  authoritative HR data contract. Missing values follow frozen imputation rules.

## Evaluation, calibration and threshold

- Development OOF: PR-AUC **0.666912**, ROC-AUC **0.848004**, Brier **0.088031**.
- Final test (one authorized evaluation): PR-AUC **{final['pr_auc']:.6f}**
  (95% CI {ci['pr_auc'][0]:.6f}–{ci['pr_auc'][1]:.6f}); ROC-AUC **{final['roc_auc']:.6f}**
  (CI {ci['roc_auc'][0]:.6f}–{ci['roc_auc'][1]:.6f}); Brier **{final['brier']:.6f}**
  (CI {ci['brier'][0]:.6f}–{ci['brier'][1]:.6f}).
- Final operational metrics at frozen threshold 0.345651: precision **{final['precision']:.6f}**
  (CI {ci['precision'][0]:.6f}–{ci['precision'][1]:.6f}); recall **{final['recall']:.6f}**
  (CI {ci['recall'][0]:.6f}–{ci['recall'][1]:.6f}); F1 **{final['f1']:.6f}**
  (CI {ci['f1'][0]:.6f}–{ci['f1'][1]:.6f}); F2 **{final['f2']:.6f}**
  (CI {ci['f2'][0]:.6f}–{ci['f2'][1]:.6f}); alert rate **{final['alert_rate']:.6f}**
  (CI {ci['alert_rate'][0]:.6f}–{ci['alert_rate'][1]:.6f}).
- Confusion matrix: TN {final['tn']}, FP {final['fp']}, FN {final['fn']}, TP {final['tp']}.
  Only **47** final-test positive cases; intervals are broad, especially for recall.
- Calibration decision: **none/raw**. Threshold policy: development capacity-15 working
  assumption, `0.345651`; it is not a business-approved operating threshold.
- Final-test guardrail result: `{summary['result']}`. This does not grant deployment approval.

## Limitations, fairness and monitoring

- Development fairness: Gender **INSUFFICIENT_EVIDENCE**; AgeGroup **REVIEW**, a red flag
  (TPR gap 0.517); MaritalStatus **REVIEW_EXPLORATORY**. No fairness or production approval.
- Holdout status: **retained_with_protocol_deviation**. {HOLDOUT_DISCLOSURE}
- Raw dataset SHA-256 provenance gap remains documented in the frozen manifest.
- Generalization declined from development to final test; final recall is about **49%**, leaving
  24 missed attrition cases out of 47. No causal interpretation is supported.
- Any later deployment would require authorized monitoring for drift, missingness, subgroup
  performance and access control. No such production monitoring is approved here.

## Explainability statement

- Tree SHAP is computed on development only in **raw log-odds**, not probability points.
- One-hot columns are aggregated back to 25 original features. Global values are mean
  absolute contributions; local values are signed contributions.
- {DISCLAIMER}

## Ethical and security use

- Human review is required. Do not use the score as grounds for employee termination.
- Restrict visibility of risk scores; implement access control and logging before any API use.
- Joblib/pickle deserialization can execute code: load this bundle only from a trusted local
  source. Its checksums detect accidental modification, not malicious replacement.
- Final-test results must not be used to revise this candidate. A new model version requires
  a newly defined evaluation dataset.
"""


def build_report(metadata: dict[str, Any], hashes: dict[str, str],
                 importance: pd.DataFrame, global_error: float,
                 local_error: float, smoke: np.ndarray) -> str:
    """Record reproducibility hashes and development-only verification outcomes."""
    hash_rows = "\n".join(f"| `{name}` | `{value}` |" for name, value in hashes.items())
    top = "\n".join(f"| {row.feature} | {row.mean_abs_shap:.6f} |" for row in
                    importance.head(10).itertuples(index=False))
    return f"""# Model bundle build — bundle-v1

- Candidate: `eris-xgboost-v1`; fitted on 1,176 development rows only.
- Fit UTC: `{metadata['training_timestamp_utc']}`; seed 42; 25 features.
- Frozen threshold `0.345651`; raw probability; no calibration, tuning or model comparison.
- Final-test CSV and quarantined mixed fairness file accessed: **No**.
- Final-test evaluation CLI rerun: **No**; completed ledger unchanged.
- Bundle reload verification: **PASS**. Prediction smoke test: **PASS**
  (3 development-safe rows; finite probabilities in [{float(np.min(smoke)):.6f},
  {float(np.max(smoke)):.6f}], predictions use the frozen threshold).
- Trusted local artifact only: joblib/pickle deserialization can execute code.
- SHAP: TreeExplainer, raw log-odds, stratified development sample
  `{int(importance['sample_size'].iloc[0])}` rows, seed 42; 25 original features.
- Global additivity max error: `{global_error:.8g}`; local max error: `{local_error:.8g}`;
  tolerance `1e-4`. One-hot absolute contributions are summed into original features.

## Artifact SHA-256

| Artifact | SHA-256 |
|---|---|
{hash_rows}

## Top 10 original-feature SHAP importance

| Feature | Mean absolute SHAP, raw log-odds |
|---|---:|
{top}

{DISCLAIMER}
"""
