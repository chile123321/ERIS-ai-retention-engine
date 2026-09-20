# Model card — eris-xgboost-v1 (bundle-v1)

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
- Development split: `existing-v1`; training file SHA-256 `5cb4c6a3226ab5e16b355af64163e5857aac0e1637eb2fb88f91fcc9f1989754`.
- Candidate manifest SHA-256 `33f0dc97a6082c5a22e4ec038386374dfb8effad6078ff003c4cb13cd9f10481`.
- This benchmark is not representative of Vietnamese enterprises. It lacks a temporal
  observation design and cannot establish true 3–6 month prospective prediction.

## Training

- Preprocessing and model are serialized together in one fitted sklearn Pipeline.
- Nominal: most-frequent imputation, one-hot encoding with unknown categories ignored.
- Ordinal/numeric: median imputation and StandardScaler.
- XGBoost configuration (frozen `configs/models/xgboost_tuned_v1.yaml`):
- `objective`: `binary:logistic`
- `eval_metric`: `logloss`
- `tree_method`: `hist`
- `scale_pos_weight`: `1.0`
- `n_jobs`: `1`
- `subsample`: `0.6`
- `reg_lambda`: `10.0`
- `reg_alpha`: `0.5`
- `n_estimators`: `500`
- `min_child_weight`: `3`
- `max_depth`: `1`
- `learning_rate`: `0.1`
- `gamma`: `0.5`
- `colsample_bytree`: `0.6`
- Seed 42; build timestamp UTC `2026-09-20T03:44:48+00:00`.
- Build environment: python 3.12.10, pandas 3.0.6, numpy 2.5.3, scikit-learn 1.9.1, xgboost 3.4.1, shap 0.52.0, joblib 1.6.0, matplotlib 3.11.2.
- Ordinal input domains in the bundle are values observed on development, **not** an
  authoritative HR data contract. Missing values follow frozen imputation rules.

## Evaluation, calibration and threshold

- Development OOF: PR-AUC **0.666912**, ROC-AUC **0.848004**, Brier **0.088031**.
- Final test (one authorized evaluation): PR-AUC **0.607967**
  (95% CI 0.487827–0.739528); ROC-AUC **0.834697**
  (CI 0.756824–0.904563); Brier **0.094169**
  (CI 0.078117–0.109943).
- Final operational metrics at frozen threshold 0.345651: precision **0.605263**
  (CI 0.473684–0.750000); recall **0.489362**
  (CI 0.340426–0.638298); F1 **0.541176**
  (CI 0.413774–0.666667); F2 **0.508850**
  (CI 0.369543–0.643794); alert rate **0.129252**
  (CI 0.098639–0.163265).
- Confusion matrix: TN 232, FP 15, FN 24, TP 23.
  Only **47** final-test positive cases; intervals are broad, especially for recall.
- Calibration decision: **none/raw**. Threshold policy: development capacity-15 working
  assumption, `0.345651`; it is not a business-approved operating threshold.
- Final-test guardrail result: `PASS_WITH_KNOWN_LIMITATIONS`. This does not grant deployment approval.

## Limitations, fairness and monitoring

- Development fairness: Gender **INSUFFICIENT_EVIDENCE**; AgeGroup **REVIEW**, a red flag
  (TPR gap 0.517); MaritalStatus **REVIEW_EXPLORATORY**. No fairness or production approval.
- Holdout status: **retained_with_protocol_deviation**. The official locked final-test file was not opened before the authorized evaluation and no final-test metric was previously computed. A mixed fairness file containing final-test rows was inadvertently loaded during schema inspection. The workflow stopped before analysis, the incident was documented, and no final-test information was used to change the model, features, preprocessing, calibration or threshold.
- Raw dataset SHA-256 provenance gap remains documented in the frozen manifest.
- Generalization declined from development to final test; final recall is about **49%**, leaving
  24 missed attrition cases out of 47. No causal interpretation is supported.
- Any later deployment would require authorized monitoring for drift, missingness, subgroup
  performance and access control. No such production monitoring is approved here.

## Explainability statement

- Tree SHAP is computed on development only in **raw log-odds**, not probability points.
- One-hot columns are aggregated back to 25 original features. Global values are mean
  absolute contributions; local values are signed contributions.
- SHAP describes how the model’s prediction changes relative to its baseline. It does not establish causality and must not be interpreted as proof that changing a feature will prevent employee attrition.

## Ethical and security use

- Human review is required. Do not use the score as grounds for employee termination.
- Restrict visibility of risk scores; implement access control and logging before any API use.
- Joblib/pickle deserialization can execute code: load this bundle only from a trusted local
  source. Its checksums detect accidental modification, not malicious replacement.
- Final-test results must not be used to revise this candidate. A new model version requires
  a newly defined evaluation dataset.
