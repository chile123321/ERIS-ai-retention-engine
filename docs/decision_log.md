# Decision log

## 0001 — Scaffold boundaries

- Status: accepted
- Decision: create interfaces and documentation only; do not perform EDA, training, tuning, SHAP analysis, or real prediction.
- Decision: preserve the supplied raw data and split, and keep the final test locked.
- Decision: package/import name remains `eris-attrition-model-api` / `eris_ml`; the repository display name is `eris-ai-retention-engine`.

## 0002 — Fairness audit input recovery before Step 10D

- Status: recovery completed on 2026-09-20; Step 10D fairness audit remains stopped.
- Incident: the mixed `fairness_audit_v1.csv` was loaded during schema inspection. It contained development and `final_test`-tagged rows.
- The locked `final_test_raw_v1.csv` was not opened. No final-test metrics were computed.
- No model, feature, calibration, or threshold decisions were changed.
- Recovery: `fairness_audit_development_v1.csv` was derived from approved development IDs and raw source attributes. It has 1,176 development rows, the expected target distribution, unique IDs, and no overlap with final-test IDs in the split manifest.
- The mixed file was moved intact to ignored `data/quarantine/fairness_audit_mixed_v1.csv`; it was not overwritten or deleted. No fairness metrics were run during recovery.

## 0003 — Step 10D resumed on development-only audit data

- Status: completed after recovery; the earlier stop and mixed-file incident in 0002 remain part of the record.
- Decision: use only `fairness_audit_development_v1.csv` and cross-fitted development predictions for subgroup audit; do not open the official final-test file or quarantined mixed file.
- Findings: Gender `INSUFFICIENT_EVIDENCE`; AgeGroup `REVIEW` (TPR gap 0.517, 95% bootstrap CI [0.330, 0.758]); MaritalStatus `REVIEW_EXPLORATORY`; intersectional `INSUFFICIENT_EVIDENCE` because Female × 50+ has 2 positive cases.
- No model, feature, calibration or threshold change followed the audit. The holdout status is `retained_with_protocol_deviation`.

## 0004 — Step 10E research-only candidate freeze

- Status: accepted with known limitations for academic capstone evaluation preparation; Step 10 development-stage work complete.
- Candidate: `eris-xgboost-v1`, XGBoost tuned raw, 25-feature v1 schema, frozen preprocessing specification, capacity-15 working threshold 0.345651.
- Fairness remains `review_known_limitation`, not PASS. The AgeGroup red flag must remain visible and must not be attributed solely to small subgroup size.
- The official locked final-test file was not opened and no final-test metric was computed. The mixed fairness file was loaded during earlier schema inspection before Step 10D; no final-test information was used to change model, features, calibration or threshold.
- `business_approved=false`, `production_approved=false`, `final_test_evaluation_authorized=false`. Production HR use and automated employment decisions are prohibited.
- The raw dataset SHA-256 was not found in approved existing evidence; the manifest marks this provenance field missing rather than fabricating it or reopening raw data. All 11 requested existing config/development/report artifacts were hashed.

## 0005 — Authorized one-time academic final-test evaluation

- Project-owner authorization was recorded at `2026-09-20T03:11:42+00:00` for `eris-xgboost-v1` only, scope `capstone_final_evaluation`, maximum one run; no business or production approval.
- Frozen candidate manifest SHA-256: `33f0dc97a6082c5a22e4ec038386374dfb8effad6078ff003c4cb13cd9f10481`. Preflight verified the 11 frozen artifact hashes, 1,176 development rows and 25-feature schema before access.
- Pre-registered guardrails: PR-AUC >= 0.45, ROC-AUC >= 0.75, Brier <= 0.135462, precision >= 0.45, recall >= 0.45, alert rate <= 0.20. They were not revised after access.
- The frozen pipeline was fitted on development before the one-time ledger transitioned to `run_count=1`, `started` at `2026-09-20T03:12:48+00:00`. The official final-test file was then opened once; its SHA-256 is `dda26134190cd4001276b96612c6020ea5ec5cd3150435d786c0e6740a4c2750`.
- Final evaluation completed at `2026-09-20T03:13:35+00:00` on 294 rows (47 positive cases) with result `PASS_WITH_KNOWN_LIMITATIONS`; all six point-estimate guardrails passed. The ledger is `completed` and must reject a second run.
- The holdout remains `retained_with_protocol_deviation`: the earlier mixed fairness file incident remains disclosed in entry 0002. No post-final tuning, feature/preprocessing change, calibration, threshold change, or candidate comparison is allowed. This academic result does not grant business or production approval.

## 0006 — Step 12A development-only research bundle

- The unchanged `eris-xgboost-v1` candidate was refitted on the 1,176 development rows only at `2026-09-20T03:44:48+00:00`, with its frozen 25-feature preprocessing/XGBoost pipeline, raw probability and threshold 0.345651. The bundle SHA-256 is `bb9c00307a36763f91de58d598caf952cca3f451e7928ca39d1a22b1763dac21`.
- Bundle load/prediction verification passed. Development-only Tree SHAP used a stratified 500-row sample, seed 42, raw log-odds output and aggregation to 25 original features; maximum global additivity error was `5.6733843e-06`.
- The model card retains the AgeGroup fairness red flag, mixed-fairness-file protocol deviation, final-test uncertainty and research-only scope. The official final-test CSV and quarantined mixed file were not accessed in Step 12A; the final-evaluation ledger remains `run_count=1`, `completed`.
- No post-final tuning, calibration, feature/preprocessing or threshold change was made. No API, business or production approval was created. Only trusted local joblib artifacts may be deserialized.

## 0007 — Step 12B local prediction API

- The existing `eris-xgboost-v1` bundle is loaded once at FastAPI startup after checksum and metadata validation. Local `/api/v1` single/batch prediction uses raw probabilities and frozen threshold 0.345651; no model or preprocessing fit occurs in the API.
- The 25-feature API contract is explicitly `dataset_derived`, `hr_business_approved=false`. Its numeric safety ranges and nominal categories were derived offline from development aggregates; serving does not read data CSV files. Unknown nominal categories receive a warning, and invalid inputs fail closed.
- Local Uvicorn smoke testing reached health, ready, model-info, single and batch endpoints successfully, then the server was stopped. Structured logs exclude employee payloads and identifiers. Swagger uses synthetic examples only.
- This is a localhost research prototype without authentication, rate limiting, production monitoring or business/production approval. The completed final-evaluation ledger was not changed or rerun.
