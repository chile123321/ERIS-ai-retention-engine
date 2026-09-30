# Local research API and backend integration contract — v1

This API serves only the trusted, checksummed `eris-xgboost-v1` bundle. It loads
once in FastAPI lifespan, never trains, and never reads raw, development or
final-test CSV at runtime. Business/production approval is **false**.

| Endpoint | Purpose |
|---|---|
| `GET /health` | Public liveness, independent of model readiness |
| `GET /ready` | Public minimal readiness; 200 only after valid auth configuration and verified bundle load, otherwise 503 |
| `GET /api/v1/model-info` | Service Bearer; allowlisted candidate metadata, no local paths |
| `POST /api/v1/predict` | Service Bearer; one record; top-five SHAP factors by default |
| `POST /api/v1/predict/batch` | Service Bearer; ordered all-or-nothing batch, max 100, or 20 with SHAP |
| `GET /metrics` | Service Bearer; process-local operational metrics without PII labels |

Swagger UI is at <http://127.0.0.1:8000/docs>. Choose the single-prediction
endpoint, **Try it out**, and paste one object from
[`manual_prediction_examples.json`](../tests/fixtures/manual_prediction_examples.json).
The three examples are synthetic, within the dataset-derived input contract,
and carry no expected Attrition label.
Select **Authorize** and supply the bare runtime service token first, without
typing the `Bearer ` prefix. Click **Authorize** in the dialog and then **Close**;
Swagger's generated Curl must contain `Authorization: Bearer ...`. A missing
header in generated Curl means the browser UI has not applied authorization to
that operation, regardless of `/ready`. The token must
be a unique 32+ character value in `ERIS_SERVICE_TOKEN`; the example placeholder
is deliberately invalid. Only use it in local Swagger for manual testing; never
embed it in frontend code or a deployed browser app.
Missing/invalid credentials return a generic 401 with `WWW-Authenticate: Bearer`.
The API is unready if no valid token is configured. `/health` remains minimal and
public. Do not place tokens in URL query parameters.

## Input and decision

Requests must contain exactly the 25 canonical feature fields shown in Swagger.
`record_id` is an optional response passthrough; it is not a model feature, logged
or persisted. Target, EmployeeNumber, source_row, Gender, MaritalStatus, and all
other extra fields are rejected. Missing features and invalid types return 422.

The schema follows `configs/features/feature_set_v1_full.yaml`, checked bundle
metadata, and `configs/data/data_contract_v1.yaml`. Numeric bounds and known
nominal categories are aggregate values observed on the IBM development set:
`contract_status=dataset_derived`, `hr_business_approved=false`. They are safety
bounds for this prototype, not validated HR business rules. Ordinal domains come
from checked bundle metadata. Null, NaN, infinite or out-of-range numeric values
are rejected. Unknown nominal categories are accepted by the frozen encoder and
produce a warning; the supplied category is not rewritten.

`probability` is uncalibrated attrition probability. `alert=true` means
`probability >= 0.345651`, nothing more; it is not certainty of resignation.
The threshold is a development capacity-15 working assumption, not a
business-approved operating policy. With `include_explanation=true`, `top_k`
may be 1–10. SHAP factors are signed **raw log-odds** contributions aggregated
to original features. They are not probability-point changes or causal evidence.
Set `include_explanation=false` to avoid SHAP computation.

Batch requests have `records` and optional `include_explanation` (default false).
Responses preserve record order. Oversized batches return 413; any invalid
record rejects the whole batch rather than returning partial results.
Each prediction includes `record_id` (passthrough only), `request_id`, raw
`probability`, `alert`, frozen `threshold`, `candidate_version`, `bundle_version`,
`feature_schema_version`, UTC `predicted_at_utc`, `warnings`, and optional
`top_factors`. Explanation space is `raw_log_odds`; SHAP is non-causal. New fields
are additive to the Step 12B response.

## Error, privacy and operational limits

Errors use `{ "error": { "code", "message", "details", "request_id" } }`.
Field-level details never echo full values or payloads. Codes include
`VALIDATION_ERROR`, `MODEL_NOT_READY`, `BUNDLE_INTEGRITY_ERROR`,
`PREDICTION_ERROR`, `EXPLANATION_ERROR`, and `BATCH_LIMIT_EXCEEDED`.
Startup checksum/metadata failure leaves the API unready. A checksum detects
accidental tampering but does not authenticate an untrusted joblib file;
deserializing joblib/pickle can execute code.

`X-Request-ID` accepts only a canonical UUID; missing/invalid IDs are replaced
with a new UUID. The response echoes the safe ID in its header, and prediction
responses include it in the body. Structured logs contain only request ID,
allowlisted endpoint, HTTP method/status, latency, candidate and bundle version.
They exclude employee inputs/IDs, Authorization, probabilities and SHAP values.
Disable server access logs if upstream URLs may contain sensitive query strings.
CORS is disabled by default; a configured origin list may not contain `*`.
Keep this prototype bound to `127.0.0.1`; service Bearer is not a replacement for
TLS, network policy, token rotation, rate limits or audit-retention policy.

`/metrics` exports Prometheus text with static endpoint/method/status labels only:
request totals, latency sum/count, prediction-request total, successful batch
record total, validation/authentication failure totals, alert total,
unknown-category warning total, bundle-load-failure total and readiness gauge.
Metrics are per process, reset on restart, and have no employee/request/feature/
probability/token labels. This is operational/input monitoring, **not** measured
performance or drift monitoring against post-deployment outcomes.

## Backend and persistence integration contract

This repository is the ML API only. It contains no React/Node web backend,
database schema/migrations or Supabase/PostgreSQL configuration. Therefore
`persistence_integration=BLOCKED_EXTERNAL_BACKEND_REQUIRED`. The ML API remains
stateless for employee profiles and does not create a substitute database.

The intended path is **frontend → authenticated web backend → canonical 25-feature
mapping → ML API with service credential → web backend → prediction database →
frontend**. The frontend must not call this ML API directly or possess the service
token. The backend must authorize HR/Admin, retrieve an employee snapshot, validate
the mapping/units/missingness against an approved HR contract, then call the API.
The backend owns idempotency, persistence, history authorization and user audit.

Proposed minimum prediction record: `prediction_id`, internal `employee_id`,
`snapshot_id` or `snapshot_date`, `request_id`, `idempotency_key`,
`candidate_version`, `bundle_version`, `feature_schema_version`, `probability`,
`threshold`, `alert`, `predicted_at`, `created_by`, `input_hash`, `warnings`.
The backend should define a unique idempotency constraint scoped to its tenant,
employee snapshot and model version, then return the existing result on retry.
Do not store the service token or, by default, all 25 feature values. If approved
to persist explanations, use child records with `prediction_id`, `rank`,
`feature_name`, optional policy-permitted `feature_value`, `shap_value`,
`direction`, `explanation_space`. Commit prediction plus explanation rows in one
database transaction; rollback both on any failure. Restrict history reads by
backend authorization. The API cannot assert these guarantees without that code.

To finish real integration, provide the web-backend repository, employee/snapshot
schema and migrations, authentication/authorization model, approved source-to-25-
feature mapping (including units and missingness), database configuration,
idempotency scope and retention policy. Backend tests must cover retry/no duplicate,
transaction rollback, authorization and redaction.

The final-test evaluation ledger remains completed and cannot be rerun through
this API. The AgeGroup fairness red flag and documented holdout protocol
deviation remain in [`model_card_v1.md`](model_card_v1.md).
