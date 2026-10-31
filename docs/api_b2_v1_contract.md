# ERIS B2 Model API v1 — API-1 contract

Status: implemented locally. The API-1 sections below retain the original B2 contract.
The additive API 1.1 extension below enables AdaBoost without changing B2 response meaning.

## API 1.1: optional AdaBoost selection (2026-10-28)

- `POST /v1/predict`, `/v1/predict/batch`, `/v1/explain` accept a top-level
  `model_id`. Omitted means `rf-v3-final15-unweighted-b2`; explicitly select
  `adaboost-v3-final15-baseline-sigmoid-s1` for calibrated AdaBoost. One model per batch.
  Explicit null is invalid. Unknown string: 422 `MODEL_UNKNOWN`. No fallback.
- `GET /v1/models` lists both locked IDs, checksums, schema `v3_final_15`, readiness
  and explain capability. It requires the existing `X-ERIS-Service-Key` header.
  `/v1/model/info` and `/ready` retain their B2-only meaning. A failed AdaBoost load
  does not disable B2, or vice versa; check the selected model in `/v1/models`.
- Input rules and two optional fields are unchanged. AdaBoost delegates validation
  to the existing codec and missing-value handling to the frozen fitted pipeline.
- B2 metadata and response fields are unchanged. AdaBoost uses the same response
  structure with its own locked ID and SHA-256
  `af7dab8f6044c9a902789fd8b69aac16c4e187157508780c54cd1b39b1a491ca`.
  Model IDs include the candidate version. Metadata remains research-only.
- Predict never calls SHAP. Explain uses the request's model and 15 values;
  it is not linked to a previous HTTP prediction or an employee history.
  s1 uses the existing full calibrated-pipeline PermutationExplainer, 64 cycles,
  seed 42, stored background. All 15 `shap_value` contributions are in probability
  units (`explanation_space=probability`); multiply by 100 for percentage points.
  `base_value + sum(shap_value)` matches probability within `1e-8` for s1.
  SHAP describes model behavior, not causes. No thresholds or risk categories.
- Each process loads each bundle once, verifies the pinned model hash and manifest,
  and reuses the s1 masker/model adapter. There is exactly one SHAP worker per process
  for both models, default two waiting slots (`ERIS_EXPLAIN_QUEUE_SIZE`). All explain
  work is serialized, including mutable masker and RNG use. Predict uses the normal
  HTTP thread pool without the explain lock. Shared CPU/GIL/native thread resources
  can still affect latency; this is not hard process isolation or a throughput SLA.
- Full queue: 429 `EXPLANATION_BUSY`. Default total queue-plus-computation wait is
  60 seconds (`ERIS_EXPLAIN_TIMEOUT_SECONDS`), then 504 `EXPLANATION_TIMEOUT`.
  Timeout/disconnection marks waiting work to be skipped. Its slot remains occupied
  until dequeued, bounding retained payloads even during repeated timeouts. Running
  computation is NOT killed and retains its slot until completion; shutdown waits for it.
  Internal explanation failures return redacted 500 `EXPLANATION_ERROR`.
- Authentication is checked before dispatch. Invalid bodies may return 422; a missing
  default B2 still returns its 503 for legacy requests. Bundle paths, input values and
  internal exceptions are not echoed in errors. No employee payload logging is added.
- Bundle path override: `ERIS_ADABOOST_BUNDLE_PATH`; only the pinned s1 binary and
  loader-compatible library versions are accepted. No training at startup.

Run commands and synthetic Postman examples: [postman/README.md](../postman/README.md).
The exported `model-api-v1.openapi.json` is now API 1.1.0; the URL prefix stays `/v1`.

## Frozen candidate and provenance

- Bundle: `artifacts/models/v3_final_15/b2/`.
- Model ID: `rf-v3-final15-unweighted-b2`; feature schema: `v3_final_15`.
- Model SHA-256: `0c72841a8505a910d31c2a687fbcbf5259e3afb60f9eddcb7586801519b5628f`.
- Raw Random Forest probability for class `1` (`Attrition=Yes`), not B1.
- API metadata: `model_status=research_candidate`, `decision_support_only=true`,
  `independent_validation=false`. The API status is a presentation label for the
  bundle's existing `research_demo_candidate` status; it does not alter the bundle.
- Baseline Git branch: `main`; HEAD: `a7a5447f4a53cb67275c0d1aaf8ce0619d9c7785`.
  Existing modified README/typing test and untracked B2/manual/audit files were
  preserved. A read-only pre-edit snapshot covers 269 existing files, including
  the B2 bundle and `audit_b2_1` outputs; no final-holdout file was read or hashed.

Contract sources (SHA-256 recorded before editing):

| Source | SHA-256 |
| --- | --- |
| `artifacts/models/v3_final_15/b2/manifest.json` | `d71f8729e55d69299c62180a5a07aae7a53147d33ee88a168f6637d7229758df` |
| `artifacts/models/v3_final_15/b2/metadata.json` | `ac674aed2b1d337f2650383dd068226be04d9aecea7cd34831de22632cd3a9ba` |
| `src/eris_ml/models/bundle_v3_b2.py` | `faafff197e6c9e13ca101775fe90aa4e185d798ede17f859cdbbb9375bd91a06` |
| `src/eris_ml/features/schema_v3.py` | `9972361bc309d80bcb1956fbb808834bf05b7bfec6d295747c8dca951f5e8037` |
| `configs/features/feature_set_v3_final_15.yaml` | `9ce2a513e692bee90d4ae97e17d6c7e176f301f249cd8bc4109a647a9134e7e5` |
| `pyproject.toml` | `5b883b9318b212f0a6cf16883e5a57d44eab99dad10735202a28b8beb7099881` |

These sources and the existing pre-API audit are evidence for this contract, not
authorization to change model behavior or claims of independent validation.

## Input contract

`EmployeeFeatures` declares these fields in this exact order. JSON object key
order is not required of callers; downstream inference must use schema order.

| # | Feature | Type / accepted values | Required |
| --- | --- | --- | --- |
| 1 | Age | strict integer, 0–2147483647 | Yes |
| 2 | Department | `Human Resources`, `Research & Development`, `Sales` | Yes |
| 3 | DistanceFromHome | strict integer, 0–2147483647 | Yes |
| 4 | Education | strict integer literal 1–5 | Yes |
| 5 | EnvironmentSatisfaction | strict integer literal 1–4 | Yes |
| 6 | JobLevel | strict integer literal 1–5 | Yes |
| 7 | JobSatisfaction | strict integer literal 1–4 | Yes |
| 8 | MonthlyIncome | strict integer, 0–2147483647 | Yes |
| 9 | OverTime | `No`, `Yes` | Yes |
| 10 | TotalWorkingYears | strict integer, 0–2147483647 | Yes |
| 11 | TrainingTimesLastYear | strict integer, 0–2147483647, or null | No |
| 12 | YearsAtCompany | strict integer, 0–2147483647 | Yes |
| 13 | YearsInCurrentRole | strict integer, 0–2147483647 | Yes |
| 14 | YearsSinceLastPromotion | strict integer, 0–2147483647 | Yes |
| 15 | YearsWithCurrManager | strict integer, 0–2147483647, or null | No |

All integer inputs reject floats (including `41.0`), fractional numbers, numeric
strings, booleans, negatives, NaN/infinity and overflow-sized integers.
`2147483647` is a **technical guardrail**, not a business maximum. No new business
upper bound or cross-field business rule is enforced in API-1.
Categories are exact, case-sensitive values; no trimming or category mapping.

Only the two optional fields may be absent or null. Schema validation preserves
missingness; it never replaces it with zero or imputes it. During later inference,
the unchanged B2 transformer uses its fitted median `3.0` for each optional field.
The future adapter must report which original fields were imputed, not infer
missingness from values equal to 3. Required fields cannot be null.

All models use `extra="forbid"`. There are no name/email/phone, target or employee
ID feature fields. `reference_id` is an opaque client token of 1–128 characters;
clients must not put names, emails or phone numbers into this token. The schema
cannot prove an arbitrary string is not PII. No input payload should be logged.

## Request and response models

| Model | Fields / constraints |
| --- | --- |
| `PredictRequest` | Optional nullable UUID `request_id`; required `reference_id`, `features: EmployeeFeatures` |
| `BatchItem` | Required `reference_id`, `features` |
| `BatchPredictRequest` | Optional nullable UUID `request_id`; `items`: 1–100 items; unique `reference_id` within batch |
| `ExplainRequest` | Same shape as `PredictRequest` |
| `ModelMetadata` | Locked model ID, SHA-256, feature schema version and status/boolean constants above |
| `PredictionResult` | `reference_id`, finite `estimated_probability` in [0,1], unique `imputed_features` drawn only from the two optional names, `metadata` |
| `PredictResponse` | Required UUID `request_id`, `result: PredictionResult` |
| `BatchPredictResponse` | Required UUID `request_id`, 1–100 `results` with unique references; same order as request |
| `ShapContribution` | `feature`, `feature_value` (number/string/null), finite `shap_value`, `direction`: increase/decrease/neutral |
| `ExplainResponse` | UUID `request_id`, `reference_id`, probability, probability-space `base_value`, 15 `contributions`, at most 5 `top_factors`, finite nonnegative `additivity_error`, `explanation_space=probability`, warning, metadata |
| `ErrorResponse` | Optional UUID `request_id`, allowlisted `code`, fixed safe `message`, optional structured `details` |

Schemas do not generate request IDs. A later service must preserve a valid supplied
UUID, or generate one when absent/null, and propagate it to responses. Response
list order is preserved by the schema; request-to-response correspondence must be
enforced and tested in the future service. Validation is atomic: if any batch
item is invalid, reject the entire request; no partial inference or success list.

SHAP contributions cover every feature exactly once in schema order. Direction
matches the contribution sign (zero means neutral). Top factors are the prefix of
contributions sorted by descending absolute SHAP value, with schema order as the
tie-breaker. `feature_value` describes the caller's original value; omitted/null
optional values remain null in explanations, even though the model imputes them.
The future service must map the existing B2 explanation into this shape, retaining
its validated probability-space additivity check; API-1 computes no SHAP values.
The mandatory warning is:

> SHAP explains model behavior; it does not establish causality.

Synthetic request example (not an employee record or a prediction):

```json
{
  "reference_id": "demo-token-01",
  "features": {
    "Age": 35,
    "Department": "Research & Development",
    "DistanceFromHome": 5,
    "Education": 3,
    "EnvironmentSatisfaction": 3,
    "JobLevel": 2,
    "JobSatisfaction": 3,
    "MonthlyIncome": 5000,
    "OverTime": "No",
    "TotalWorkingYears": 10,
    "TrainingTimesLastYear": null,
    "YearsAtCompany": 5,
    "YearsInCurrentRole": 3,
    "YearsSinceLastPromotion": 1,
    "YearsWithCurrManager": 3
  }
}
```

## Endpoint contract — not implemented

| Endpoint | Authentication | Intended semantics |
| --- | --- | --- |
| `GET /health` | None | 200 if process is alive; not a bundle-readiness assertion |
| `GET /ready` | None | 200 only when model is ready and integrity verified; otherwise 503 |
| `GET /v1/model/info` | Service key | 200 with locked `ModelMetadata` when ready |
| `POST /v1/predict` | Service key | `PredictRequest` → `PredictResponse` |
| `POST /v1/predict/batch` | Service key | `BatchPredictRequest` → `BatchPredictResponse`, atomic validation |
| `POST /v1/explain` | Service key | `ExplainRequest` → `ExplainResponse`; SHAP only on explicit request |

All `/v1/*` require `X-ERIS-Service-Key`. Authentication and health/readiness
handlers are future work, not provided by these schemas. Node/NestJS is the
intended client; the frontend must not call Model API directly. Model API has no
database connection. Secrets must not appear in responses, examples or logs.

## HTTP and safe error contract

| HTTP | Code | Fixed safe message |
| --- | --- | --- |
| 200 | — | Success |
| 401 | `AUTH_MISSING` | Service key is required. |
| 401 | `AUTH_INVALID` | Service key is invalid. |
| 422 | `VALIDATION_ERROR` | Request validation failed. |
| 422 | `BATCH_TOO_LARGE` | Batch exceeds 100 items. |
| 503 | `MODEL_NOT_READY` | Model is not ready. |
| 503 | `BUNDLE_INTEGRITY_ERROR` | Model bundle integrity check failed. |
| 500 | `INFERENCE_ERROR` | Prediction could not be completed. |
| 500 | `EXPLANATION_ERROR` | Explanation could not be completed. |

Empty batches, duplicate references, invalid UUIDs and invalid individual items
are 422 `VALIDATION_ERROR`; more than 100 items is 422 `BATCH_TOO_LARGE`.
HTTP mappings are obligations for API-2 and later, not behavior of Pydantic itself.

`details`, when present, contains 1–100 `ErrorDetail` records with only an
allowlisted field, an optional zero-based item index (0–99), and an allowlisted
issue code. Issues: `required`, `invalid_type`, `out_of_range`, `invalid_value`,
`extra_fields`, `duplicate_reference_id`, `empty_batch`, `batch_too_large`.
Unknown submitted field names must be represented by a safe container location
(for example `features`), not echoed back. No submitted values, traceback,
exception text or full input are allowed. The future handler must not expose
raw `ValidationError.errors()` output, which can contain input/context despite
`hide_input_in_errors=True`. Convert errors to this safe allowlist instead.

## Limits and decisions still open

Output contains no `will_leave`, threshold, High/Medium/Low label or automated HR
decision. `estimated_probability` is a raw model estimate, not an independently
validated probability at a new company. IBM evidence does not establish a 3–6
month forecast horizon, intervention benefit or fairness at a new employer.
No final-holdout evaluation was performed or reused to certify this API contract.

Still open: business upper bounds; MonthlyIncome unit/currency; DistanceFromHome
unit; Department mapping for target companies; cross-field business validation.
The technical integer guardrail does not resolve those business/data decisions.

## API-1 checks

Use the existing `.venv` from repository root (Python 3.12.10, Pydantic 2.13.5).
Tests use synthetic dictionaries, never load B2 and never fit a model.

```powershell
.\.venv\Scripts\python.exe -B -m pytest tests/unit/test_api_b2_schemas.py -q -p no:cacheprovider
.\.venv\Scripts\python.exe -B -m ruff check --no-cache src/eris_ml/api_b2 tests/unit/test_api_b2_schemas.py
.\.venv\Scripts\python.exe -B -m mypy --strict --no-incremental --cache-dir=nul src/eris_ml/api_b2
git diff --check
```

The lowercase Windows device name `nul` avoids cache writes. Mypy 2.3.1's
case-sensitive device check does not recognize `NUL` and attempts to create
`NUL/3.11`, causing an internal error; no dependency or typing rule is changed
to work around this invocation issue. Python 3.11 runtime is not verified here.
Stop after API-1; no app/routes/service, OpenAPI generation, Docker or Postman.

Verified in this API-1 run: **238 tests passed**, Ruff passed, strict Mypy passed
for 2 source files, and `git diff --check` passed. Git emitted only existing-file
LF/CRLF notices for `README.md` and `tests/unit/test_typing_boundary.py`; neither
was edited. All 269 protected file hashes matched the pre-edit snapshot,
including the expected B2 model hash. Only the four authorized API-1 files were
added; no dependencies installed, model loaded/fitted, full suite run, final
holdout accessed, commit or push performed. Contract/schema status:
**READY_FOR_API_2**, without implementing API-2 in this step.
