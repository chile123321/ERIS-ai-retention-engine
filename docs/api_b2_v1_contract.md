# ERIS Random Forest B2 Model API v1

Status: local research service; decision support only; not production-approved.

## Locked model

- Model ID: `rf-v3-final15-unweighted-b2`
- Bundle: `artifacts/models/v3_final_15/b2/`
- Feature schema: `v3_final_15`
- `pipeline.joblib` SHA-256:
  `0c72841a8505a910d31c2a687fbcbf5259e3afb60f9eddcb7586801519b5628f`
- Output: raw probability for class `Attrition=1`
- No threshold, risk band, `will_leave`, automatic action or model fallback

## Endpoints

| Method | Path | Auth | Purpose |
| --- | --- | --- | --- |
| GET | `/health` | No | Process liveness |
| GET | `/ready` | No | Bundle/key readiness |
| GET | `/v1/model/info` | Yes | Locked model metadata |
| POST | `/v1/predict` | Yes | One probability |
| POST | `/v1/predict/batch` | Yes | 1–100 ordered predictions |
| POST | `/v1/explain` | Yes | One probability-space SHAP explanation |

Protected endpoints require `X-ERIS-Service-Key`. The key is supplied only through
`ERIS_SERVICE_API_KEY`; an empty key keeps protected endpoints unavailable. The full
machine-readable contract is [model-api-v1.openapi.json](model-api-v1.openapi.json).

## Input

Exact feature order:

1. `Age`
2. `Department`
3. `DistanceFromHome`
4. `Education`
5. `EnvironmentSatisfaction`
6. `JobLevel`
7. `JobSatisfaction`
8. `MonthlyIncome`
9. `OverTime`
10. `TotalWorkingYears`
11. `TrainingTimesLastYear` (optional)
12. `YearsAtCompany`
13. `YearsInCurrentRole`
14. `YearsSinceLastPromotion`
15. `YearsWithCurrManager` (optional)

`Department` accepts `Human Resources`, `Research & Development`, or `Sales`.
`OverTime` accepts `No` or `Yes`. Numeric fields are strict non-negative integers;
survey/code ranges are enforced by the Pydantic schema. Unknown fields, identifiers,
PII, coercion and invalid categories are rejected. Omitted/null optional values are
removed before inference and imputed only by the frozen pipeline.

## Responses and safety

- Prediction responses include `estimated_probability`, model metadata and the names
  of optional features imputed by the bundle.
- Batch preserves input/reference order and is atomic at validation time.
- Explain returns all 15 contributions in probability units, `base_value`, top five
  factors and additivity error. SHAP describes model behavior, not causality.
- Errors use stable, redacted codes; request bodies, feature values, filesystem paths,
  secrets and tracebacks are not logged or echoed.
- Bundle is loaded once during FastAPI lifespan and its model ID/manifest/hash are
  verified. Load failure leaves `/health` alive while `/ready` and `/v1/*` return 503.

## Runtime configuration

| Variable | Meaning | Default |
| --- | --- | --- |
| `ERIS_BUNDLE_PATH` | B2 bundle directory | repository B2 path |
| `ERIS_EXPECTED_MODEL_SHA256` | pinned pipeline hash | locked B2 hash |
| `ERIS_SERVICE_API_KEY` | service-to-service key | empty/unready |
| `ERIS_MAX_BATCH_SIZE` | batch cap | `100` |
| `ERIS_LOG_LEVEL` | log level | `INFO` |

The service does not read datasets, train, calibrate or evaluate a model at startup.
