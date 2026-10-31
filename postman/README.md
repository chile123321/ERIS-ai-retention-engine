# ERIS Model API v1 — local Docker/Postman handoff

## AdaBoost s1 extension (2026-10-28)

API 1.1 keeps existing B2 requests/responses and adds optional `model_id` selection.
The existing collection now includes **AdaBoost s1**: list models, predict, explain.
Existing B2 examples remain unchanged. Import the updated collection JSON explicitly
into Postman; previously imported/workspace copies are not updated automatically.

Local PowerShell from the repository root (existing `.venv`, no installation):

```powershell
$env:ERIS_SERVICE_API_KEY = & .\.venv\Scripts\python.exe -B -c "import secrets; print(secrets.token_urlsafe(40))"
.\.venv\Scripts\python.exe -m uvicorn eris_ml.api_b2.main:app --host 127.0.0.1 --port 8000 --workers 1 --no-access-log
```

Use the same key in your client shell or Postman's local-only `service_key` variable.
Never export a populated key. `GET /v1/models` uses `X-ERIS-Service-Key` and lists
per-model readiness. In the existing predict/explain JSON add:

```json
"model_id": "adaboost-v3-final15-baseline-sigmoid-s1"
```

Omit it for B2. Predict does not calculate SHAP. Explain returns 15 contributions in
probability units, not percentages. Default limits per process: one active explain,
two waiting, 60-second total wait. Handle 429 by retrying later; a 504 does **not**
mean running SHAP has stopped, so avoid immediate duplicate requests. Do not increase
Uvicorn workers without accounting for multiplied memory and explanation capacity.

Both frozen bundles are explicitly allowed by `.dockerignore` and copied by the
Dockerfile. Compose remains loopback-only. Docker verification completed on 2026-10-29:
both models ready, legacy B2 request preserved, AdaBoost prediction identical to local
(0.04492076225919636; tolerance 1e-12), batches select the correct model and preserve
order/optional missingness, and SHAP returns 15 features with additivity error 2.78e-17.
Invalid key/model/category return 401/422/422 with the documented error codes.
Image: `sha256:f0abeba28a82807016afe7c0a93a1caa38fe0f21d396151a18e662d44a550931`.
Python 3.12.10, runtime dependency pins and `pip check` passed; all nine bundle hashes
matched the host. No datasets/prediction artifacts were packaged. Container logs did
not contain the temporary key, synthetic reference or rejected category. Docker health
was healthy. Test project `eris-adaboost-docker-verify` was removed with `down`;
the image/build cache remain local. No packaging/runtime fixes were needed.
Illustrative Docker HTTP timings: predict 30 ms; first/warm SHAP 9.31/2.48 s.
This synthetic local verification is not production approval or a throughput SLA.

Measured locally on Windows 11/Python 3.12.10, FastAPI 0.141.1, SHAP 0.50.0,
sklearn 1.9.1, NumPy 2.4.6, one synthetic profile and 64 SHAP cycles:
TestClient predict warm mean 46 ms, first/warm explain 12.76/3.23 s. A separate
Uvicorn loopback HTTP smoke test measured startup 11.04 s, first/warm-mean predict
78/66 ms, first/warm explain 19.48/3.92 s. AdaBoost returned 0.04492076225919636;
additivity error was 2.78e-17. The B2 legacy request also succeeded. These are local
illustrations under the current machine workload, not latency guarantees.
Both bundles' nine component/manifest hashes were unchanged. Compose config validation
and subsequent Docker build/runtime checks passed (see above).
Strict Mypy for the API passes; whole-src Mypy still reports the existing missing
`streamlit` dependency for the unrelated V2 UI. No environment changes were made.
No development/final-test rows were used.

Status (2026-10-18): **MODEL_API_V1_READY_FOR_INTEGRATION**, not production ready.
B2 remains a research candidate: decision support only, no independent/external
validation, no automated HR decisions. No backend, database or frontend is connected.

## Start and stop from the repository root (PowerShell)

Generate a fresh local key in process memory, without printing it or writing a file:

```powershell
$env:ERIS_SERVICE_API_KEY = & .\.venv\Scripts\python.exe -B -c "import secrets; print(secrets.token_urlsafe(40))"
docker compose --env-file .env.api.example -f compose.model-api.yml up --build -d
Invoke-RestMethod http://127.0.0.1:8000/health
Invoke-RestMethod http://127.0.0.1:8000/ready
docker compose --env-file .env.api.example -f compose.model-api.yml down
Remove-Item Env:ERIS_SERVICE_API_KEY
```

Keep the key available until Compose is stopped. The supplied example has an
empty value, not a working credential; shell environment takes precedence.
Compose fails when no nonempty key is supplied. Explicit `--env-file` avoids
implicitly reading a personal `.env`. No key is baked into the image or collection.
The existing `.gitignore` rule `.env.*` also ignores `.env.api.example`. It exists
locally; that rule was not changed and no file was staged. Include this empty
example explicitly in any later reviewed handoff; never include a populated secret file.
Port 8000 binds to **127.0.0.1 only**. Do not use Compose `config` without
`--quiet` when a real key is set: rendered configuration contains environment values.

The service has one worker, non-root UID/GID 10001, read-only root and bundle,
a writable temporary `/tmp`, dropped capabilities, no-new-privileges and no
source/data mounts. Image HEALTHCHECK calls `/ready` using urllib every 15 s.
Do not increase workers or horizontal replicas before measuring memory for that
deployment. There is no SLA or approved production capacity.

## Postman

1. Import `ERIS-Model-API-v1.postman_collection.json` (21 synthetic requests) and
   `ERIS-Model-API-Local.postman_environment.json` from this directory.
2. Select the local environment; set `base_url=http://127.0.0.1:8000`.
3. Set `service_key` to the same temporary key **locally only**, as a sensitive
   value. Do not export/share/sync an environment containing it. Do not print it
   in terminal history, screenshots, test output or logs.
4. Run Public → Protected model API → Negative tests. Ready must be 200 first;
   otherwise fix startup, rather than treating expected 401/422 checks as passed.
5. Use Send for individual requests or Collection Runner for all 21. The oversized
   batch is generated in a pre-request script; the extreme-integer case is a
   400-digit synthetic JSON integer (expected 422, never 500).
6. Alternatively import `../docs/model-api-v1.openapi.json` in Postman to inspect
   the API. **OpenAPI is the canonical contract**; this collection is only a
   manual test tool.

Newman/Postman CLI was not installed on the audited machine and was not installed
by this task. JSON was validated with Python; the original 18 B2 HTTP scenarios were
executed against the earlier container, not through Postman's JavaScript runner.
The AdaBoost extension was smoke-tested via local HTTP, not a new container. If Newman is already
available, run the collection with its programmatic API and an in-memory environment
whose `service_key` value comes from `process.env.ERIS_SERVICE_API_KEY`; use only the
CLI reporter and do not persist environment or response reports. Do not pass the
secret in `--env-var` on a command line or save it in an exported environment.

## Historical B2 packaging rationale (API-3, before AdaBoost)

This section and its image inventory describe the original B2-only build. The new
Docker context adds s1 and its 40-module loaded import closure; the new verified image
is recorded above. Historical image hashes below do not describe API 1.1.

`deploy/api_b2/requirements-runtime.txt` is the one additional file beyond the
initial API-3 list. The repo has broad dependency ranges but no existing runtime
lock; B2 refuses mismatched Python/library versions. This file pins the Linux
runtime dependency closure from the validated environment, with Python 3.12.10
and exact B2 ML versions. No dev tools, Jupyter, Streamlit or XGBoost are installed.
Base image is pinned by digest. Only packages needed for RF/SHAP/API imports are
installed; pip check passes. System libgomp is installed inside the image only.

The existing loader imports shared helpers through `baseline_v3`, `bundle_v2`
and related modules, including `models/training.py`. Those unchanged Python
modules are required imports, not training entry-point scripts; nothing fits or
trains. They are included explicitly instead of modifying model core. No project
`scripts/`, tests, UI, notebooks, datasets, predictions, reports or other bundles
are in `/app`. No runtime config YAML is required: the frozen feature config is
already embedded in B2 metadata.

An initial image failed content inspection because Docker directory include rules
also included unrelated source and bytecode. That image was not accepted. Rules
were corrected and the final image rebuilt: an exact 34-file application inventory
was verified, with all source/bundle hashes matching the host and no project
cache/UI/dataset/secret content. The final image and build cache remain local;
nothing was pushed. No system prune was run.

## Historical B2 verified evidence (API-3)

- Branch `main`; HEAD `a7a5447f4a53cb67275c0d1aaf8ce0619d9c7785`.
- Image: `eris-model-api-b2:api3-local`.
- Image ID: `sha256:5b9371f2b1666afe81f4938037d33963b45822104f6af643ebbd379fc50a5741`.
- Docker-reported Size: 213827256 bytes; the largest uncompressed
  history layer is the runtime dependencies (~558 MB). Initial build took roughly
  one minute; subsequent cached builds were shorter. This is not a benchmark.
- Bundle: `/app/artifacts/models/v3_final_15/b2/`; UID/GID `10001:10001`;
  Docker healthcheck `healthy`, including after restart.
- Model SHA-256 on host and in container:
  `0c72841a8505a910d31c2a687fbcbf5259e3afb60f9eddcb7586801519b5628f`.
- Synthetic host probability: `0.05957326622379898`; container:
  `0.05957326622379898`; absolute difference `0`
  (tolerance 1e-12). Four concurrent requests also matched.
- 18/18 HTTP scenarios passed: health/ready/info/predict/optional/batch/explain 200,
  missing/wrong key 401, all nine invalid input/batch scenarios 422.
- Two startup completions (initial + deliberate restart); no secrets, submitted
  fields, SHAP output or validation tracebacks in container logs. Readiness proves
  bundle loading succeeded. API regression tests verify one load per service start;
  image source is byte-identical to that tested implementation.
- Tests API-1/API-2/API-3: **285 passed**. Ruff across src/scripts/tests passed;
  strict Mypy on API plus exporter passed (6 source files); in-memory compile
  and `git diff --check` passed. No full research suite was run because it includes
  model fitting outside this task.
- Existing warnings: Starlette/httpx and AnyIO deprecations, SHAP/Matplotlib pending
  deprecations, Git LF/CRLF notices for pre-existing changes. Build pip emitted
  its root-install warning (build stage only; runtime is non-root) and update notice.
  No dependency was changed to silence warnings.
- Temporary Compose containers/network were removed; no test service remains running.
  To remove the retained tagged image later: `docker image rm eris-model-api-b2:api3-local`.
  Do not run global prune commands.

Latency in milliseconds, 10 measured requests per operation after warm-up,
P95 by nearest rank (the maximum of 10). Local Docker Desktop only, not an SLA:

| Operation | Median ms | P95 ms |
| --- | ---: | ---: |
| health | 2.69 | 23.04 |
| predict_one | 179.68 | 182.86 |
| batch_10 | 186.56 | 203.49 |
| batch_100 | 205.05 | 221.05 |
| explain_one | 377.78 | 401.55 |

Actual server-process RSS from `/proc/1/status`: startup 287220 KiB;
after warm-up 292724 KiB; after smoke/benchmark
294004 KiB. The small warm-up increase does not establish a
memory leak or prove long-duration stability. No timeout or stated latency warning
threshold was reached.

## Contract export and provenance

```powershell
.\.venv\Scripts\python.exe -B scripts/export_api_b2_openapi.py
.\.venv\Scripts\python.exe -B -m pytest tests/unit/test_api_b2_schemas.py tests/integration/test_api_b2.py tests/integration/test_api_b2_openapi.py -q -p no:cacheprovider
.\.venv\Scripts\python.exe -B -m ruff check --no-cache src scripts tests
.\.venv\Scripts\python.exe -B -m mypy --strict --no-incremental --cache-dir=nul src/eris_ml/api_b2 scripts/export_api_b2_openapi.py
git diff --check
```

Current export has seven endpoints including `/v1/models`; it matches `app.openapi()`.
The following hashes/diff are retained as historical API-3 provenance, not current hashes.

Historical export: `docs/model-api-v1.openapi.json`, exactly six endpoints, no public OpenAPI
HTTP route added. Repeated exports are byte-identical and match `app.openapi()`;
no lifespan/model call or service key is needed. Existing differing files are
refused, not overwritten. SHA-256:
`8539ffeac048ae173dab4abab97e411831bdf484e3fb8a887ad15abec78e2fd0`.

Only existing file intentionally changed: `src/eris_ml/api_b2/main.py`.
The user authorized metadata-only APIKeyHeader(auto_error=False), Security and
response declarations. Runtime auth, comparison, request/response schemas,
lifespan, probability/SHAP and API-1 contract remain unchanged.
- Before SHA-256: `530996474a6ecb031bb1b98f62f80053f9e2e4f85c8ddb2007d579828571c4aa`.
- After SHA-256: `5b7ed28b3a7d692c26274c2df78c2c1c66ac2cdda995e0cec17051ebf570099e`.

Exact metadata diff (the reconstructed pre-edit bytes were checked against the
recorded pre-edit SHA-256; only line endings are normalized in this display):

<details>
<summary>main.py metadata-only diff</summary>

```diff
--- main.py before API-3
+++ main.py after API-3
@@ -9 +9 @@
-from fastapi import FastAPI, Request, Response
+from fastapi import FastAPI, Request, Response, Security
@@ -11,0 +12 @@
+from fastapi.security import APIKeyHeader
@@ -53,0 +55,4 @@
+    )
+    # Documentation-only header dependency. Authentication remains in boundary().
+    api_key_header = APIKeyHeader(
+        name="X-ERIS-Service-Key", scheme_name="ERISServiceKey", auto_error=False
@@ -107 +112 @@
-    @application.get("/ready")
+    @application.get("/ready", responses={503: {"model": ErrorResponse}})
@@ -112 +117,6 @@
-    @application.get("/v1/model/info", response_model=ModelMetadata)
+    @application.get(
+        "/v1/model/info",
+        response_model=ModelMetadata,
+        dependencies=[Security(api_key_header)],
+        responses={401: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
+    )
@@ -116 +126,11 @@
-    @application.post("/v1/predict", response_model=PredictResponse)
+    @application.post(
+        "/v1/predict",
+        response_model=PredictResponse,
+        dependencies=[Security(api_key_header)],
+        responses={
+            401: {"model": ErrorResponse},
+            422: {"model": ErrorResponse},
+            503: {"model": ErrorResponse},
+            500: {"model": ErrorResponse},
+        },
+    )
@@ -122 +142,11 @@
-    @application.post("/v1/predict/batch", response_model=BatchPredictResponse)
+    @application.post(
+        "/v1/predict/batch",
+        response_model=BatchPredictResponse,
+        dependencies=[Security(api_key_header)],
+        responses={
+            401: {"model": ErrorResponse},
+            422: {"model": ErrorResponse},
+            503: {"model": ErrorResponse},
+            500: {"model": ErrorResponse},
+        },
+    )
@@ -128 +158,11 @@
-    @application.post("/v1/explain", response_model=ExplainResponse)
+    @application.post(
+        "/v1/explain",
+        response_model=ExplainResponse,
+        dependencies=[Security(api_key_header)],
+        responses={
+            401: {"model": ErrorResponse},
+            422: {"model": ErrorResponse},
+            503: {"model": ErrorResponse},
+            500: {"model": ErrorResponse},
+        },
+    )
```

</details>

All other protected hashes must match the pre-edit snapshot (276 existing files,
including B2 and the earlier audit outputs). No final-holdout/f1 data was read.
No train/refit/tuning/calibration, secret file, backend/database/frontend change,
commit, push, image push or production deployment occurred.
