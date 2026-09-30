# ERIS AI Retention Engine

AI/ML service for validating HR data, developing employee-attrition models, packaging approved models, and serving inference through FastAPI.

## Scope

This repository owns data contracts and validation, development-only EDA, reusable preprocessing, model training/evaluation, calibration, threshold selection, explainability, model bundles, and the inference API. React UI, Node.js business services, Supabase migrations, authentication, user management, dashboards, and application reporting are out of scope.

## Current data state

- IBM HR raw data: 1,470 rows and 35 columns; target `Attrition` maps `No=0`, `Yes=1`.
- Candidate feature set v0: at most 25 features.
- Development set: 1,176 rows (`986 No`, `190 Yes`).
- Locked final test: 294 rows (`247 No`, `47 Yes`).

The existing raw data and split are immutable. The locked final test must not be opened or used for EDA, feature decisions, tuning, calibration, or threshold selection. It is reserved for a final, explicitly authorized evaluation.

## Feature strategies

- **V1 Full:** the approved set of up to 25 IBM HR candidate features.
- **V2 Practical:** features that ERIS can reliably collect and represent in its application schema.
- **V3 Minimal:** a small, easy-to-collect and easy-to-standardize feature set.

V2 and V3 remain intentionally undefined until their owners approve them.

## Planned setup

Requires Python 3.11 or 3.12. Dependency installation is intentionally not performed by this scaffold task.

```bash
python -m venv .venv
pip install -e ".[api,ml,dev]"
```

Planned commands:

```bash
make lint
make test
make serve
python scripts/prepare_data.py --help
python scripts/train.py --help
```

## Layout

`configs/` contains versioned contracts and experiment configuration; `src/eris_ml/` contains production code; `scripts/` contains entry points; `notebooks/` is exploration-only; `tests/` contains automated checks; local data and generated artifacts live under `data/` and `artifacts/` and are ignored by Git.

## Manual model testing on localhost

```powershell
.\scripts\start_manual_model_ui.ps1
```

The script opens <http://127.0.0.1:8501/> after its separate server is ready.
It binds only to `127.0.0.1`, requires no API token, and uses the checked frozen
bundle directly. Load a low/high synthetic example, change one of the 25
features, and press Predict. This is research-only manual testing, not a
production service or a substitute for the locked final-test evaluation. A few
manually entered profiles cannot be used to calculate Accuracy. The production
API still requires Bearer authentication. Press Ctrl+C to stop the local UI.

## Manual model testing without API authentication

```powershell
.\scripts\manual_model_test.ps1
```

This local research tool loads the checked, frozen bundle directly and offers a
25-feature guided menu. It does not start Uvicorn, call HTTP, test API
authentication or require a service token. Use synthetic examples with
`-Example low` or `-Example high`, a single JSON object with `-InputFile
".\employee.json"`, and `-NoExplanation` to skip SHAP. `-JsonOutput` with a
direct example or JSON input produces machine-readable output. Manual inputs and
the synthetic examples do not replace the locked final-test evaluation, and a
few predictions cannot measure Accuracy. The API below still requires Bearer
service authentication; this tool does not change it. Do not use predictions for
automated employment decisions.

## Local research API (Steps 12B–12C)

The development-only `eris-xgboost-v1` bundle is available for local, research-only
inference. It is **not** business or production approved. The frozen raw-probability
threshold is `0.345651`; an alert is a human-review signal, not a prediction of
certain resignation. Do not use it for automated employment decisions.

In PowerShell, from the repository root:

```powershell
.\.venv\Scripts\Activate.ps1
$env:ERIS_MODEL_BUNDLE_PATH="artifacts/models/eris_xgboost_v1.joblib"
$env:ERIS_MODEL_METADATA_PATH="artifacts/models/eris_xgboost_v1.metadata.json"
$env:ERIS_MODEL_CHECKSUM_PATH="artifacts/models/eris_xgboost_v1.sha256"
# Generate a temporary token before starting Uvicorn. Never print or commit it.
$token = ([guid]::NewGuid().ToString("N") + [guid]::NewGuid().ToString("N"))
$env:ERIS_SERVICE_TOKEN = $token
Set-Clipboard -Value $token
.\.venv\Scripts\python.exe -m uvicorn eris_ml.api.main:app `
  --host 127.0.0.1 `
  --port 8000 --no-access-log
```

Open <http://127.0.0.1:8000/docs>, choose `POST /api/v1/predict`, select
**Try it out**, paste `synthetic_low_signal` from
[`tests/fixtures/manual_prediction_examples.json`](tests/fixtures/manual_prediction_examples.json),
select **Authorize**, paste the bare token (without `Bearer `), click **Authorize**
inside the dialog, then **Close** and **Execute**. The generated Curl must include
`Authorization: Bearer ...`. If it does not, the UI request is not authorized;
reopen the dialog before debugging prediction input. Check `probability`,
`threshold`, `alert`, and the top SHAP factors. SHAP values are raw log-odds
contributions, not causal effects or
percentage-point changes in risk. The three examples are synthetic demonstrations,
not labelled evaluation cases.

PowerShell request using the synthetic fixture (use a second shell while the
clipboard still holds the token; copying another value overwrites it):

```powershell
$token = [string](Get-Clipboard -Raw)
$token = $token.TrimEnd("`r", "`n")
if ($token -notmatch '^[0-9a-fA-F]{64}$') { throw 'Clipboard does not hold a 64-character token.' }
$headers = @{ Authorization = "Bearer $token" }
$examples = Get-Content tests/fixtures/manual_prediction_examples.json -Raw | ConvertFrom-Json
$body = $examples.synthetic_low_signal | ConvertTo-Json -Depth 5
Invoke-RestMethod -Uri http://127.0.0.1:8000/api/v1/predict `
  -Method Post -ContentType application/json -Body $body `
  -Headers $headers
```

`/ready` confirms only that a token is configured and the bundle is valid; it
cannot confirm that a second shell or Swagger holds the same token. If a server
was already using port 8000, stop that process you own before restarting, so
requests do not reach an older app with a different token. Do not print the
token or copy a JSON payload over it before the second shell reads it.

`GET /health` is process liveness; `GET /ready` reports bundle readiness; and
`GET /api/v1/model-info` exposes safe model metadata. The server loads the trusted,
checksummed local joblib bundle once at startup and reads no employee data CSV.
Input boundaries are derived from the IBM development benchmark, **not** an
HR-approved data contract. Protected routes use Bearer service authentication;
operational metrics are process-local. Rate limiting, web-backend persistence and
production monitoring are not implemented; keep the prototype on localhost.
See [API contract](docs/api_contract.md).
