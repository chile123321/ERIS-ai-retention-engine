# ERIS AI Retention Engine

AI/ML service for validating HR data, developing employee-attrition models, packaging approved models, and serving inference through FastAPI.

## Scope

This repository owns data contracts and validation, development-only EDA, reusable preprocessing, model training/evaluation, calibration, threshold selection, explainability, model bundles, and the inference API. React UI, Node.js business services, Supabase migrations, authentication, user management, dashboards, and application reporting are out of scope.

## Current data state

- IBM HR raw data: 1,470 rows and 35 columns; target `Attrition` maps `No=0`, `Yes=1`.
- V1 model input: 25 features; V2 has four locked variants (18/17/17/16).
- Development set: 1,176 rows (`986 No`, `190 Yes`).
- Locked final test: 294 rows (`247 No`, `47 Yes`).

The existing raw data and split are immutable. The V1 final test was evaluated once on 2026-09-20, as recorded in [the evaluation ledger](configs/release/final_evaluation_record_v1.yaml). It is not a fresh independent holdout for V2. Do not open it or use its scores for V2 feature/model decisions, tuning, calibration, or policy selection.

## Feature strategies

- **V1 Full:** the approved set of up to 25 IBM HR candidate features.
- **V2 Practical:** four locked feature sets: `v2_full_18`, `v2_no_department_17`, `v2_no_jobrole_17`, and `v2_general_16`. Enterprise collection semantics still need confirmation.
- **V3 Minimal:** a small, easy-to-collect and easy-to-standardize feature set.

V2 development experiments are complete through baseline feature ablation, three-model comparison, Logistic nested tuning, calibration, hypothetical capacity-policy evaluation, and an HR discussion package. Logistic baseline **raw `v2_general_16`** is the portability-motivated working candidate; `v2_no_department_17` is the JobRole comparator. Tuned Logistic was not retained and calibration retained raw for both. The 5%/10%/15% capacities are assumptions, not HR approval or a production policy. V3 remains undefined.

See [V2 status and quality gate](docs/v2_status.md) for evidence, pending decisions and the next development-only fairness/subgroup audit. IBM cross-validation does not establish performance at a new employer or prediction 3–6 months ahead.

The original API and V1 manual UI still use the **V1, 25-feature bundle**. A separate
Logistic raw `general_16` **research/demo candidate bundle** now exists at
`artifacts/models/eris_logistic_general16_v2_r1.joblib`; it returns probabilities
only, with no default threshold. See [V2 status](docs/v2_status.md#researchdemo-final-fit--2026-10-07).
There is no V2 API, model card or approved production policy. IBM-coded candidate
packaging is not HR/data-contract approval and does not update serving. Stop and
notify the project owner before any API/backend design or change.

## Local setup

Declared support is Python 3.11 or 3.12. Windows Python 3.12 is the runtime used
for the installation audit; see [environment audit](docs/environment_audit_v2.md)
for the separate, non-runtime Python 3.11 resolver check. Do not recreate an
existing environment in place: use a new path for upgrades and compare results.

Fresh checkout, reproducible Windows Python 3.12 setup (from the repo root):

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[api,ml,explain,dev]" -c requirements/quality-py312.txt
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m mypy src
.\.venv\Scripts\python.exe -m pytest -v
.\.venv\Scripts\python.exe -m ruff check .
git diff --check
```

All four extras are needed for the complete research/API test environment;
`explain` supplies XGBoost and SHAP. No manual stub installation or relaxed Mypy
flag is needed. `dev` also includes Streamlit for the V2 UI typing/widget tests;
`manual-ui` makes it available without the full development toolchain.
The local `typings/` directory is part of the checkout. A plain
`python -m pip install -e ".[api,ml,explain,dev]"` also installs all declared
dependencies; constraints pin the audited resolution, not historical training.
Do not use the Python-3.12 lock as a Python-3.11 lock.

Data, OOF, and the trusted V1 bundle/sidecars are Git-ignored and must be supplied
separately through an authorized local transfer; installing Python packages does
not create them. Tests requiring those assets cannot be certified from a Git-only
clone. Never obtain them by opening final test or rerunning experiments implicitly.
No production dependency installation is implied by this research setup.

For an alternate environment, invoke its Python directly; the manual PowerShell
wrappers accept `-PythonExecutable 'C:\path\to\venv\Scripts\python.exe'`.
They default to the repo `.venv` when this option is omitted.

Repository commands:

```bash
make lint
make test
make serve
python scripts/prepare_data.py --help
python scripts/train.py --help
```

## Layout

`configs/` contains versioned contracts and experiment configuration; `src/eris_ml/` contains production code; `scripts/` contains entry points; `notebooks/` is exploration-only; `tests/` contains automated checks; local data and generated artifacts live under `data/` and `artifacts/` and are ignored by Git.

## Offline V2 prediction testing (16 features)

This is a separate Streamlit smoke-test page, not a product UI, API V2 or backend
integration. It loads the existing trusted bundle
`artifacts/models/eris_logistic_general16_v2_r1.joblib` and its sidecars through the
V2 loader. No fitting, calibration, classification threshold or top-k is applied.
Do not substitute the V1 bundle or rebuild anything to resolve a loading error.

Use the **same Python 3.12.10 / ML versions as the bundle**. In PowerShell, from
the repo root, with `$uiPython` pointing to that environment's `python.exe`:

```powershell
# One-time installation; this step needs package access, prediction does not.
& $uiPython -m pip install -e ".[api,ml,explain,dev,manual-ui]" -c requirements/quality-py312.txt
& $uiPython scripts/manual_model_ui_v2.py
```

Open **http://127.0.0.1:8502**; Ctrl+C stops the server. The launcher pins loopback,
disables telemetry/file watching/static file serving and uses bundled fonts.
See Streamlit's [telemetry configuration](https://docs.streamlit.io/develop/concepts/configuration/options)
and [local default fonts](https://docs.streamlit.io/develop/concepts/configuration/theming-customize-fonts).
There is no HTTP prediction API call or automatic input logging/persistence.
Input and the last result live only in process/session memory. Do not expose this
unauthenticated testing tool on a network or use it for real personnel decisions.

Try: **Hồ sơ mẫu IBM → Mẫu 1 → Dự đoán**; then **Nhập thủ công**, change a field
and predict; finally clear `Age` and submit to see validation. Samples are physical
development CSV positions **0, 1, 2**, chosen without labels/scores. Only the 16
feature columns are parsed/displayed, never IDs or target; the development hash
must match bundle provenance. Manual defaults come from sample 1. Missing/invalid
input is rejected, not replaced with zero or `OverTime=No`.

The page displays the raw class-1 probability, a 0–100% bar, full-precision value,
prediction time and the 16 submitted values. IBM codes/units are not an approved
enterprise data contract; these probabilities are not validated at a new employer.
The screenshot and test results are recorded in [V2 status](docs/v2_status.md).
This stops at offline testing; notify the owner before any API/backend design/change.

## V1 manual model testing on localhost

```powershell
.\scripts\start_manual_model_ui.ps1
```

The script opens <http://127.0.0.1:8501/> after its separate server is ready.
It binds only to `127.0.0.1`, requires no API token, and uses the checked frozen
bundle directly. Load a low/high synthetic example, change one of the 25
features, and press Predict. This is research-only manual testing, not a
production service or a new independent evaluation of V2. A few
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
the synthetic examples are not independent model evaluation, and a
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
