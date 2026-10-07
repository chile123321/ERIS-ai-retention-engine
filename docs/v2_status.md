# ERIS V2 status and repository quality gate — 2026-10-07

V2 is a development research candidate, **not business/production approved**.
This page is the current progress reference; experiment documents preserve their
historical results and next-step recommendations.

**Current gate:** clean Windows Python 3.12.10 installation passes full pytest
(283 passed, 1 existing skip), strict `mypy src` (73 files), Ruff and diff check.
Python 3.11 dependency resolution was checked, but its runtime is unavailable.
See [environment audit](environment_audit_v2.md) for exact commands, versions,
root causes, warnings, source fingerprint drift and preservation evidence.
The older failed gate below is historical, not the current status.
The development-only fairness audit is now complete; see [decision summary](fairness_v2.md).
Next is HR confirmation or an explicitly confirmed capstone simulation, not deployment.
The HR package now includes fairness evidence and a fillable decision form.
`general_16` + top-k 10% is **proposed, not approved**, for a possible capstone
simulation; decision, owner and confirmation date remain blank. Data-contract
finalization is not yet authorized by a confirmed scope/policy decision.
Stop and notify the project owner before any API/backend design or changes.
The user has now separately authorized and completed an IBM-coded **research/demo
final fit and bundle** for Logistic raw general_16. This limited technical
authorization is not HR/policy approval or completion of enterprise data mappings.
See [final-fit record](#researchdemo-final-fit--2026-10-07); no serving was changed.

## Completed and current decisions

| Completed work | Evidence | Current interpretation |
|---|---|---|
| Four-set feature ablation, 18/17/17/16 features | [Feature strategy](feature_strategy_v2.md), [baseline ablation](baseline_feature_ablation_v2.md) | Feature definitions remain locked; removing JobRole costs predictive performance on IBM. |
| Logistic, Random Forest, XGBoost fixed baselines | [Model comparison](baseline_model_comparison_v2.md) | Retain Logistic baseline `v2_general_16` as the portability-motivated working candidate, `v2_no_department_17` as comparator; not a production choice. |
| Logistic nested tuning | [Nested tuning](logistic_nested_v2.md) | Do not retain tuned Logistic: discrimination/probability trade-offs do not justify replacing either baseline. |
| Cross-fitted calibration | [Calibration](calibration_v2.md) | Retain **raw** for both baselines; calibration did not meet improvement/guardrail requirements. |
| Cross-fitted hypothetical capacity policies | [Policy evaluation](threshold_policy_v2.md) | 5%/10%/15% are assumptions. Top-k and fixed probability thresholds differ; no production threshold selected. |
| HR discussion package, updated with fairness | [HR package](hr_policy_decision_package_v2.md) | Six capacity options, reviews/TP, overlap and subgroup support are documented. General_16 + top-k 10% is a proposed academic illustration only; all confirmation fields remain blank. No HR or simulation decision has been confirmed. |
| Development-only fairness/subgroup audit | [Fairness decision summary](fairness_v2.md), [technical report](../artifacts/reports/fairness_v2.md) | Saved raw/top-k predictions only. Age disparities and sparse fold support require review; no blanket fairness approval or automatic model/policy change. |
| Authorized research/demo final fit and packaging | `artifacts/models/eris_logistic_general16_v2_r1.joblib`, build preflight/trace below | Fixed Logistic baseline raw general_16 fit on all development rows. No default threshold, production approval or V2 API. |

The unchanged development split has 1,176 rows, target `{0: 986, 1: 190}`.
All variants reuse the saved five-fold assignment, originally stratified/shuffled
with seed 42; inner CV uses four stratified folds with seed 43 where applicable.
Logistic baseline remains liblinear, C=1, max_iter=2000, class_weight=None,
random_state=42, effective L2. No model/protocol was changed in this quality gate.

API and manual UI still use `artifacts/models/eris_xgboost_v1.joblib`, **25 features**;
see `src/eris_ml/api/contract.py`, `src/eris_ml/manual_ui/app.py` and
`scripts/start_manual_model_ui.ps1`. The separate V2 research bundle is not wired
into those services. There is no V2 API, model card or approved production policy.
Do not send a 16-feature payload to the V1 contract.

## Pending and next step

- Human review of V2 subgroup findings and low-support limitations; V1 fairness is not transferable.
- HR confirmation, or an explicitly confirmed capstone simulation scenario.
- Standardized input data contract: income units, overtime window, survey scales,
  snapshot timing, year-based features, missingness and any JobRole mapping.
- Business-scope candidate freeze and V2 model card; explicit owner checkpoint
  before V2 API/web/backend design or integration. Research bundle now exists separately.
- Evaluation on new employer data with event/snapshot timestamps and a defined horizon.

**Next after the completed audit:** HR confirmation or an explicitly confirmed
capstone simulation, using the HR package and subgroup findings for the two
Logistic baseline raw candidates. Do not access final test or treat completion
of technical auditing as HR, fairness or production approval.
The proposal is not permission to use alerts for real employees. The current
user request explicitly authorizes IBM-only research/demo final fit and packaging,
without settling HR/capstone policy or enterprise data-contract decisions. Those
decisions remain necessary before approval for the corresponding real-use scope.
API/backend work has an explicit owner notification and instruction checkpoint;
do not proceed automatically, including API schema design.

IBM is cross-sectional: these results do not validate 3–6 month prediction,
successful intervention or generalization to Vietnamese/new employers. The V1
[evaluation record](../configs/release/final_evaluation_record_v1.yaml) records
`run_count=1`, `status=completed`, completed 2026-09-20. That final test is **not a
new independent V2 holdout**. Only the record was read, not the final-test CSV;
its model scores were not used for any V2 decision.

## Dictionary provenance

- Name/version: `ERIS_Feature_Category_Dictionary_v2.xlsx`, user-supplied v2 workbook.
- Source: the user's local `Documents` directory adjacent to this repository;
  not a published IBM specification and not an approved enterprise data contract.
- SHA-256: `05e2e70455dbc1047075e8a930a85aa37c72cb133a60ea8d7ffa40555481bc02`.
- The previous `../Documents/...` path resolves from this repository root on this
  machine, but not from the registry's directory or necessarily another clone.
  Registry metadata now names the external workbook and expected hash, not a
  supposedly portable filesystem location. No workbook or personal file was added to Git.

Obtain this version from its owner through an authorized channel and retain it
outside Git. Set a local path explicitly and compare its hash before using it:

```powershell
$dictionaryPath = 'C:\path\provided-by-owner\ERIS_Feature_Category_Dictionary_v2.xlsx'
$expected = '05e2e70455dbc1047075e8a930a85aa37c72cb133a60ea8d7ffa40555481bc02'
if ((Get-FileHash -LiteralPath $dictionaryPath -Algorithm SHA256).Hash.ToLowerInvariant() -ne $expected) {
    throw 'Dictionary version/hash mismatch; do not silently replace the feature definition.'
}
```

Local file availability and hash were verified in this gate. Schema unit tests
check versioned provenance/configuration without requiring a personal folder.
The four actual feature definitions, including all 18 names, remain byte-unchanged.

## Historical consistency audit — 2026-09-30

Initial branch `main`, HEAD `91eb94f709f5188ce2c2c5db5edc28ec9c1bd6e6`, clean
worktree. Local `origin/main` points to the same commit as the previously observed
GitHub `91eb94f`; no fetch/live remote verification was performed. No commit/push.

Before editing, all 109 input-hash entries and 32 output-hash entries in the five
current V2 traces matched their local files; all declared output files were present.
The inspected trace families are `baseline_feature_ablation_v2`,
`baseline_model_comparison_v2`, `logistic_nested_v2`, `calibration_v2`, and
`threshold_policy_v2` under `artifacts/reports/`. Their JSON `input_sha256` and
`output_sha256` were checked against SHA-256; output keys were resolved through
each corresponding script's `OUTPUTS` declaration, without calling its run function.
Historical recovery/preformat runs were not substituted for current traces.

**Intentional registry metadata change:** each historical trace pins the old full
`configs/features/feature_set_v2_practical.yaml` hash
`45dabe9e999ee5bbe346d7f29a7e82edbf71965f90f660b87bba7a58b5d877ec`.
It remains recoverable from the initial HEAD. The status/provenance-only update
changes that hash, not the four feature schemas, model parameters, split or folds.
Do not rewrite historical traces or suppress fingerprint checks. Existing experiment
CLIs still reject full-registry drift (and existing outputs); this is expected,
not permission to overwrite/rerun them. A future fairness trace must record the
current registry and explicitly validate the unchanged experimental fields against
that historical revision before accepting prior OOF artifacts.

After editing: 104/109 input-hash entries still match; the other five are exactly
that one registry metadata revision, with current SHA-256
`ef19d1038911b28edeffe3eb69647d00595dda18f79afdb5f95c0fc34720a81f`.
All 32 trace output-hash entries still match. All 44 declared output files across
the five runs are locally present and Git-ignored (`git check-ignore -q`); absence
from tracked Git content is **not** absence from this machine. Reports themselves
are locally present but not every report is output-hashed by its historical trace;
the 32 count must not be read as a claim of 44 historical digest checks.
74 literal/brace-expanded document references were checked with no missing files;
the HR package's wildcard artifact family was covered by the threshold script's
explicit output map. No final-test CSV reference was opened, hashed or statted.

The registry's version, preferred-feature rationale, four configuration paths and
entire comparison protocol were compared with initial HEAD and are unchanged.
The existing feature/split tests also pass. No published experimental number was
modified, and no numerical conflict was found in the reviewed decisions.

## Historical quality gate before typing/environment repair

Commands below were run from the repo root using the existing `.venv`:
Python 3.12.10, pytest 9.1.1, Ruff 0.16.8, Mypy 2.3.1, NumPy 2.5.3,
pandas 3.0.6, scikit-learn 1.9.1. No dependency or tool configuration was changed.

| Actual command | Result |
|---|---|
| `.\.venv\Scripts\python.exe -m pytest -v` | Exit 0: **222 passed, 1 skipped, 0 failed**, 55 warnings; 223 collected, 69.53 seconds. |
| `.\.venv\Scripts\python.exe -m ruff check .` | Initially 41 errors; after scoped corrections, exit 0, **all checks passed**, including notebook. |
| `.\.venv\Scripts\python.exe -m mypy` | Exit 1: **72 errors in 32 files**, further checking prevented. Uses unmodified pyproject configuration. |
| `.\.venv\Scripts\python.exe -m mypy src` | Same failure; this is the Makefile typecheck target with the local interpreter. |
| `.\.venv\Scripts\python.exe -m mypy --python-version 3.12 src` | Diagnostic only, not a replacement gate: exit 1, **71 errors in 31 files**, 69 source files checked. |
| `git diff --check` | Exit 0; no whitespace errors. Git reports its existing LF-to-CRLF conversion warnings. |

The single existing skip is `tests/unit/test_data_contract.py:6`: authoritative
contract columns have not been approved. No tests were newly skipped/disabled.
Warnings: 25 sklearn penalty deprecations, 25 L1/default-l1_ratio warnings,
two Starlette/AnyIO deprecations and three SHAP/matplotlib deprecations.
Synthetic training/serialization tests ran in temporary fixtures; no development
experiment, model release or actual fairness evaluation was run.

Ruff corrections cover the two known 101-character lines in logging/preprocessing
tests plus 39 notebook findings (imports, wrapping, one-line statements, explicit
`zip(strict=False)` preserving truncation behavior for plot axes). The notebook was
not executed. AST comparison confirmed equivalent code after normalizing import
order and explicit default `strict=False`; all outputs, execution counts, markdown
and metadata remain unchanged. No lint rule was disabled or widened.

**Remaining blocker: strict Mypy environment/typing compatibility.** The repository
targets Python 3.11, but installed NumPy stubs use Python-3.12-only `type` syntax.
There are also 71 `import-untyped` diagnostics for absent stubs/typing metadata in
PyYAML, pandas, scikit-learn, scipy, SHAP and joblib. Changing only the command-line
target to 3.12 removes the syntax blocker but does not solve those diagnostics.
Earlier experiment-specific Mypy passes used different command-line options;
they do not certify this unmodified strict repository gate. No
`--ignore-missing-imports`, exclusion or relaxed strictness was used to claim a pass.
Resolve the supported Python/dependency matrix and typing dependencies in a scoped
follow-up; then rerun the configured Mypy gate before starting the planned audit.

**Readiness: not yet through the complete quality gate.** Documentation, feature
scope, local artifacts, tests and lint are consistent; Mypy remains unresolved.
HR confirmation and enterprise data semantics remain pending independently.

Final branch/HEAD unchanged; worktree has seven modified tracked files plus this
one new document. No staged changes, commit, push, final-test access, experimental
retraining, production-policy selection, bundle creation or API change. Raw data,
development membership and prior experiment artifacts were not changed.

## Current installation and typing gate — 2026-10-01

This audit began on the same HEAD with 23 modified tracked files and five
untracked entries from ongoing work; it did not reset them or replace `.venv`.
A separate temporary environment installed all four declared extras successfully.
The reproducible Windows 3.12 profile is `requirements/quality-py312.txt`;
README documents installation, required local assets and interpreter selection.

The original 72 Mypy errors were 71 missing typing metadata/stub errors plus a
Python-3.12-only NumPy typing syntax error against target 3.11. Published stubs,
a tested narrow SHAP boundary and precise source typing repair replace the old
relaxed command. NumPy/SciPy/SHAP bounds preserve declared Python 3.11 support;
SHAP 0.50 fixes the observed XGBoost base-score failure. sklearn is bounded below
1.10 to protect the unchanged historical penalty grid. No strictness relaxation,
package exclusions or broad type ignores were introduced.

Verified runtime: Python 3.12.10, NumPy 2.4.6, SciPy 1.16.3, pandas 3.0.6,
sklearn 1.9.1, XGBoost 3.4.1, SHAP 0.50.0, Pydantic 2.13.5, Mypy 2.3.1.
`python -m mypy src`, `python -m pytest -v`, `python -m ruff check .`,
`python -m pip check` and `git diff --check` pass in the clean profile.
The existing skip is the unapproved data contract; 54 warnings remain visible
(50 historical Logistic, one test-client, three SHAP plotting warnings).
Python 3.11 resolves with XGBoost 3.2.0, but has **not** been runtime tested.

All 32 historical V2 output hashes still match; 44 outputs are locally present.
Current input matches are 83/109, not the historical 104/109 above: 26 entries
refer to 11 registry/source files with reviewed metadata/typing edits. Mapping:
`docs/quality/environment_audit_20261001_readonly.json`. Preserve old traces and
guards; the next audit must record its own environment/source hashes and verify
unchanged schemas, split and OOF against this revision evidence.
Five V2 entry points imported and passed the separate read-only config/alignment
checks; their experiment writers and training entry points were not run.

The installation audit established readiness for development-only fairness work,
**not** production or Python 3.11 runtime certification. No fairness audit ran
as part of that installation task; the subsequent audit is recorded below.
The final constrained test rerun passed 224 tests with one existing skip in
146.62 seconds. Before/after hashes match for all 126 protected files; the old
environment's package versions, development data, configs and stored artifacts
remain unchanged during this clean-audit pass.

## Completed development-only fairness audit — 2026-10-01

Audit branch `main`, HEAD `91eb94f709f5188ce2c2c5db5edc28ec9c1bd6e6`, unchanged.
All 26 previously modified tracked files were preserved byte-for-byte. This
status document was already untracked and is updated intentionally. No commit/push.

Preflight verified 1,176 development rows, target 986/190, unique identities,
saved fold/target alignment, baseline/calibration raw agreement and exact saved
top-k decisions (floor budget, score descending/source_row ascending ties).
Two raw and six policy candidates have one prediction per row. Historical traces
are unchanged: 20 output-hash entries match; 15 reviewed source-revision entries
match the environment audit mapping and historical HEAD. No generic drift bypass.

The protocol was frozen before subgroup metrics. AgeGroup/Gender are primary,
MaritalStatus exploratory; Age remains an existing model feature, whereas Gender,
MaritalStatus and derived AgeGroup are audit-only. V1 support rules are retained:
n>=50, positives>=10, negatives>=10; precision additionally needs >=10 alerts.
Unsupported performance metrics are undefined, not zero. No new model fit,
policy/threshold selection, calibration, employee-level output or final-test access.

At hypothetical 5%/10%/15%, both candidates retain 55/115/175 alerts. TP counts
are 35/67/84 (16) and 41/67/90 (17). At 10%, the same 67 total TP mask a shift:
17-minus-16 is +2 Divorced, +4 Single, -6 Married. At 15%, pooled Gender TPR
gap shrinks from 10.4 to 3.1 percentage points, but AgeGroup gap remains 52.9
points in both; the Gender gap also varies substantially by fold. No single
metric establishes fairness or a universally preferable feature set.

All nine pooled subgroups pass support; Age 50+ and Divorced pass in 0/5 folds,
Age 40–49 in 1/5, Age 18–29 in 2/5. Full age/marital TPR/FPR gaps by fold are
therefore undefined, not evidence of stability or absence of disparity.
Ten pooled precision cells have fewer than ten alerts and are undefined.

New local evidence: `artifacts/reports/fairness_v2_preflight.json`,
`fairness_v2_trace.json`, `fairness_v2.md`, nine aggregate metric/stability CSVs,
`fairness_v2_initial_state.json` and `fairness_v2_verification.json`.
Independent crosstab/metric reconstruction checked 120 probability rows,
360 policy rows, 180 paired deltas, all partition totals and fold summaries.
All 126 protected historical file hashes, 26 pre-existing tracked edits,
62 current input hashes and 10 audit output hashes match after execution.
No old output or trace was overwritten. See [summary](fairness_v2.md).

Commands run from repo root in the same audited clean interpreter:

```powershell
$auditPython = 'C:\Users\Chile\AppData\Local\Temp\eris-env-audit-e01b331ff3c6437c83a882b24bb920c2\venv\Scripts\python.exe'
& $auditPython -m pytest tests/integration/test_fairness_v2.py -v
& $auditPython scripts/audit_fairness_v2.py
& $auditPython -m pytest -v
& $auditPython -m mypy src
& $auditPython -m ruff check .
& $auditPython -m pip check
git diff --check
```

The audit script above ran once successfully and now intentionally refuses
existing output paths; these commands document the executed run, not permission
to overwrite/recreate evidence. Tests use synthetic fixtures, not final-test data.
Related tests: 28 passed. Full suite: **252 passed, 1 skipped, 0 failed**,
54 visible warnings, 131.69 seconds. Strict Mypy: **71 source files pass**;
Ruff, pip check and diff check pass. The existing data-contract skip and warnings
remain as described in the installation audit; none were hidden for this gate.
Python 3.12.10 is runtime tested; Python 3.11 remains dependency-resolution-only.

Technical evidence is sufficient to proceed to HR/simulation discussion, not to
assert fairness/production approval. HR must still confirm batch definition,
capacity, policy/feature preference, ties, JobRole mapping and intervention process.
The 16-feature portability preference is unchanged; both remain research candidates.

## Research/demo final fit — 2026-10-07

**Status: `research_demo_candidate`, not production.** The current user authorized
this technical packaging step explicitly despite unresolved employer mappings;
do not interpret that request as a signed HR policy or fill confirmation fields.
Only baseline raw `v2_general_16` was fit, not the 17-feature comparator, nested
tuned models or calibrated variants. The 18-feature dictionary and all four
feature configs remain unchanged.

Preflight checked 1,176 development rows, target `{0: 986, 1: 190}`, unique IDs,
saved assignment/OOF target and fold alignment, all baseline output hashes and
locked split fingerprints. Four historical source revisions match the exact
old/current pairs in the environment bridge and historical HEAD; old traces
remain unchanged. Full resolved parameters match the saved baseline trace.
The 16 columns are projected **before** fit; Department, JobRole, Gender,
MaritalStatus, EmployeeNumber, source_row and Attrition never enter the transformer.

Fixed estimator parameters (scikit-learn 1.9.1):

```text
solver=liblinear, C=1.0, max_iter=2000, class_weight=None, random_state=42
dual=False, fit_intercept=True, intercept_scaling=1, tol=0.0001
warm_start=False, verbose=0, n_jobs=None
penalty='deprecated', l1_ratio=0.0 => effective L2, unchanged baseline defaults
```

Pipeline is the existing `preprocessor` + `model`: nominal most-frequent imputation
and one-hot encoding, ordinal/numeric median imputation and StandardScaler, all
fit solely on development. The full transformer signature and config/source/data
hashes are captured in preflight before either fit. Two identical full-development
fits check determinism; only the first pipeline is saved. Both converged, with
six Logistic iterations and no fit warnings. No training-set performance metric
was calculated.

| Evidence | Result |
|---|---|
| Reload same process, 32 fixed first development rows | Maximum absolute probability difference 0.0 |
| Reload separate isolated Python process (`-I`), same 32 rows | Difference 0.0; finite probabilities in [0,1], class order [0,1], positive class Attrition=1 |
| Repeat full-development fit, 1,176 probabilities | Difference 0.0; tolerance rtol=1e-12, atol=1e-12 |
| Actual reloaded-bundle schema checks | 15 invalid-input cases rejected; column reordering preserves probabilities exactly |
| Preserved historical evidence | 140 protected file hashes unchanged, including V1 bundle, split/configs and prior experiment/fairness artifacts |

Local, Git-ignored artifacts (no existing bundle/output overwritten):

- `artifacts/models/eris_logistic_general16_v2_r1.joblib`
- `artifacts/models/eris_logistic_general16_v2_r1.metadata.json`
- `artifacts/models/eris_logistic_general16_v2_r1.sha256`
- `artifacts/reports/logistic_general16_v2_r1_{initial_state,preflight,trace,quality_gate}.json`

Bundle SHA-256: `6da528c7cb26884813b1bdb0ab3665d83eb1a0ee5be26d73c5a380db37f15d52`.
Use only trusted locally produced joblib: checksums detect alteration, not origin
authentication, and pickle can execute code. The V2 loader checks both sidecars,
metadata/runtime versions, schema, fitted classes, model and preprocessing before
returning probabilities. It rejects any existing output on a repeat build; inspect
provenance and request a separately versioned run rather than overwrite.

This local bundle boundary is **not a new enterprise data contract or API schema**:
exactly 16 named columns; missing/extra/duplicate columns rejected; nonnegative
finite integer-valued numeric inputs (not strings/bools), supported IBM ordinal
codes and OverTime No/Yes. PerformanceRating currently accepts the observed IBM
codes 3/4. Nulls are rejected at the bundle boundary although baseline imputers
are retained inside the unchanged pipeline. Future missingness, income/currency,
survey, overtime, job-grade and snapshot mappings still require confirmation.

The bundle exposes raw class-1 probability only and has **no classification method
or default threshold** (`threshold=null`). Hypothetical top-k 10% must be a separate
whole-batch operation; it is not executed by the bundle and remains unapproved.
No employee-level smoke prediction output was written.

**Evaluation evidence remains saved development OOF**, not smoke/train metrics:
AP/PR-AUC **0.477584**, ROC-AUC **0.775729**, Brier **0.110733**, from
`baseline_feature_ablation_v2_metrics.csv`. These describe cross-fitted baseline
evaluation; the final-fit checks above only certify packaging consistency.
Fairness evidence likewise comes from the saved OOF audit, not a new fairness claim
about the final fitted object. IBM does not establish new-employer fairness,
production performance, a 3–6 month horizon or intervention one month in advance.
The previously used V1 final-test CSV was not opened, statted or hashed.

Tested runtime: Python 3.12.10, NumPy 2.4.6, pandas 3.0.6, SciPy 1.16.3,
scikit-learn 1.9.1, joblib 1.6.0. Historical baseline used NumPy 2.5.3; the reviewed
typing/environment repair is documented, not hidden by rewriting old traces.
Python 3.11 remains resolve-only, not runtime certified. No dependency changed here.

Commands executed in the same clean interpreter listed in the fairness section:

```powershell
& $auditPython -m pytest tests/integration/test_logistic_bundle_v2.py -q
& $auditPython -m pytest -v
& $auditPython -m mypy src
& $auditPython -m ruff check .
& $auditPython -m pip check
& $auditPython scripts/build_logistic_bundle_v2.py
git diff --check
```

Related tests: **31 passed**. Full suite: **283 passed, 1 existing skip, 0 failed**,
54 existing warnings, 180.35 seconds. Strict Mypy: **73 source files pass**.
Ruff, pip check and diff check pass. Warnings remain the historical 50 Logistic
grid, one test-client and three SHAP plotting notices; the unapproved data-contract
test remains skipped. No warnings were suppressed to claim success.

Code added: `models/bundle_v2.py`, `models/bundle_v2_build.py` under `src/eris_ml`,
`scripts/build_logistic_bundle_v2.py`, `tests/integration/test_logistic_bundle_v2.py`.
V1 persistence/serving code was not changed. Only existing README, feature strategy,
HR package and this status document were updated; no duplicate report/model card
or data contract was created. Branch main and HEAD remain unchanged; no commit/push.

**Stop here:** inform the project owner before any API/backend design or change.
Do not automatically integrate this candidate or turn the demo policy into HR approval.

## Offline V2 prediction-testing UI — 2026-10-07

Added a separate Streamlit entry point `scripts/manual_model_ui_v2.py` and
`src/eris_ml/manual_ui/v2_streamlit.py`, with tests in
`tests/integration/test_manual_ui_v2.py`. Optional extra `manual-ui` pins
Streamlit **1.65.0**; `dev` also includes it so the published full typing/test gate
works without an undeclared manual installation. Existing V1 UI/API, model code,
bundle and research outputs
are unchanged. This is **local research/demo testing**, not V2 product/backend
integration or business approval. See the README for installation/start commands.

The page calls the existing V2 `load_bundle`, `input_schema`, `validate_input` and
`predict_proba`; no second preprocessing pipeline is constructed. Bundle identity
is `logistic-general16-v2-r1`, raw probability for class 1, SHA256 prefix
`6da528c7cb268848`. It validates bundle sidecars/runtime and development fingerprint,
and fails closed without falling back to V1 or fitting a replacement. Samples are
fixed CSV positions 0/1/2, with only the 16 features parsed; manual defaults are
sample 1. No identifiers or labels are displayed. Input/errors are not written to
files/logs; session memory only. The launcher binds `127.0.0.1:8502`, disables
telemetry and uses local fonts. No final-test file was opened or hashed.

Verified with Python **3.12.10**, NumPy **2.4.6**, pandas **3.0.6**, SciPy **1.16.3**,
scikit-learn **1.9.1**, joblib **1.6.0**; the ML versions remain unchanged after
installing Streamlit with `requirements/quality-py312.txt`. This task adds only
the UI/development dependency declarations to project metadata; old provenance/trace fingerprints
are retained verbatim, not rewritten to reflect this authorized metadata change.
Python 3.11 runtime remains unverified.

Actual Edge browser checks (Playwright 1.63.0, test driver only): sample 1 returned
**0.14552073424746556 (14.55%)**; repeat was identical. Manual input with `Age=45`
and all other defaults from sample 1 returned **0.12693268873539174 (12.69%)**.
Both match direct calls to the same bundle within 1e-15. Blank Age was rejected
and the prior result cleared. External HTTP requests were blocked during browser
checks and none were observed. The integrated browser connection was unavailable;
these were actual local-browser checks, not just widget simulations. A requested
sample-only screenshot is local/ignored at
`artifacts/reports/manual_ui_v2_checked.png`; no manual-input record was persisted.
These examples are smoke results, **not generalization metrics** or risk categories.

Using the audited interpreter
`C:\Users\Chile\AppData\Local\Temp\eris-env-audit-e01b331ff3c6437c83a882b24bb920c2\venv\Scripts\python.exe`:

- `python -m pytest tests/integration/test_manual_ui_v2.py -q`: **20 passed**.
- `python -m pytest -q`: **303 passed, 1 existing skip, 54 existing warnings**.
- `python -m mypy src`: strict, **74 source files pass**.
- `python -m ruff check .`, `python -m pip check`, `git diff --check`: **pass**.

The skip remains the unapproved data contract; warnings remain the historical
Logistic grid, Starlette test-client and SHAP plotting notices. No warnings were
suppressed. **147 protected hashes** (historical inventory plus V2 bundle/sidecars
and final-fit records) matched before/after. No training/repackaging, V1 changes,
API/backend work, commit or push. Stop here and notify the owner before any
API/backend design or change; enterprise data mappings and HR policy remain open.

## Ordered continuation plan

Plans only: no policy, candidate or HR approval is implied.

| Order | Inputs and work | Outputs | Condition to advance |
|---|---|---|---|
| 1. Development fairness/subgroup audit — completed | Saved baseline raw and hypothetical 5%/10%/15% policy OOF for `v2_general_16` and `v2_no_department_17`; saved folds and development-only audit attributes. | Aggregate report, support/disparity/fold tables, frozen protocol, preflight and verified hashes; [summary](fairness_v2.md). | Technical checks passed; disparities/low support require human review, not a blanket fairness certificate. No automatic model/policy change. |
| 2. HR confirmation or explicit capstone simulation | HR package and subgroup evidence; confirm batch scope, capacity, response workflow, ties and JobRole mapping. Without HR, separately confirm a labeled academic simulation and assumptions. | Dated record with owner, scope, policy/feature option or unresolved items. | HR confirmation or separately confirmed academic simulation; simulation is never labeled enterprise approval. |
| 3. Data contract | Confirmed HR/capstone scope and policy/feature preference, dictionary/IBM semantic gaps, audit findings; settle units, surveys/overtime, snapshots, missingness, stable IDs/batch rules and optional JobRole mapping. | Approved versioned data rules and validation requirements; not API/backend schema design. | Human confirmation remains missing. Contract must be finalized before candidate selection/freeze or final fitting. |
| 4. Research/demo bundle completed; business freeze/model card pending | User separately authorized IBM-only general_16 final fit, without enterprise data/policy approval. | Checked research bundle/preflight/trace above; V1 intact. No V2 model card or production candidate. | Real-use scope still needs HR/capstone policy and enterprise data-contract decisions; packaging alone grants no production authority. |
| 5. Owner checkpoint before API/backend | Completed decision/contract and any separately authorized candidate packaging. | Notify project owner that the API/backend milestone has been reached and request direction. | **Stop before any API/backend design or changes**, including API contracts or web integration; await separate instructions. No assumption that stable IDs/batches already exist. |
| 6. Later integration/demo/new-data evaluation, only if authorized | Owner instructions, finalized scope/data rules and suitable data with event/snapshot timestamps. | Deliverables to be agreed in a later task, not designed or implemented here. | Independent employer evidence is required before portability, intervention or 3–6 month claims. |

**Current stopping point:** notify the project owner before any API/backend
design or changes and wait for coordination/instructions. HR/simulation policy
and enterprise data mapping questions remain open; leave decision/owner/date blank
until actually confirmed. No final-test access or automatic integration.

## HR decision-package refresh — documentation only

The package now separates observed IBM evidence, unconfirmed assumptions and
confirmed decisions (the last group remains empty). It includes precision,
recall and reviews/TP for all six top-k options; 115 alerts at 10% is the sum of
five technical validation folds, not a monthly workload forecast. It adds age
and MaritalStatus support/fold limitations, overlap, and human responsibility
for all personnel decisions. No demographic quota or policy rule was changed.

The fillable form covers batch membership/cutoff, actual review capacity, policy
and feature choice, rounding, stable-ID ties, JobRole mapping, access, actions,
feedback and authority/date. Source_row is an experimental tie key, not an
assumed backend employee ID. The proposed general_16/top-k 10% academic scenario
is not approved or permitted for real employees. Decision/name/date fields were
not filled by the assistant.

This refresh changes only `docs/hr_policy_decision_package_v2.md` and this page.
Saved aggregate metrics, overlap and subgroup support were cross-checked against
existing policy/fairness artifacts; protected historical hashes and all unrelated
worktree content were preserved. `git diff --check` passed. No training, audit
rerun, new threshold, artifact generation, final-test CSV access, API/backend
design/change, external messages or commit/push. The earlier pytest/Mypy/Ruff
results above are retained historical gate results, **not rerun in this docs-only
task**. Python 3.11 still lacks runtime verification.

Ready for a human decision meeting, **not yet for finalizing the data contract**:
missing HR/project-owner confirmation of real-use versus simulation scope,
batch/capacity, policy/feature preference, tie rules and alert-handling workflow.
