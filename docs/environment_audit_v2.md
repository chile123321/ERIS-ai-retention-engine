# Environment and installation audit — 2026-10-01

Scope: environment/typing/compatibility and planning only. No research experiment,
fairness audit, policy selection, V2 API work or production deployment was performed.
Synthetic tests may fit small models and serialize temporary fixtures; they do
not regenerate any saved V1/V2 experiment.

## Preservation and starting state

Branch main; HEAD 91eb94f709f5188ce2c2c5db5edc28ec9c1bd6e6.
Initial worktree: 23 tracked modified files and five untracked entries from
ongoing consistency/typing work. No reset, fetch, commit or push was performed.
The [before snapshot](quality/environment_audit_20261001_before.json) records
status, initial diff hash, old environment versions and SHA-256 of 126 protected
files. These include raw/development data, split manifest, development-only audit
input, existing non-final-test reports/predictions/bundle and YAML configurations.
Final-test CSV was excluded before file-content or metadata access.
The [after snapshot](quality/environment_audit_20261001_after.json) confirms
**126/126 protected hashes unchanged**, including the development split, all
captured prior experiment artifacts and bundle. Old .venv package versions also
remain identical to the initial snapshot.
Existing .venv was not reinstalled or removed during this clean audit.

The 72-error Mypy baseline predates this restart; the initial snapshot already
contains the in-progress NumPy 2.4.6 / SHAP 0.50 repair. Earlier full evidence:
[environment before repair](quality/typing_repair_v2_environment_before.txt),
[Mypy diagnostics before repair](quality/typing_repair_v2_mypy_before.txt).

## Causes, fixes and outcomes

| Initial failure/risk | Cause and scoped repair | Outcome |
|---|---|---|
| Original 72 Mypy errors / 32 files | 71 missing/untyped imports and one NumPy syntax blocker against target 3.11. Original NumPy 2.5.3, SciPy 1.18.1, SHAP 0.52 required 3.12. Keep Python >=3.11,<3.13; bound NumPy <2.5, SciPy <1.17, SHAP >=0.50,<0.51. | Strict gate: 0 errors / 69 files. |
| Earlier Mypy passes were misleading | Reproduced --ignore-missing-imports --python-version 3.12: pass, but not equivalent to configured gate. Source counts grew as V2 modules were added. | Final command uses neither override; strict=true and target 3.11 retained. |
| Stubs exposed 1,151 diagnostics at 42 sites / 15 files | Mostly repeated pandas scalar-union overload errors, reused variable types and imprecise estimator annotations. Add precise casts/annotations, rename locals, use public get_n_splits() for progress. | 15 modified runtime ASTs equivalent after reviewed normalization; tests pass. No model/protocol change. |
| SHAP 0.49.1 with XGBoost 3.4.1 | Intermediate repair caused 5 failing tests: vector base_score '[2.5E-1]' failed float conversion. Use SHAP 0.50.0. | V1 bundle, SHAP/additivity, API and UI tests pass. |
| Missing typing metadata | Declare five published stub packages in dev; add narrowly scoped SHAP stub with signature and invalid-argument tests. | Fresh install needs no manual stubs, blanket ignores or exclusions. |
| Clean tests secretly used old .venv | PowerShell wrappers hardcoded interpreter. Add optional -PythonExecutable; tests pass sys.executable. Default unchanged. | CLI/API agreement tested in clean interpreter; wrappers parse. |
| Upcoming sklearn removal | Historical penalty API scheduled for removal in 1.10. Add sklearn <1.10 bound; do not change historical grid. | 1.9.1 passes; warnings still visible. |
| Installation drift | Record complete clean resolution as Windows 3.12 constraints; document extras and local assets. | Plain and constrained installs plus pip check pass. |

Upstream context: [SHAP release notes](https://shap.readthedocs.io/en/stable/release_notes.html)
and [base-score issue](https://github.com/shap/shap/issues/4202).
The retained combination is supported by observed failures and regression tests.

## Clean installation and reproduction

Only Windows Python 3.12.10 is installed (py -0p). A new venv was created without
system site packages. Commands from repository root:

~~~powershell
$auditRoot = 'C:\Users\Chile\AppData\Local\Temp\eris-env-audit-e01b331ff3c6437c83a882b24bb920c2'
.\.venv\Scripts\python.exe -m venv "$auditRoot\venv"
$auditPython = "$auditRoot\venv\Scripts\python.exe"
& $auditPython -m pip install -e ".[api,ml,explain,dev]"
# After recording resolution and adding the sklearn upper bound:
& $auditPython -m pip install -e ".[api,ml,explain,dev]" -c requirements/quality-py312.txt
& $auditPython -m pip check
& $auditPython -m mypy src
& $auditPython -m pytest -v
& $auditPython -m ruff check .
git diff --check
~~~

Both installs exited 0 without resolver conflicts. pip 25.0.1 and isolated
setuptools build were used. No dependency was manually injected. The constrained
install reused the same dependency versions and reinstalled project metadata only.
A final test rerun additionally used --junitxml in the temporary audit directory.

[Clean versions](quality/environment_audit_20261001_clean.json) and
[constraints](../requirements/quality-py312.txt) include transitive dependencies,
excluding pip and the separately installed project. This is a quality/runtime
profile, not a rewrite of historical training versions.

| Component | Verified clean version |
|---|---|
| NumPy / SciPy | 2.4.6 / 1.16.3 |
| pandas / scikit-learn | 3.0.6 / 1.9.1 |
| XGBoost / SHAP | 3.4.1 / 0.50.0 |
| Numba / llvmlite | 0.68.0 / 0.50.0 |
| Pydantic / settings | 2.13.5 / 2.15.0 |
| FastAPI / Starlette / httpx | 0.142.2 / 1.7.0 / 0.28.1 |
| pytest / Mypy / Ruff | 9.1.1 / 2.3.1 / 0.16.9 |

Stubs: types-PyYAML 6.0.12.20260906, pandas-stubs 3.0.5.260914,
scikit-learn-stubs 0.0.3, scipy-stubs 1.16.3.3, joblib-stubs
1.5.3.1.20260117; local typings/shap covers the binary-XGBoost surface used.

Normalized package comparison found no old-only dependency. Ten packages resolved
to newer versions; opentelemetry-api was added transitively. Differences are in
the [read-only audit](quality/environment_audit_20261001_readonly.json).
Previously hand-installed stubs are now declared in pyproject.

### Python 3.11: resolve only

~~~powershell
& $auditPython -m pip install --dry-run --ignore-installed --python-version 3.11 --only-binary=:all: ".[api,ml,explain,dev]" --report "$auditRoot\resolve311-final.json"
~~~

The [all-extras resolution](quality/environment_audit_20261001_resolve311.json)
succeeds for Windows cp311, selecting XGBoost 3.2.0 and
contourpy 1.3.3 (versus 3.4.1 and 1.4.0 on 3.12). A previous attempt to force the
3.12 XGBoost 3.4.1 pin onto 3.11 failed. Do not apply the 3.12 lock to 3.11;
quality-common.txt is only a partial core constraint set.

There is no Python 3.11 executable: its pytest, imports, bundle loading and runtime
Mypy execution were NOT run. Cross-version resolution verifies wheels and
Requires-Python, not every marker/runtime behavior. Declared support is retained
with this gap explicit. The verified 3.12 Mypy process targets 3.11 syntax.

## Quality gate and warnings

| Clean-environment check | Actual result |
|---|---|
| Full pytest -v | 224 passed, 1 skipped, 0 failed; 225 collected, 54 warnings. First clean run: 199.04 seconds; final constrained rerun: 146.62 seconds. |
| python -m mypy src | Exit 0, 69 files; strict enabled, target 3.11. |
| python -m ruff check . | Exit 0, all checks passed, including notebook. |
| python -m pip check | Exit 0, no broken requirements. |
| git diff --check | Exit 0; Git line-ending notices only. |
| Imports/app/trusted V1 bundle | Pass; integrity guards unchanged. |
| Synthetic API smoke | Health/ready/model-info/predict/batch 200; unauthenticated 401; invalid record 422; SHAP available. |
| Manual CLI/UI tests | Pass using clean interpreter; unchanged V1 25-feature contract. |
| V2 read-only checks | Five imports, configs/protocols, feature counts 18/17/17/16, saved ID/target/fold and raw alignment pass. No run() call. |

Old and clean environments return identical V1 probabilities for synthetic
low/medium/high records: 0.0013171798782423139, 0.3134416341781616,
0.9948952794075012. This is smoke evidence, not statistical model evaluation.

Existing skip: tests/unit/test_data_contract.py:6, authoritative HR columns
unapproved. No new skip, typing exclusion or broad warning suppression.

- Historical grid: 25 penalty FutureWarnings and 25 L1/l1_ratio UserWarnings in
  synthetic V1 tuning tests. Preserve grid; bound versions pending scoped migration.
- Upcoming API: one Starlette warning recommending httpx2. Current httpx passes;
  migrate deliberately before unpinning.
- Upstream plotting: three SHAP/matplotlib colormap deprecations; additivity passes.
- The SHAP/base_score exception was fixed, not suppressed.

## Read-only behavior and provenance review

Imported baseline_ablation_v2, compare_baselines_v2, tune_logistic_v2,
calibrate_v2 and evaluate_threshold_policy_v2 via runpy.run_path with a non-main
name. Called config validators, validate_reference and validate_inputs separately
on development and saved OOF. No training, selection, writer or write-producing
script preflight was called.

All 44 declared outputs exist locally; all 32 recorded output hashes match.
Development and saved assignment each contain 1,176 rows; target is 986/190.
Loaded baseline/calibration raw OOF probabilities align exactly.

83/109 historical input hashes match current files. The other 26 entries concern
11 distinct files: registry metadata and 10 typing-edited source files.
Read-only JSON maps old/current hashes and traces. Historical CLIs still reject
those fingerprints and existing outputs: do not rewrite traces or suppress guards.
The next audit must explicitly record source revision mapping and independently
validate unchanged data/config/OOF hashes.

All 15 edited ML/reporting sources match HEAD after runtime AST normalization of
reviewed casts/annotations, imports, local renames and get_n_splits()/n_splits.
This supports source equivalence, not reproduction across library versions.
No development model was refitted. Notebook code also matches after reviewed
import order and explicit default zip(strict=False); outputs, execution counts,
metadata and markdown are unchanged.

## Limits and continuation

Ignored data, OOF and V1 bundle/sidecars are present locally; a Git-only clone
requires authorized copies for asset-dependent tests. Python installation does
not supply those assets. No missing local asset blocked this audit.

No final-test CSV content, metadata or hash was accessed. V1 used it once; it is
not a fresh V2 holdout. IBM does not establish performance at new employers or a
3–6 month prediction horizon. The [ordered V2 plan](v2_status.md#ordered-continuation-plan)
is planning only, not HR approval or production readiness.

## Files and final worktree

This clean-audit pass changed README.md, pyproject.toml (sklearn upper bound),
requirements/quality-common.txt and quality-py312.txt, docs/v2_status.md,
scripts/manual_model_test.ps1, scripts/start_manual_model_ui.ps1 and
tests/integration/test_manual_model_cli.py. It added this report and five
environment_audit_20261001 JSON evidence files under docs/quality/.
Other existing source typing edits, local SHAP stubs, typing tests and previous
documentation/notebook changes were reviewed and preserved, not reset.

Final branch and HEAD unchanged; 26 modified tracked files and six untracked
directory/file entries, no staged changes. The clean temporary venv remains for
inspection; no old environment or artifact was deleted.
