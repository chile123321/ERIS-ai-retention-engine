# V2 Logistic nested-CV tuning — development decision

## Protocol frozen before training

The run uses the unchanged 1,176 IBM development rows (986 negatives, 190 positives)
and `baseline_feature_ablation_v2_assignments.csv`. All input fingerprints and
baseline IDs/targets/folds matched before training. No new outer split was made.

Both general_16 and no_department_17 use the unchanged V1 grid:
`C = [0.001, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]`, penalty `[l1, l2]`.
Fixed: liblinear, random_state=42, class_weight=None, max_iter=3000, other constructor
defaults recorded in trace. Baseline used max_iter=2000; no convergence warning
occurred in this run. Inner CV: 4 stratified folds, shuffle=True, random_state=43,
created only from outer-training rows and shared between both feature sets.

Selection maximizes mean inner average precision. Within absolute 1e-12 of the
maximum, choose smaller C, then L2. The complete grid, tie rule, versions, defaults,
features and hashes were written to an immutable preflight JSON before the first
fit. The grid was not changed. There were 800 inner fits plus 10 outer-training
refits, no full-development search and no single final tuned configuration chosen.

All preprocessing remains in each pipeline. General_16 receives neither Department
nor JobRole. No outer-validation row participates in inner selection or any fit.

## Pooled OOF results

Each candidate has 1,176 OOF rows. Baseline probabilities were reused unchanged.
AP is average precision, not trapezoidal PR area; lower Brier is better.

| Feature set | Procedure | AP | ROC-AUC | Brier |
|---|---|---:|---:|---:|
| general_16 | Baseline | 0.477584 | 0.775729 | 0.110733 |
| general_16 | Nested tuned | 0.486468 | 0.737926 | 0.173905 |
| no_department_17 | Baseline | 0.521276 | 0.792692 | 0.105940 |
| no_department_17 | Nested tuned | 0.471179 | 0.753480 | 0.122428 |

| Comparison | Delta AP | Delta ROC | Delta Brier | Improved folds AP / ROC / Brier |
|---|---:|---:|---:|---|
| general_16 tuned minus baseline | +0.008884 | -0.037803 | +0.063172 | 4 / 2 / 0 |
| no_department_17 tuned minus baseline | -0.050097 | -0.039212 | +0.016488 | 4 / 3 / 0 |
| general_16 tuned minus no_department_17 tuned | +0.015289 | -0.015555 | +0.051477 | 0 / 1 / 0 |

The positive pooled AP difference in the last row does NOT mean general_16 is
consistently better: its AP is lower in every paired outer fold. Fold-specific
score scales differ, so pooled ranking can disagree with within-fold ranking.

## Hyperparameters and stability

| Feature set | Outer fold | C | Penalty | Inner mean AP |
|---|---:|---:|---|---:|
| general_16 | 1 | 0.001 | L2 | 0.527203 |
| general_16 | 2 | 0.001 | L2 | 0.503281 |
| general_16 | 3 | 0.001 | L2 | 0.489292 |
| general_16 | 4 | 0.010 | L2 | 0.525485 |
| general_16 | 5 | 0.001 | L2 | 0.529529 |
| no_department_17 | 1 | 0.100 | L2 | 0.542532 |
| no_department_17 | 2 | 0.100 | L2 | 0.521789 |
| no_department_17 | 3 | 0.001 | L2 | 0.511481 |
| no_department_17 | 4 | 0.100 | L2 | 0.553964 |
| no_department_17 | 5 | 0.030 | L2 | 0.558770 |

L2 wins 5/5 folds in both sets. General_16 selects C=0.001 in 4/5 folds and 0.01
once; no_department_17 spans C=0.001 to 0.1, a 100-fold range. Selection of the
lower grid boundary is reported, not used as permission to expand the grid.

| Feature set / procedure | Mean AP ± sample SD | Mean ROC ± sample SD | Mean Brier ± sample SD |
|---|---:|---:|---:|
| general_16 baseline | 0.497061 ± 0.078957 | 0.778366 ± 0.062846 | 0.110730 ± 0.009063 |
| general_16 tuned | 0.514895 ± 0.084079 | 0.769185 ± 0.062380 | 0.173892 ± 0.027082 |
| no_department_17 baseline | 0.541276 ± 0.077389 | 0.794255 ± 0.055394 | 0.105938 ± 0.009427 |
| no_department_17 tuned | 0.550783 ± 0.082188 | 0.795908 ± 0.063104 | 0.122439 ± 0.034916 |

Full baseline/tuned metrics for all 20 outer-fold records and all 200 inner search
candidate records are in the detailed report/CSVs. SD uses ddof=1. No significance
test or noninferiority margin was preregistered, and no such claim is made.

## Why pooled and fold scores disagree here

Read-only inspection of the stored OOF probabilities found:

- General_16 tuned mean probabilities by fold: 0.409371, 0.404436, 0.407692,
  0.246669, 0.405489. Observed positive rates are approximately 0.161 in every fold.
- No_department_17 tuned: 0.180224, 0.167536, 0.401828, 0.182994, 0.183701.
  Fold 3 selects C=0.001 and has Brier 0.182776 versus baseline 0.105642.
- General_16 Brier is worse in 5/5 folds; no_department_17 also worsens in 5/5.

The installed scikit-learn source documents regularization of liblinear's synthetic
intercept feature (`intercept_scaling=1`). The observed probability inflation under
very strong regularization is consistent with this mechanism. AP-only selection
rewards ordering, not accurate probability levels. This explanation is not a new
experiment: no refit, grid modification, calibration or threshold was introduced.
All identity/hash checks and independent metric recomputations passed.

## Decision

Do not replace either baseline with this tuned procedure. General_16's pooled AP
gain of 0.008884 is insufficient to justify its ROC loss and large Brier increase;
no_department_17 loses all three pooled metrics. Keep tuned outputs as negative
experimental evidence, not production candidates.

Retain Logistic baseline general_16 as the portability-motivated working candidate
and baseline no_department_17 as the JobRole comparator. On IBM, removing JobRole
from that baseline costs AP 0.043691, ROC-AUC 0.016964 and adds Brier 0.004793.
Avoiding organization-specific mapping is a practical motivation, not proof of
cross-company performance. IBM does not validate Vietnamese employers or a 3–6
month prediction horizon. Repeated development comparisons can introduce selection
optimism; the already-used V1 final test is not a new independent V2 holdout.

Exactly one next step: cross-fitted calibration assessment of the two retained
Logistic baselines (general_16 and no_department_17), using the saved outer folds
and calibration fit only inside outer-training. Do not run that step here.

## Files, reproduction and checks

Run `.\.venv\Scripts\python.exe scripts/tune_logistic_v2.py`. Existing outputs,
including preflight, are refused. A new experiment requires an explicit version;
the current run must not be overwritten.

- `configs/tuning/logistic_nested_v2.yaml`: frozen protocol referencing unchanged V1 grid.
- `src/eris_ml/models/logistic_nested_v2.py`: nested search, validation and paired metrics.
- `src/eris_ml/evaluation/logistic_nested_reporting.py`: report generation.
- `scripts/tune_logistic_v2.py`: guarded preflight, immutable trace, execution and outputs.
- `tests/integration/test_logistic_nested_v2.py`: seven tests including parametrized cases.
- `artifacts/reports/logistic_nested_v2.md`: detailed report with full trace.
- `artifacts/reports/logistic_nested_v2_{preflight,trace}.json`: frozen/run provenance.
- `artifacts/reports/logistic_nested_v2_{metrics,folds,parameters,search,deltas}.csv`.
- `artifacts/predictions/logistic_nested_v2_oof.csv`: 4,704 rows, including baseline references;
  exactly 1,176 rows for each of the two tuned candidates.
- `artifacts/predictions/logistic_nested_v2_inner_folds.csv`: 4,704 row memberships,
  shared across feature sets and containing only the relevant outer-training rows.

Checks: 41 relevant tests passed (including all seven new cases), changed-file Ruff
passed, mypy passed for 65 source files. Whole-repository Ruff retains two pre-existing
E501 errors in monitoring/logging.py and test_preprocessing.py; not changed here.
The full suite was not rerun. Input/output hashes, baseline identity, inner coverage,
OOF validity and independently recomputed metrics all passed.

Versions: Python 3.12.10, scikit-learn 1.9.1, NumPy 2.5.3, pandas 3.0.6. Warnings:
810 penalty-deprecation FutureWarnings and 400 L1/default-l1_ratio UserWarnings.
The installed source still honors explicit penalty in this API; the L1 candidates
were genuinely L1. No convergence warning occurred. Warning messages/counts are
preserved in the trace; no out-of-scope API migration was made.

No final-test access/scores, tree tuning, calibration, threshold selection, model
serialization, API changes, commit or push. Raw/development and all earlier
artifacts remain unchanged. New artifacts are local and ignored.
