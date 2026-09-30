# V2 Logistic baseline calibration

## Scope and implementation

Only the two fixed Logistic baselines were calibrated: general_16 and
no_department_17. No estimator, selected parameter or probability from
logistic_nested_v2 was used. Development remains 1,176 rows, target {0:986, 1:190}.
Saved outer assignment and all baseline/data/config fingerprints passed preflight.

Logistic is unchanged: liblinear, C=1, max_iter=2000, class_weight=None,
random_state=42, effective L2. Complete constructor defaults and library versions
are in the immutable preflight trace. Python 3.12.10, scikit-learn 1.9.1,
NumPy 2.5.3, pandas 3.0.6 were used.

For each outer fold, raw fits a fresh complete pipeline on all outer-training.
All ten raw regressions pass before the first calibrator is fitted. Sigmoid and
isotonic use CalibratedClassifierCV with ensemble=False, n_jobs=1, and four
stratified inner folds (shuffle=True, random_state=43). Calibration learns from
inner-OOF decision_function scores on outer-training only. A fresh final base
pipeline is fitted on all outer-training; its scores are transformed by the fitted
calibrator for outer-validation prediction. Unlike V1 ensemble=True, this does
not average four smaller-training estimators. There is no full-development fit.

Each imputer, category vocabulary and scaler is fitted inside the relevant pipeline.
General_16 supplies neither Department nor JobRole to any fit; no_department_17
supplies JobRole but not Department. Outer-validation is predicted once per candidate
and never used for fitting. Each of six candidates has 1,176 valid OOF probabilities.

## Pooled results

Observed positive rate is 0.161565 for all candidates. AP means average precision.

| Feature set | Method | Brier | AP | ROC-AUC | Mean probability | ECE |
|---|---|---:|---:|---:|---:|---:|
| general_16 | raw | 0.110733 | 0.477584 | 0.775729 | 0.164164 | 0.028005 |
| general_16 | sigmoid | 0.110895 | 0.479074 | 0.776273 | 0.162464 | 0.034465 |
| general_16 | isotonic | 0.110633 | 0.466568 | 0.765013 | 0.160218 | 0.018683 |
| no_department_17 | raw | 0.105940 | 0.521276 | 0.792692 | 0.164115 | 0.020961 |
| no_department_17 | sigmoid | 0.106237 | 0.522892 | 0.792965 | 0.162052 | 0.021322 |
| no_department_17 | isotonic | 0.106645 | 0.499210 | 0.785347 | 0.162061 | 0.012857 |

| Feature set | Method vs raw | Delta Brier | Delta AP | Delta ROC | Brier improved folds |
|---|---|---:|---:|---:|---:|
| general_16 | sigmoid | +0.000162 | +0.001490 | +0.000544 | 3/5 |
| general_16 | isotonic | -0.000100 | -0.011016 | -0.010716 | 3/5 |
| no_department_17 | sigmoid | +0.000297 | +0.001616 | +0.000272 | 3/5 |
| no_department_17 | isotonic | +0.000705 | -0.022066 | -0.007345 | 2/5 |

All 30 outer-fold metric rows and mean/sample SD (ddof=1) appear in the detailed
report and CSVs. Sigmoid preserves within-fold AP and ROC ordering in this run;
small pooled changes arise from fold-specific transformations. Isotonic introduces
ties and loses discrimination. Closer overall mean probability does not guarantee
better Brier or reliable predictions within each probability range.

## Reliability and support

Ten quantile bins per candidate, with tied probabilities kept together, duplicate
edges collapsed and empty bins omitted. Every candidate's bins sum to 1,176 rows
and 190 positives. The report/CSV lists count, positive count, mean predicted
probability, observed rate and absolute gap for every one of the 60 bins.

| Feature set | Method | Samples per bin | Positives per bin | Low-support bins |
|---|---|---:|---:|---:|
| general_16 | raw | 117–118 | 3–67 | 5/10 |
| general_16 | sigmoid | 117–118 | 3–66 | 4/10 |
| general_16 | isotonic | 41–226 | 2–68 | 4/10 |
| no_department_17 | raw | 117–118 | 3–67 | 5/10 |
| no_department_17 | sigmoid | 117–118 | 4–68 | 5/10 |
| no_department_17 | isotonic | 80–170 | 4–69 | 3/10 |

Low support was predefined as count<30 or positives<10. Here sparse positives
drive the warnings. These descriptive bins are not precise evidence of calibration;
ECE depends on candidate-specific bins, and lower ECE alone cannot choose a method.
No causal interpretation or calibration training on these pooled bins is made.

## Decision

- general_16: retain raw. Sigmoid passes discrimination guards but worsens Brier.
  Isotonic gains only 0.000100 Brier while losing AP 0.011016 and ROC 0.010716;
  both exceed the maximum permitted 0.005 drop. Its fold gains are also unstable.
- no_department_17: retain raw. Both methods worsen Brier; isotonic additionally
  violates both discrimination guardrails.

Before training, the protocol required pooled Brier improvement, AP/ROC loss no
greater than 0.005, at least 3/5 improved Brier folds for sigmoid or 4/5 for isotonic,
no individual fold Brier deterioration above 0.005, and positive mean Brier gain
after leaving any one fold out. These are conservative descriptive heuristics,
not statistical significance tests. No method is accepted even under the simpler
pooled Brier plus discrimination checks, so the added stability rules do not
change the ultimate decision.

No production model or feature-set winner is selected. General_16 remains the
portability-motivated candidate, with no_department_17 as comparator. IBM results
do not validate performance at new/Vietnamese employers or a 3–6 month horizon.
V1 final test was previously used once, is not an independent new V2 holdout, and
was neither accessed nor used for these choices.

Exactly one next step: business-aware cross-fitted threshold policy evaluation
for these retained raw baselines, using explicit HR capacity/cost assumptions and
inner-only policy selection. Do not run threshold selection or open final test here.

## Artifacts and verification

Run `.\.venv\Scripts\python.exe scripts/calibrate_v2.py` from the repository root.
Existing artifacts/preflight cause refusal; do not overwrite this completed run.

- `artifacts/reports/calibration_v2.md`: detailed report, all fold/reliability tables, trace.
- `artifacts/reports/calibration_v2_{preflight,trace}.json`: frozen/run provenance.
- `artifacts/reports/calibration_v2_{metrics,folds,reliability,deltas,decisions,raw_regression}.csv`.
- `artifacts/predictions/calibration_v2_oof.csv`: 7,056 rows, 1,176 per candidate.
- `artifacts/predictions/calibration_v2_inner_folds.csv`: 4,704 outer-training memberships,
  shared between feature sets and methods.

42 relevant tests passed, including eight new cases. Tests inspect inner scores
passed to the actual calibrator, train-only encoder/imputer/scaler statistics,
full-outer-training base refit, deterministic output, raw regression failure,
baseline drift, invalid probabilities, guardrail rules and final-path rejection.
The initial instrumentation test needed its wrapper name kept as decision_function
because scikit-learn selects the response method by name; no model code/protocol
change was needed for that test correction.

All input/output/preflight hashes verified; saved metrics independently recomputed;
inner memberships and reliability totals checked. Raw maximum absolute error
versus baseline: 9.8879238130678e-17 (tolerance rtol=1e-7, atol=1e-9). No warnings.
Ruff for all four new Python files passes; full-repository Ruff still has two
pre-existing E501 errors in monitoring/logging.py and test_preprocessing.py.
Mypy passes for 67 source files. Full suite was not rerun; tests were scoped.

All generated artifacts are ignored/local. V1, baseline V2, nested tuning V2,
raw/development and split membership remain unchanged. No tuning, threshold search,
model serialization, API edits, final-test access, commit or push occurred.
