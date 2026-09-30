# V2 fixed-baseline model comparison — 2026-09-30

Development only: 1,176 rows, target `{0: 986, 1: 190}`, prevalence 16.156463%.
The existing `baseline_feature_ablation_v2_assignments.csv` was consumed unchanged;
no fold was generated. IDs, order, target, folds, locked development/manifest hashes,
and Logistic artifact/config/preprocessing hashes were checked before training.
Logistic probabilities were reused, not refitted. All twelve candidates have 1,176
valid OOF predictions on the same five validation folds.

## Fixed parameters, not search results

- Random Forest: `n_estimators=500`, `max_depth=None`, `min_samples_split=2`,
  `min_samples_leaf=1`, `max_features='sqrt'`, `class_weight=None`, `n_jobs=-1`,
  `random_state=42`. Unchanged `configs/models/random_forest.yaml`.
- XGBoost: `n_estimators=300`, `max_depth=3`, `learning_rate=0.05`, `subsample=0.8`,
  `colsample_bytree=0.8`, `min_child_weight=1`, `reg_lambda=1.0`,
  `scale_pos_weight=1.0`, `eval_metric='logloss'`, `tree_method='hist'`,
  `n_jobs=-1`, `random_state=42`. Unchanged `configs/models/xgboost.yaml`.
- Logistic reference: `solver='liblinear'`, `C=1`, `max_iter=2000`,
  `class_weight=None`, `random_state=42`, effective L2.
- Python 3.12.10, scikit-learn 1.9.1, XGBoost 3.4.1, NumPy 2.5.3, pandas 3.0.6.
  All constructor defaults and native XGBoost default configuration are in the
  trace/report; native defaults were inspected with synthetic data before training.
  The XGBoost `missing=NaN` sentinel is represented by the metadata string `"nan"`.

Every fold fits a fresh complete pipeline. Column projection precedes preprocessing;
general_16 receives neither Department nor JobRole. Imputation, one-hot categories,
and scaling are learned from the training partition only. No forced IBM category
mapping, class-weight search, tuning, calibration or threshold selection occurred.

## Pooled OOF results

PR-AUC means average precision. Lower Brier is better.

| Feature set | Model | PR-AUC | ROC-AUC | Brier |
|---|---|---:|---:|---:|
| full_18 | Logistic | 0.520679 | 0.792500 | 0.106081 |
| full_18 | Random Forest | 0.483639 | 0.778133 | 0.110093 |
| full_18 | XGBoost | 0.509091 | 0.772473 | 0.110168 |
| no_department_17 | Logistic | 0.521276 | 0.792692 | 0.105940 |
| no_department_17 | Random Forest | 0.490017 | 0.783888 | 0.109114 |
| no_department_17 | XGBoost | 0.511621 | 0.772793 | 0.109673 |
| no_jobrole_17 | Logistic | 0.489113 | 0.783730 | 0.109064 |
| no_jobrole_17 | Random Forest | 0.473742 | 0.764725 | 0.111783 |
| no_jobrole_17 | XGBoost | 0.484472 | 0.761487 | 0.113430 |
| general_16 | Logistic | 0.477584 | 0.775729 | 0.110733 |
| general_16 | Random Forest | 0.467778 | 0.756889 | 0.112449 |
| general_16 | XGBoost | 0.478959 | 0.750246 | 0.114108 |

## What removing features changes

Candidate minus full_18 within the same model; AP improved-fold counts are descriptive.

| Model | Removed | Delta AP | Delta ROC | Delta Brier | AP improved folds |
|---|---|---:|---:|---:|---:|
| Logistic | Department | +0.000597 | +0.000192 | -0.000140 | 2/5 |
| Logistic | JobRole | -0.031566 | -0.008770 | +0.002983 | 0/5 |
| Logistic | Both | -0.043094 | -0.016772 | +0.004652 | 0/5 |
| Random Forest | Department | +0.006378 | +0.005754 | -0.000979 | 4/5 |
| Random Forest | JobRole | -0.009897 | -0.013409 | +0.001690 | 2/5 |
| Random Forest | Both | -0.015861 | -0.021245 | +0.002356 | 2/5 |
| XGBoost | Department | +0.002530 | +0.000320 | -0.000496 | 2/5 |
| XGBoost | JobRole | -0.024619 | -0.010985 | +0.003261 | 0/5 |
| XGBoost | Both | -0.030132 | -0.022227 | +0.003939 | 1/5 |

Trees narrow the within-model AP/Brier gap between 16 and 18 features, but not the
ROC-AUC gap. This is not evidence that trees recover the lost information: their
18-feature baselines are also weaker than Logistic. On general_16, RF versus
Logistic changes AP/ROC/Brier by -0.009806/-0.018840/+0.001716; XGBoost changes
them by +0.001375/-0.025483/+0.003375. XGBoost's small AP gain is accompanied by
worse ROC-AUC and Brier, so it is not a demonstrated overall improvement.

Removing Department has small pooled gains in all models, but fold signs are mixed
(RF Brier improves in 5/5 folds). JobRole retains predictive value on this IBM
benchmark: removing it worsens pooled AP/ROC/Brier in all three models. AP and Brier
worsen in all five Logistic and XGBoost folds; RF AP worsens in 3/5 and ROC in 5/5.
This is predictive association, not causal evidence or proof of portability.

## Fold stability

All 60 fold records, metric means and sample SD (`ddof=1`), and 69 metric-delta
records with five paired differences are available in the technical artifacts.

| general_16 model | Mean AP ± SD | Mean ROC ± SD | Mean Brier ± SD |
|---|---:|---:|---:|
| Logistic | 0.497061 ± 0.078957 | 0.778366 ± 0.062846 | 0.110730 ± 0.009063 |
| Random Forest | 0.482689 ± 0.060896 | 0.757711 ± 0.038854 | 0.112445 ± 0.007463 |
| XGBoost | 0.495417 ± 0.053570 | 0.752149 ± 0.049417 | 0.114098 ± 0.009647 |

General_16 AP is strongest in fold 2 for all models. The largest AP loss from
removing both columns occurs in fold 5 for Logistic (-0.059120), fold 4 for RF
(-0.076682), and fold 2 for XGBoost (-0.100541). General_16 XGBoost beats Logistic
AP in 3/5 folds but Brier in only 1/5. Lower fold SD alone does not justify choosing
a lower-performing model. No statistical significance or equivalence is claimed.

## Decision and exactly one next step

Retain Logistic general_16 as the primary portability-motivated candidate for
further development; retain no_department_17 as the predictive comparator. The
small AP advantage of XGBoost general_16 does not offset its worse probability
error and ROC-AUC here; RF is not prioritized. Neither general_16 nor a tree model
is declared the production winner. The 17-feature comparator still needs JobRole
mapping, so its higher IBM AP is not sufficient to make it the deployment choice.

Next step: a bounded, development-only nested-CV Logistic regularization tuning
experiment for general_16 with no_department_17 as the comparator, using these
unchanged outer folds and inner-only parameter selection. Do not execute it as
part of this baseline comparison.

Cross-validation on IBM does not establish performance at new/Vietnamese employers
or validate a 3–6 month attrition horizon. Dictionary income, overtime, survey and
snapshot semantics remain unresolved for enterprise deployment. The V1 final test
was evaluated once previously; it is not a new independent V2 holdout. Its file was
not accessed and none of its scores entered this decision.

## Reproduction and artifacts

Run `.\.venv\Scripts\python.exe scripts/compare_baselines_v2.py` from the repository
root. The CLI refuses existing outputs; preserve this run and use an explicit new
version namespace for future experiments. No model is serialized.

- `artifacts/reports/baseline_model_comparison_v2.md`: detailed tables and trace.
- `artifacts/reports/baseline_model_comparison_v2_metrics.csv`: 12 pooled/summary rows.
- `artifacts/reports/baseline_model_comparison_v2_folds.csv`: 60 fold records.
- `artifacts/reports/baseline_model_comparison_v2_deltas.csv`: pooled/paired differences.
- `artifacts/reports/baseline_model_comparison_v2_trace.json`: parameters, versions, hashes.
- `artifacts/predictions/baseline_model_comparison_v2_oof.csv`: 14,112 rows including
  unchanged Logistic reference probabilities; 1,176 rows per candidate.

The first run completed training but failed while writing JSON for XGBoost's NaN
missing sentinel. Its four complete CSVs and partial trace were moved, not deleted,
to ignored `artifacts/reports/recovery/baseline_model_comparison_v2_metadata_failure/`.
The corrected run uses identical model settings; these earlier CSVs are retained
only for reproducibility checks, not as an additional model-selection experiment.
All artifacts are local/ignored. Raw data, split membership, V1, Logistic V2
artifacts and unrelated working-tree changes are preserved. No commit/push.

## Verification

- 42 relevant tests passed, including 9 new tests for saved-fold identity, corruption
  rejection before fitting, train-only learned preprocessing, exact feature projection,
  deterministic real RF/XGB on synthetic data, final-path guard and JSON metadata.
- Two complete development training runs produced identical OOF probabilities
  (maximum absolute difference 0); all four numeric output tables match.
- All input/output hashes verified, saved metrics independently recomputed from OOF,
  Logistic probabilities exactly unchanged, 1,176 valid rows for each candidate.
- Ruff on all four added Python files passed. Whole-repository Ruff still reports
  the two pre-existing E501 issues in monitoring/logging.py and test_preprocessing.py;
  these unrelated files were not changed.
- Mypy passed for all 63 source files; `git diff --check` passed. Final run warnings: none.
- The full repository test suite was not rerun; checks were targeted to this change.
