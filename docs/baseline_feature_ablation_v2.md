# V2 baseline feature-ablation — 2026-09-30

Completed on the unchanged IBM development split: 1,176 rows, 986 negatives,
190 positives (16.156463%). One StratifiedKFold assignment (5 folds, shuffled,
seed 42) was generated once in the existing row order and reused for all eight
candidates. Validation sizes are 236/235/235/235/235; each fold has 38 positives.
No final-test file was accessed. Existing V1 artifacts and serving code are unchanged.

## Pooled OOF results

PR-AUC denotes average precision; each candidate has 1,176 OOF probabilities.

| Feature set | Model | PR-AUC | ROC-AUC | Brier |
|---|---|---:|---:|---:|
| v2_full_18 | Dummy prior | 0.161455 | 0.499594 | 0.135462 |
| v2_no_department_17 | Dummy prior | 0.161455 | 0.499594 | 0.135462 |
| v2_no_jobrole_17 | Dummy prior | 0.161455 | 0.499594 | 0.135462 |
| v2_general_16 | Dummy prior | 0.161455 | 0.499594 | 0.135462 |
| v2_full_18 | Logistic | 0.520679 | 0.792500 | 0.106081 |
| v2_no_department_17 | Logistic | 0.521276 | 0.792692 | 0.105940 |
| v2_no_jobrole_17 | Logistic | 0.489113 | 0.783730 | 0.109064 |
| v2_general_16 | Logistic | 0.477584 | 0.775729 | 0.110733 |

The reused V1 Logistic config is fixed: solver=liblinear, C=1.0,
max_iter=2000, class_weight=None, random_state=42. Effective regularization is
L2; tol=0.0001, fit_intercept=True, intercept_scaling=1, dual=False.
The run used scikit-learn 1.9.1, whose constructor default records
penalty="deprecated", l1_ratio=0.0. All resolved parameters and versions are
saved in the trace. The real-data run emitted no warnings.

Nominal most-frequent imputation/one-hot encoding and numeric/ordinal median
imputation/scaling are fitted within each fold. Exactly the configured columns
enter each pipeline, including Dummy. No vocabulary is supplied from validation
or from a forced enterprise-to-IBM category mapping.

## Interpretation

- Removing Department changes pooled AP by +0.000597, ROC-AUC by +0.000192
  and Brier by -0.000140: very small observed differences, with mixed fold signs.
- Removing JobRole changes AP by -0.031566, ROC-AUC by -0.008770 and Brier
  by +0.002983; AP and Brier worsen in all five folds.
- Removing both changes AP by -0.043094, ROC-AUC by -0.016772 and Brier
  by +0.004652 versus full_18; AP, ROC-AUC and Brier worsen in all five folds.
- General_16 versus no_department_17: AP -0.043691, ROC-AUC -0.016964,
  Brier +0.004793. Versus no_jobrole_17: AP -0.011528, ROC-AUC -0.008001,
  Brier +0.001669. These deltas use unrounded values.
- General_16 fold AP is 0.469765/0.624650/0.498272/0.483461/0.409160;
  mean 0.497061, sample SD 0.078957. Fold 2 is strongest; fold 5 is weakest
  and has the largest AP loss versus full_18 (-0.059120). Fold 4 has the
  largest ROC-AUC loss (-0.040208).

General_16 remains worth evaluating as a portability-motivated candidate, but
there is insufficient evidence to prefer it as the sole V2 model input on
predictive performance. Retain the 18/17-feature comparators. No significance
test or noninferiority margin was pre-registered. These are IBM CV observations,
not evidence of performance at new employers or a validated 3–6 month horizon.
No V1 final-test scores were used for selection or numerical comparison.

## Reproduction and artifacts

Run from the repository root:

```powershell
.\.venv\Scripts\python.exe scripts/baseline_ablation_v2.py
```

The script refuses to overwrite any existing output and checks the locked
development/manifest fingerprints before fitting. Preserve the present artifacts;
an explicit new version/run namespace is needed for a later experiment.

- Detailed report: `artifacts/reports/baseline_feature_ablation_v2.md`
- Pooled and fold-summary metrics: `artifacts/reports/baseline_feature_ablation_v2_metrics.csv`
- All 40 fold records: `artifacts/reports/baseline_feature_ablation_v2_folds.csv`
- Versions, parameters, input/output hashes: `artifacts/reports/baseline_feature_ablation_v2_trace.json`
- Long-form OOF (9,408 rows): `artifacts/predictions/baseline_feature_ablation_v2_oof.csv`
- Shared assignment (1,176 rows): `artifacts/predictions/baseline_feature_ablation_v2_assignments.csv`

The artifacts remain local under existing ignore rules. Tests exercise training
row isolation, fold-specific imputer/scaler/encoder statistics, feature projection,
unseen validation categories, deterministic results and OOF corruption rejection.

## One next step

Compare fixed Random Forest and XGBoost baselines across the same four feature
sets on the saved assignment, retaining these Logistic results as references.
No hyperparameter search is needed in that next comparison.
