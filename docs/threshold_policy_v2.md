# V2 cross-fitted capacity policy evaluation

## Scope and rules locked before fitting

Development only: 1,176 rows, target {0:986, 1:190}. The original V2 five-fold
assignment is unchanged. Baseline IDs/targets/folds/fingerprints and calibration
raw OOF probabilities matched before evaluation. Only fixed Logistic baseline
raw is used (liblinear, C=1, max_iter=2000, class_weight=None, random_state=42).
No tuned or calibrated model is used. Saved outer probabilities are reused without
an outer refit. Forty inner pipeline fits generate threshold-training OOF scores.

The 5%, 10%, 15% capacities are hypothetical, not HR-confirmed. All rules, parameters,
seeds, library versions and hashes were written to preflight before the first fit.

- Inner CV: four stratified folds, shuffled with seed 43, inside outer-training only.
  Each pipeline independently fits imputation, category encoding, scaling and Logistic.
- Threshold candidates: unique inner probabilities plus 0 and 1, and a finite
  `nextafter(1,+inf)` sentinel meaning no alerts. The sentinel never won in this run.
  Alert when probability >= threshold; all equal-boundary scores are included.
- Feasible inner alerts <= floor(rows*capacity_percent/100). Maximize recall, then
  precision, then threshold. Apply unchanged to outer-validation raw probabilities.
  Never adjust using outer labels or observed alert volume.
- Top-k treats each outer-validation fold as one simulated batch. k uses the same
  integer floor rule. Sort probability descending, source_row ascending for ties;
  this ID is a deterministic tie-breaker, never a model feature or target proxy.
  Select exactly k. Top-k requires the full batch and is not a common per-person cutoff.
- Reference is probability >=0.5, not a selected policy. The artifact capacity=0
  marks this unconstrained reference, not a zero-alert budget.
- Reviews per true positive = alerts/TP. For TP=0 this is undefined (blank plus
  `undefined_no_tp` status), not zero. Precision/recall/F scores use zero denominators
  safely. No invented FP/FN cost ratio or business-optimal threshold is calculated.

## Capacity scenarios: pooled evaluation over five simulated batches

Each batch has 235 or 236 rows. Per-batch top-k limits are 11/23/35, summing to
55/115/175 alerts. These equal 4.677%/9.779%/14.881% of all development rows because
rounding occurs per batch, not on pooled OOF.

| Features | Policy | Capacity | Alerts | TP | FP | FN | TN | Precision | Recall | F1 | F2 | Reviews/TP | Over-budget folds |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | threshold | 5% | 63 | 39 | 24 | 151 | 962 | 0.6190 | 0.2053 | 0.3083 | 0.2369 | 1.615 | 3/5 |
| 17 | threshold | 5% | 59 | 44 | 15 | 146 | 971 | 0.7458 | 0.2316 | 0.3534 | 0.2686 | 1.341 | 3/5 |
| 16 | threshold | 10% | 118 | 67 | 51 | 123 | 935 | 0.5678 | 0.3526 | 0.4351 | 0.3815 | 1.761 | 1/5 |
| 17 | threshold | 10% | 110 | 64 | 46 | 126 | 940 | 0.5818 | 0.3368 | 0.4267 | 0.3678 | 1.719 | 2/5 |
| 16 | threshold | 15% | 166 | 80 | 86 | 110 | 900 | 0.4819 | 0.4211 | 0.4494 | 0.4320 | 2.075 | 2/5 |
| 17 | threshold | 15% | 171 | 88 | 83 | 102 | 903 | 0.5146 | 0.4632 | 0.4875 | 0.4726 | 1.943 | 2/5 |
| 16 | top-k | 5% | 55 | 35 | 20 | 155 | 966 | 0.6364 | 0.1842 | 0.2857 | 0.2147 | 1.571 | 0/5 |
| 17 | top-k | 5% | 55 | 41 | 14 | 149 | 972 | 0.7455 | 0.2158 | 0.3347 | 0.2515 | 1.341 | 0/5 |
| 16 | top-k | 10% | 115 | 67 | 48 | 123 | 938 | 0.5826 | 0.3526 | 0.4393 | 0.3829 | 1.716 | 0/5 |
| 17 | top-k | 10% | 115 | 67 | 48 | 123 | 938 | 0.5826 | 0.3526 | 0.4393 | 0.3829 | 1.716 | 0/5 |
| 16 | top-k | 15% | 175 | 84 | 91 | 106 | 895 | 0.4800 | 0.4421 | 0.4603 | 0.4492 | 2.083 | 0/5 |
| 17 | top-k | 15% | 175 | 90 | 85 | 100 | 901 | 0.5143 | 0.4737 | 0.4932 | 0.4813 | 1.944 | 0/5 |

All 70 per-fold metric records, actual alert rates, confusion counts, F1/F2 and
reviews/TP are in the detailed report and fold CSV. Each policy has 1,176 OOF rows.

## Inner-selected thresholds

These are fold-specific evaluation policies, not production thresholds.

| Features | Capacity | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 |
|---|---:|---:|---:|---:|---:|---:|
| 16 | 5% | 0.527083 | 0.489963 | 0.511423 | 0.569463 | 0.538848 |
| 16 | 10% | 0.405699 | 0.377815 | 0.415622 | 0.402348 | 0.407805 |
| 16 | 15% | 0.326223 | 0.312439 | 0.323533 | 0.331024 | 0.331115 |
| 17 | 5% | 0.581756 | 0.510550 | 0.573985 | 0.584074 | 0.570477 |
| 17 | 10% | 0.439895 | 0.425248 | 0.447551 | 0.440351 | 0.432323 |
| 17 | 15% | 0.351442 | 0.338308 | 0.347714 | 0.350201 | 0.358719 |

All inner budgets were respected. Outer alert counts by fold reveal why a threshold
is not a strict future-batch capacity policy:

| Features | Capacity | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Per-fold limit |
|---|---:|---:|---:|---:|---:|---:|---:|
| 16 | 5% | 14 | 11 | 14 | 15 | 9 | 11 |
| 16 | 10% | 22 | 22 | 20 | 34 | 20 | 23 |
| 16 | 15% | 38 | 29 | 30 | 40 | 29 | 35 |
| 17 | 5% | 12 | 11 | 12 | 14 | 10 | 11 |
| 17 | 10% | 26 | 18 | 20 | 28 | 18 | 23 |
| 17 | 15% | 34 | 36 | 33 | 41 | 27 | 35 |

Even pooled alert rates below capacity can hide per-batch overruns. For example,
16-feature 15% threshold averages 14.116% alerts but exceeds capacity in folds 1/4.
No thresholds or outputs were corrected after observing these overruns.

## Reference at 0.5

| Features | Alerts | Alert rate | TP | FP | FN | TN | Precision | Recall | F1 | F2 | Reviews/TP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | 70 | 0.059524 | 41 | 29 | 149 | 957 | 0.585714 | 0.215789 | 0.315385 | 0.246988 | 1.707 |
| 17 | 79 | 0.067177 | 54 | 25 | 136 | 961 | 0.683544 | 0.284211 | 0.401487 | 0.321812 | 1.463 |

The same numeric threshold does not imply the same workload and is not used to
claim same-capacity superiority.

## Paired comparison: the price and benefit of JobRole

For equal-workload top-k, adding JobRole yields net +6/0/+6 true positives at
5%/10%/15%. At 5%, TP changes by fold are +2/0/+2/-1/+3. At 10% they are
0/0/-1/0/+1. At 15% they are +1/+2/+1/+2/0. Thus gains are not universal.

At 10% equal totals hide different cases: each feature set uniquely identifies
12 positives missed by the other. At 5%, JobRole-only positives=12 versus
general-only=6; at 15%, 15 versus 9. These are paired observations, not significance
tests. The threshold policies give net JobRole TP changes +5/-3/+8 but also change
alert volume by -4/-8/+5; those are not equal-workload comparisons.

The 17-feature model requires mapping IBM job titles, while general_16 avoids that
model input. Its IBM advantage does not imply the same advantage at another employer.

## Discussion with HR, not production selection

The 10% top-k scenario is a useful starting point for discussion, not an optimum:
both models review 115 people and identify 67 IBM positives (recall 35.26%). At 5%,
only 35/41 positives are identified. Moving 10% to 15% adds 60 reviews and finds
17 additional positives for general_16 or 23 for the comparator, with 43/37 additional
false positives. HR must judge whether this workload and trade-off are acceptable.
Even at 15%, 106/100 positive labels are missed, so these policies are not high-recall.

If batch capacity is a hard cap, top-k enforces it by construction; thresholds do
not guarantee it under score-distribution shifts. Top-k also forces k reviews even
if absolute scores are low, and needs a defined batch and tie rule. No policy or
threshold is approved here, and there is no fabricated business cost optimization.

IBM is cross-sectional. Detected cases are positive IBM labels, not validated
departures 3–6 months after a prediction or cases known to be actionable one month
before departure. No claim of successful intervention or Vietnamese-company
generalization is made. V1 final test was evaluated previously; neither its file
nor its results were used here.

Exactly one next step: review the 5/10/15% scenarios with HR and record batch
definition, confirmed review capacity, acceptable missed/false alerts, policy type
and tie handling before freezing any policy.

## Files and validation

New implementation: configs/threshold/threshold_policy_v2.yaml,
src/eris_ml/evaluation/threshold_policy_v2.py,
src/eris_ml/evaluation/threshold_policy_v2_reporting.py,
scripts/evaluate_threshold_policy_v2.py, tests/integration/test_threshold_policy_v2.py.

Artifacts (local/ignored):

- artifacts/reports/threshold_policy_v2.md
- artifacts/reports/threshold_policy_v2_{preflight,trace}.json
- artifacts/reports/threshold_policy_v2_{metrics,folds,thresholds,stability,paired,raw_regression}.csv
- artifacts/predictions/threshold_policy_v2_oof.csv (16,464 rows: 14 policies x 1,176)
- artifacts/predictions/threshold_policy_v2_inner_oof.csv (9,408 rows: two sets x five outer trains)

The CLI refuses existing outputs. Reproduction needs a separately authorized run
namespace; do not overwrite this experiment. One initial run was retained under
ignored artifacts/reports/threshold_policy_v2_preformat_run/ before a reporting-only
formatting fix. Repeating the same frozen protocol produced identical numeric tables,
including inner probabilities and every prediction. No grid or policy rule changed.

36 relevant tests passed (including nine new cases). Tests cover inner-only fitting,
feature projection, learned preprocessing statistics, selection/tie/budget rules,
raw alignment and failure before fit, deterministic output and final-path rejection.
A label-perturbation test changes outer-fold-1 labels and verifies fold-1 thresholds
and predictions are unchanged. No full repository suite was rerun.

Input/output/preflight hashes verified, raw OOF exactly matches saved baseline,
baseline versus calibration maximum error <1e-16. Every threshold was recomputed
from saved inner OOF; every outer prediction and pooled/fold metric independently
checked. All top-k fold budgets are exact. No warning occurred. Mypy passes 69
source files; changed-file Ruff passes. Prior artifacts/raw/split/dirty worktree
changes remain preserved. No tuning, calibration, new model selection, serialization,
API edits, final-test access, commit or push.
