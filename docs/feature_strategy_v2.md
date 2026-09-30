# ERIS model v2 feature decision

Status: feature scope locked; fixed-baseline ablation completed on 2026-09-30.
Results and the next comparison are recorded in `docs/baseline_feature_ablation_v2.md`.

## Decision

The approved dictionary contains 18 exact IBM feature names. Four model-input
variants are registered under `configs/features/`:

| Variant | Count | Department | JobRole | Role |
|---|---:|:---:|:---:|---|
| `v2_full_18` | 18 | Yes | Yes | comparison reference |
| `v2_no_department_17` | 17 | No | Yes | ablation comparison |
| `v2_no_jobrole_17` | 17 | Yes | No | ablation comparison |
| `v2_general_16` | 16 | No | No | preferred deployment candidate |

`v2_general_16` is preferred because the IBM `Department` and `JobRole`
taxonomies are tied to one organization and do not yet have a stable,
versioned cross-employer mapping. ERIS may store both fields for HR operations,
but the 16-feature model config lists them as excluded and its preprocessing
pipeline receives neither field.

The 18 dictionary features are:

`Age`, `Department`, `DistanceFromHome`, `Education`,
`EnvironmentSatisfaction`, `JobInvolvement`, `JobLevel`, `JobRole`,
`JobSatisfaction`, `MonthlyIncome`, `OverTime`, `PerformanceRating`,
`TotalWorkingYears`, `TrainingTimesLastYear`, `YearsAtCompany`,
`YearsInCurrentRole`, `YearsSinceLastPromotion`, `YearsWithCurrManager`.

All names occur in the 35-column IBM source header and in the existing v1
25-feature pipeline. In the development split, `Department`, `JobRole`, and
`OverTime` are strings; the other 15 fields are integer-valued. The target is
exactly `Attrition`: the IBM source uses `No`/`Yes`, while the processed
development split uses integer `0`/`1` through the existing mapping.

## IBM observations are not ERIS collection rules

The workbook `ERIS_Feature_Category_Dictionary_v2.xlsx` distinguishes IBM
reference values from proposed future ERIS collection. The following items are
not semantically equivalent yet and must remain versioned assumptions:

| Area | Available in IBM/development | Proposed ERIS rule | Decision still needed |
|---|---|---|---|
| `MonthlyIncome` | Integer, development range 1,009–19,973; IBM currency/unit is undocumented. The dictionary's full-IBM reference reaches 19,999. | Monthly base income plus ISO currency and pay-period lineage at `snapshot_at`. | One comparable monetary representation; never mix raw VND/KRW/USD. |
| `OverTime` | Static `Yes`/`No`; IBM does not document the measurement window or cutoff used here. | Derive from a defined pre-snapshot window (dictionary proposes 30 days) and preserve hours/window. | HR-approved window and Yes/No threshold; missing must not become `No`. |
| Survey scores | `EnvironmentSatisfaction`, `JobInvolvement`, and `JobSatisfaction` are integer levels 1–4. | Use a fixed question set, scale, survey version/date, consent and freshness rule. | Wording, scoring direction, validity period and missing/stale handling. |
| `Education` / `JobLevel` | Integer levels 1–5 under IBM semantics. | Map verified degree and company grade/band through versioned tables. | Vietnamese degree mapping and cross-company job architecture mapping. |
| `PerformanceRating` | Schema is 1–4, but development contains only 3 and 4. | Use finalized review before snapshot with scale version. | Mapping and whether low variance makes the feature useful. |
| Snapshot timing | IBM file is a cross-sectional snapshot; no event-time lineage or validated prediction horizon is supplied. | Every feature is computed as-of an explicit `snapshot_at`; never use later events. | Define label window and operational prediction horizon before company-data training. |
| Year-based tenure | IBM values are integer years; rounding and event history are not documented. | Calculate source months as of `snapshot_at`, then apply one versioned conversion rule. | Rounding convention, verified prior experience and overlapping employment rules. |
| `TrainingTimesLastYear` | Integer count 0–6. | Count completed, de-duplicated courses in `[snapshot_at-365d, snapshot_at)`. | LMS coverage; distinguish true zero from missing integration. |
| `YearsSinceLastPromotion` | Integer years; zero is semantically ambiguous. | Store source months and `never_promoted_flag`. | Approved representation for never promoted versus just promoted. |
| `DistanceFromHome` | Integer 1–29; the original unit cannot be treated as a universal business rule. | Approximate distance in documented km without sending full address to ML. | Geocoding method, privacy controls and acceptable technical range. |

Therefore IBM results are only experimental evidence on this benchmark. They
do not show that the feature meanings transfer to Vietnamese employers, and do
not validate a 3–6 month attrition horizon.

## Locked split reuse and v1 evidence

No split is regenerated. All four variants must use the same immutable
`existing-v1` development membership and identical `StratifiedKFold` assignments:
five folds, `shuffle=true`, `random_state=42`. The pinned split-manifest SHA-256
is `d019d05d95d997dd3e6f85983cadc3eb0547c8a9946c2a4d11ab08af1b92749b`;
the development CSV SHA-256 is
`5cb4c6a3226ab5e16b355af64163e5857aac0e1637eb2fb88f91fcc9f1989754`.
Development remains 1,176 rows with target counts `{0: 986, 1: 190}`.
The unchanged manifest has 1,470 unique `source_row` and `EmployeeNumber`
values, assigns 294 rows to `final_test`, and has zero ID overlap between splits.

The prompt's statement that v1 final test has not been evaluated no longer
matches repository state. `configs/release/final_evaluation_record_v1.yaml` and
decision-log entry 0005 record one authorized academic evaluation on 294 rows
(`{0: 247, 1: 47}`), completed on 2026-09-20. This feature-lock step did not
open that CSV. The v1 result cannot select or validate any v2 variant.

## What must be rerun for v2

The split, fold protocol, target normalization, leakage-safe pipeline structure,
metric implementations, and model factories can be reused. Starting from the
four-way baseline comparison, v2 must regenerate OOF predictions and rerun model
comparison, any chosen hyperparameter tuning, calibration assessment, business
threshold selection, subgroup/fairness audit, candidate freeze, model card, and
bundle/API contract for the ultimately selected schema. V1 probabilities,
preprocessor fits, tuned parameters, calibration/threshold decisions, SHAP
values, fairness results, bundle and final-test metrics are not transferable to
the changed feature set.

No tuning, calibration, threshold selection, model fitting, or final-test access
was performed during the initial feature-lock step. The subsequent fixed-baseline
development-only run is documented separately above.
