"""Human-readable Step 10E candidate-freeze decision report."""

from pathlib import Path
from typing import Any

from eris_ml.utils.freeze import INCIDENT_STATEMENT, validate_freeze_manifest


def write_freeze_report(path: Path, manifest: dict[str, Any]) -> None:
    """Write one report grounded in the already frozen development evidence."""
    validate_freeze_manifest(manifest)
    threshold = manifest["threshold"]
    candidate = threshold["full_development_oof_candidate"]
    cross_fitted = threshold["cross_fitted_policy_evaluation"]
    fairness = manifest["fairness"]
    rows = [
        "| Artifact | Status | SHA-256 |",
        "| --- | --- | --- |",
    ]
    for name, item in manifest["artifact_hashes"].items():
        rows.append(f"| `{name}` | {item['status']} | `{item['sha256'] or 'missing'}` |")
    libraries = ", ".join(f"{name} {version}" for name, version in
                          manifest["library_versions"].items())
    hash_count = sum(item["status"] == "present" for item in manifest["artifact_hashes"].values())
    lines = [
        "# Step 10E — Governance decision and candidate freeze", "",
        "## A. Candidate specification", "",
        f"- Candidate: `{manifest['candidate_name']}` (`{manifest['candidate_version']}`); "
        f"created {manifest['created_at_utc']}.",
        f"- Model: XGBoost tuned raw; config `{manifest['model']['config']}`. "
        "No fitted model is included.",
        f"- Feature schema: `{manifest['feature_schema']['config']}` "
        f"({manifest['feature_schema']['version']}, 25 features). Gender and MaritalStatus "
        "are audit attributes, not model features.",
        "- Preprocessing `preprocessing-v1`: nominal most-frequent imputation plus "
        "one-hot encoding with unknown values ignored; ordinal/numeric median imputation "
        "plus StandardScaler. All learned operations remain inside the model pipeline.",
        "- Calibration: none; probability type: raw.",
        f"- Frozen working threshold: `{threshold['value']:.6f}`, capacity-15 policy; "
        "not a business-approved or production threshold.",
        "- Seeds: outer CV 42; inner CV 43.",
        f"- Library versions: {libraries}.",
        "- Dataset: IBM HR Attrition public/synthetic benchmark; development split "
        "`existing-v1` (1,176 rows, 190 positives).",
        "- Raw data SHA-256: **missing from approved existing evidence**. It was not "
        "inferred from the development CSV or computed by reopening raw data. This is "
        "an explicit provenance gap, not a claimed hash match.",
        "", "## B. Development evidence", "",
        "| Evidence | PR-AUC | ROC-AUC | Brier | Alert rate | Precision | Recall |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        "| XGBoost tuned raw OOF | 0.666912 | 0.848004 | 0.088031 | — | — | — |",
        "| Full-development OOF threshold candidate | — | — | — | "
        f"{candidate['expected_alert_rate']:.4f} | "
        f"{candidate['expected_precision']:.4f} | "
        f"{candidate['expected_recall']:.4f} |",
        "| Cross-fitted capacity-15 policy evaluation | — | — | — | "
        f"{cross_fitted['alert_rate']:.4f} | "
        f"{cross_fitted['precision']:.4f} | "
        f"{cross_fitted['recall']:.4f} |",
        "", "The full-development row describes a candidate threshold applied to development "
        "OOF probabilities; the cross-fitted row evaluates thresholds selected in outer-training "
        "and applied to outer-validation. Their small metric differences are expected and must "
        "not be conflated.",
        "", "## C. Fairness governance decision", "",
        "- Governance decision: `accepted_with_known_limitations`, restricted to an academic "
        "research prototype. Fairness status: `review_known_limitation`, **not PASS**.",
        f"- Gender: `{fairness['gender']['status']}`; TPR gap 0.079, FPR gap 0.003, "
        "selection-rate ratio 0.999, TPR-gap 95% CI [0.005, 0.220]. Do not claim gender fairness.",
        f"- AgeGroup: `{fairness['age_group']['status']}`; TPR gap 0.517, FPR gap 0.112, "
        "equalized-odds gap 0.517, TPR-gap 95% CI [0.330, 0.758]. TPR is 0.784 for "
        "18–29 versus 0.267 for 50+. This is a red flag and cannot be dismissed as "
        "sample-size noise alone.",
        f"- MaritalStatus: `{fairness['marital_status']['status']}`; TPR gap 0.328.",
        f"- Gender × AgeGroup: `{fairness['intersectional']['status']}`; Female × 50+ "
        "has only 2 positive cases.",
        "- The dataset is synthetic/public, not representative of Vietnamese enterprises. "
        "Only 190 development cases are positive; some subgroups are small. Mitigation on "
        "this benchmark could overfit synthetic patterns. Re-audit on representative real "
        "company data before any deployment.",
        "- Freeze is allowed only for capstone evaluation preparation: the system offers a "
        "risk signal for human research review, never an automatic personnel decision.",
        "", "## D. Holdout protocol deviation", "",
        f"> {INCIDENT_STATEMENT}", "",
        "- `holdout_status: retained_with_protocol_deviation`; incident documented. "
        "Do not state that the final test was completely untouched.",
        "- `final_test_file_opened=false`, `final_test_metrics_computed=false`, "
        "`final_test_used_for_model_selection=false`.",
        "", "## E. Allowed and prohibited uses", "",
        "Allowed: academic capstone evaluation, prototype demonstration, decision-support "
        "research and technical integration testing.", "",
        "Prohibited: production HR deployment; automated employment or termination decisions; "
        "promotion, salary or disciplinary decisions; and claims of fairness for Vietnamese "
        "enterprises.",
        "", "## F. Freeze manifest and traceability", "",
        "- Manifest: `configs/release/candidate_v1.yaml`. Hashes below cover only the approved "
        "config, development and report files. No locked final-test or quarantined file was read "
        "or hashed.",
        "",
        *rows,
        "", f"- Hashed artifacts: {hash_count}/"
        f"{len(manifest['artifact_hashes'])}. Missing artifacts: "
        f"{manifest['missing_artifacts'] or 'none'}.",
        "- Raw dataset SHA-256 remains `missing_from_approved_evidence`; this is separately "
        "disclosed and should be supplied from a trusted prior record before stronger provenance "
        "claims are made.",
        "", "## G. Decision", "",
        "- **Step 10: COMPLETE** for development-stage candidate selection and governance record, "
        "with the disclosed raw-hash provenance gap and known fairness red flag.",
        "- Candidate frozen for academic capstone final-evaluation preparation: **YES**.",
        "- Business approved: **NO**. Production approved: **NO**. "
        "Final-test evaluation authorized: **NO**.",
        "- No retraining, tuning, calibration, threshold change, fairness mitigation or model "
        "serialization was performed in Step 10E.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
