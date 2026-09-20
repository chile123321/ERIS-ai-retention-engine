"""Audit frozen development-only capacity-15 prediction policies by subgroup."""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from eris_ml.data.fairness_recovery import (
    load_manifest_ids,
    validate_development_fairness_frame,
)
from eris_ml.data.splitting import assert_development_path
from eris_ml.evaluation.fairness import join_audit_inputs, run_fairness_audit, validate_protocol
from eris_ml.evaluation.fairness_reporting import (
    plot_fairness_dashboard,
    write_fairness_report,
)

ROOT = Path(__file__).resolve().parents[1]
INPUTS = {
    "development": ROOT / "data/processed/development_raw_v1.csv",
    "fairness_data": ROOT / "data/processed/fairness_audit_development_v1.csv",
    "calibration_oof": ROOT / "artifacts/predictions/calibration_oof_v1.csv",
    "threshold_policy_oof": ROOT / "artifacts/predictions/threshold_policy_oof_v1.csv",
    "threshold_candidates": ROOT / "configs/threshold/threshold_candidates_v1.yaml",
    "fairness_protocol": ROOT / "configs/evaluation/fairness_protocol_v1.yaml",
}
OUTPUTS = {
    "report": ROOT / "artifacts/reports/fairness_audit_v1.md",
    "dashboard": ROOT / "artifacts/reports/fairness_dashboard_v1.png",
    "metrics": ROOT / "artifacts/reports/fairness_metrics_v1.csv",
    "predictions": ROOT / "artifacts/predictions/fairness_policy_oof_v1.csv",
}
MANIFEST = ROOT / "data/processed/split_manifest_v1.csv"


def parse_args() -> argparse.Namespace:
    """Require all approved Step 10D paths explicitly for auditability."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (*INPUTS, *OUTPUTS):
        parser.add_argument(f"--{name.replace('_', '-')}", type=Path, required=True)
    return parser.parse_args()


def validate_paths(args: argparse.Namespace) -> None:
    """Reject locked/quarantine paths before touching their metadata or opening files."""
    for name, expected in {**INPUTS, **OUTPUTS}.items():
        path: Path = getattr(args, name)
        normalized = os.path.normcase(os.path.abspath(path)).replace("\\", "/")
        if "/data/quarantine/" in normalized.lower():
            raise ValueError("Quarantined fairness data is prohibited.")
        assert_development_path(path)
        if normalized != os.path.normcase(os.path.abspath(expected)).replace("\\", "/"):
            raise ValueError(f"Only the approved {name} path is permitted.")
    for path in (*INPUTS.values(), MANIFEST):
        if not path.is_file():
            raise FileNotFoundError(f"Required development input is missing: {path}")


def validate_candidates(config: dict[str, object], working_threshold: float) -> None:
    """Check the unchanged unapproved Step 10C working candidate."""
    if (config.get("model") != "xgboost_tuned_raw"
        or config.get("business_approved") is not False
        or config.get("production_approved") is not False
        or config.get("final_test_evaluated") is not False
        or config.get("recommended_provisional_candidate") != "capacity_15_percent"):
        raise ValueError("Threshold candidate status or recommendation differs from Step 10C.")
    candidates = config.get("candidates")
    if not isinstance(candidates, list):
        raise ValueError("Threshold candidates are missing.")
    matches = [row for row in candidates
               if isinstance(row, dict) and row.get("scenario") == "capacity_15_percent"]
    if len(matches) != 1 or not np.isclose(
        float(matches[0]["threshold"]), working_threshold, rtol=0, atol=5e-7
    ):
        raise ValueError("Working threshold differs from frozen capacity-15 candidate.")


def main() -> None:
    args = parse_args()
    validate_paths(args)
    with args.fairness_protocol.open(encoding="utf-8") as stream:
        protocol = validate_protocol(yaml.safe_load(stream))
    with args.threshold_candidates.open(encoding="utf-8") as stream:
        validate_candidates(yaml.safe_load(stream), protocol["working_threshold"])
    development = pd.read_csv(args.development,
                              usecols=["source_row", "EmployeeNumber", "Age", "Attrition"])
    fairness = pd.read_csv(args.fairness_data)
    manifest_development, final_source_ids, final_employee_ids = load_manifest_ids(MANIFEST)
    expected = {(int(row.source_row), int(row.EmployeeNumber)): int(row.Attrition)
                for row in development.itertuples(index=False)}
    if len(expected) != len(development) or expected != manifest_development:
        raise ValueError("Development IDs or target differ from the split manifest.")
    validate_development_fairness_frame(
        fairness, development, forbidden_source_rows=final_source_ids,
        forbidden_employee_numbers=final_employee_ids,
    )
    calibration = pd.read_csv(
        args.calibration_oof,
        usecols=["source_row", "EmployeeNumber", "Attrition", "outer_fold",
                 "xgboost_raw_probability", "logistic_raw_probability"],
    )
    policy = pd.read_csv(
        args.threshold_policy_oof,
        usecols=["source_row", "EmployeeNumber", "Attrition", "outer_fold",
                 "xgboost_raw_probability", "logistic_raw_probability",
                 "xgboost_capacity_15_percent_threshold",
                 "xgboost_capacity_15_percent_prediction",
                 "logistic_capacity_15_percent_threshold",
                 "logistic_capacity_15_percent_prediction"],
    )
    frame = join_audit_inputs(
        development, fairness, calibration, policy,
        forbidden_source_rows=final_source_ids,
        forbidden_employee_numbers=final_employee_ids,
    )
    results = run_fairness_audit(frame, protocol)
    minimal = frame[["source_row", "EmployeeNumber", "Attrition", "outer_fold",
                     "xgboost_raw_probability", "xgboost_capacity_15_percent_prediction",
                     "Gender", "AgeGroup", "MaritalStatus", "Gender_AgeGroup"]].rename(
        columns={"xgboost_raw_probability": "working_probability",
                 "xgboost_capacity_15_percent_prediction": "working_prediction"}
    )
    minimal["working_prediction"] = minimal["working_prediction"].astype(int)
    for path in (args.report, args.dashboard, args.metrics, args.predictions):
        path.parent.mkdir(parents=True, exist_ok=True)
    results["metrics"].to_csv(args.metrics, index=False)
    minimal.to_csv(args.predictions, index=False)
    plot_fairness_dashboard(args.dashboard, results)
    write_fairness_report(args.report, frame, results, protocol, args.dashboard.name)
    print("Fairness audit completed on 1,176 development rows; no final-test metrics.")
    print("Holdout status: retained_with_protocol_deviation; business/production approval: false.")


if __name__ == "__main__":
    main()
