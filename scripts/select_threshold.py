"""Evaluate raw-model threshold policies with nested development-only CV."""

import argparse
import json
import platform
from pathlib import Path

import matplotlib
import pandas as pd
import sklearn
import xgboost
import yaml
from train import _validate_development_frame

from eris_ml.data.loaders import load_development_data
from eris_ml.data.splitting import assert_development_path
from eris_ml.evaluation.threshold_reporting import (
    candidate_configuration,
    plot_tradeoff,
    recommend_candidates,
    write_threshold_report,
)
from eris_ml.evaluation.thresholding import (
    run_cross_fitted_policies,
    validate_calibration_oof,
    validate_policy,
)
from eris_ml.features.definitions import load_feature_definition
from eris_ml.models.calibration import load_outer_best_parameters
from eris_ml.models.tuning import load_yaml_config
from eris_ml.utils.hashing import sha256_file

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_DEVELOPMENT = ROOT / "data/processed/development_raw_v1.csv"
OOF_COLUMNS = [
    "source_row", "EmployeeNumber", "Attrition", "outer_fold",
    "xgboost_raw_probability", "logistic_raw_probability",
]
PROTECTED_NAMES = {
    "calibration_oof_v1.csv", "calibration_v1.md", "calibration_reliability_v1.png",
    "tuning_nested_oof_v1.csv", "xgboost_search_results_v1.csv",
    "logistic_search_results_v1.csv", "tuning_v1.md", "baseline_cv_v1.md",
    "development_oof_v1.csv", "tree_baseline_oof_v1.csv",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "development", "feature-config", "xgboost-config", "logistic-config",
        "xgboost-search-results", "calibration-oof", "threshold-policy",
        "report", "plot", "predictions", "threshold-curve", "candidate-config",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    return parser.parse_args()


def validate_paths(args: argparse.Namespace) -> None:
    """Reject final-test aliases and any overwrite of prior-step artifacts."""
    inputs = (
        args.development, args.feature_config, args.xgboost_config,
        args.logistic_config, args.xgboost_search_results, args.calibration_oof,
        args.threshold_policy,
    )
    outputs = (
        args.report, args.plot, args.predictions, args.threshold_curve,
        args.candidate_config,
    )
    for path in (*inputs, *outputs):
        assert_development_path(path)
    if args.development.resolve() != EXPECTED_DEVELOPMENT.resolve():
        raise ValueError("Only the approved development_raw_v1.csv may be loaded.")
    if args.calibration_oof.name != "calibration_oof_v1.csv" or (
        args.xgboost_search_results.name != "xgboost_search_results_v1.csv"
    ):
        raise ValueError("Approved Step 10A/10B artifact filenames are required.")
    permitted = (
        ROOT / "artifacts/reports", ROOT / "artifacts/reports",
        ROOT / "artifacts/predictions", ROOT / "artifacts/reports",
        ROOT / "configs/threshold",
    )
    for output, parent in zip(outputs, permitted, strict=True):
        if not output.resolve().is_relative_to(parent.resolve()):
            raise ValueError(f"Threshold output must stay under {parent}: {output}")
    resolved_inputs = {path.resolve() for path in inputs}
    if len({path.resolve() for path in outputs}) != len(outputs):
        raise ValueError("Threshold output paths must be distinct.")
    for output in outputs:
        if output.resolve() in resolved_inputs or output.name in PROTECTED_NAMES:
            raise ValueError(f"Cannot overwrite an input or prior-step artifact: {output}")
    for path in inputs:
        if not path.is_file():
            raise FileNotFoundError(f"Required threshold input does not exist: {path}")


def _require_raw_decisions() -> None:
    for name in (
        "xgboost_tuned_calibrated_v1.yaml", "logistic_calibrated_reference_v1.yaml"
    ):
        config = load_yaml_config(ROOT / "configs/models" / name)
        if config.get("calibration", {}).get("method") != "none":
            raise ValueError(f"Step 10B did not select raw probability for {name}.")


def main() -> None:
    args = _parse_args()
    validate_paths(args)
    _require_raw_decisions()
    frame = load_development_data(args.development)
    definition = load_feature_definition(args.feature_config)
    target = _validate_development_frame(frame, definition)
    policy = validate_policy(load_yaml_config(args.threshold_policy))
    logistic_config = load_yaml_config(args.logistic_config)
    xgboost_config = load_yaml_config(args.xgboost_config)
    try:
        calibration_oof = pd.read_csv(args.calibration_oof, usecols=OOF_COLUMNS)
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        raise ValueError("Cannot read approved raw calibration OOF columns.") from exc
    validate_calibration_oof(frame, calibration_oof)
    try:
        search = pd.read_csv(args.xgboost_search_results)
    except (OSError, pd.errors.ParserError) as exc:
        raise ValueError("Cannot read Step 10A XGBoost search results.") from exc
    fold_parameters = load_outer_best_parameters(search, 5)
    print("Running nested raw threshold-policy evaluation (no tuning/calibration)...",
          flush=True)
    results = run_cross_fitted_policies(
        frame, definition, logistic_config, xgboost_config,
        fold_parameters, calibration_oof, policy,
        progress=lambda message: print(message, flush=True),
    )
    oof = results["oof"]
    if len(oof) != 1176 or not oof["source_row"].is_unique or (
        oof["Attrition"].value_counts().sort_index().to_dict() != {0: 986, 1: 190}
    ):
        raise RuntimeError("Threshold OOF failed the approved development contract.")
    for column in oof.columns:
        if column.endswith("_prediction") and oof[column].isna().any():
            print(f"No feasible threshold in every fold for: {column}", flush=True)
    decision = recommend_candidates(results)
    metadata = {
        "negative_count": int((target == 0).sum()),
        "positive_count": int(target.sum()),
        "feature_version": definition.version,
        "feature_count": len(definition.all_features),
        "development_sha256": sha256_file(args.development),
        "calibration_oof_sha256": sha256_file(args.calibration_oof),
        "xgboost_search_sha256": sha256_file(args.xgboost_search_results),
        "xgboost_config_sha256": sha256_file(args.xgboost_config),
        "logistic_config_sha256": sha256_file(args.logistic_config),
        "policy_sha256": sha256_file(args.threshold_policy),
        "fold_parameters": json.dumps(fold_parameters, sort_keys=True),
        "python_version": platform.python_version(),
        "pandas_version": pd.__version__,
        "sklearn_version": sklearn.__version__,
        "xgboost_version": xgboost.__version__,
        "matplotlib_version": matplotlib.__version__,
    }
    args.predictions.parent.mkdir(parents=True, exist_ok=True)
    oof.to_csv(args.predictions, index=False)
    args.threshold_curve.parent.mkdir(parents=True, exist_ok=True)
    results["full_curve"].to_csv(args.threshold_curve, index=False)
    plot_tradeoff(results, args.plot)
    write_threshold_report(results, decision, metadata, args.report, args.plot)
    args.candidate_config.parent.mkdir(parents=True, exist_ok=True)
    args.candidate_config.write_text(
        yaml.safe_dump(candidate_configuration(results, decision), sort_keys=False),
        encoding="utf-8",
    )
    print(f"Provisional candidate: {decision['recommended_provisional_candidate']}")
    print(f"Report: {args.report}")
    print(f"Policy OOF: {args.predictions}")


if __name__ == "__main__":
    main()
