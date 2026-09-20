"""Run development-only cross-fitted calibration without retuning or final-test access."""

import argparse
import json
import platform
from pathlib import Path
from typing import Any

import matplotlib
import pandas as pd
import sklearn
import xgboost
import yaml
from train import _validate_development_frame

from eris_ml.data.loaders import load_development_data
from eris_ml.data.splitting import assert_development_path
from eris_ml.evaluation.calibration_reporting import (
    candidate_configuration,
    plot_reliability,
    summarize_calibration,
    write_calibration_report,
)
from eris_ml.features.definitions import load_feature_definition
from eris_ml.models.calibration import (
    load_outer_best_parameters,
    run_cross_fitted_calibration,
    validate_calibration_protocol,
)
from eris_ml.models.tuning import load_yaml_config
from eris_ml.utils.hashing import sha256_file

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_DEVELOPMENT = ROOT / "data/processed/development_raw_v1.csv"
PROTECTED_ARTIFACT_NAMES = {
    "baseline_cv_v1.md", "class_weight_experiment_v1.md",
    "tree_baseline_comparison_v1.md", "development_oof_v1.csv",
    "class_weight_oof_v1.csv", "tree_baseline_oof_v1.csv",
    "tuning_nested_oof_v1.csv", "xgboost_search_results_v1.csv",
    "logistic_search_results_v1.csv", "tuning_v1.md",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in (
        "development", "feature-config", "logistic-config", "xgboost-config",
        "tuning-oof", "xgboost-search-results", "calibration-protocol",
        "report", "predictions", "reliability-plot",
    ):
        parser.add_argument(f"--{flag}", type=Path, required=True)
    return parser.parse_args()


def validate_paths(args: argparse.Namespace) -> None:
    """Reject final-test aliases, missing inputs, and output collisions before opening files."""
    inputs = (
        args.development, args.feature_config, args.logistic_config,
        args.xgboost_config, args.tuning_oof, args.xgboost_search_results,
        args.calibration_protocol,
    )
    outputs = (args.report, args.predictions, args.reliability_plot)
    for path in (*inputs, *outputs):
        assert_development_path(path)
    if args.development.resolve() != EXPECTED_DEVELOPMENT.resolve():
        raise ValueError("Only data/processed/development_raw_v1.csv is approved.")
    if args.tuning_oof.name != "tuning_nested_oof_v1.csv":
        raise ValueError("Step 10A OOF artifact filename must be tuning_nested_oof_v1.csv.")
    if args.xgboost_search_results.name != "xgboost_search_results_v1.csv":
        raise ValueError("Step 10A XGBoost search artifact filename is required.")
    resolved_inputs = {path.resolve() for path in inputs}
    resolved_outputs = [path.resolve() for path in outputs]
    if len(set(resolved_outputs)) != len(resolved_outputs):
        raise ValueError("Calibration outputs must have distinct destinations.")
    approved_parents = (
        ROOT / "artifacts/reports", ROOT / "artifacts/predictions",
        ROOT / "artifacts/reports",
    )
    for path, approved_parent in zip(outputs, approved_parents, strict=True):
        if not path.resolve().is_relative_to(approved_parent.resolve()):
            raise ValueError(f"Calibration output must stay under {approved_parent}: {path}")
    for path in outputs:
        if path.resolve() in resolved_inputs or path.name in PROTECTED_ARTIFACT_NAMES:
            raise ValueError(f"Output cannot overwrite an input or prior-step artifact: {path}")
    for path in inputs:
        if not path.is_file():
            raise FileNotFoundError(f"Required input does not exist: {path}")


def _load_previous(path: Path, columns: set[str]) -> pd.DataFrame:
    try:
        frame = pd.read_csv(path)
    except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        raise ValueError(f"Cannot read Step 10A artifact: {path}") from exc
    if not columns.issubset(frame.columns):
        missing = sorted(columns - set(frame.columns))
        raise ValueError(f"Step 10A artifact missing columns: {missing}.")
    return frame


def main() -> None:
    args = _parse_args()
    validate_paths(args)
    frame = load_development_data(args.development)
    definition = load_feature_definition(args.feature_config)
    target = _validate_development_frame(frame, definition)
    protocol = validate_calibration_protocol(load_yaml_config(args.calibration_protocol))
    if protocol["outer_cv"]["n_splits"] != 5 or protocol["calibration_cv"]["n_splits"] != 4:
        raise ValueError("Production calibration requires 5 outer and 4 inner folds.")
    logistic_config = load_yaml_config(args.logistic_config)
    xgboost_config = load_yaml_config(args.xgboost_config)
    previous_oof = _load_previous(
        args.tuning_oof,
        {"source_row", "EmployeeNumber", "Attrition", "outer_fold",
         "logistic_baseline_probability", "xgboost_tuned_probability"},
    )
    search_results = _load_previous(
        args.xgboost_search_results, {"outer_fold", "parameters", "rank"}
    )
    fold_parameters = load_outer_best_parameters(search_results, 5)
    print("Running nested development-only calibration (no search)...", flush=True)
    oof = run_cross_fitted_calibration(
        frame, definition, logistic_config, xgboost_config, fold_parameters,
        previous_oof, protocol, progress=lambda message: print(message, flush=True),
    )
    if len(oof) != 1176 or oof["Attrition"].value_counts().sort_index().to_dict() != {
        0: 986, 1: 190
    }:
        raise RuntimeError("Calibration OOF does not match the approved development split.")
    summary = summarize_calibration(oof, bins=protocol["calibration_bins"]["count"])
    metadata: dict[str, Any] = {
        "development_rows": len(frame),
        "negative_count": int((target == 0).sum()),
        "positive_count": int(target.sum()),
        "feature_set_version": definition.version,
        "feature_count": len(definition.all_features),
        "development_sha256": sha256_file(args.development),
        "step10a_oof_sha256": sha256_file(args.tuning_oof),
        "step10a_search_sha256": sha256_file(args.xgboost_search_results),
        "logistic_config_sha256": sha256_file(args.logistic_config),
        "xgboost_config_sha256": sha256_file(args.xgboost_config),
        "xgboost_fold_parameters": json.dumps(fold_parameters, sort_keys=True),
        "outer_splits": 5, "outer_seed": 42,
        "calibration_splits": 4, "calibration_seed": 43,
        "bins": protocol["calibration_bins"]["count"],
        "python_version": platform.python_version(),
        "pandas_version": pd.__version__,
        "sklearn_version": sklearn.__version__,
        "xgboost_version": xgboost.__version__,
        "matplotlib_version": matplotlib.__version__,
    }
    args.predictions.parent.mkdir(parents=True, exist_ok=True)
    oof.to_csv(args.predictions, index=False)
    plot_reliability(summary, args.reliability_plot)
    write_calibration_report(summary, metadata, args.report, args.reliability_plot)
    destinations = {
        "xgboost": ROOT / "configs/models/xgboost_tuned_calibrated_v1.yaml",
        "logistic": ROOT / "configs/models/logistic_calibrated_reference_v1.yaml",
    }
    for model, destination in destinations.items():
        payload = candidate_configuration(model, summary["decisions"][model]["method"])
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    for model, decision in summary["decisions"].items():
        print(f"{model} selected method: {decision['method']}")
    print(f"Report: {args.report}")
    print(f"OOF predictions: {args.predictions}")


if __name__ == "__main__":
    main()
