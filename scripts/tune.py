"""Run development-only nested tuning for Logistic Regression and XGBoost."""

import argparse
import platform
from pathlib import Path

import pandas as pd
import sklearn
import xgboost
import yaml
from sklearn.model_selection import ParameterGrid
from train import _validate_development_frame

from eris_ml.data.loaders import load_development_data
from eris_ml.evaluation.tuning_reporting import write_tuning_report
from eris_ml.features.definitions import load_feature_definition
from eris_ml.models.factory import create_estimator
from eris_ml.models.tuning import (
    candidate_configuration,
    load_yaml_config,
    run_full_development_search,
    run_nested_cv,
    validate_search_config,
    validate_tuning_protocol,
)
from eris_ml.utils.hashing import sha256_file

PROTECTED_ARTIFACT_NAMES = {
    "baseline_cv_v1.md",
    "class_weight_experiment_v1.md",
    "tree_baseline_comparison_v1.md",
    "development_oof_v1.csv",
    "class_weight_oof_v1.csv",
    "tree_baseline_oof_v1.csv",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in (
        "development", "feature-config", "evaluation-config", "tuning-protocol",
        "logistic-baseline-config", "xgboost-baseline-config",
        "logistic-search-config", "xgboost-search-config", "report", "predictions",
        "search-results-dir",
    ):
        parser.add_argument(f"--{flag}", type=Path, required=True)
    return parser.parse_args()


def _validate_paths(args: argparse.Namespace) -> None:
    inputs = {
        Path(getattr(args, name)).resolve()
        for name in (
            "development", "feature_config", "evaluation_config", "tuning_protocol",
            "logistic_baseline_config", "xgboost_baseline_config",
            "logistic_search_config", "xgboost_search_config",
        )
    }
    for output in (args.report, args.predictions):
        if output.resolve() in inputs:
            raise ValueError(f"Output cannot overwrite an input file: {output}")
        if output.name in PROTECTED_ARTIFACT_NAMES:
            raise ValueError(f"Prior-step artifact cannot be overwritten: {output}")
    if args.report.resolve() == args.predictions.resolve():
        raise ValueError("Report and predictions must have different destinations.")


def main() -> None:
    """Validate inputs, evaluate nested OOF, then search all development rows."""
    args = _parse_args()
    _validate_paths(args)
    frame = load_development_data(args.development)
    definition = load_feature_definition(args.feature_config)
    target = _validate_development_frame(frame, definition)
    features = frame.loc[:, definition.all_features].copy()

    evaluation = load_yaml_config(args.evaluation_config)
    protocol = validate_tuning_protocol(load_yaml_config(args.tuning_protocol))
    expected_cv = evaluation.get("cross_validation", {})
    outer = protocol["outer_cv"]
    if (
        evaluation.get("metrics", {}).get("primary") != "average_precision"
        or evaluation.get("classification_threshold") != protocol["threshold"]
        or expected_cv.get("n_splits") != outer["n_splits"]
        or expected_cv.get("shuffle") != outer["shuffle"]
        or expected_cv.get("random_seed") != outer["random_state"]
    ):
        raise ValueError("Tuning protocol must match the approved evaluation baseline.")

    baseline_configs = {
        "logistic_regression": load_yaml_config(args.logistic_baseline_config),
        "xgboost": load_yaml_config(args.xgboost_baseline_config),
    }
    for family, config in baseline_configs.items():
        if config.get("model") != family:
            raise ValueError(f"Wrong baseline model in {family} config.")
        create_estimator(config)
    if baseline_configs["logistic_regression"]["parameters"]["class_weight"] is not None:
        raise ValueError("Logistic baseline must be unweighted.")
    if baseline_configs["xgboost"]["parameters"]["scale_pos_weight"] != 1.0:
        raise ValueError("XGBoost baseline must be unweighted.")

    search_configs = {
        "logistic_regression": load_yaml_config(args.logistic_search_config),
        "xgboost": load_yaml_config(args.xgboost_search_config),
    }
    for family, config in search_configs.items():
        validate_search_config(config, family)

    print("Running nested development-only evaluation...", flush=True)
    nested = run_nested_cv(
        features, target, definition, baseline_configs, search_configs, protocol,
        progress=lambda message: print(message, flush=True),
    )
    print("Nested OOF complete. Running full-development searches...", flush=True)
    full = run_full_development_search(
        features, target, definition, search_configs, protocol,
        progress=lambda message: print(message, flush=True),
    )

    metadata = {
        "development_rows": len(frame),
        "negative_count": int((target == 0).sum()),
        "positive_count": int(target.sum()),
        "feature_set_version": definition.version,
        "feature_count": len(definition.all_features),
        "outer_splits": outer["n_splits"],
        "inner_splits": protocol["inner_cv"]["n_splits"],
        "outer_seed": outer["random_state"],
        "inner_seed": protocol["inner_cv"]["random_state"],
        "threshold": protocol["threshold"],
        "development_sha256": sha256_file(args.development),
        "python_version": platform.python_version(),
        "pandas_version": pd.__version__,
        "sklearn_version": sklearn.__version__,
        "xgboost_version": xgboost.__version__,
        "logistic_combinations": len(
            ParameterGrid(search_configs["logistic_regression"]["search_space"])
        ),
    }
    predictions = pd.DataFrame(
        {
            "source_row": frame["source_row"].to_numpy(),
            "EmployeeNumber": frame["EmployeeNumber"].to_numpy(),
            "Attrition": target.to_numpy(),
            "outer_fold": nested["outer_fold"],
            **{
                f"{name}_probability": probabilities
                for name, probabilities in nested["oof_probabilities"].items()
            },
        }
    )
    if len(predictions) != 1176 or not predictions["source_row"].is_unique:
        raise RuntimeError("Nested predictions failed the approved development row contract.")

    args.predictions.parent.mkdir(parents=True, exist_ok=True)
    args.search_results_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.predictions, index=False)
    for family, label in (("logistic_regression", "logistic"), ("xgboost", "xgboost")):
        rows = nested["search_results"][family] + full[family]["search_results"]
        pd.DataFrame(rows).to_csv(
            args.search_results_dir / f"{label}_search_results_v1.csv", index=False
        )
        candidate = candidate_configuration(
            family, search_configs[family], full[family]["best_parameters"]
        )
        destination = Path("configs/models") / f"{family}_tuned_v1.yaml"
        destination.write_text(yaml.safe_dump(candidate, sort_keys=False), encoding="utf-8")
    decision = write_tuning_report(nested, full, search_configs, metadata, args.report)
    for name, metrics in nested["oof_metrics"].items():
        print(f"{name} OOF PR-AUC: {metrics['average_precision']:.6f}")
    print(f"Champion: {decision['champion']}")
    print(f"Challenger: {decision['challenger']}")
    print(f"Report: {args.report}")
    print(f"OOF predictions: {args.predictions}")


if __name__ == "__main__":
    main()
