"""Evaluate Dummy and Logistic Regression baselines on development folds."""

import argparse
import platform
from pathlib import Path
from typing import Any

import pandas as pd
import sklearn
import yaml

from eris_ml.data.loaders import load_development_data
from eris_ml.evaluation.reporting import (
    write_class_weight_report,
    write_evaluation_report,
    write_tree_baseline_report,
)
from eris_ml.features.definitions import FeatureDefinition, load_feature_definition
from eris_ml.models.training import (
    evaluate_baselines,
    evaluate_candidates,
    normalize_binary_target,
)
from eris_ml.utils.hashing import sha256_file


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"Could not read configuration: {path}") from exc
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML configuration: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Configuration must contain a mapping: {path}")
    return value


def _validate_development_frame(
    frame: pd.DataFrame, definition: FeatureDefinition
) -> pd.Series:
    expected_columns = set(definition.all_features) | {
        "source_row",
        "EmployeeNumber",
        "Attrition",
    }
    missing = sorted(expected_columns - set(frame.columns))
    unexpected = sorted(set(frame.columns) - expected_columns)
    if frame.shape != (1176, 28):
        raise ValueError(
            f"Approved development data must have shape (1176, 28), received {frame.shape}."
        )
    if missing or unexpected:
        raise ValueError(
            f"Development columns mismatch; missing={missing}, unexpected={unexpected}."
        )
    if frame["source_row"].isna().any() or not frame["source_row"].is_unique:
        raise ValueError("source_row must be complete and unique.")
    if frame["EmployeeNumber"].isna().any() or not frame["EmployeeNumber"].is_unique:
        raise ValueError("EmployeeNumber must be complete and unique.")
    target = normalize_binary_target(frame["Attrition"])
    distribution = target.value_counts().sort_index().to_dict()
    if distribution != {0: 986, 1: 190}:
        raise ValueError(
            "Approved development target distribution must be {0: 986, 1: 190}; "
            f"received {distribution}."
        )
    return target


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate development-only Dummy and Logistic Regression baselines."
    )
    parser.add_argument("--development", type=Path, required=True)
    parser.add_argument("--feature-config", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--balanced-model-config", type=Path)
    parser.add_argument("--random-forest-config", type=Path)
    parser.add_argument("--xgboost-config", type=Path)
    parser.add_argument("--experiment", choices=["tree-baselines"])
    parser.add_argument("--evaluation-config", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    """Run approved development-only baseline evaluation and write local artifacts."""
    args = _parse_args()
    frame = load_development_data(args.development)
    definition = load_feature_definition(args.feature_config)
    target = _validate_development_frame(frame, definition)
    model_config = _load_yaml(args.model_config)
    evaluation_config = _load_yaml(args.evaluation_config)

    cv_config = evaluation_config["cross_validation"]
    threshold = float(evaluation_config["classification_threshold"])
    metadata = {
        "development_rows": len(frame),
        "negative_count": int((target == 0).sum()),
        "positive_count": int(target.sum()),
        "positive_rate": float(target.mean()),
        "development_sha256": sha256_file(args.development),
        "feature_set_version": definition.version,
        "feature_count": len(definition.all_features),
        "n_splits": cv_config["n_splits"],
        "shuffle": cv_config["shuffle"],
        "random_seed": cv_config["random_seed"],
        "threshold": threshold,
        "python_version": platform.python_version(),
        "pandas_version": pd.__version__,
        "sklearn_version": sklearn.__version__,
    }
    if args.experiment == "tree-baselines":
        required_tree_configs = {
            "--balanced-model-config": args.balanced_model_config,
            "--random-forest-config": args.random_forest_config,
            "--xgboost-config": args.xgboost_config,
        }
        missing = [name for name, value in required_tree_configs.items() if value is None]
        if missing:
            raise ValueError(
                "tree-baselines experiment requires: " + ", ".join(missing)
            )
        if args.balanced_model_config is None:
            raise AssertionError("balanced model config was validated above")
        if args.random_forest_config is None:
            raise AssertionError("Random Forest config was validated above")
        if args.xgboost_config is None:
            raise AssertionError("XGBoost config was validated above")
        balanced_config = _load_yaml(args.balanced_model_config)
        random_forest_config = _load_yaml(args.random_forest_config)
        xgboost_config = _load_yaml(args.xgboost_config)
        candidate_configs = {
            "dummy_prior": {
                "model": "dummy",
                "parameters": {"strategy": "prior"},
            },
            "logistic_regression": model_config,
            "logistic_regression_balanced": balanced_config,
            "random_forest": random_forest_config,
            "xgboost": xgboost_config,
        }
        features = frame.loc[:, definition.all_features].copy()
        results = evaluate_candidates(
            candidate_configs,
            definition,
            features,
            frame["Attrition"],
            cv_config,
            threshold,
        )
        import xgboost

        metadata["xgboost_version"] = xgboost.__version__
        write_tree_baseline_report(results, args.report, metadata, candidate_configs)
        fold_assignments = results["dummy_prior"]["fold_assignments"]
        dummy_probabilities = results["dummy_prior"]["oof_probabilities"]
    elif args.random_forest_config is not None or args.xgboost_config is not None:
        raise ValueError(
            "Tree model configs require --experiment tree-baselines."
        )
    elif args.balanced_model_config is None:
        results = evaluate_baselines(frame, definition, model_config, evaluation_config)
        write_evaluation_report(results, args.report, metadata)
        fold_assignments = results["dummy"]["fold_assignments"]
        dummy_probabilities = results["dummy"]["oof_probabilities"]
    else:
        balanced_config = _load_yaml(args.balanced_model_config)
        candidate_configs = {
            "dummy_prior": {
                "model": "dummy",
                "parameters": {"strategy": "prior"},
            },
            "logistic_regression": model_config,
            "logistic_regression_balanced": balanced_config,
        }
        features = frame.loc[:, definition.all_features].copy()
        results = evaluate_candidates(
            candidate_configs,
            definition,
            features,
            frame["Attrition"],
            cv_config,
            threshold,
        )
        write_class_weight_report(results, args.report, metadata)
        fold_assignments = results["dummy_prior"]["fold_assignments"]
        dummy_probabilities = results["dummy_prior"]["oof_probabilities"]

    predictions = pd.DataFrame(
        {
            "source_row": frame["source_row"].to_numpy(),
            "EmployeeNumber": frame["EmployeeNumber"].to_numpy(),
            "Attrition": target.to_numpy(),
            "fold": fold_assignments,
            "dummy_probability": dummy_probabilities,
            "logistic_probability": results["logistic_regression"][
                "oof_probabilities"
            ],
        }
    )
    if args.balanced_model_config is not None:
        predictions["logistic_balanced_probability"] = results[
            "logistic_regression_balanced"
        ]["oof_probabilities"]
    if args.experiment == "tree-baselines":
        predictions["random_forest_probability"] = results["random_forest"][
            "oof_probabilities"
        ]
        predictions["xgboost_probability"] = results["xgboost"]["oof_probabilities"]
    args.predictions.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.predictions, index=False)

    dummy_key = "dummy" if args.balanced_model_config is None else "dummy_prior"
    dummy_metrics = results[dummy_key]["oof_metrics"]
    logistic_metrics = results["logistic_regression"]["oof_metrics"]
    print(f"Dummy Accuracy: {dummy_metrics['accuracy']:.6f}")
    print(f"Dummy PR-AUC: {dummy_metrics['average_precision']:.6f}")
    print(f"Logistic Accuracy: {logistic_metrics['accuracy']:.6f}")
    print(f"Logistic PR-AUC: {logistic_metrics['average_precision']:.6f}")
    print(f"Logistic Recall: {logistic_metrics['recall']:.6f}")
    if args.balanced_model_config is not None:
        balanced_metrics = results["logistic_regression_balanced"]["oof_metrics"]
        print(f"Logistic balanced PR-AUC: {balanced_metrics['average_precision']:.6f}")
        print(f"Logistic balanced Recall: {balanced_metrics['recall']:.6f}")
    if args.experiment == "tree-baselines":
        random_forest_metrics = results["random_forest"]["oof_metrics"]
        xgboost_metrics = results["xgboost"]["oof_metrics"]
        print(f"Random Forest PR-AUC: {random_forest_metrics['average_precision']:.6f}")
        print(f"XGBoost PR-AUC: {xgboost_metrics['average_precision']:.6f}")
    print(f"Report path: {args.report}")
    print(f"OOF prediction path: {args.predictions}")


if __name__ == "__main__":
    main()
