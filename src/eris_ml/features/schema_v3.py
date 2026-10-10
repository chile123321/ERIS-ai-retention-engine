"""Config-driven V3 input boundary and train-only preprocessing, separate from V1/V2."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import yaml
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from eris_ml.data.splitting import assert_development_path
from eris_ml.models.factory import create_estimator


def load_protocol(path: Path) -> dict[str, Any]:
    assert_development_path(path)
    config: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("V3 protocol must be a mapping.")
    validate_protocol(config)
    return cast(dict[str, Any], config)


def validate_protocol(config: dict[str, Any]) -> None:
    features = config.get("features", [])
    if (
        config.get("version") != "v3_final_15"
        or not isinstance(features, list)
        or len(features) != 15
        or not all(isinstance(item, dict) for item in features)
    ):
        raise ValueError("Expected v3_final_15 with fifteen feature definitions.")
    names = [item.get("name") for item in features]
    if len(set(names)) != 15 or not all(
        isinstance(name, str) and name.strip() == name for name in names
    ):
        raise ValueError("Duplicate or invalid V3 feature names.")
    blocked = {config["target"]["name"], *sum(config["non_features"].values(), [])}
    if set(names) & blocked:
        raise ValueError("Target, ID or excluded feature in V3 X.")
    if sum(item.get("required") is True for item in features) != 13:
        raise ValueError("V3 requires thirteen required and two optional features.")
    for item in features:
        if type(item.get("required")) is not bool or item.get("type") not in {
            "integer",
            "categorical",
        }:
            raise ValueError("Invalid feature type/required declaration.")
        if item["type"] == "categorical":
            vocab = item.get("vocabulary", [])
            if (
                not item["required"]
                or not vocab
                or len(vocab) != len(set(vocab))
                or not all(isinstance(value, str) for value in vocab)
            ):
                raise ValueError("Invalid categorical vocabulary.")
    if (
        config["target"] != {"name": "Attrition", "mapping": {"No": 0, "Yes": 1}}
        or config["cv"]
        != {"type": "StratifiedKFold", "n_splits": 5, "shuffle": True, "random_state": 42}
        or config["execution"] != {"estimator_n_jobs": 1, "outer_parallelism": 1}
        or config["threshold_reference"] != 0.5
        or config["primary_metric"] != "average_precision"
    ):
        raise ValueError("Unapproved target/CV/execution/metric protocol.")
    if set(config["models"]) != {"dummy_prior", "logistic_regression", "random_forest", "xgboost"}:
        raise ValueError("All four baseline models are required.")
    if (
        any(model["random_seed"] != 42 for model in config["models"].values())
        or config["models"]["logistic_regression"]["parameters"]["class_weight"] is not None
    ):
        raise ValueError("Baseline seeds/weights changed.")


def validate_features(frame: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Reject unexpected columns, missing required data, invalid codes and coercion."""
    if not isinstance(frame, pd.DataFrame) or frame.empty or frame.columns.has_duplicates:
        raise ValueError("X must be a nonempty DataFrame with unique columns.")
    names = [item["name"] for item in config["features"]]
    extra = sorted(set(frame.columns) - set(names))
    if extra:
        raise ValueError(f"Unexpected/non-feature columns: {extra}.")
    result = frame.copy()
    for item in config["features"]:
        name = item["name"]
        if name not in result:
            if item["required"]:
                raise ValueError(f"{name}: missing required column.")
            result[name] = np.nan
        column = result[name]
        if item["required"] and column.isna().any():
            raise ValueError(f"{name}: required values cannot be null.")
        if item["type"] == "categorical":
            if not column.isin(item["vocabulary"]).all():
                raise ValueError(f"{name}: category outside the IBM vocabulary.")
        else:
            for value in column.dropna():
                if (
                    isinstance(value, (bool, np.bool_))
                    or not isinstance(value, (int, float, np.integer, np.floating))
                    or not np.isfinite(value)
                    or value != int(value)
                ):
                    raise ValueError(f"{name}: expected a finite integer, not a string/bool.")
                if value < item.get("minimum", 0) or (
                    "values" in item and value not in item["values"]
                ):
                    raise ValueError(f"{name}: outside declared IBM code/nonnegative domain.")
            result[name] = column.to_numpy(dtype=float, na_value=np.nan)
    return result.loc[:, names]


class V3InputValidator(TransformerMixin, BaseEstimator):
    """Runs again at predict time, adding only the explicitly optional NaN columns."""

    def __init__(self, config: dict[str, Any]):
        self.config = config

    def fit(self, X: pd.DataFrame, y: Any = None) -> V3InputValidator:
        frame = validate_features(X, self.config)
        for item in self.config["features"]:
            if not item["required"] and frame[item["name"]].isna().all():
                raise ValueError(f"{item['name']}: no training observations to estimate a median.")
        self.feature_names_in_ = np.asarray(frame.columns, dtype=object)
        self.n_features_in_ = len(frame.columns)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return validate_features(X, self.config)


def build_pipeline(config: dict[str, Any], model: str) -> Pipeline:
    """Reuse the validated V1 estimator factory; only thread count/seed are resolved here."""
    estimator = create_estimator(config["models"][model])
    parameters = estimator.get_params(deep=False)
    if model in {"random_forest", "xgboost"} and "n_jobs" in parameters:
        estimator.set_params(n_jobs=config["execution"]["estimator_n_jobs"])
    if "random_state" in parameters:
        estimator.set_params(random_state=config["models"][model]["random_seed"])
    specs = config["features"]
    categorical = [item["name"] for item in specs if item["type"] == "categorical"]
    required = [item["name"] for item in specs if item["type"] == "integer" and item["required"]]
    optional = [item["name"] for item in specs if not item["required"]]
    scale = model == "logistic_regression"
    optional_steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        optional_steps.append(("scaler", StandardScaler()))
    transformer = ColumnTransformer(
        [
            ("required_numeric", StandardScaler() if scale else "passthrough", required),
            ("optional_numeric", Pipeline(optional_steps), optional),
            (
                "categorical",
                OneHotEncoder(
                    categories=[
                        item["vocabulary"] for item in specs if item["type"] == "categorical"
                    ],
                    handle_unknown="error",
                    sparse_output=False,
                ),
                categorical,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return Pipeline(
        [
            ("validate", V3InputValidator(config)),
            ("preprocessor", transformer),
            ("model", estimator),
        ]
    )
