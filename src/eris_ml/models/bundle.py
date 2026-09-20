"""Validated, development-trained research bundle; no serving side effects."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

from eris_ml.utils.freeze import CANDIDATE_NAME, MODEL_PARAMETERS

THRESHOLD = 0.345651


def validate_bundle_metadata(metadata: dict[str, Any]) -> None:
    """Reject model, feature, threshold or governance drift at load time."""
    features = metadata.get("feature_order")
    types = metadata.get("feature_types")
    if (metadata.get("candidate_version") != CANDIDATE_NAME
        or metadata.get("bundle_version") != "bundle-v1"
        or metadata.get("feature_schema_version") != "v1-full"
        or metadata.get("feature_count") != 25
        or not isinstance(features, list) or len(features) != 25
        or len(set(features)) != 25
        or not isinstance(types, dict) or set(types) != set(features)
        or set(types.values()) != {"nominal", "ordinal", "numeric"}
        or metadata.get("threshold") != THRESHOLD
        or metadata.get("calibration") != "none"
        or metadata.get("probability_type") != "raw"
        or metadata.get("model_family") != "xgboost"
        or metadata.get("model_parameters") != MODEL_PARAMETERS
        or metadata.get("fairness_status") != "review_known_limitation"
        or metadata.get("holdout_status") != "retained_with_protocol_deviation"
        or metadata.get("business_approved") is not False
        or metadata.get("production_approved") is not False
        or metadata.get("training_scope") != "development_only"
        or metadata.get("trusted_artifact_only") is not True):
        raise ValueError("Bundle metadata differs from the frozen research candidate.")
    domains = metadata.get("ordinal_domains")
    ordinal = {name for name, kind in types.items() if kind == "ordinal"}
    if not isinstance(domains, dict) or set(domains) != ordinal or any(
        not isinstance(values, list) or not values for values in domains.values()
    ):
        raise ValueError("Ordinal validation domains are missing.")
    if metadata.get("class_mapping") != {"0": "No", "1": "Yes"}:
        raise ValueError("Class mapping differs from the frozen target.")


@dataclass
class ModelBundle:
    """One fitted preprocessing/classifier pipeline and its validated schema."""

    pipeline: Pipeline
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        validate_bundle_metadata(self.metadata)
        if list(self.pipeline.named_steps) != ["preprocessor", "model"]:
            raise ValueError("Bundle must contain one fitted preprocessing/model pipeline.")
        from xgboost import XGBClassifier

        model = self.pipeline.named_steps["model"]
        if not isinstance(model, XGBClassifier) or any(
            model.get_params().get(key) != value
            for key, value in self.metadata["model_parameters"].items()
        ) or model.get_params().get("random_state") != 42:
            raise ValueError("Fitted XGBoost parameters differ from the frozen config.")
        transformer = self.pipeline.named_steps["preprocessor"]
        configured = {name: list(columns) for name, _, columns in transformer.transformers}
        types = self.metadata["feature_types"]
        expected = {kind: [name for name in self.metadata["feature_order"]
                           if types[name] == kind]
                    for kind in ("nominal", "ordinal", "numeric")}
        if configured != expected:
            raise ValueError("Preprocessing feature groups differ from the frozen schema.")
        check_is_fitted(self.pipeline)

    def validate_input(self, records: dict[str, Any] | list[dict[str, Any]] |
                       pd.DataFrame) -> pd.DataFrame:
        """Normalize field order; reject missing/extra fields and invalid scalar types."""
        if isinstance(records, pd.DataFrame):
            frame = records.copy()
        elif isinstance(records, dict):
            frame = pd.DataFrame([records])
        elif isinstance(records, list) and records and all(isinstance(x, dict) for x in records):
            frame = pd.DataFrame(records)
        else:
            raise ValueError("Input must be a record or a non-empty table of records.")
        if frame.empty or frame.columns.has_duplicates:
            raise ValueError("Input is empty or contains duplicate field names.")
        expected = set(self.metadata["feature_order"])
        missing = sorted(expected - set(frame.columns))
        extra = sorted(set(frame.columns) - expected)
        if missing or extra:
            raise ValueError(f"Invalid fields: missing={missing}; extra={extra}.")
        frame = frame.loc[:, self.metadata["feature_order"]]
        for name in self.metadata["feature_order"]:
            kind = self.metadata["feature_types"][name]
            for value in frame[name]:
                if value is None or value is pd.NA or (
                    isinstance(value, (float, np.floating)) and np.isnan(value)
                ):
                    continue  # Frozen pipeline imputes missing values.
                if kind == "nominal":
                    if not isinstance(value, str):
                        raise ValueError(
                            f"{name}: invalid value/type {value!r}; expected string or missing."
                        )
                elif (isinstance(value, (bool, np.bool_))
                      or not isinstance(value, (int, float, np.integer, np.floating))
                      or not np.isfinite(value)):
                    raise ValueError(
                        f"{name}: invalid value/type {value!r}; expected finite numeric or missing."
                    )
                elif kind == "ordinal" and value not in self.metadata["ordinal_domains"][name]:
                    raise ValueError(
                        f"{name}: invalid value {value!r}; expected one of "
                        f"{self.metadata['ordinal_domains'][name]} or missing."
                    )
        return frame

    def predict_proba(self, records: dict[str, Any] | list[dict[str, Any]] |
                      pd.DataFrame) -> np.ndarray:
        """Return uncalibrated positive-class probabilities in input order."""
        values = np.asarray(self.pipeline.predict_proba(self.validate_input(records))[:, 1],
                            dtype=float)
        if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
            raise ValueError("Fitted pipeline returned invalid probabilities.")
        return values

    def predict(self, records: dict[str, Any] | list[dict[str, Any]] |
                pd.DataFrame) -> np.ndarray:
        """Apply only the frozen attrition threshold, without risk-level cutoffs."""
        return (self.predict_proba(records) >= THRESHOLD).astype(int)

    def explain(self, records: dict[str, Any] | list[dict[str, Any]] |
                pd.DataFrame, *, top_k: int = 5) -> list[dict[str, Any]]:
        """Return original-feature SHAP factors in raw log-odds space."""
        from eris_ml.models.explainability import local_explanations

        return local_explanations(self, self.validate_input(records), top_k=top_k)
