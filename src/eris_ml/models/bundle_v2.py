"""Probability-only IBM research bundle, deliberately separate from V1 serving."""

from __future__ import annotations

import io
import json
import os
import platform
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any, cast

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

from eris_ml.evaluation.feature_ablation import GROUPS
from eris_ml.features.definitions import FeatureDefinition
from eris_ml.features.preprocessing import build_preprocessor
from eris_ml.models.persistence import sidecar_paths
from eris_ml.utils.hashing import sha256_file

CANDIDATE = "logistic-general16-v2-r1"
FEATURE_SET = "v2_general_16"
FEATURE_GROUPS = {kind: [name for name in names if name not in {"Department", "JobRole"}]
                  for kind, names in GROUPS.items()}
FEATURE_ORDER = [name for names in FEATURE_GROUPS.values() for name in names]
FORBIDDEN = {"Department", "JobRole", "Gender", "MaritalStatus", "EmployeeNumber",
             "source_row", "Attrition"}
LIMITATIONS = (
    "Research/demo only; IBM encoding and semantics, not an approved enterprise contract. "
    "Income, surveys, overtime, job grade, snapshot and missing-data mappings remain unresolved. "
    "No validated 3-6 month horizon, one-month early intervention, new-employer fairness or "
    "production performance. V1 final test was previously used and does not certify V2."
)


def runtime_versions() -> dict[str, str]:
    return {"python": platform.python_version(), **{
        name: version(name) for name in ("numpy", "pandas", "scipy", "scikit-learn", "joblib")
    }}


def input_schema() -> dict[str, Any]:
    """Local IBM boundary, not an API or approved employer data contract."""
    return {
        "version": "ibm-general16-research-input-v1",
        "feature_order": FEATURE_ORDER.copy(), "feature_groups": FEATURE_GROUPS,
        "additional_columns": "reject", "duplicate_columns": "reject",
        "missing_columns": "reject", "null_values": "reject",
        "numeric_type": "finite_integer_valued_number_no_bool_no_string",
        "column_order": "normalize_to_feature_order",
        "nominal_domains": {"OverTime": ["No", "Yes"]},
        "ordinal_domains": {
            name: ([1, 2, 3, 4, 5] if name in {"Education", "JobLevel"} else
                   [3, 4] if name == "PerformanceRating" else [1, 2, 3, 4])
            for name in FEATURE_GROUPS["ordinal"]
        },
        "domain_source": "IBM_encoding_and_observed_PerformanceRating_not_HR_approval",
        "numeric_ranges": "nonnegative_no_claim_of_enterprise_validity",
        "imputation_note": "baseline_imputers_retained_but_public_boundary_rejects_nulls",
    }


def validate_input(records: pd.DataFrame | dict[str, Any] | list[dict[str, Any]]) -> pd.DataFrame:
    if isinstance(records, pd.DataFrame):
        frame = records.copy()
    elif isinstance(records, dict):
        frame = pd.DataFrame([records])
    elif isinstance(records, list) and records and all(isinstance(x, dict) for x in records):
        frame = pd.DataFrame(records)
    else:
        raise ValueError("Input must be a nonempty DataFrame, record or list of records.")
    if frame.empty or frame.columns.has_duplicates:
        raise ValueError("Input is empty or contains duplicate columns.")
    if not all(isinstance(x, str) for x in frame.columns):
        raise ValueError("Column names must be strings.")
    missing = sorted(set(FEATURE_ORDER) - set(frame.columns))
    extra = sorted(set(frame.columns) - set(FEATURE_ORDER))
    if missing or extra:
        raise ValueError(f"Invalid columns: missing={missing}; extra={extra}.")
    frame = frame.loc[:, FEATURE_ORDER]
    schema = input_schema()
    for name in FEATURE_ORDER:
        for value in frame[name]:
            if value is None or value is pd.NA:
                raise ValueError(f"{name}: null values are unsupported at this research boundary.")
            if name == "OverTime":
                if not isinstance(value, str) or value not in schema["nominal_domains"][name]:
                    raise ValueError("OverTime: expected IBM string No or Yes.")
            elif (isinstance(value, (bool, np.bool_))
                  or not isinstance(value, (int, float, np.integer, np.floating))
                  or not np.isfinite(value) or float(value) != int(value) or value < 0):
                raise ValueError(f"{name}: expected a finite nonnegative integer-valued number.")
            elif name in schema["ordinal_domains"] and value not in schema["ordinal_domains"][name]:
                raise ValueError(f"{name}: value outside supported IBM ordinal domain.")
    return frame


def preprocessing_signature(transformer: ColumnTransformer) -> dict[str, Any]:
    """Resolve every transformer setting without embedding estimator objects in JSON."""
    def encode(value: Any) -> Any:
        if isinstance(value, BaseEstimator):
            return {"class": f"{type(value).__module__}.{type(value).__name__}",
                    "params": encode(value.get_params(deep=False))}
        if isinstance(value, dict):
            return {key: encode(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [encode(item) for item in value]
        if isinstance(value, type):
            return f"{value.__module__}.{value.__name__}"
        if isinstance(value, float) and np.isnan(value):
            return "NaN_missing_sentinel"
        return value
    return cast(dict[str, Any], encode(transformer))


def validate_parameters(parameters: dict[str, Any]) -> None:
    fixed = {"solver": "liblinear", "C": 1.0, "max_iter": 2000,
             "class_weight": None, "random_state": 42, "dual": False,
             "fit_intercept": True, "intercept_scaling": 1, "tol": .0001,
             "warm_start": False, "verbose": 0, "n_jobs": None}
    if any(parameters.get(k) != v for k, v in fixed.items()):
        raise ValueError("Logistic parameters are not the fixed unweighted baseline.")
    if (parameters.get("penalty") not in {"deprecated", "l2"}
        or parameters.get("l1_ratio", 0.0) not in (0.0, None)):
        raise ValueError("Logistic regularization must be effective L2.")


def validate_metadata(metadata: dict[str, Any]) -> None:
    expected = {
        "candidate_version": CANDIDATE, "bundle_version": "probability-bundle-v2-r1",
        "status": "research_demo_candidate", "feature_set": FEATURE_SET,
        "feature_count": 16, "feature_order": FEATURE_ORDER, "input_schema": input_schema(),
        "model_family": "logistic_regression", "probability_type": "raw",
        "calibration": "none", "effective_regularization": "L2",
        "class_order": [0, 1], "positive_class": 1, "target": "Attrition",
        "class_mapping": {"0": "No", "1": "Yes"}, "threshold": None,
        "business_approved": False, "production_approved": False,
        "enterprise_data_contract_approved": False, "trusted_artifact_only": True,
        "training_scope": "development_only", "limitations": LIMITATIONS,
    }
    if any(metadata.get(key) != value for key, value in expected.items()):
        raise ValueError("V2 bundle metadata/schema/governance mismatch.")
    validate_parameters(metadata["model_parameters"])
    if (metadata.get("library_versions") != runtime_versions()
        or not isinstance(metadata.get("provenance"), dict)
        or not metadata["provenance"].get("preflight_sha256")):
        raise ValueError("Bundle runtime versions or provenance mismatch.")


@dataclass
class LogisticResearchBundle:
    """No predict/threshold API: only raw positive-class probabilities."""

    pipeline: Pipeline
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        validate_metadata(self.metadata)
        if not isinstance(self.pipeline, Pipeline) or list(self.pipeline.named_steps) != [
            "preprocessor", "model"
        ]:
            raise ValueError("Expected one preprocessing/Logistic pipeline.")
        model = self.pipeline.named_steps["model"]
        preprocessor = self.pipeline.named_steps["preprocessor"]
        if (not isinstance(model, LogisticRegression)
            or model.get_params(deep=False) != self.metadata["model_parameters"]
            or not isinstance(preprocessor, ColumnTransformer)):
            raise ValueError("Fitted model differs from baseline parameters/type.")
        definition = FeatureDefinition(
            version=FEATURE_SET, nominal=tuple(FEATURE_GROUPS["nominal"]),
            ordinal=tuple(FEATURE_GROUPS["ordinal"]), numeric=tuple(FEATURE_GROUPS["numeric"]),
            target=("Attrition",), identifiers=("EmployeeNumber",), provenance=("source_row",),
            excluded=("Department", "JobRole", "Gender", "MaritalStatus"),
        )
        signature = preprocessing_signature(preprocessor)
        if (signature != preprocessing_signature(build_preprocessor(definition))
            or signature != self.metadata["preprocessing"]):
            raise ValueError("Preprocessing differs from the fixed baseline.")
        check_is_fitted(model)
        check_is_fitted(preprocessor)
        if (list(getattr(preprocessor, "feature_names_in_", [])) != FEATURE_ORDER
            or list(self.pipeline.feature_names_in_) != FEATURE_ORDER
            or not np.array_equal(model.classes_, [0, 1])
            or not np.isfinite(model.coef_).all() or not np.isfinite(model.intercept_).all()):
            raise ValueError("Fitted features/classes/coefficients are invalid.")

    def predict_proba(self, records: pd.DataFrame | dict[str, Any] |
                      list[dict[str, Any]]) -> np.ndarray:
        self.validate()
        frame = validate_input(records)
        matrix = np.asarray(self.pipeline.predict_proba(frame), dtype=float)
        if (matrix.shape != (len(frame), 2) or not np.isfinite(matrix).all()
            or ((matrix < 0) | (matrix > 1)).any()
            or not np.allclose(matrix.sum(axis=1), 1., rtol=1e-12, atol=1e-12)):
            raise ValueError("Pipeline returned invalid class probabilities.")
        return matrix[:, 1]


def write_exclusive(path: Path, payload: bytes) -> None:
    """Create without replacement; an interrupted partial artifact must be investigated."""
    if path.is_symlink():
        raise ValueError("Refusing symlink output.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def write_json(path: Path, value: dict[str, Any]) -> None:
    write_exclusive(path, (json.dumps(value, indent=2, ensure_ascii=False,
                                     allow_nan=False) + "\n").encode("utf-8"))


def save_bundle(bundle: LogisticResearchBundle, path: Path) -> None:
    """V1-compatible sidecar names, separate V2 type; checksum manifest written last."""
    bundle.validate()
    metadata_path, checksum_path = sidecar_paths(path)
    if any(p.exists() or p.is_symlink() for p in (path, metadata_path, checksum_path)):
        raise ValueError("Refusing existing V2 bundle/sidecar; investigate before a new version.")
    data = io.BytesIO()
    joblib.dump(bundle, data, compress=3)
    encoded = (json.dumps(bundle.metadata, indent=2, allow_nan=False) + "\n").encode("utf-8")
    write_exclusive(path, data.getvalue())
    write_exclusive(metadata_path, encoded)
    checksum = (f"{sha256_file(path)}  {path.name}\n"
                f"{sha256_file(metadata_path)}  {metadata_path.name}\n")
    write_exclusive(checksum_path, checksum.encode("ascii"))


def load_bundle(path: Path) -> LogisticResearchBundle:
    """Only for trusted local pickle/joblib: checksums are not origin authentication."""
    metadata_path, checksum_path = sidecar_paths(path)
    if any(p.is_symlink() for p in (path, metadata_path, checksum_path)):
        raise ValueError("Symlink bundle/sidecar rejected.")
    lines = checksum_path.read_text(encoding="ascii").splitlines()
    if len(lines) != 2:
        raise ValueError("Expected two checksum entries.")
    for line, item in zip(lines, (path, metadata_path), strict=True):
        if line != f"{sha256_file(item)}  {item.name}":
            raise ValueError("Bundle checksum mismatch; refusing deserialization.")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise ValueError("Invalid metadata object.")
    validate_metadata(metadata)
    bundle = joblib.load(path)
    if not isinstance(bundle, LogisticResearchBundle) or bundle.metadata != metadata:
        raise ValueError("Serialized object differs from metadata.")
    bundle.validate()
    return bundle
