"""Trusted local RF candidate: raw probability and probability-space SHAP, no policy/API."""

from __future__ import annotations

import io
import json
from dataclasses import dataclass, field
from importlib.metadata import version
from pathlib import Path
from typing import Any, cast

import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from eris_ml.features.schema_v3 import validate_features, validate_protocol
from eris_ml.models.baseline_v3 import json_safe
from eris_ml.models.bundle_v2 import runtime_versions, write_exclusive
from eris_ml.utils.hashing import sha256_file

ATOL = RTOL = 1e-6
BUNDLE_FILES = {"pipeline.joblib", "metadata.json", "background.npy", "feature_mapping.json"}


def versions() -> dict[str, str]:
    return {**runtime_versions(), "shap": version("shap")}


def feature_mapping(pipeline: Pipeline) -> list[str]:
    """Map fitted output slices, never infer source names by splitting encoded labels."""
    pre = pipeline.named_steps["preprocessor"]
    names = list(pipeline.feature_names_in_)
    mapping: list[str | None] = [None] * pipeline.named_steps["model"].n_features_in_
    for name, transformer, columns in pre.transformers_:
        indices = list(range(*pre.output_indices_[name].indices(len(mapping))))
        if not indices:
            continue
        if name == "categorical":
            if (
                not isinstance(transformer, OneHotEncoder)
                or transformer.get_params()["drop"] is not None
            ):
                raise ValueError("Expected undropped fitted one-hot encoding.")
            categories = cast(list[np.ndarray], transformer.categories_)
            sources = [
                column for column, cats in zip(columns, categories, strict=True) for _ in cats
            ]
        elif name in {"required_numeric", "optional_numeric"}:
            sources = list(columns)
        else:
            raise ValueError("Unexpected fitted transformer in RF candidate.")
        for index, source in zip(indices, sources, strict=True):
            if mapping[index] is not None or source not in names:
                raise ValueError("Duplicate/unrecognized transformed-feature mapping.")
            mapping[index] = source
    if any(x is None for x in mapping) or set(mapping) != set(names):
        raise ValueError("Transformed mapping does not cover all fifteen inputs exactly.")
    return [str(x) for x in mapping]


@dataclass
class CandidateBundle:
    pipeline: Pipeline
    metadata: dict[str, Any]
    background: np.ndarray
    mapping: list[str]
    _explainer: Any = field(default=None, init=False, repr=False)

    def validate(self) -> None:
        meta = self.metadata
        config = meta["feature_config"]
        validate_protocol(config)
        names = [f["name"] for f in config["features"]]
        expected = {
            "model_family": "random_forest",
            "calibration": "none",
            "evaluation_status": "development_only",
            "final_holdout_evaluated": False,
            "production_approved": False,
            "business_approved": False,
            "status": "research_demo_candidate",
            "positive_class": 1,
            "class_order": [0, 1],
            "feature_order": names,
            "feature_set": "v3_final_15",
            "threshold": None,
            "target_mapping": {"No": 0, "Yes": 1},
            "library_versions": versions(),
            "shap_tolerance": {"atol": ATOL, "rtol": RTOL},
        }
        if any(meta.get(k) != v for k, v in expected.items()):
            raise ValueError("V3 metadata, governance, schema or runtime mismatch.")
        if list(self.pipeline.named_steps) != ["validate", "preprocessor", "model"]:
            raise ValueError("Expected a single V3 preprocessing/RF pipeline.")
        model = self.pipeline.named_steps["model"]
        if (
            not isinstance(model, RandomForestClassifier)
            or model.get_params() != meta["resolved_model_parameters"]
            or json_safe(self.pipeline.named_steps["validate"].config) != json_safe(config)
            or list(self.pipeline.feature_names_in_) != names
            or list(self.pipeline.named_steps["preprocessor"].feature_names_in_) != names
            or not np.array_equal(model.classes_, meta["class_order"])
            or self.mapping != feature_mapping(self.pipeline)
            or self.background.shape != (meta["background"]["count"], len(self.mapping))
            or not np.isfinite(self.background).all()
        ):
            raise ValueError("Fitted pipeline/classes/background/mapping differs from metadata.")

    def inputs(self, records: pd.DataFrame | list[dict[str, Any]] | dict[str, Any]) -> pd.DataFrame:
        if isinstance(records, dict):
            frame = pd.DataFrame([records])
        elif isinstance(records, pd.DataFrame):
            frame = records.copy()
        elif isinstance(records, list) and records and all(isinstance(x, dict) for x in records):
            frame = pd.DataFrame(records)
        else:
            raise ValueError("Expected a nonempty record, list of records or DataFrame.")
        return validate_features(frame, self.metadata["feature_config"])

    def probabilities(self, frame: pd.DataFrame) -> np.ndarray:
        model = self.pipeline.named_steps["model"]
        positive = list(model.classes_).index(self.metadata["positive_class"])
        matrix = np.asarray(self.pipeline.predict_proba(frame), dtype=float)
        if (
            matrix.shape != (len(frame), len(model.classes_))
            or not np.isfinite(matrix).all()
            or ((matrix < 0) | (matrix > 1)).any()
            or not np.allclose(matrix.sum(axis=1), 1, rtol=1e-12, atol=1e-12)
        ):
            raise ValueError("Invalid class probability matrix.")
        return matrix[:, positive]


def predict_batch(
    bundle: CandidateBundle, records: pd.DataFrame | list[dict[str, Any]] | dict[str, Any]
) -> list[dict[str, Any]]:
    frame = bundle.inputs(records)
    probability = bundle.probabilities(frame)
    optional = [
        f["name"] for f in bundle.metadata["feature_config"]["features"] if not f["required"]
    ]
    return [
        {
            "probability": float(p),
            "model_version": bundle.metadata["model_version"],
            "imputed_features": [name for name in optional if pd.isna(frame.iloc[i][name])],
        }
        for i, p in enumerate(probability)
    ]


def predict_one(bundle: CandidateBundle, record: dict[str, Any]) -> dict[str, Any]:
    return predict_batch(bundle, record)[0]


def explain_one(bundle: CandidateBundle, record: dict[str, Any]) -> dict[str, Any]:
    frame = bundle.inputs(record)
    result = predict_one(bundle, record)
    model = bundle.pipeline.named_steps["model"]
    transformed = np.asarray(bundle.pipeline[:-1].transform(frame), dtype=float)
    if bundle._explainer is None:
        bundle._explainer = shap.TreeExplainer(
            model,
            data=bundle.background,
            feature_perturbation="interventional",
            model_output="probability",
        )
    explanation = bundle._explainer(transformed, check_additivity=True)
    values, bases = np.asarray(explanation.values), np.asarray(explanation.base_values)
    # SHAP 0.50 multi-output RF contract: samples x transformed inputs x model.classes_.
    # Validate both class reconstructions independently before selecting mapped Attrition=1.
    if values.shape != (1, len(bundle.mapping), len(model.classes_)) or bases.shape != (1, 2):
        raise ValueError(f"Unsupported SHAP output axes: values={values.shape}, base={bases.shape}")
    matrix = np.asarray(bundle.pipeline.predict_proba(frame), dtype=float)
    if (
        not np.isfinite(values).all()
        or not np.isfinite(bases).all()
        or not np.allclose(bases + values.sum(axis=1), matrix, atol=ATOL, rtol=RTOL)
    ):
        raise ValueError("Probability-space SHAP additivity failed; tolerance is locked.")
    positive = list(model.classes_).index(bundle.metadata["positive_class"])
    contributions = {
        name: float(
            sum(values[0, i, positive] for i, source in enumerate(bundle.mapping) if source == name)
        )
        for name in bundle.metadata["feature_order"]
    }
    base = float(bases[0, positive])
    reconstruction = base + sum(contributions.values())
    if not np.isclose(reconstruction, result["probability"], atol=ATOL, rtol=RTOL):
        raise ValueError("Original-feature SHAP aggregation failed additivity.")
    optional = [
        f["name"] for f in bundle.metadata["feature_config"]["features"] if not f["required"]
    ]
    medians = (
        bundle.pipeline.named_steps["preprocessor"]
        .named_transformers_["optional_numeric"]
        .named_steps["imputer"]
        .statistics_
    )
    input_values = {k: (None if pd.isna(v) else v) for k, v in frame.iloc[0].to_dict().items()}
    factors = [{"feature": k, "contribution": v} for k, v in contributions.items()]
    return {
        **result,
        "base_value": base,
        "contributions": contributions,
        "increasing": sorted(
            [x for x in factors if x["contribution"] > 0], key=lambda x: -abs(x["contribution"])
        ),
        "decreasing": sorted(
            [x for x in factors if x["contribution"] < 0], key=lambda x: -abs(x["contribution"])
        ),
        "input_values": input_values,
        "imputed_values": {
            k: float(v)
            for k, v in zip(optional, medians, strict=True)
            if k in result["imputed_features"]
        },
        "absolute_additivity_error": abs(reconstruction - result["probability"]),
        "units": "probability; multiply signed contributions by 100 for percentage points",
        "interpretation": "Model behavior, not causality or intervention benefit.",
    }


def save_bundle(bundle: CandidateBundle, directory: Path) -> None:
    bundle.validate()
    if directory.exists() or directory.is_symlink():
        raise FileExistsError("Bundle output exists; refusing replacement.")
    buffer = io.BytesIO()
    joblib.dump(bundle.pipeline, buffer, compress=3)
    payloads = {
        "pipeline.joblib": buffer.getvalue(),
        "metadata.json": json.dumps(bundle.metadata, indent=2, allow_nan=False).encode(),
        "feature_mapping.json": json.dumps(bundle.mapping, indent=2).encode(),
    }
    background = io.BytesIO()
    np.save(background, bundle.background, allow_pickle=False)
    payloads["background.npy"] = background.getvalue()
    for name, payload in payloads.items():
        write_exclusive(directory / name, payload)
    manifest = {name: sha256_file(directory / name) for name in sorted(BUNDLE_FILES)}
    write_exclusive(directory / "manifest.json", json.dumps(manifest, indent=2).encode())


def load_bundle(directory: Path) -> CandidateBundle:
    """Trusted local joblib only: checksums detect corruption, not malicious origins."""
    if directory.is_symlink() or any(p.is_symlink() for p in directory.iterdir()):
        raise ValueError("Symlink bundle rejected.")
    manifest = json.loads((directory / "manifest.json").read_text("utf-8"))
    if set(manifest) != BUNDLE_FILES or {p.name for p in directory.iterdir()} != (
        BUNDLE_FILES | {"manifest.json"}
    ):
        raise ValueError("Unexpected bundle manifest/files.")
    for name, expected in manifest.items():
        if sha256_file(directory / name) != expected:
            raise ValueError(f"Bundle checksum mismatch: {name}")
    meta = json.loads((directory / "metadata.json").read_text("utf-8"))
    if meta["library_versions"] != versions():
        raise ValueError("Bundle dependency versions differ; refusing deserialization.")
    bundle = CandidateBundle(
        joblib.load(directory / "pipeline.joblib"),
        meta,
        np.load(directory / "background.npy", allow_pickle=False),
        json.loads((directory / "feature_mapping.json").read_text("utf-8")),
    )
    bundle.validate()
    return bundle
