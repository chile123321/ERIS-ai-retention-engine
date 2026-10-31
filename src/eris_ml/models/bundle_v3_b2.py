"""Local research bundle for the fixed, unweighted V3 candidate b2."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from eris_ml.models import bundle_v3 as v3
from eris_ml.utils.hashing import sha256_file

MODEL_VERSION = "rf-v3-final15-unweighted-b2"
PROBABILITY_STATUS = "development_validated_estimate_not_production"
FIXED_PARAMETERS: dict[str, Any] = {
    "n_estimators": 600,
    "max_depth": 4,
    "min_samples_leaf": 12,
    "min_samples_split": 10,
    "max_features": "sqrt",
    "class_weight": None,
    "bootstrap": True,
    "random_state": 42,
    "n_jobs": 4,
}


class B2Bundle(v3.CandidateBundle):
    def validate(self) -> None:
        meta = self.metadata
        required = {
            "model_version": MODEL_VERSION,
            "model_family": "random_forest",
            "class_weight": None,
            "calibration": "none",
            "feature_set": "v3_final_15",
            "evaluation_status": "development_remediation_only",
            "final_holdout_evaluated_for_this_version": False,
            "external_validation_required": True,
            "parent_candidate": "rf-v3-final15-b1",
            "derived_after_final_holdout_consumed": True,
            "estimated_probability_status": PROBABILITY_STATUS,
            "production_approved": False,
            "business_approved": False,
            "threshold": None,
        }
        if any(meta.get(key) != value for key, value in required.items()):
            raise ValueError("B2 version, governance or probability metadata differs.")
        if any(
            meta.get("resolved_model_parameters", {}).get(key) != value
            for key, value in FIXED_PARAMETERS.items()
        ):
            raise ValueError("B2 RF parameters differ from the fixed devfix1 candidate.")
        if (
            len(meta.get("feature_order", [])) != 15
            or sum(bool(f["required"]) for f in meta["feature_config"]["features"]) != 13
            or meta.get("positive_class") != 1
            or meta.get("class_order") != [0, 1]
            or meta.get("background", {}).get("count") != 100
            or len(meta["background"].get("development_source_rows", [])) != 100
            or not meta.get("provenance", {}).get("development_sha256")
            or not meta.get("provenance", {}).get("config_sha256")
        ):
            raise ValueError("B2 schema, background, class or fingerprints are incomplete.")
        # Reuse the complete b1 fitted-pipeline/SHAP integrity validator without
        # changing its frozen source or accepting b1's evaluation label for b2.
        v3.CandidateBundle(
            self.pipeline,
            {**meta, "evaluation_status": "development_only", "final_holdout_evaluated": False},
            self.background,
            self.mapping,
        ).validate()


def load_bundle(directory: Path) -> B2Bundle:
    """Read a trusted local joblib only after checking the bundle manifest."""
    if (
        directory.is_symlink()
        or not directory.is_dir()
        or any(path.is_symlink() for path in directory.iterdir())
    ):
        raise ValueError("Missing or symlinked B2 bundle directory.")
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if set(manifest) != v3.BUNDLE_FILES or {p.name for p in directory.iterdir()} != (
        v3.BUNDLE_FILES | {"manifest.json"}
    ):
        raise ValueError("Unexpected B2 manifest or files.")
    for name, digest in manifest.items():
        if sha256_file(directory / name) != digest:
            raise ValueError(f"B2 checksum mismatch: {name}")
    meta = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    if meta.get("library_versions") != v3.versions():
        raise ValueError("B2 dependency versions differ; refusing deserialization.")
    bundle = B2Bundle(
        joblib.load(directory / "pipeline.joblib"),
        meta,
        np.load(directory / "background.npy", allow_pickle=False),
        json.loads((directory / "feature_mapping.json").read_text(encoding="utf-8")),
    )
    bundle.validate()
    return bundle


def predict_batch(
    bundle: B2Bundle, records: pd.DataFrame | list[dict[str, Any]] | dict[str, Any]
) -> list[dict[str, Any]]:
    return [
        {
            "model_version": item["model_version"],
            "risk_score": item["probability"],
            "estimated_probability": item["probability"],
            "estimated_probability_status": PROBABILITY_STATUS,
            "imputed_features": item["imputed_features"],
        }
        for item in v3.predict_batch(bundle, records)
    ]


def predict_one(bundle: B2Bundle, record: dict[str, Any]) -> dict[str, Any]:
    return predict_batch(bundle, record)[0]


def explain_one(bundle: B2Bundle, record: dict[str, Any]) -> dict[str, Any]:
    explanation = v3.explain_one(bundle, record)
    value = explanation.pop("probability")
    return {
        **explanation,
        "risk_score": value,
        "estimated_probability": value,
        "estimated_probability_status": PROBABILITY_STATUS,
    }


save_bundle = v3.save_bundle
