"""Original-feature permutation SHAP for a trusted, calibrated AdaBoost demo bundle."""

from __future__ import annotations

import json
import platform
import threading
import time
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.calibration import CalibratedClassifierCV
from threadpoolctl import threadpool_limits

from eris_ml.features.schema_v3 import validate_features, validate_protocol
from eris_ml.models import adaboost_v3 as a1
from eris_ml.models import baseline_v3 as baseline
from eris_ml.models.tuning_v3 import inner_splits
from eris_ml.utils.hashing import sha256_file

MODEL_ID = "adaboost-v3-final15-baseline-sigmoid-s1"
FILES = {"pipeline.joblib", "background.npy", "metadata.json"}
RNG_LOCK = threading.Lock()
PROTOCOL: dict[str, Any] = {
    "explainer": "shap.PermutationExplainer; identity link; scalar calibrated P(Attrition=1)",
    "sample": "first 3 source_row ascending per saved outer-validation fold; no labels/scores",
    "background": "32 outer-training rows sampled without replacement; seed 42",
    "background_count": 32,
    "sample_per_fold": 3,
    "low_cycles": 16,
    "high_cycles": 64,
    "seeds": [42, 43],
    "max_evals": "cycles * (2*15+1); each cycle includes forward and reverse traversal",
    "additivity_atol": 1e-8,
    "max_seed_difference": 0.01,
    "max_budget_difference": 0.02,
    "minimum_material_sign_agreement": 0.9,
    "material_contribution": 0.005,
    "minimum_mean_top5_overlap": 0.8,
    "minimum_top5_overlap": 0.6,
    "missing_stress": "first validation sample/fold: each optional missing and both missing",
    "latency": "warm-up excluded; high-budget per-record wall time; informational, no SLA",
    "semantics": "interventional marginal replacement; not causal/conditional SHAP",
    "production_approved": False,
}


def versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        **{
            p: version(p)
            for p in ("numpy", "pandas", "scipy", "scikit-learn", "joblib", "shap", "numba")
        },
    }


def fit_candidate(X: pd.DataFrame, y: np.ndarray, config: dict[str, Any]) -> Any:
    """Exactly a1 baseline sigmoid: four inner folds then full-training base refit."""
    model: Any = CalibratedClassifierCV(
        estimator=a1.pipeline_for(config),
        method="sigmoid",
        cv=inner_splits(y),
        ensemble=False,
        n_jobs=1,
    )
    with threadpool_limits(limits=1):
        model.fit(X, y)
    return model


class FeatureCodec:
    """Lossless masker transport only; NOT an alternative model preprocessing pipeline."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.specs = config["features"]
        self.names = [f["name"] for f in self.specs]

    def encode(self, frame: pd.DataFrame) -> np.ndarray:
        try:
            checked = validate_features(frame, self.config)
        except (ValueError, TypeError, OverflowError) as error:
            raise ValueError("Invalid IBM input: " + str(error)) from None
        columns = []
        for spec in self.specs:
            column = checked[spec["name"]]
            if spec["type"] == "categorical":
                column = column.map({v: i for i, v in enumerate(spec["vocabulary"])})
            values = column.to_numpy(dtype=float, na_value=np.nan)
            if not spec["required"]:
                values = np.where(np.isnan(values), -1.0, values)
            columns.append(values)
        encoded = np.column_stack(columns)
        self.decode(encoded)  # Same validity boundary used for every masked input.
        return encoded

    def decode(self, encoded: Any) -> pd.DataFrame:
        try:
            values = np.asarray(encoded, dtype=float)
        except (TypeError, ValueError, OverflowError):
            raise ValueError("Masker input must contain finite numeric codes.") from None
        if (
            values.ndim != 2
            or values.shape[1] != 15
            or not len(values)
            or not np.isfinite(values).all()
            or np.any(np.abs(values) > 2**53 - 1)
        ):
            raise ValueError(
                "Masker input has invalid shape, nonfinite or excessively large values."
            )
        data: dict[str, Any] = {}
        for i, spec in enumerate(self.specs):
            column = values[:, i]
            if np.any(column != np.floor(column)):
                raise ValueError(f"{spec['name']}: expected integer/code.")
            if spec["type"] == "categorical":
                vocab = spec["vocabulary"]
                if np.any((column < 0) | (column >= len(vocab))):
                    raise ValueError(f"{spec['name']}: invalid category code.")
                data[spec["name"]] = np.asarray(vocab, dtype=object)[column.astype(int)]
            else:
                missing = (column == -1) & (not spec["required"])
                valid = column[~missing]
                if np.any(valid < spec.get("minimum", 0)) or (
                    "values" in spec and not np.isin(valid, spec["values"]).all()
                ):
                    raise ValueError(f"{spec['name']}: outside the IBM numeric domain.")
                data[spec["name"]] = np.where(missing, np.nan, column)
        return pd.DataFrame(data, columns=self.names)


class ProbabilityFunction:
    def __init__(self, model: Any, codec: FeatureCodec):
        self.model, self.codec = model, codec
        self.masked_rows_checked = 0

    def __call__(self, encoded: np.ndarray) -> np.ndarray:
        frame = self.codec.decode(encoded)
        self.masked_rows_checked += len(frame)
        return a1.positive_probability(self.model, frame)


class Explainer:
    def __init__(self, model: Any, config: dict[str, Any], background: pd.DataFrame):
        self.codec = FeatureCodec(config)
        self.function = ProbabilityFunction(model, self.codec)
        self.background = self.codec.encode(background)
        # Repository SHAP stubs describe TreeExplainer only; this tested optional API is dynamic.
        shap_runtime: Any = shap
        self.masker = shap_runtime.maskers.Independent(
            self.background, max_samples=len(self.background)
        )

    def explain(self, record: pd.DataFrame, cycles: int = 64, seed: int = 42) -> dict[str, Any]:
        if len(record) != 1 or type(cycles) is not int or not 1 <= cycles <= 256:
            raise ValueError("Explain exactly one record; cycles must be an integer in [1,256].")
        encoded = self.codec.encode(record)
        before = self.function.masked_rows_checked
        started = time.perf_counter()
        # SHAP 0.50 seeds numpy's global RNG; serialize and restore it for local callers.
        with RNG_LOCK, threadpool_limits(limits=1):
            state = np.random.get_state()
            try:
                shap_runtime: Any = shap
                explainer = shap_runtime.PermutationExplainer(
                    self.function,
                    self.masker,
                    feature_names=self.codec.names,
                    seed=seed,
                )
                explanation = explainer(encoded, max_evals=cycles * 31, silent=True)
            finally:
                np.random.set_state(state)
        contribution = np.asarray(explanation.values[0], dtype=float)
        base = float(explanation.base_values[0])
        probability = float(self.function(encoded)[0])
        expected_base = float(self.function(self.background).mean())
        error = abs(base + float(contribution.sum()) - probability)
        if (
            contribution.shape != (15,)
            or not np.isfinite(contribution).all()
            or not np.isfinite(base)
            or error > PROTOCOL["additivity_atol"]
            or abs(base - expected_base) > PROTOCOL["additivity_atol"]
        ):
            raise ValueError("SHAP shape/finiteness/additivity/background check failed.")
        frame = self.codec.decode(encoded)
        values = {
            name: None if pd.isna(frame.iloc[0][name]) else frame.iloc[0][name]
            for name in self.codec.names
        }
        return {
            "estimated_probability": probability,
            "base_value": base,
            "feature_values": baseline.json_safe(values),
            "feature_order": self.codec.names,
            "contributions": contribution.tolist(),
            "contributions_percentage_points": (contribution * 100).tolist(),
            "additivity_error": error,
            "seconds": time.perf_counter() - started,
            "masked_rows_checked": self.function.masked_rows_checked - before,
            "cycles": cycles,
            "seed": seed,
            "imputed_features": [name for name, value in values.items() if value is None],
        }


def stability(first: dict[str, Any], second: dict[str, Any]) -> dict[str, float]:
    a, b = np.asarray(first["contributions"]), np.asarray(second["contributions"])
    material = np.abs(b) >= PROTOCOL["material_contribution"]
    top_a, top_b = (
        np.argsort(-np.abs(a), kind="stable")[:5],
        np.argsort(-np.abs(b), kind="stable")[:5],
    )
    return {
        "max_abs_difference": float(np.max(np.abs(a - b))),
        "mean_abs_difference": float(np.mean(np.abs(a - b))),
        "top5_overlap": len(set(top_a) & set(top_b)) / 5,
        "material_sign_agreement": float(np.mean(np.sign(a[material]) == np.sign(b[material])))
        if material.any()
        else 1.0,
        "material_count": int(material.sum()),
    }


@dataclass
class Bundle:
    model: Any
    metadata: dict[str, Any]
    background: pd.DataFrame

    def explainer(self) -> Explainer:
        return Explainer(self.model, self.metadata["feature_config"], self.background)


def save_bundle(
    directory: Path, model: Any, background: pd.DataFrame, metadata: dict[str, Any]
) -> None:
    if directory.exists():
        raise ValueError("Bundle already exists; refusing overwrite.")
    directory.mkdir(parents=True, exist_ok=False)
    joblib.dump(model, directory / "pipeline.joblib", compress=3)
    np.save(
        directory / "background.npy",
        FeatureCodec(metadata["feature_config"]).encode(background),
        allow_pickle=False,
    )
    baseline.write_json(directory / "metadata.json", metadata)
    baseline.write_json(
        directory / "manifest.json", {name: sha256_file(directory / name) for name in sorted(FILES)}
    )


def load_bundle(directory: Path) -> Bundle:
    """Only load trusted local joblib. Checksums detect corruption, not malicious provenance."""
    baseline.safe_path(directory.parent, directory.name)
    if not directory.is_dir() or {p.name for p in directory.iterdir()} != FILES | {"manifest.json"}:
        raise ValueError("Invalid candidate bundle file set.")
    if any(p.is_symlink() for p in directory.iterdir()):
        raise ValueError("Symlinked bundle component rejected.")
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if set(manifest) != FILES or any(sha256_file(directory / n) != manifest[n] for n in FILES):
        raise ValueError("Bundle checksum mismatch.")
    meta = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    if (
        meta.get("model_id") != MODEL_ID
        or meta.get("versions") != versions()
        or meta.get("status") != "research_demo_candidate"
        or meta.get("class_order") != [0, 1]
        or meta.get("positive_class") != 1
        or meta.get("production_approved") is not False
        or meta.get("threshold") is not None
    ):
        raise ValueError("Bundle identity/version/governance mismatch.")
    config = meta["feature_config"]
    validate_protocol(config)
    if meta.get("feature_order") != [f["name"] for f in config["features"]]:
        raise ValueError("Bundle feature order mismatch.")
    model = joblib.load(directory / "pipeline.joblib")
    if (
        not isinstance(model, CalibratedClassifierCV)
        or model.get_params(deep=False)["method"] != "sigmoid"
        or model.get_params(deep=False)["ensemble"] is not False
        or not np.array_equal(model.classes_, [0, 1])
        or len(model.calibrated_classifiers_) != 1
    ):
        raise ValueError("Expected one fitted sigmoid-calibrated binary estimator.")
    base = model.calibrated_classifiers_[0].estimator
    if baseline.json_safe(base) != baseline.json_safe(a1.pipeline_for(config)):
        raise ValueError("Candidate is not the frozen a1 baseline pipeline.")
    background = FeatureCodec(config).decode(
        np.load(directory / "background.npy", allow_pickle=False)
    )
    a1.positive_probability(model, background)
    return Bundle(model, meta, background)


def parse_inputs(values: dict[str, str], config: dict[str, Any]) -> pd.DataFrame:
    """Friendly console/UI conversion, preserving optional missing instead of inserting zero."""
    result: dict[str, Any] = {}
    for spec in config["features"]:
        name = spec["name"]
        text = values.get(name, "").strip()
        if not spec["required"] and text.lower() in {"", "missing", "null", "none"}:
            result[name] = None
        elif spec["type"] == "categorical":
            result[name] = text
        else:
            try:
                number = float(text)
                if not np.isfinite(number) or abs(number) > 2**53 - 1 or number != int(number):
                    raise ValueError
                result[name] = int(number)
            except (ValueError, TypeError, OverflowError):
                raise ValueError(
                    f"{name}: enter a valid integer, not an empty/huge value."
                ) from None
    codec = FeatureCodec(config)
    return codec.decode(codec.encode(pd.DataFrame([result])))
