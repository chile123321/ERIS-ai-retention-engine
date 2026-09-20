"""Shared pytest configuration for the src-layout package."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


@pytest.fixture(scope="session")
def synthetic_bundle() -> tuple[object, pd.DataFrame, object]:
    """Fitted frozen-parameter pipeline on synthetic rows; never reads holdout data."""
    from xgboost import XGBClassifier

    from eris_ml.features.definitions import load_feature_definition
    from eris_ml.models.bundle import ModelBundle
    from eris_ml.models.bundle_build import bundle_metadata
    from eris_ml.models.training import build_model_pipeline
    from eris_ml.utils.freeze import MODEL_PARAMETERS

    root = Path(__file__).resolve().parents[1]
    definition = load_feature_definition(root / "configs/features/feature_set_v1_full.yaml")
    manifest = yaml.safe_load((root / "configs/release/candidate_v1.yaml").read_text(
        encoding="utf-8"
    ))
    rng = np.random.default_rng(42)
    size = 80
    data: dict[str, object] = {}
    for name in definition.nominal:
        data[name] = rng.choice(["Alpha", "Beta"], size=size)
    for name in definition.ordinal:
        data[name] = rng.choice([1, 2, 3], size=size)
    for name in definition.numeric:
        data[name] = rng.normal(5, 2, size=size)
    frame = pd.DataFrame(data)
    target = pd.Series(([0] * 60) + ([1] * 20))
    pipeline = build_model_pipeline(
        definition, XGBClassifier(random_state=42, **MODEL_PARAMETERS)
    )
    pipeline.fit(frame.loc[:, list(definition.all_features)], target)
    metadata = bundle_metadata(
        manifest, frame, definition, manifest_hash="a" * 64,
        development_hash="b" * 64, timestamp="2026-09-20T04:00:00+00:00",
    )
    return ModelBundle(pipeline, metadata), frame, definition
