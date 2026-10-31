"""Synthetic calibrated-model, original-feature masking and trusted-bundle checks."""

import json
import runpy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import sklearn.calibration as calibration

from eris_ml.explainability import adaboost_shap as study
from eris_ml.features.schema_v3 import V3InputValidator, load_protocol, validate_features
from eris_ml.models.tuning_v3 import inner_splits

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def fixture():
    config = load_protocol(ROOT / "configs/features/feature_set_v3_final_15.yaml")
    rng = np.random.default_rng(31)
    frame = pd.DataFrame(
        {
            f["name"]: rng.choice(f.get("vocabulary", f.get("values", [1, 2, 3, 4])), 80)
            for f in config["features"]
        }
    )
    target = np.asarray([0, 0, 0, 1] * 20)
    model = study.fit_candidate(frame.iloc[:64], target[:64], config)
    return config, frame, target, model


def test_fit_isolation_and_sigmoid(fixture, monkeypatch):
    config, frame, y, _ = fixture
    splits = inner_splits(y[:64])
    expected = [set(frame.iloc[t].index) for t, _ in splits] + [set(range(64))]
    seen, labels = [], []
    original = V3InputValidator.fit
    sigmoid = calibration._SigmoidCalibration.fit

    def capture(self, X, y=None):
        seen.append(set(X.index))
        assert list(X.columns) == [f["name"] for f in config["features"]]
        return original(self, X, y)

    def capture_sigmoid(self, X, y, sample_weight=None):
        labels.append(y.copy())
        return sigmoid(self, X, y, sample_weight=sample_weight)

    monkeypatch.setattr(V3InputValidator, "fit", capture)
    monkeypatch.setattr(calibration._SigmoidCalibration, "fit", capture_sigmoid)
    model = study.fit_candidate(frame.iloc[:64], y[:64], config)
    assert seen == expected and len(labels) == 1
    np.testing.assert_array_equal(labels[0], y[:64])
    assert model.ensemble is False and model.method == "sigmoid"


def test_codec_roundtrip_masks_and_missing(fixture):
    config, frame, _, _ = fixture
    codec = study.FeatureCodec(config)
    encoded = codec.encode(frame)
    pd.testing.assert_frame_equal(codec.decode(encoded), validate_features(frame, config))
    mixed = np.where(
        np.random.default_rng(42).integers(0, 2, encoded.shape), encoded, encoded[::-1]
    )
    validate_features(codec.decode(mixed), config)
    missing = frame.iloc[[0]].drop(columns=["TrainingTimesLastYear", "YearsWithCurrManager"])
    decoded = codec.decode(codec.encode(missing))
    assert decoded[["TrainingTimesLastYear", "YearsWithCurrManager"]].isna().all().all()
    for name, value in (("Department", "unknown"), ("Age", -1), ("Age", float("inf"))):
        bad = frame.iloc[[0]].copy()
        bad[name] = value
        with pytest.raises(ValueError):
            codec.encode(bad)
    corrupt = encoded[:1].copy()
    corrupt[0, 1] = 0.5
    with pytest.raises(ValueError, match="integer"):
        codec.decode(corrupt)


def test_shap_calibrated_probability_seed_rng_and_optional(fixture):
    config, frame, _, model = fixture
    explainer = study.Explainer(model, config, frame.iloc[:4])
    np.random.seed(29)
    expected_rng = np.random.get_state()
    result = explainer.explain(frame.iloc[[65]], cycles=2)
    actual_rng = np.random.get_state()
    np.testing.assert_array_equal(expected_rng[1], actual_rng[1])
    assert expected_rng[2:] == actual_rng[2:]
    repeated = explainer.explain(frame.iloc[[65]], cycles=2)
    np.testing.assert_allclose(result["contributions"], repeated["contributions"], atol=1e-12)
    assert len(result["contributions"]) == len(result["feature_order"]) == 15
    assert result["additivity_error"] < 1e-8 and result["masked_rows_checked"] > 0
    np.testing.assert_allclose(
        result["estimated_probability"], model.predict_proba(frame.iloc[[65]])[0, 1]
    )
    missing = frame.iloc[[65]].copy()
    missing[["TrainingTimesLastYear", "YearsWithCurrManager"]] = np.nan
    result = explainer.explain(missing, cycles=2)
    assert result["imputed_features"] == ["TrainingTimesLastYear", "YearsWithCurrManager"]
    assert result["additivity_error"] < 1e-8


def test_additivity_alone_cannot_pass_stability():
    first = {"contributions": [0.1, -0.1] + [0.0] * 13}
    second = {"contributions": [-0.1, 0.1] + [0.0] * 13}
    assert sum(first["contributions"]) == sum(second["contributions"])
    stability = study.stability(first, second)
    assert stability["max_abs_difference"] == 0.2
    assert stability["material_sign_agreement"] == 0
    gate = runpy.run_path(str(ROOT / "scripts/explain_adaboost_v3.py"))["gate"]
    result = {"additivity_error": 0, "seconds": 0.1, "masked_rows_checked": 1}
    assert not gate(
        [
            {
                "seed_stability": stability,
                "budget_stability": stability,
                "low": result,
                "high": result,
                "other_seed": result,
            }
        ]
    )["passed"]


def test_bundle_roundtrip_overwrite_and_checksum(fixture, tmp_path, monkeypatch):
    config, frame, _, model = fixture
    directory = tmp_path / "bundle"
    meta = {
        "model_id": study.MODEL_ID,
        "status": "research_demo_candidate",
        "versions": study.versions(),
        "class_order": [0, 1],
        "positive_class": 1,
        "production_approved": False,
        "threshold": None,
        "feature_config": config,
        "feature_order": [f["name"] for f in config["features"]],
    }
    study.save_bundle(directory, model, frame.iloc[:4], meta)
    loaded = study.load_bundle(directory)
    np.testing.assert_allclose(loaded.model.predict_proba(frame), model.predict_proba(frame))
    with pytest.raises(ValueError, match="overwrite"):
        study.save_bundle(directory, model, frame.iloc[:4], meta)
    (directory / "metadata.json").write_text(json.dumps({**meta, "positive_class": 0}))

    def forbidden_load(*args, **kwargs):
        raise AssertionError("Must reject corrupt bundle before deserialization")

    monkeypatch.setattr(study.joblib, "load", forbidden_load)
    with pytest.raises(ValueError, match="checksum"):
        study.load_bundle(directory)


@pytest.mark.parametrize("value", ["not a number", "1e999", "9" * 1000, "1.5", ""])
def test_friendly_manual_validation(fixture, value):
    config, frame, _, _ = fixture
    values = {k: str(v) for k, v in frame.iloc[0].to_dict().items()}
    values["Age"] = value
    with pytest.raises(ValueError, match="Age"):
        study.parse_inputs(values, config)


def test_manual_optional_and_category(fixture):
    config, frame, _, _ = fixture
    values = {k: str(v) for k, v in frame.iloc[0].to_dict().items()}
    values["YearsWithCurrManager"] = "Missing"
    assert pd.isna(study.parse_inputs(values, config).iloc[0].YearsWithCurrManager)
    values["OverTime"] = "Sometimes"
    with pytest.raises(ValueError, match="OverTime"):
        study.parse_inputs(values, config)
