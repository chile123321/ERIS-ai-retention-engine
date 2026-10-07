"""Synthetic-only final-fit/bundle safety tests; never access the locked holdout."""

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from sklearn.pipeline import Pipeline

from eris_ml.features.definitions import load_feature_definition
from eris_ml.features.preprocessing import build_preprocessor
from eris_ml.models import bundle_v2_build as builder
from eris_ml.models.bundle_v2 import (
    FEATURE_ORDER,
    FORBIDDEN,
    load_bundle,
    preprocessing_signature,
    runtime_versions,
    save_bundle,
    validate_input,
    write_json,
)
from eris_ml.models.factory import create_estimator
from eris_ml.models.persistence import sidecar_paths

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def inputs():
    rng = np.random.default_rng(42)
    definition = load_feature_definition(ROOT / "configs/features/feature_set_v2_general_16.yaml")
    config = yaml.safe_load((ROOT / "configs/models/logistic_regression.yaml").read_text())
    frame = pd.DataFrame({name: rng.integers(1, 4, size=80) for name in FEATURE_ORDER})
    frame["OverTime"] = rng.choice(["No", "Yes"], size=80)
    frame["PerformanceRating"] = rng.choice([3, 4], size=80)
    frame["Attrition"] = [0, 0, 0, 1] * 20
    frame["source_row"] = np.arange(80)
    frame["EmployeeNumber"] = np.arange(100, 180)
    frame["Department"] = "must_not_enter_fit"
    frame["JobRole"] = "must_not_enter_fit"
    trace = {
        "model_parameters": create_estimator(config).get_params(deep=False),
        "preprocessing": preprocessing_signature(build_preprocessor(definition)),
        "development_rows": 80, "target_distribution": {"0": 60, "1": 20},
        "library_versions": runtime_versions(), "saved_oof_evaluation": {},
        "evaluation_kind": "synthetic_fixture_no_performance_claim",
        "input_sha256": {}, "protected_sha256": {}, "head": "synthetic",
        "reviewed_source_revisions": [], "status": "research_demo_candidate",
    }
    return frame, definition, config, trace


def test_build_preflight_before_fit_clean_reload_and_no_overwrite(tmp_path, inputs, monkeypatch):
    frame, definition, config, trace = inputs
    monkeypatch.setattr(builder, "prepare", lambda root: inputs)
    fit_calls = []
    original_fit = Pipeline.fit

    def audited_fit(self, features, target, **kwargs):
        assert (tmp_path / builder.PREFLIGHT).exists()
        assert list(features.columns) == FEATURE_ORDER
        assert set(features.columns).isdisjoint(FORBIDDEN)
        assert len(features) == len(frame)
        fit_calls.append(features.copy())
        return original_fit(self, features, target, **kwargs)

    monkeypatch.setattr(Pipeline, "fit", audited_fit)
    output = builder.build(tmp_path)
    assert len(fit_calls) == 2
    assert output["checks"]["repeat_fit_max_absolute_error"] < 1e-12
    assert output["checks"]["fresh_process"]["rows"] == 32
    assert output["checks"]["fresh_process"]["max_absolute_error"] < 1e-12
    assert output["training_performance_metrics_computed"] is False
    bundle = load_bundle(tmp_path / builder.BUNDLE)
    assert bundle.metadata["threshold"] is None
    assert bundle.metadata["business_approved"] is False
    assert bundle.metadata["production_approved"] is False
    assert not hasattr(bundle, "predict")
    assert bundle.metadata["model_parameters"] == create_estimator(config).get_params(deep=False)
    scores = bundle.predict_proba(frame.loc[:, FEATURE_ORDER])
    np.testing.assert_allclose(scores, bundle.predict_proba(frame.loc[:, FEATURE_ORDER[::-1]]))
    assert len(scores) == 80 and np.isfinite(scores).all() and ((0 <= scores) & (scores <= 1)).all()
    paths = [tmp_path / builder.BUNDLE, *sidecar_paths(tmp_path / builder.BUNDLE),
             tmp_path / builder.PREFLIGHT, tmp_path / builder.TRACE]
    before = {p: p.read_bytes() for p in paths}
    with pytest.raises(ValueError, match="Output exists"):
        builder.build(tmp_path)
    assert before == {p: p.read_bytes() for p in paths}
    assert len(fit_calls) == 2


@pytest.mark.parametrize("kind", ["missing", "duplicate", "string", "bool", "infinite", "null",
                                  "fraction", "category", "ordinal", "empty", *sorted(FORBIDDEN)])
def test_input_rejects_invalid_fields_and_types(inputs, kind):
    frame = inputs[0].loc[:, FEATURE_ORDER].iloc[:2].copy()
    if kind == "missing":
        frame = frame.drop(columns="Age")
    elif kind == "duplicate":
        frame = pd.concat([frame, frame[["Age"]]], axis=1)
    elif kind in FORBIDDEN:
        frame[kind] = 1
    elif kind == "empty":
        frame = frame.iloc[:0]
    else:
        column = "OverTime" if kind == "category" else "Education" if kind == "ordinal" else "Age"
        values = {"string": "35", "bool": True, "infinite": np.inf, "null": None,
                  "fraction": 3.5, "category": "unknown", "ordinal": 99}
        frame[column] = frame[column].astype(object)
        frame.iloc[0, frame.columns.get_loc(column)] = values[kind]
    with pytest.raises(ValueError):
        validate_input(frame)


def test_saved_object_checksum_metadata_and_classes_fail_closed(inputs, tmp_path):
    frame, definition, config, trace = inputs
    metadata = builder.metadata_from_preflight(trace, "a" * 64)
    bundle, _ = builder.fit_and_check(frame, definition, config, metadata)
    path = tmp_path / "test.joblib"
    save_bundle(bundle, path)
    before = path.read_bytes()
    with pytest.raises(ValueError, match="existing"):
        save_bundle(bundle, path)
    assert path.read_bytes() == before
    path.write_bytes(before + b"tampered")
    with pytest.raises(ValueError, match="checksum"):
        load_bundle(path)
    changed = copy.deepcopy(bundle)
    changed.pipeline.named_steps["model"].classes_ = np.array([1, 0])
    with pytest.raises(ValueError, match="classes"):
        changed.validate()
    for key, value in [("threshold", .5), ("status", "production"), ("probability_type", "sigmoid"),
                       ("feature_order", FEATURE_ORDER[::-1]), ("business_approved", True)]:
        changed = copy.deepcopy(bundle)
        changed.metadata[key] = value
        with pytest.raises(ValueError, match="metadata"):
            changed.validate()


def aligned_tables(frame):
    assignment = frame[["source_row", "EmployeeNumber", "Attrition"]].copy()
    assignment["development_position"] = np.arange(len(frame))
    assignment["fold"] = np.repeat(np.arange(1, 6), 16)
    oof = assignment.drop(columns="development_position").copy()
    oof["feature_set"] = "v2_general_16"
    oof["model"] = "logistic_regression"
    oof["probability"] = .25
    return assignment, oof


@pytest.mark.parametrize("kind", ["id", "target", "fold", "probability", "duplicate", "assignment"])
def test_alignment_drift_rejected(inputs, kind):
    frame = inputs[0]
    assignment, oof = aligned_tables(frame)
    builder.validate_alignment(frame, assignment, oof, rows=80, counts={0: 60, 1: 20})
    if kind == "assignment":
        assignment.loc[0, "development_position"] = 99
    elif kind == "duplicate":
        oof = pd.concat([oof, oof.iloc[:1]], ignore_index=True)
    else:
        column = {"id": "EmployeeNumber", "target": "Attrition", "fold": "fold",
                  "probability": "probability"}[kind]
        oof.loc[0, column] = {"id": 999, "target": 1, "fold": 5, "probability": np.inf}[kind]
    with pytest.raises(ValueError, match="assignment|alignment|probability"):
        builder.validate_alignment(frame, assignment, oof, rows=80, counts={0: 60, 1: 20})


def test_provenance_bridge_requires_exact_hash_and_historical_head(tmp_path, monkeypatch):
    source = tmp_path / "source.py"
    source.write_bytes(b"old\n")
    historical = hashlib.sha256(source.read_bytes()).hexdigest()
    source.write_bytes(b"new\n")
    actual = hashlib.sha256(source.read_bytes()).hexdigest()
    trace = {"input_sha256": {"source.py": historical}, "outputs": {}, "output_sha256": {},
             "versions": {}}
    mapping = {"source.py": {"current": actual, "historical": historical,
                             "traces": ["baseline_ablation_v2"]}}
    write_json(tmp_path / builder.BASELINE, trace)
    write_json(tmp_path / builder.BRIDGE, {"historical_source_drift": mapping})
    monkeypatch.setattr(builder.subprocess, "check_output", lambda *a, **k: b"old\n")
    _, result = builder.verify_baseline(tmp_path)
    assert len(result["reviewed_source_revisions"]) == 1
    source.write_bytes(b"unreviewed\n")
    with pytest.raises(ValueError, match="Unreviewed"):
        builder.verify_baseline(tmp_path)
    source.write_bytes(b"new\n")
    monkeypatch.setattr(builder.subprocess, "check_output", lambda *a, **k: b"wrong HEAD")
    with pytest.raises(ValueError, match="HEAD"):
        builder.verify_baseline(tmp_path)


@pytest.mark.parametrize("path", ["data/final_test_raw_v1.csv", "../outside.csv", "/absolute.csv"])
def test_path_guard_before_io(tmp_path, monkeypatch, path):
    def forbidden(*args, **kwargs):
        raise AssertionError("Guard must reject before filesystem access")
    monkeypatch.setattr(Path, "is_symlink", forbidden)
    monkeypatch.setattr(Path, "read_bytes", forbidden)
    with pytest.raises(ValueError):
        builder.safe_path(tmp_path, path)


def test_drift_stops_before_preflight_or_training(tmp_path, monkeypatch):
    def refuse(root):
        raise ValueError("Fingerprint mismatch")
    def no_fit(*args, **kwargs):
        raise AssertionError("Training is forbidden after failed preflight")
    monkeypatch.setattr(builder, "prepare", refuse)
    monkeypatch.setattr(Pipeline, "fit", no_fit)
    with pytest.raises(ValueError, match="Fingerprint"):
        builder.build(tmp_path)
    assert not (tmp_path / builder.PREFLIGHT).exists()
    assert not (tmp_path / builder.BUNDLE).exists()


def test_probability_parity_and_exclusive_json_failures(tmp_path):
    with pytest.raises(ValueError, match="parity"):
        builder.assert_equal_probabilities(np.array([.1]), np.array([.2]))
    path = tmp_path / "state.json"
    with pytest.raises(ValueError):
        write_json(path, {"invalid": float("nan")})
    assert not path.exists()
    write_json(path, {"stage": "preflight"})
    with pytest.raises(FileExistsError):
        write_json(path, {"stage": "overwritten"})
    assert json.loads(path.read_text()) == {"stage": "preflight"}
