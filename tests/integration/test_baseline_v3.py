"""Small synthetic V3 schema, isolation, OOF and provenance regression tests."""

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from sklearn.pipeline import Pipeline

from eris_ml.evaluation.feature_ablation import make_shared_folds
from eris_ml.features.schema_v3 import (
    build_pipeline,
    load_protocol,
    validate_features,
    validate_protocol,
)
from eris_ml.models import baseline_v3 as baseline
from eris_ml.utils.hashing import sha256_file

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/features/feature_set_v3_final_15.yaml"
NAMES = [
    "Age",
    "Department",
    "DistanceFromHome",
    "Education",
    "EnvironmentSatisfaction",
    "JobLevel",
    "JobSatisfaction",
    "MonthlyIncome",
    "OverTime",
    "TotalWorkingYears",
    "TrainingTimesLastYear",
    "YearsAtCompany",
    "YearsInCurrentRole",
    "YearsSinceLastPromotion",
    "YearsWithCurrManager",
]
OPTIONAL = ["TrainingTimesLastYear", "YearsWithCurrManager"]


@pytest.fixture
def config():
    return load_protocol(CONFIG)


@pytest.fixture
def inputs(config):
    rng = np.random.default_rng(123)
    features = {}
    for spec in config["features"]:
        if spec["type"] == "categorical":
            features[spec["name"]] = rng.choice(spec["vocabulary"], 80)
        elif "values" in spec:
            features[spec["name"]] = rng.choice(spec["values"], 80)
        else:
            features[spec["name"]] = rng.integers(1, 20, 80).astype(float)
    frame = pd.DataFrame(features)
    frame.loc[::7, OPTIONAL] = np.nan
    frame["source_row"] = np.arange(80)
    frame["EmployeeNumber"] = np.arange(1000, 1080)
    frame["Attrition"] = [0, 0, 0, 1] * 20
    assignments = frame[baseline.IDS].copy()
    assignments["fold"] = make_shared_folds(frame.Attrition.to_numpy())
    assignments["development_position"] = np.arange(80)
    manifest = frame[baseline.IDS[:2]].copy().assign(split="development")
    # Only synthetic membership IDs, not any final-test file or employee attributes.
    manifest.loc[80] = [999, 9999, "final_test"]
    return frame, assignments, manifest


def small_pipeline(config, model):
    pipeline = build_pipeline(config, model)
    if model in {"random_forest", "xgboost"}:
        # Test fixture only: production config/factory and parameters are unchanged.
        pipeline.set_params(model__n_estimators=7)
    return pipeline


def test_locked_feature_order_required_and_optional(config, inputs):
    frame, _, _ = inputs
    assert [f["name"] for f in config["features"]] == NAMES
    assert [f["name"] for f in config["features"] if not f["required"]] == OPTIONAL
    result = validate_features(frame[NAMES[::-1]], config)
    assert list(result.columns) == NAMES and result.shape == (80, 15)
    assert not set(baseline.IDS + ["Gender", "MaritalStatus", "JobRole"]) & set(result.columns)
    for name in baseline.IDS + ["Gender", "MaritalStatus", "JobRole", "JobInvolvement"]:
        with pytest.raises(ValueError, match="Unexpected"):
            validate_features(result.assign(**{name: 1}), config)
    for model in config["models"]:
        pipeline = build_pipeline(config, model)
        params = pipeline.named_steps["model"].get_params()
        assert params["random_state"] == 42
        if model in {"random_forest", "xgboost"}:
            assert params["n_jobs"] == 1
            assert pipeline.named_steps["preprocessor"].transformers[0][1] == "passthrough"


@pytest.mark.parametrize("missing", [[OPTIONAL[0]], [OPTIONAL[1]], OPTIONAL])
@pytest.mark.parametrize(
    "model", ["dummy_prior", "logistic_regression", "random_forest", "xgboost"]
)
def test_optional_missing_prediction_and_single_batch(config, inputs, missing, model):
    frame, _, _ = inputs
    pipeline = small_pipeline(config, model).fit(frame[NAMES], frame.Attrition)
    batch = frame[NAMES].iloc[:3].drop(columns=missing)
    normalized = validate_features(batch, config)
    assert normalized[missing].isna().all().all()
    prediction = pipeline.predict_proba(batch)
    assert prediction.shape == (3, 2) and np.isfinite(prediction).all()
    assert ((prediction >= 0) & (prediction <= 1)).all()
    np.testing.assert_allclose(prediction[:1], pipeline.predict_proba(batch.iloc[:1]), atol=1e-12)
    np.testing.assert_allclose(prediction, pipeline.predict_proba(normalized), atol=1e-12)


@pytest.mark.parametrize(
    "field,value",
    [
        ("Age", np.nan),
        ("Age", np.inf),
        ("Age", "42"),
        ("Age", True),
        ("Age", 1.5),
        ("MonthlyIncome", -1),
        ("Education", 6),
        ("Department", "New company department"),
        ("OverTime", "yes"),
        ("TrainingTimesLastYear", np.inf),
    ],
)
def test_invalid_values_rejected_with_field(config, inputs, field, value):
    frame, _, _ = inputs
    record = frame[NAMES].iloc[[1]].astype(object)
    record.loc[record.index[0], field] = value
    with pytest.raises(ValueError, match=field):
        validate_features(record, config)


def test_required_missing_duplicate_and_empty_optional_training(config, inputs):
    frame, _, _ = inputs
    with pytest.raises(ValueError, match="Age"):
        validate_features(frame[NAMES].drop(columns="Age"), config)
    with pytest.raises(ValueError, match="unique columns"):
        validate_features(pd.concat([frame[NAMES], frame[["Age"]]], axis=1), config)
    with pytest.raises(ValueError, match="training observations"):
        small_pipeline(config, "logistic_regression").fit(
            frame[NAMES].drop(columns=OPTIONAL), frame.Attrition
        )
    invalid = copy.deepcopy(config)
    invalid["features"][0]["name"] = "Attrition"
    with pytest.raises(ValueError, match="Target"):
        validate_protocol(invalid)


def test_scaler_and_imputer_do_not_learn_validation(config, inputs):
    frame, _, _ = inputs
    train = frame[NAMES].iloc[:60].copy()
    held_out = frame[NAMES].iloc[60:].copy()
    held_out.loc[:, ["Age", *OPTIONAL]] = 999999
    pipeline = small_pipeline(config, "logistic_regression").fit(train, frame.Attrition.iloc[:60])
    transformer = pipeline.named_steps["preprocessor"]
    imputer = transformer.named_transformers_["optional_numeric"].named_steps["imputer"]
    np.testing.assert_allclose(imputer.statistics_, train[OPTIONAL].median().to_numpy())
    required = [s["name"] for s in config["features"] if s["type"] == "integer" and s["required"]]
    scaler = transformer.named_transformers_["required_numeric"]
    np.testing.assert_allclose(scaler.mean_, train[required].mean().to_numpy())
    before = scaler.mean_.copy(), imputer.statistics_.copy()
    pipeline.predict_proba(held_out)
    np.testing.assert_array_equal(scaler.mean_, before[0])
    np.testing.assert_array_equal(imputer.statistics_, before[1])
    held_out["Department"] = "Foreign department"
    with pytest.raises(ValueError, match="Department"):
        pipeline.predict_proba(held_out)


@pytest.mark.parametrize("kind", ["fold", "target", "id", "manifest", "duplicate"])
def test_saved_fold_and_membership_mismatch_stops(inputs, kind):
    frame, assignments, manifest = inputs
    baseline.validate_alignment(frame, assignments, manifest)
    if kind == "fold":
        assignments.loc[0, "fold"] = 9
    elif kind == "target":
        assignments.loc[0, "Attrition"] = 1
    elif kind == "id":
        assignments.loc[0, "EmployeeNumber"] = 999
    elif kind == "manifest":
        manifest.loc[0, "split"] = "final_test"
    else:
        assignments.loc[0, "source_row"] = assignments.loc[1, "source_row"]
    with pytest.raises(ValueError):
        baseline.validate_alignment(frame, assignments, manifest)


def test_four_model_oof_end_to_end_isolated_deterministic(config, inputs, monkeypatch):
    frame, assignments, _ = inputs
    monkeypatch.setattr(baseline, "build_pipeline", small_pipeline)
    calls = []
    original_fit = Pipeline.fit

    def audited_fit(self, X, y=None, **kwargs):
        if "model" in self.named_steps:
            assert list(X.columns) == NAMES
            matching = [
                fold
                for fold in range(1, 6)
                if set(X.index) == set(frame.index[assignments.fold != fold])
            ]
            assert len(matching) == 1
            validation_ids = set(frame.index[assignments.fold == matching[0]])
            assert not set(X.index) & validation_ids
            calls.append(matching[0])
        return original_fit(self, X, y, **kwargs)

    monkeypatch.setattr(Pipeline, "fit", audited_fit)
    first, metrics = baseline.evaluate(frame, assignments, config)
    second, repeated = baseline.evaluate(frame, assignments, config)
    pd.testing.assert_frame_equal(first, second)
    assert metrics == repeated
    assert len(calls) == 40 and first.groupby("model").size().to_dict() == {
        model: 80 for model in config["models"]
    }
    assert not first.isna().any().any()
    assert first.probability.between(0, 1).all()
    for record in metrics["models"]:
        values = [f["metrics"]["average_precision"] for f in record["folds"]]
        assert record["fold_mean_std"]["average_precision"]["sample_std"] == pytest.approx(
            np.std(values, ddof=1)
        )
        pooled = record["pooled_oof"]
        assert sum(pooled[name] for name in ("tp", "fp", "tn", "fn")) == 80
    bad = first.copy()
    bad.loc[0, "probability"] = np.inf
    with pytest.raises(ValueError, match="finite"):
        baseline.validate_oof(bad, assignments, config)
    with pytest.raises(ValueError, match="coverage"):
        baseline.validate_oof(first.iloc[1:], assignments, config)


def test_prepare_pins_only_development_and_manifest_ids(config, inputs, tmp_path, monkeypatch):
    frame, assignments, manifest = inputs
    config["data"].update(
        expected_rows=80, expected_target={0: 60, 1: 20}, expected_manifest_rows=81
    )
    data = config["data"]
    for key, table in (
        ("development", frame),
        ("manifest", manifest),
        ("assignments", assignments),
    ):
        path = tmp_path / data[key]
        path.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(path, index=False)
    trace = {
        "input_sha256": {
            data[key]: sha256_file(tmp_path / data[key]) for key in ("development", "manifest")
        },
        "output_sha256": {"assignments": sha256_file(tmp_path / data["assignments"])},
        "cv": config["cv"],
    }
    trace_path = tmp_path / data["provenance_trace"]
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    trace_path.write_text(json.dumps(trace), encoding="utf-8")
    ledger = tmp_path / data["holdout_ledger"]
    ledger.parent.mkdir(parents=True, exist_ok=True)
    ledger.write_text(
        yaml.safe_dump({"run_count": 1, "status": "completed", "completed_at_utc": "synthetic"}),
        encoding="utf-8",
    )
    original_read = pd.read_csv
    reads = []

    def read(path, **kwargs):
        assert "final_test" not in str(path)
        reads.append(Path(path))
        if Path(path) == tmp_path / data["manifest"]:
            assert kwargs["usecols"] == ["source_row", "EmployeeNumber", "split"]
        return original_read(path, **kwargs)

    monkeypatch.setattr(pd, "read_csv", read)
    actual, saved, provenance = baseline.prepare(tmp_path, config)
    assert len(actual) == len(saved) == 80 and not provenance["folds_recreated"]
    assert len(reads) == 3
    trace["output_sha256"]["assignments"] = "wrong"
    trace_path.write_text(json.dumps(trace), encoding="utf-8")
    with pytest.raises(ValueError, match="assignment hash"):
        baseline.prepare(tmp_path, config)
    assert len(reads) == 3  # Hash mismatch stops before any data read or fit.


def test_preflight_is_written_before_fit_and_output_guards(config, inputs, monkeypatch, tmp_path):
    for relative in ("data/processed/final_test_raw_v1.csv", "../escape.csv"):
        with pytest.raises(ValueError):
            baseline.safe_path(tmp_path, relative)
    frame, assignments, _ = inputs
    monkeypatch.setattr(
        baseline, "prepare", lambda root, cfg: (frame, assignments, {"input_sha256": {}})
    )
    original_writer = baseline.write_json
    events = []
    monkeypatch.setattr(baseline, "write_json", lambda path, payload: events.append(path.name))

    def stop_before_fit(*args):
        assert events == ["preflight.json"]
        raise ValueError("stop before synthetic fit")

    monkeypatch.setattr(baseline, "evaluate", stop_before_fit)
    with pytest.raises(ValueError, match="stop before synthetic fit"):
        baseline.run(ROOT, CONFIG, "test_preflight_only")
    # Exclusive writer also preserves existing technical output bytes.
    output = tmp_path / "existing.json"
    output.write_text("preserve", encoding="utf-8")
    with pytest.raises(FileExistsError):
        original_writer(output, {"must_not_replace": True})
    assert output.read_text(encoding="utf-8") == "preserve"
