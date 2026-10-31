"""Synthetic checks for the fixed b2 fit, local inference and integrity boundary."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.impute import SimpleImputer

from eris_ml.features.schema_v3 import V3InputValidator, load_protocol
from eris_ml.models import candidate_v3_b2 as candidate
from eris_ml.models import remediation_v3 as study
from eris_ml.models.bundle_v3 import feature_mapping
from eris_ml.models.bundle_v3_b2 import (
    FIXED_PARAMETERS,
    B2Bundle,
    explain_one,
    load_bundle,
    predict_batch,
    predict_one,
    save_bundle,
)

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def fitted(tmp_path_factory):
    config = load_protocol(ROOT / "configs/features/feature_set_v3_final_15.yaml")
    rng = np.random.default_rng(37)
    frame = pd.DataFrame(
        {
            feature["name"]: rng.choice(
                feature.get("vocabulary", feature.get("values", [1, 2, 3])), 1176
            )
            for feature in config["features"]
        }
    )
    frame["Age"] = rng.integers(18, 60, 1176)
    frame["Attrition"] = np.array([0] * 986 + [1] * 190)[rng.permutation(1176)]
    frame["source_row"] = np.arange(1176)
    frame["EmployeeNumber"] = np.arange(2000, 3176)
    base = study.pipeline_for(
        config, {**FIXED_PARAMETERS, "class_weight": "balanced"}, "rf_unweighted"
    )
    pipeline, X, details = candidate.fit_fixed(
        config, frame, base.named_steps["model"].get_params()
    )
    positions = np.random.default_rng(42).choice(len(X), 100, replace=False).tolist()
    provenance = {
        "data_provenance": {
            "input_sha256": {config["data"]["development"]: "synthetic-development-sha"}
        }
    }
    metadata = candidate.metadata(
        config,
        details["resolved_model_parameters"],
        provenance,
        frame,
        positions,
        "synthetic-preflight-sha",
        "synthetic-config-sha",
    )
    bundle = B2Bundle(
        pipeline, metadata, pipeline[:-1].transform(X.iloc[positions]), feature_mapping(pipeline)
    )
    directory = tmp_path_factory.mktemp("b2_bundle") / "model"
    save_bundle(bundle, directory)
    return bundle, X, frame, directory, details


def test_fixed_fit_schema_parameters_and_development_medians(fitted):
    bundle, X, frame, _, details = fitted
    assert details["fit_count"] == 1 and details["fit_rows"] == 1176
    assert details["fit_target_distribution"] == {0: 986, 1: 190}
    assert bundle.pipeline.named_steps["model"].get_params()["class_weight"] is None
    assert all(
        bundle.pipeline.named_steps["model"].get_params()[k] == v
        for k, v in FIXED_PARAMETERS.items()
    )
    assert list(X) == bundle.metadata["feature_order"] and len(X.columns) == 15
    assert not {"source_row", "EmployeeNumber", "Attrition"} & set(X)
    assert bundle.metadata["class_order"] == [0, 1]
    assert bundle.metadata["positive_class"] == 1
    assert bundle.metadata["background"]["development_source_rows"] == (
        frame.iloc[bundle.metadata["background"]["development_positions"]].source_row.tolist()
    )
    assert len(bundle.mapping) == 18 and set(bundle.mapping) == set(X)


@pytest.mark.parametrize(
    "missing",
    [
        [],
        ["TrainingTimesLastYear"],
        ["YearsWithCurrManager"],
        ["TrainingTimesLastYear", "YearsWithCurrManager"],
    ],
)
def test_reload_inference_key_order_optional_and_shap(fitted, missing):
    bundle, X, _, directory, _ = fitted
    loaded = load_bundle(directory)
    record = X.iloc[0].to_dict()
    for field in missing:
        record.pop(field)
    one = predict_one(loaded, record)
    backwards = predict_one(loaded, dict(reversed(list(record.items()))))
    batch = predict_batch(loaded, [record, record])
    for result in (backwards, *batch):
        assert result["imputed_features"] == one["imputed_features"]
        assert result["risk_score"] == pytest.approx(one["risk_score"], abs=1e-12)
    assert one["risk_score"] == one["estimated_probability"]
    assert one["imputed_features"] == missing
    assert one["estimated_probability_status"] == "development_validated_estimate_not_production"
    np.testing.assert_allclose(
        one["risk_score"], bundle.pipeline.predict_proba(bundle.inputs(record))[0, 1], atol=1e-12
    )
    explanation = explain_one(loaded, record)
    assert set(explanation["contributions"]) == set(X)
    assert explanation["base_value"] + sum(explanation["contributions"].values()) == (
        pytest.approx(one["risk_score"], abs=1e-6, rel=1e-6)
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("Age", None),
        ("Age", float("nan")),
        ("Age", "30"),
        ("Department", "Unknown"),
        ("TrainingTimesLastYear", float("inf")),
        ("source_row", 0),
    ],
)
def test_input_rejections(fitted, field, value):
    _, X, _, directory, _ = fitted
    with pytest.raises(ValueError):
        predict_one(load_bundle(directory), {**X.iloc[0].to_dict(), field: value})


def test_manifest_detects_change_and_existing_outputs_stop_before_fit(
    fitted, tmp_path, monkeypatch
):
    _, _, _, directory, _ = fitted
    with pytest.raises(FileExistsError):
        save_bundle(load_bundle(directory), directory)
    background = directory / "background.npy"
    original = background.read_bytes()
    try:
        with background.open("ab") as stream:
            stream.write(b"tampered")
        with pytest.raises(ValueError, match="checksum"):
            load_bundle(directory)
    finally:
        background.write_bytes(original)
    (tmp_path / "artifacts/reports/v3_final_15/b2").mkdir(parents=True)

    def forbidden(*args, **kwargs):
        pytest.fail("Existing output must stop before data preflight or fitting")

    monkeypatch.setattr(candidate, "preflight", forbidden)
    with pytest.raises(FileExistsError):
        candidate.run(tmp_path)


def test_final_and_f1_path_guards(tmp_path):
    for relative in (
        "data/processed/final_test_raw_v1.csv",
        "artifacts/predictions/v3_final_15/f1/holdout_predictions.csv",
        "artifacts/reports/v3_final_15/f1/final_metrics.json",
        "artifacts/reports/v3_final_15/final_evaluation_consumed.json",
    ):
        with pytest.raises(ValueError):
            candidate.protected_path(tmp_path, relative)


def test_train_only_validator_and_imputer_fits_on_all_development(fitted, monkeypatch):
    _, _, frame, _, _ = fitted
    config = load_protocol(ROOT / "configs/features/feature_set_v3_final_15.yaml")
    seen = []
    original_validator, original_imputer = V3InputValidator.fit, SimpleImputer.fit

    def validate(self, X, y=None):
        seen.append(("validate", set(X.index), list(X.columns)))
        return original_validator(self, X, y)

    def impute(self, X, y=None):
        seen.append(("impute", set(X.index), list(X.columns)))
        return original_imputer(self, X, y)

    monkeypatch.setattr(V3InputValidator, "fit", validate)
    monkeypatch.setattr(SimpleImputer, "fit", impute)
    base = study.pipeline_for(
        config, {**FIXED_PARAMETERS, "class_weight": "balanced"}, "rf_unweighted"
    )
    candidate.fit_fixed(config, frame, base.named_steps["model"].get_params())
    assert [(kind, indices) for kind, indices, _ in seen] == [
        ("validate", set(frame.index)),
        ("impute", set(frame.index)),
    ]
    assert seen[0][2] == [f["name"] for f in config["features"]]
    assert seen[1][2] == ["TrainingTimesLastYear", "YearsWithCurrManager"]
