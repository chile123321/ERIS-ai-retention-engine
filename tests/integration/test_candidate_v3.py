"""Synthetic-only selection isolation, RF probability SHAP and immutable bundle tests."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.impute import SimpleImputer

from eris_ml.features.schema_v3 import V3InputValidator, load_protocol
from eris_ml.models import baseline_v3 as baseline
from eris_ml.models import candidate_v3 as candidate
from eris_ml.models.bundle_v3 import (
    CandidateBundle,
    explain_one,
    feature_mapping,
    load_bundle,
    predict_batch,
    predict_one,
    save_bundle,
)
from eris_ml.utils.hashing import sha256_file

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def setup():
    config = load_protocol(ROOT / "configs/features/feature_set_v3_final_15.yaml")
    rng = np.random.default_rng(17)
    frame = pd.DataFrame(
        {
            f["name"]: rng.choice(f.get("vocabulary", f.get("values", [1, 2, 3])), 120)
            for f in config["features"]
        }
    )
    frame["Age"] = rng.integers(18, 62, 120)
    frame["Attrition"] = [0, 0, 0, 1] * 30
    frame["source_row"] = np.arange(100, 220)
    frame["EmployeeNumber"] = np.arange(1000, 1120)
    parameters = {"model__n_estimators": 7, "model__max_depth": 3, "model__min_samples_leaf": 2}
    return config, frame, parameters


@pytest.fixture
def fitted(setup):
    config, frame, params = setup
    pipeline, result = candidate.select_and_fit(frame, config, [params], forest_jobs=1)
    X = frame[[f["name"] for f in config["features"]]]
    positions = np.random.default_rng(42).choice(len(X), 100, replace=False).tolist()
    meta = candidate.bundle_metadata(config, result, {"synthetic": True}, "btest", positions)
    bundle = CandidateBundle(
        pipeline, meta, pipeline[:-1].transform(X.iloc[positions]), feature_mapping(pipeline)
    )
    return bundle, X


def test_cv_isolation_medians_and_first_candidate_tie(setup, monkeypatch):
    config, frame, params = setup
    splits = candidate.tuning.inner_splits(frame.Attrition.to_numpy())
    expected = [set(frame.iloc[t].index) for t, _ in splits] * 2 + [set(frame.index)]
    validator_calls, median_calls = [], []
    validate_fit, impute_fit = V3InputValidator.fit, SimpleImputer.fit

    def validator(self, X, y=None):
        assert list(X.columns) == [f["name"] for f in config["features"]]
        assert not {"source_row", "EmployeeNumber", "Attrition", "Gender", "MaritalStatus"} & set(X)
        validator_calls.append(set(X.index))
        return validate_fit(self, X, y)

    def imputer(self, X, y=None):
        median_calls.append(set(X.index))
        result = impute_fit(self, X, y)
        np.testing.assert_array_equal(result.statistics_, X.median().to_numpy())
        return result

    monkeypatch.setattr(V3InputValidator, "fit", validator)
    monkeypatch.setattr(SimpleImputer, "fit", imputer)
    _, result = candidate.select_and_fit(frame, config, [params, params], forest_jobs=1)
    assert validator_calls == median_calls == expected
    assert result["best_index"] == 0
    assert result["cv_fits"] == 8 and result["refits"] == 1
    assert len(result["cv_results"]["params"]) == 2


@pytest.mark.parametrize(
    "missing",
    [
        [],
        ["TrainingTimesLastYear"],
        ["YearsWithCurrManager"],
        ["TrainingTimesLastYear", "YearsWithCurrManager"],
    ],
)
def test_missing_one_batch_key_order_and_shap(fitted, missing):
    bundle, X = fitted
    record = X.iloc[0].to_dict()
    for name in missing:
        record.pop(name)
    one = predict_one(bundle, record)
    batch = predict_batch(bundle, [record, dict(reversed(list(record.items())))])
    assert batch == [one, one]
    assert set(one["imputed_features"]) == set(missing)
    assert predict_one(bundle, {**record, **dict.fromkeys(missing)}) == one
    explanation = explain_one(bundle, record)
    assert set(explanation["contributions"]) == set(X)
    assert len(bundle.mapping) == 18 and set(bundle.mapping) == set(X)
    assert explanation["probability"] == one["probability"]
    assert explanation["base_value"] + sum(explanation["contributions"].values()) == pytest.approx(
        one["probability"], abs=1e-6, rel=1e-6
    )
    model = bundle.pipeline.named_steps["model"]
    index = list(model.classes_).index(1)
    assert one["probability"] == bundle.pipeline.predict_proba(bundle.inputs(record))[0, index]


@pytest.mark.parametrize(
    "field,value",
    [
        ("Age", None),
        ("Age", "30"),
        ("Age", True),
        ("Age", float("nan")),
        ("Department", "New department"),
        ("TrainingTimesLastYear", float("inf")),
        ("Gender", "Female"),
        ("EmployeeNumber", 1),
    ],
)
def test_invalid_inputs(fitted, field, value):
    bundle, X = fitted
    with pytest.raises(ValueError):
        predict_one(bundle, {**X.iloc[0].to_dict(), field: value})


def test_fresh_process_manifest_and_no_overwrite(fitted, tmp_path):
    bundle, X = fitted
    directory, report = tmp_path / "bundle", tmp_path / "report"
    save_bundle(bundle, directory)
    hashes = {p.name: sha256_file(p) for p in directory.iterdir()}
    result = candidate.verify_runtime(bundle, directory, X, report)
    assert result["fresh_process_max_absolute_difference"] <= 1e-12
    with pytest.raises(FileExistsError):
        save_bundle(bundle, directory)
    assert hashes == {p.name: sha256_file(p) for p in directory.iterdir()}
    with (directory / "background.npy").open("ab") as stream:
        stream.write(b"corruption")
    with pytest.raises(ValueError, match="checksum"):
        load_bundle(directory)


def test_mapping_missing_column_and_metadata_rejected(fitted):
    bundle, X = fitted
    with pytest.raises(ValueError, match="Age"):
        predict_one(bundle, X.drop(columns="Age").iloc[0].to_dict())
    with pytest.raises(ValueError, match="unique"):
        predict_batch(bundle, pd.concat([X, X[["Age"]]], axis=1))
    bundle.mapping[0] = "Gender"
    with pytest.raises(ValueError, match="mapping"):
        bundle.validate()


def test_alignment_subgroup_denominators_and_guard(setup, tmp_path):
    _, frame, _ = setup
    audit = frame[[*baseline.IDS, "Age"]].assign(
        split="development", Gender="Female", MaritalStatus="Single"
    )
    candidate.aligned_audit(frame, audit.sample(frac=1, random_state=2))
    bad = audit.copy()
    bad.loc[0, "split"] = "final_test"
    with pytest.raises(ValueError, match="development-only"):
        candidate.aligned_audit(frame, bad)
    with pytest.raises(ValueError):
        baseline.safe_path(tmp_path, "final_test_raw_v1.csv")
    cohort = frame[[*baseline.IDS, "Age"]].copy()
    cohort["Age"] = 35
    raw = (
        frame[baseline.IDS]
        .rename(columns={"Attrition": "y_true"})
        .assign(probability=0.2, fold=np.tile(np.arange(1, 6), 24), top_10=0)
    )
    table = candidate.subgroup_table(raw, cohort)
    empty = table.loc[table.n == 0]
    assert empty[["tpr", "fpr", "average_precision", "brier"]].isna().all().all()
    pooled = table.loc[table.fold == 0]
    assert pooled.n.sum() == 120 and pooled.positives.sum() == 30
    assert pooled.alerts.sum() == 0
    assert table.precision.isna().all()
    raw["y_true"] = 0
    cohort["Attrition"] = 0
    single = candidate.subgroup_table(raw, cohort)
    assert single.average_precision.isna().all() and single.tpr.isna().all()
    raw.loc[0, "EmployeeNumber"] = -1
    with pytest.raises(ValueError, match="alignment"):
        candidate.subgroup_table(raw, cohort)


def test_existing_run_rejected_before_preflight(tmp_path, monkeypatch):
    (tmp_path / "artifacts/reports/v3_final_15/b1").mkdir(parents=True)

    def never_prepare(root):
        pytest.fail("Existing output should be rejected before any data read or fit.")

    monkeypatch.setattr(candidate, "prepare", never_prepare)
    with pytest.raises(FileExistsError):
        candidate.run(tmp_path, "b1")
