"""Synthetic proof of calibration score provenance, raw regression and OOF integrity."""

from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import sklearn.calibration as calibration
from sklearn.pipeline import Pipeline

from eris_ml.evaluation.calibration_v2_reporting import choose_method, reliability, summarize
from eris_ml.evaluation.feature_ablation import GROUPS, LOGISTIC_PARAMETERS, REMOVED
from eris_ml.features.definitions import load_feature_definition
from eris_ml.models.calibration_v2 import (
    FEATURES,
    PROTOCOL,
    calibration_splits,
    run_calibration,
    validate_calibration_oof,
    validate_protocol,
)
from eris_ml.models.factory import create_estimator
from eris_ml.models.training import build_model_pipeline

ROOT = Path(__file__).resolve().parents[2]
DEFINITIONS = {name: load_feature_definition(ROOT / f"configs/features/feature_set_{name}.yaml")
               for name in REMOVED}
LOGISTIC = {"model": "logistic_regression", "random_seed": 42, "parameters": LOGISTIC_PARAMETERS}


def fixture() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(12)
    data = pd.DataFrame({name: rng.uniform(0, 10, 60)
                         for group in ("numeric", "ordinal") for name in GROUPS[group]})
    for name in GROUPS["nominal"]:
        data[name] = [f"{name}-{i}" for i in range(60)]
    data.loc[0, "Age"] = np.nan
    data["source_row"], data["EmployeeNumber"] = np.arange(60), np.arange(100, 160)
    data["Attrition"] = np.tile([0, 0, 1], 20)
    assignment = data[["source_row", "EmployeeNumber", "Attrition"]].copy()
    assignment["fold"] = np.repeat([3, 1, 5, 2, 4], 12)
    assignment["development_position"] = np.arange(60)
    tables = []
    for feature in REMOVED:
        probability = np.full(60, .3)
        if feature in FEATURES:
            for fold in range(1, 6):
                train, validation = assignment.fold != fold, assignment.fold == fold
                model = build_model_pipeline(DEFINITIONS[feature], create_estimator(LOGISTIC))
                features = data[list(DEFINITIONS[feature].all_features)]
                model.fit(features.loc[train], data.loc[train, "Attrition"])
                probability[validation] = model.predict_proba(features.loc[validation])[:, 1]
        for name in ("dummy_prior", "logistic_regression"):
            table = assignment.drop(columns="development_position").copy()
            table["feature_set"], table["model"] = feature, name
            table["probability"] = probability if name == "logistic_regression" else .3
            tables.append(table)
    return data, assignment, pd.concat(tables, ignore_index=True)


def test_calibrator_inner_oof_provenance_and_reproducibility(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data, assignment, baseline = fixture()
    original_fit = Pipeline.fit
    original_decision = Pipeline.decision_function
    original_calibration_fit = calibration.CalibratedClassifierCV.fit
    original_calibrator = calibration._fit_calibrator
    state: dict[str, Any] = {"active": False, "calls": 0, "calibrators": 0, "raw_fits": 0}
    validation_by_model: dict[int, list[int]] = {}

    def calibrated_fit(self: Any, features: pd.DataFrame, target: Any, **kwargs: Any) -> Any:
        position = state["calls"] % 20
        feature, fold = FEATURES[position // 10], position % 5 + 1
        train = np.flatnonzero(assignment.fold.to_numpy() != fold)
        assert features.index.tolist() == train.tolist()
        assert list(features.columns) == list(DEFINITIONS[feature].all_features)
        expected = calibration_splits(np.asarray(target))
        for actual, wanted in zip(self.cv, expected, strict=True):
            assert all(np.array_equal(a, b) for a, b in zip(actual, wanted, strict=True))
        state.update(active=True, train=train, feature=feature, scores={}, full_fits=0)
        state["allowed"] = {tuple(train[t]): train[v].tolist() for t, v in expected}
        state["allowed"][tuple(train)] = []
        state["calls"] += 1
        result = original_calibration_fit(self, features, target, **kwargs)
        assert state["full_fits"] == 1 and len(self.calibrated_classifiers_) == 1
        state["active"] = False
        return result

    def fit(self: Pipeline, features: pd.DataFrame, target: Any, **kwargs: Any) -> Any:
        if "model" not in self.named_steps:
            return original_fit(self, features, target, **kwargs)
        if state["active"]:
            feature = state["feature"]
            assert tuple(features.index) in state["allowed"]
            validation_by_model[id(self)] = state["allowed"][tuple(features.index)]
            if tuple(features.index) == tuple(state["train"]):
                state["full_fits"] += 1
        else:
            position = state["raw_fits"] % 10
            feature, fold = FEATURES[position // 5], position % 5 + 1
            assert features.index.tolist() == np.flatnonzero(assignment.fold != fold).tolist()
            state["raw_fits"] += 1
        definition = DEFINITIONS[feature]
        assert list(features.columns) == list(definition.all_features)
        assert "Department" not in features
        if feature == FEATURES[0]:
            assert "JobRole" not in features and len(features.columns) == 16
        assert np.array_equal(target, data.loc[features.index, "Attrition"])
        value = original_fit(self, features, target, **kwargs)
        processor = self.named_steps["preprocessor"]
        encoder = processor.named_transformers_["nominal"].named_steps["encoder"]
        for column, categories in zip(definition.nominal, encoder.categories_, strict=True):
            assert set(categories) == set(features[column])
        for group in ("numeric", "ordinal"):
            columns = list(getattr(definition, group))
            steps = processor.named_transformers_[group].named_steps
            medians = features[columns].median()
            assert np.allclose(steps["imputer"].statistics_, medians)
            assert np.allclose(steps["scaler"].mean_, features[columns].fillna(medians).mean())
        return value

    def decision_function(self: Pipeline, features: pd.DataFrame, **kwargs: Any) -> Any:
        values = original_decision(self, features, **kwargs)
        if state["active"]:
            assert features.index.tolist() == validation_by_model[id(self)]
            for row, value in zip(features.index, values, strict=True):
                assert row not in state["scores"]
                state["scores"][row] = value
        return values

    def calibrator(clf: Any, predictions: Any, y: Any, *args: Any, **kwargs: Any) -> Any:
        assert state["active"] and state["full_fits"] == 1
        assert set(state["scores"]) == set(state["train"])
        assert np.array_equal(y, data.loc[state["train"], "Attrition"])
        expected = [state["scores"][i] for i in state["train"]]
        assert np.allclose(np.asarray(predictions).ravel(), expected)
        state["calibrators"] += 1
        return original_calibrator(clf, predictions, y, *args, **kwargs)

    monkeypatch.setattr(Pipeline, "fit", fit)
    monkeypatch.setattr(Pipeline, "decision_function", decision_function)
    monkeypatch.setattr(calibration.CalibratedClassifierCV, "fit", calibrated_fit)
    monkeypatch.setattr(calibration, "_fit_calibrator", calibrator)
    first, regression, inner = run_calibration(data, DEFINITIONS, assignment, baseline, LOGISTIC)
    second, _, _ = run_calibration(data, DEFINITIONS, assignment, baseline, LOGISTIC)
    pd.testing.assert_frame_equal(first, second)
    assert state["calibrators"] == 40 and state["raw_fits"] == 20
    assert regression.max_abs_difference.max() < 1e-12
    assert len(inner) == 240 and len(first) == 360
    assert first.groupby(["feature_set", "method"]).size().eq(60).all()
    tables = summarize(first)
    assert len(tables["summary"]) == 6 and len(tables["folds"]) == 30
    for _, bins in tables["bins"].groupby(["feature_set", "method"]):
        assert bins['count'].sum() == 60 and bins.positives.sum() == 20


@pytest.mark.parametrize("damage", ["probability", "target", "fold", "duplicate"])
def test_drift_stops_before_calibration(damage: str, monkeypatch: pytest.MonkeyPatch) -> None:
    data, assignment, baseline = fixture()
    row = baseline.index[(baseline.feature_set == FEATURES[0])
                         & (baseline.model == "logistic_regression")][0]
    if damage == "probability":
        baseline.loc[row, "probability"] = .999
    elif damage == "target":
        baseline.loc[row, "Attrition"] = 1
    elif damage == "fold":
        baseline.loc[row, "fold"] = 4
    else:
        baseline.loc[row + 1] = baseline.loc[row]

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("Calibrator must not fit after reference drift")

    monkeypatch.setattr(calibration.CalibratedClassifierCV, "fit", forbidden)
    with pytest.raises(ValueError):
        run_calibration(data, DEFINITIONS, assignment, baseline, LOGISTIC)


def test_reliability_duplicates_endpoints_and_guards() -> None:
    bins = reliability(np.array([0, 1, 0, 1]), np.array([0., 0., 1., 1.]))
    assert bins['count'].sum() == 4 and bins.positives.sum() == 2
    assert not bins.isna().any().any()
    assert len(reliability(np.array([0, 1]), np.array([.5, .5]))) == 1
    summary = pd.DataFrame([
        {"method": m, "average_precision": .6, "roc_auc": .8, "brier_score": b}
        for m, b in [("raw", .12), ("sigmoid", .11), ("isotonic", .13)]])
    folds = pd.concat([summary.assign(fold=i) for i in range(1, 6)], ignore_index=True)
    assert choose_method(summary, folds)[0] == "sigmoid"
    summary.loc[summary.method == "sigmoid", "average_precision"] = .59
    assert choose_method(summary, folds)[0] == "raw"
    summary.loc[summary.method == "sigmoid", "average_precision"] = .6
    summary.loc[summary.method == "sigmoid", "roc_auc"] = .79
    assert choose_method(summary, folds)[0] == "raw"
    summary.loc[summary.method == "sigmoid", "roc_auc"] = .8
    folds.loc[(folds.method == "sigmoid") & (folds.fold == 1), "brier_score"] = .14
    assert choose_method(summary, folds)[0] == "raw"


def test_protocol_tuned_config_and_final_guard(monkeypatch: pytest.MonkeyPatch) -> None:
    validate_protocol(PROTOCOL)
    bad = deepcopy(PROTOCOL)
    bad["ensemble"] = True
    with pytest.raises(ValueError):
        validate_protocol(bad)
    data, assignment, baseline = fixture()
    tuned = deepcopy(LOGISTIC)
    tuned["parameters"]["C"] = .001
    with pytest.raises(ValueError, match="unchanged Logistic baseline"):
        run_calibration(data, DEFINITIONS, assignment, baseline, tuned)
    from scripts.calibrate_v2 import run

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("No file access allowed")

    monkeypatch.setattr(Path, "open", forbidden)
    monkeypatch.setattr(Path, "is_symlink", forbidden)
    with pytest.raises(ValueError, match="Locked final test"):
        run(Path("data/processed/final_test_raw_v1.csv"))


def test_output_rejects_nonfinite_or_alignment() -> None:
    data, assignment, baseline = fixture()
    tables = []
    for feature in FEATURES:
        for method in ("raw", "sigmoid", "isotonic"):
            part = assignment.drop(columns="development_position").copy()
            part["feature_set"], part["method"], part["probability"] = feature, method, .3
            tables.append(part)
    oof = pd.concat(tables, ignore_index=True)
    validate_calibration_oof(oof, assignment)
    for column, value in [("probability", np.inf), ("probability", -1.), ("EmployeeNumber", -1)]:
        broken = oof.copy()
        broken.loc[0, column] = value
        with pytest.raises(ValueError):
            validate_calibration_oof(broken, assignment)
