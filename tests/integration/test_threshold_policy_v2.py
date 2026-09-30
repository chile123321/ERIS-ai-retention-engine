"""Capacity/tie rules and inner-only policy selection with unchanged outer probabilities."""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from eris_ml.evaluation import threshold_policy_v2 as policy
from eris_ml.evaluation.feature_ablation import GROUPS, LOGISTIC_PARAMETERS, REMOVED
from eris_ml.features.definitions import load_feature_definition
from eris_ml.models.calibration_v2 import FEATURES, calibration_splits

ROOT = Path(__file__).resolve().parents[2]
DEFINITIONS = {
    name: load_feature_definition(ROOT / f"configs/features/feature_set_{name}.yaml")
    for name in REMOVED
}
LOGISTIC = {"model": "logistic_regression", "random_seed": 42, "parameters": LOGISTIC_PARAMETERS}


def fixture() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(62)
    frame = pd.DataFrame(
        {name: rng.uniform(1, 20, 60) for group in ("numeric", "ordinal") for name in GROUPS[group]}
    )
    for name in GROUPS["nominal"]:
        frame[name] = [f"{name}-{i}" for i in range(60)]
    frame.loc[0, "Age"] = np.nan
    frame["Attrition"] = np.tile([0, 0, 1], 20)
    frame["source_row"], frame["EmployeeNumber"] = np.arange(60), np.arange(100, 160)
    assignment = frame[policy.IDS[:3]].copy()
    assignment["fold"], assignment["development_position"] = (
        np.repeat([3, 1, 5, 2, 4], 12),
        range(60),
    )
    tables = []
    for feature in REMOVED:
        for model in ("dummy_prior", "logistic_regression"):
            table = assignment[policy.IDS].copy()
            table["feature_set"], table["model"] = feature, model
            table["probability"] = rng.uniform(0.05, 0.95, 60)
            tables.append(table)
    baseline = pd.concat(tables, ignore_index=True)
    raw = baseline.loc[
        baseline.feature_set.isin(FEATURES) & (baseline.model == "logistic_regression")
    ].copy()
    raw = raw.rename(columns={"model": "method"})
    raw["method"] = "raw"
    return frame, assignment, baseline, raw


def test_inner_isolation_features_and_reproducibility(monkeypatch: pytest.MonkeyPatch) -> None:
    data, assignment, baseline, raw = fixture()
    fit_original, predict_original = Pipeline.fit, Pipeline.predict_proba
    select_original = policy.select_capacity
    state = {"fits": 0, "selections": 0}
    predict_indices: dict[int, list[int]] = {}

    def fit(self: Pipeline, features: pd.DataFrame, target: Any, **kwargs: Any) -> Any:
        if "model" not in self.named_steps:
            return fit_original(self, features, target, **kwargs)
        position = state["fits"] % 40
        name = FEATURES[position // 20]
        fold, inner_fold = position % 20 // 4 + 1, position % 4
        outer_train = np.flatnonzero(assignment.fold != fold)
        train, held = calibration_splits(data.Attrition.to_numpy()[outer_train])[inner_fold]
        assert features.index.tolist() == outer_train[train].tolist()
        assert list(features.columns) == list(DEFINITIONS[name].all_features)
        assert not set(features.columns) & REMOVED[name]
        assert np.array_equal(target, data.Attrition.to_numpy()[outer_train[train]])
        predict_indices[id(self)] = outer_train[held].tolist()
        state["fits"] += 1
        value = fit_original(self, features, target, **kwargs)
        preprocessor = self.named_steps["preprocessor"]
        encoder = preprocessor.named_transformers_["nominal"].named_steps["encoder"]
        for column, categories in zip(DEFINITIONS[name].nominal, encoder.categories_, strict=True):
            assert set(categories) == set(features[column])
        for group in ("numeric", "ordinal"):
            columns = list(getattr(DEFINITIONS[name], group))
            fitted = preprocessor.named_transformers_[group].named_steps
            medians = features[columns].median()
            assert np.allclose(fitted["imputer"].statistics_, medians)
            assert np.allclose(fitted["scaler"].mean_, features[columns].fillna(medians).mean())
        return value

    def predict(self: Pipeline, features: pd.DataFrame, **kwargs: Any) -> Any:
        assert features.index.tolist() == predict_indices[id(self)]
        return predict_original(self, features, **kwargs)

    def select(target: np.ndarray, scores: np.ndarray, percent: int) -> dict[str, Any]:
        fold = state["selections"] % 15 // 3 + 1
        assert np.array_equal(target, data.loc[assignment.fold != fold, "Attrition"])
        state["selections"] += 1
        chosen = select_original(target, scores, percent)
        assert chosen["inner_alerts"] <= policy.budget(len(target), percent)
        return chosen

    monkeypatch.setattr(Pipeline, "fit", fit)
    monkeypatch.setattr(Pipeline, "predict_proba", predict)
    monkeypatch.setattr(policy, "select_capacity", select)
    first = policy.evaluate(data, DEFINITIONS, assignment, baseline, raw, LOGISTIC)
    second = policy.evaluate(data, DEFINITIONS, assignment, baseline, raw, LOGISTIC)
    assert state["fits"] == 80 and state["selections"] == 60
    for key in first:
        pd.testing.assert_frame_equal(first[key], second[key])
    policy.validate_policy_oof(first["oof"], assignment)
    assert len(first["oof"]) == 840 and len(first["inner_oof"]) == 480
    tables = policy.summarize(first["oof"], first["selections"])
    assert len(tables["summary"]) == 14 and len(tables["folds"]) == 70
    assert not tables["folds"].loc[tables["folds"].policy == "top_k", "budget_exceeded"].any()


def test_outer_labels_do_not_change_selected_policy() -> None:
    data, assignment, baseline, raw = fixture()
    first = policy.evaluate(data, DEFINITIONS, assignment, baseline, raw, LOGISTIC)
    changed_ids = set(assignment.loc[assignment.fold == 1, "source_row"])
    for frame in (data, assignment, baseline, raw):
        mask = frame.source_row.isin(changed_ids)
        frame.loc[mask, "Attrition"] = 1 - frame.loc[mask, "Attrition"]
    second = policy.evaluate(data, DEFINITIONS, assignment, baseline, raw, LOGISTIC)
    pd.testing.assert_frame_equal(
        first["selections"].query("fold == 1"), second["selections"].query("fold == 1")
    )
    a = first["oof"].query("fold == 1").drop(columns="Attrition")
    b = second["oof"].query("fold == 1").drop(columns="Attrition")
    pd.testing.assert_frame_equal(a, b)


def test_capacity_threshold_boundaries_and_ties() -> None:
    labels = np.array([1, 1, 0] + [0] * 17)
    scores = np.array([0.9, 0.8, 0.7] + [0.1] * 17)
    chosen = policy.select_capacity(labels, scores, 15)
    assert chosen["threshold"] == 0.8 and chosen["inner_alerts"] == 2
    assert (scores >= chosen["threshold"]).sum() == 2
    tied = policy.select_capacity(labels, np.ones(20), 5)
    assert tied["no_alert"] and tied["threshold"] == policy.NO_ALERT
    assert tied["inner_alerts"] == 0  # do not split a boundary tie to meet capacity
    zero = policy.select_capacity(np.zeros(20), np.zeros(20), 5)
    assert zero["threshold"] == policy.NO_ALERT
    with pytest.raises(ValueError):
        policy.select_capacity(labels, scores, 20)
    with pytest.raises(ValueError):
        policy.select_capacity(labels, np.full(20, np.inf), 5)
    assert policy.operational(np.array([1, 0]), np.array([0, 0]))["reviews_per_tp"] is None
    assert policy.operational(np.array([1, 0]), np.array([1, 1]))["reviews_per_tp"] == 2


def test_top_k_rounding_stable_id_ties_and_no_label_input() -> None:
    scores, ids = np.ones(20), np.arange(20)[::-1]
    chosen = policy.top_k(scores, ids, 15)
    assert set(ids[chosen == 1]) == {0, 1, 2}
    assert chosen.sum() == 3
    assert policy.top_k(np.ones(10), np.arange(10), 5).sum() == 0
    assert policy.budget(236, 5) == 11 and policy.budget(235, 15) == 35
    with pytest.raises(ValueError):
        policy.top_k(scores, np.zeros(20), 5)
    with pytest.raises(ValueError):
        policy.top_k(np.full(20, 1.1), ids, 5)


@pytest.mark.parametrize("damage", ["probability", "target", "fold", "duplicate"])
def test_drift_stops_before_any_fit(damage: str, monkeypatch: pytest.MonkeyPatch) -> None:
    data, assignment, baseline, raw = fixture()
    raw = raw.reset_index(drop=True)
    if damage == "probability":
        raw.loc[0, "probability"] = 0.999
    elif damage == "target":
        raw.loc[0, "Attrition"] = 1
    elif damage == "fold":
        raw.loc[0, "fold"] = 4
    else:
        raw.loc[1] = raw.loc[0]

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("No fit after baseline/calibration drift")

    monkeypatch.setattr(Pipeline, "fit", forbidden)
    with pytest.raises(ValueError):
        policy.evaluate(data, DEFINITIONS, assignment, baseline, raw, LOGISTIC)


def test_final_guard(monkeypatch: pytest.MonkeyPatch) -> None:
    from scripts.evaluate_threshold_policy_v2 import run

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("No file access")

    monkeypatch.setattr(Path, "open", forbidden)
    monkeypatch.setattr(Path, "is_symlink", forbidden)
    with pytest.raises(ValueError, match="Locked final test"):
        run(Path("data/processed/final_test_raw_v1.csv"))
