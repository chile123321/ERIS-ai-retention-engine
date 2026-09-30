"""Small synthetic nested searches prove inner/outer isolation, alignment and determinism."""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from eris_ml.evaluation.feature_ablation import GROUPS, REMOVED
from eris_ml.features.definitions import load_feature_definition
from eris_ml.models.logistic_nested_v2 import (
    FEATURES,
    make_inner_splits,
    run_nested,
    select_best,
    validate_nested_oof,
)
from eris_ml.models.tuning import LOGISTIC_FIXED

ROOT = Path(__file__).resolve().parents[2]
DEFINITIONS = {name: load_feature_definition(ROOT / f"configs/features/feature_set_{name}.yaml")
               for name in REMOVED}
SEARCH = {"model": "logistic_regression", "search_type": "grid", "scoring": "average_precision",
          "fixed_parameters": LOGISTIC_FIXED, "search_space": {"C": [.1], "penalty": ["l1", "l2"]}}


def fixture() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(82)
    data = pd.DataFrame({name: rng.uniform(0, 20, 60)
                         for group in ("numeric", "ordinal") for name in GROUPS[group]})
    for name in GROUPS["nominal"]:
        data[name] = [f"{name}-{i}" for i in range(60)]
    data.loc[0, "Age"] = np.nan
    data["source_row"], data["EmployeeNumber"] = np.arange(60), np.arange(1000, 1060)
    data["Attrition"] = np.tile([0, 0, 1], 20)
    assignment = data[["source_row", "EmployeeNumber", "Attrition"]].copy()
    assignment["fold"] = np.repeat([3, 1, 5, 2, 4], 12)
    assignment["development_position"] = range(60)
    tables = []
    for name in REMOVED:
        for model in ("dummy_prior", "logistic_regression"):
            part = assignment.drop(columns="development_position").copy()
            part["feature_set"], part["model"], part["probability"] = name, model, .3
            tables.append(part)
    return data, assignment, pd.concat(tables, ignore_index=True)


@pytest.mark.filterwarnings("ignore:.*penalty.*:FutureWarning")
@pytest.mark.filterwarnings("ignore:.*l1_ratio.*:UserWarning")
def test_nested_fit_isolation_and_determinism(monkeypatch: pytest.MonkeyPatch) -> None:
    data, assignment, baseline = fixture()
    original_search, original_fit = GridSearchCV.fit, Pipeline.fit
    original_predict = Pipeline.predict_proba
    state: dict[str, Any] = {"searches": 0, "fits": 0}
    predictions: dict[int, list[int]] = {}

    def search_fit(self: Any, features: pd.DataFrame, target: Any, **kwargs: Any) -> Any:
        position = state["searches"] % 10
        feature, fold = FEATURES[position // 5], position % 5 + 1
        train = np.flatnonzero(assignment.fold.to_numpy() != fold)
        validation = np.flatnonzero(assignment.fold.to_numpy() == fold)
        assert features.index.tolist() == train.tolist()
        assert list(features.columns) == list(DEFINITIONS[feature].all_features)
        assert len(features.columns) == (16 if feature == FEATURES[0] else 17)
        assert "Department" not in features
        if feature == FEATURES[0]:
            assert "JobRole" not in features
        expected = make_inner_splits(np.asarray(target))
        for (actual_t, actual_v), (expected_t, expected_v) in zip(self.cv, expected, strict=True):
            assert np.array_equal(actual_t, expected_t) and np.array_equal(actual_v, expected_v)
            assert not set(train[actual_t]) & set(validation)
            assert not set(train[actual_v]) & set(validation)
        state["allowed"] = {tuple(train[t]): train[v].tolist() for t, v in expected}
        state["allowed"][tuple(train)] = validation.tolist()
        state["definition"] = DEFINITIONS[feature]
        state["searches"] += 1
        return original_search(self, features, target, **kwargs)

    def fit(self: Pipeline, features: pd.DataFrame, target: Any, **kwargs: Any) -> Any:
        if "model" not in self.named_steps:
            return original_fit(self, features, target, **kwargs)
        assert tuple(features.index) in state["allowed"]
        assert list(features.columns) == list(state["definition"].all_features)
        assert np.array_equal(target, data.loc[features.index, "Attrition"])
        predictions[id(self)] = state["allowed"][tuple(features.index)]
        state["fits"] += 1
        value = original_fit(self, features, target, **kwargs)
        preprocessor = self.named_steps["preprocessor"]
        encoder = preprocessor.named_transformers_["nominal"].named_steps["encoder"]
        for column, categories in zip(
            state["definition"].nominal, encoder.categories_, strict=True,
        ):
            assert set(categories) == set(features[column])
        for group in ("numeric", "ordinal"):
            cols = list(getattr(state["definition"], group))
            learned = preprocessor.named_transformers_[group].named_steps
            medians = features[cols].median()
            assert np.allclose(learned["imputer"].statistics_, medians)
            assert np.allclose(learned["scaler"].mean_, features[cols].fillna(medians).mean())
        return value

    def predict(self: Pipeline, features: pd.DataFrame, **kwargs: Any) -> Any:
        assert features.index.tolist() == predictions[id(self)]
        return original_predict(self, features, **kwargs)

    monkeypatch.setattr(GridSearchCV, "fit", search_fit)
    monkeypatch.setattr(Pipeline, "fit", fit)
    monkeypatch.setattr(Pipeline, "predict_proba", predict)
    first = run_nested(data, DEFINITIONS, assignment, baseline, SEARCH)
    second = run_nested(data, DEFINITIONS, assignment, baseline, SEARCH)
    assert state["searches"] == 20 and state["fits"] == 180
    pd.testing.assert_frame_equal(first.oof, second.oof)
    pd.testing.assert_frame_equal(first.selected, second.selected)
    assert len(first.oof) == 240 and len(first.search) == 20
    assert len(first.inner_assignments) == 240
    assert first.oof.groupby(["feature_set", "model"]).size().eq(60).all()
    assert np.isfinite(first.oof.probability).all() and first.oof.probability.between(0, 1).all()
    broken = first.oof.copy()
    broken.loc[0, "probability"] = np.inf
    with pytest.raises(ValueError):
        validate_nested_oof(broken, assignment)


@pytest.mark.parametrize("damage", ["id", "target", "fold", "order"])
def test_reference_drift_stops_before_fit(damage: str, monkeypatch: pytest.MonkeyPatch) -> None:
    data, assignment, baseline = fixture()
    if damage == "id":
        baseline.loc[60, "EmployeeNumber"] = -1
    elif damage == "target":
        baseline.loc[60, "Attrition"] = 1
    elif damage == "fold":
        baseline.loc[60, "fold"] = 4
    else:
        assignment = assignment.iloc[::-1].reset_index(drop=True)

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("Must reject before fitting")

    monkeypatch.setattr(GridSearchCV, "fit", forbidden)
    with pytest.raises(ValueError):
        run_nested(data, DEFINITIONS, assignment, baseline, SEARCH)


def test_ties_and_invalid_scores() -> None:
    params = [{"model__C": c, "model__penalty": p} for c, p in [(1, "l2"), (.1, "l1"), (.1, "l2")]]
    assert select_best({"params": params, "mean_test_score": [.7, .7, .7 - 1e-13]}) == 2
    assert select_best({"params": params, "mean_test_score": [.8, .7, .7]}) == 0
    with pytest.raises(ValueError):
        select_best({"params": params, "mean_test_score": [.8, np.nan, .7]})


def test_final_path_guard(monkeypatch: pytest.MonkeyPatch) -> None:
    from scripts.tune_logistic_v2 import run

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("No filesystem access")

    monkeypatch.setattr(Path, "open", forbidden)
    monkeypatch.setattr(Path, "is_symlink", forbidden)
    with pytest.raises(ValueError, match="Locked final test"):
        run(Path("data/processed/final_test_raw_v1.csv"))
