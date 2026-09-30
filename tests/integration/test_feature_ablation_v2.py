"""Prove V2 fold/preprocessing isolation and complete comparable OOF coverage."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import yaml
from sklearn.pipeline import Pipeline

from eris_ml.evaluation.feature_ablation import (
    GROUPS,
    REMOVED,
    evaluate_feature_ablation,
    make_shared_folds,
    validate_definitions,
    validate_oof,
)
from eris_ml.evaluation.feature_ablation_reporting import build_report
from eris_ml.features.definitions import load_feature_definition

ROOT = Path(__file__).resolve().parents[2]
DEFINITIONS = {name: load_feature_definition(ROOT / f"configs/features/feature_set_{name}.yaml")
               for name in REMOVED}
LOGISTIC = yaml.safe_load((ROOT / "configs/models/logistic_regression.yaml").read_text())


def synthetic() -> pd.DataFrame:
    rng = np.random.default_rng(114)
    data = pd.DataFrame(index=np.arange(100))
    for name in GROUPS["numeric"]:
        data[name] = rng.uniform(1, 50, len(data))
    for name in GROUPS["ordinal"]:
        data[name] = rng.integers(1, 5, len(data))
    data["OverTime"] = rng.choice(["Yes", "No"], len(data))
    # Validation-only categories must never enter any encoder's learned vocabulary.
    data["Department"] = [f"department-{i}" for i in data.index]
    data["JobRole"] = [f"role-{i}" for i in data.index]
    data.loc[0, "Age"] = np.nan
    data["Attrition"] = np.tile([0, 0, 0, 0, 1], 20)
    data["source_row"] = np.arange(1000, 1100)
    data["EmployeeNumber"] = np.arange(2000, 2100)
    return data


def test_fold_isolation_feature_projection_and_learned_preprocessing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = synthetic()
    expected_folds = make_shared_folds(data.Attrition.to_numpy())
    original_fit, original_predict = Pipeline.fit, Pipeline.predict_proba
    fitted: list[Pipeline] = []
    fold_by_pipeline: dict[int, int] = {}

    def fit(self: Pipeline, features: pd.DataFrame, target: Any, **kwargs: Any) -> Any:
        if "model" not in self.named_steps:
            return original_fit(self, features, target, **kwargs)
        position = len(fitted)
        name = list(REMOVED)[position // 10]
        fold = position % 5 + 1
        definition = DEFINITIONS[name]
        train = np.flatnonzero(expected_folds != fold)
        assert list(features.columns) == list(definition.all_features)
        assert features.index.tolist() == train.tolist()
        assert set(features.columns).isdisjoint(REMOVED[name])
        assert np.array_equal(target, data.Attrition.to_numpy()[train])
        assert all(self is not old for old in fitted)
        fitted.append(self)
        fold_by_pipeline[id(self)] = fold
        result = original_fit(self, features, target, **kwargs)
        preprocessor = self.named_steps["preprocessor"]
        nominal = preprocessor.named_transformers_["nominal"].named_steps["encoder"]
        for column, categories in zip(definition.nominal, nominal.categories_, strict=True):
            assert set(categories) == set(features[column].dropna())
        for kind in ("numeric", "ordinal"):
            columns = list(getattr(definition, kind))
            part = preprocessor.named_transformers_[kind]
            median = features[columns].median().to_numpy()
            assert np.allclose(part.named_steps["imputer"].statistics_, median)
            imputed = features[columns].fillna(features[columns].median())
            assert np.allclose(part.named_steps["scaler"].mean_, imputed.mean().to_numpy())
        return result

    def predict(self: Pipeline, features: pd.DataFrame, **kwargs: Any) -> Any:
        fold = fold_by_pipeline[id(self)]
        assert features.index.tolist() == np.flatnonzero(expected_folds == fold).tolist()
        return original_predict(self, features, **kwargs)

    monkeypatch.setattr(Pipeline, "fit", fit)
    monkeypatch.setattr(Pipeline, "predict_proba", predict)
    result = evaluate_feature_ablation(data, DEFINITIONS, LOGISTIC)
    assert len(fitted) == 40
    assert len(result.oof) == 800
    assert result.oof.groupby(["feature_set", "model"]).size().eq(100).all()
    assert np.array_equal(result.assignments.fold, expected_folds)


def test_deterministic_oof_metrics_and_report() -> None:
    data = synthetic()
    original = data.copy(deep=True)
    first = evaluate_feature_ablation(data, DEFINITIONS, LOGISTIC)
    second = evaluate_feature_ablation(data, DEFINITIONS, LOGISTIC)
    pd.testing.assert_frame_equal(first.oof, second.oof)
    pd.testing.assert_frame_equal(data, original)
    assert first.oof.probability.between(0, 1).all()
    assert np.isfinite(first.oof.probability).all()
    # Dummy's validation probability must equal its training prevalence in each fold.
    for fold in range(1, 6):
        expected = data.loc[first.assignments.fold != fold, "Attrition"].mean()
        rows = first.oof.loc[(first.oof.model == "dummy_prior") & (first.oof.fold == fold)]
        assert np.allclose(rows.probability, expected)
    report = build_report(first, {
        "development_rows": 100, "target_distribution": {0: 80, 1: 20},
        "positive_rate": 0.2, "logistic_parameters": LOGISTIC,
    })
    assert all(name in report for name in REMOVED)
    assert "Paired Logistic fold differences" in report
    assert "ddof=1" in report


@pytest.mark.parametrize("damage", ["duplicate", "fold", "target", "inf", "missing"])
def test_oof_rejects_corruption(damage: str) -> None:
    # Construct a valid table without training; these checks guard persisted identity.
    data = synthetic()
    assignments = data[["source_row", "EmployeeNumber", "Attrition"]].copy()
    assignments["fold"] = make_shared_folds(data.Attrition.to_numpy())
    tables = []
    for name in REMOVED:
        for model in ("dummy_prior", "logistic_regression"):
            table = assignments.copy()
            table["feature_set"], table["model"], table["probability"] = name, model, 0.2
            tables.append(table)
    oof = pd.concat(tables, ignore_index=True)
    validate_oof(oof, assignments)
    if damage == "duplicate":
        oof.loc[1] = oof.loc[0]
    elif damage == "fold":
        oof.loc[0, "fold"] = 1 + int(oof.loc[0, "fold"]) % 5
    elif damage == "target":
        oof.loc[0, "Attrition"] = 1 - oof.loc[0, "Attrition"]
    elif damage == "inf":
        oof.loc[0, "probability"] = np.inf
    else:
        oof = oof.iloc[1:]
    with pytest.raises(ValueError):
        validate_oof(oof, assignments)


def test_schema_drift_and_final_path_fail_before_fit_or_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts.baseline_ablation_v2 import run

    altered = dict(DEFINITIONS)
    altered["v2_general_16"] = replace(DEFINITIONS["v2_general_16"],
                                       nominal=("Department", "OverTime"))
    with pytest.raises(ValueError, match="feature group"):
        validate_definitions(altered)

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("File inspection/fit must not occur")

    monkeypatch.setattr(Path, "is_symlink", forbidden)
    monkeypatch.setattr(Path, "open", forbidden)
    monkeypatch.setattr(Pipeline, "fit", forbidden)
    with pytest.raises(ValueError, match="Locked final test"):
        run(Path("data/processed/final_test_raw_v1.csv"))
