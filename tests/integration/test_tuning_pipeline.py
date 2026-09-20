"""Small synthetic end-to-end checks for nested tuning; no real data is loaded."""

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from eris_ml.evaluation.tuning_reporting import write_tuning_report
from eris_ml.features.definitions import load_feature_definition
from eris_ml.models.tuning import (
    candidate_configuration,
    load_yaml_config,
    run_full_development_search,
    run_nested_cv,
)

ROOT = Path(__file__).resolve().parents[2]


def _fixture_frame() -> pd.DataFrame:
    generator = np.random.default_rng(42)
    rows = 80
    target = np.zeros(rows, dtype=int)
    target[generator.choice(rows, size=20, replace=False)] = 1
    definition = load_feature_definition(ROOT / "configs/features/feature_set_v1_full.yaml")
    frame = pd.DataFrame(index=range(rows))
    for column in definition.nominal:
        frame[column] = generator.choice(["A", "B"], rows)
    for column in definition.ordinal:
        frame[column] = generator.integers(1, 5, rows)
    for column in definition.numeric:
        frame[column] = generator.normal(0, 1, rows)
    frame["Age"] += target * 1.5
    frame["Attrition"] = target
    frame["source_row"] = np.arange(1, rows + 1)
    frame["EmployeeNumber"] = np.arange(1001, 1001 + rows)
    return frame


def _small_configs():
    protocol = load_yaml_config(ROOT / "configs/tuning/tuning_protocol_v1.yaml")
    protocol["outer_cv"]["n_splits"] = 2
    protocol["inner_cv"]["n_splits"] = 2
    protocol["execution"]["search_n_jobs"] = 1
    logistic = load_yaml_config(ROOT / "configs/tuning/logistic_regression_search_v1.yaml")
    logistic["search_space"] = {"penalty": ["l1"], "C": [0.1, 1.0]}
    xgboost = load_yaml_config(ROOT / "configs/tuning/xgboost_search_v1.yaml")
    xgboost["n_iter"] = 2
    xgboost["search_space"] = {
        name: [5, 10] if name == "n_estimators" else [values[0]]
        for name, values in xgboost["search_space"].items()
    }
    baselines = {
        "logistic_regression": load_yaml_config(ROOT / "configs/models/logistic_regression.yaml"),
        "xgboost": load_yaml_config(ROOT / "configs/models/xgboost.yaml"),
    }
    return protocol, {"logistic_regression": logistic, "xgboost": xgboost}, baselines


def test_nested_cv_full_search_report_and_candidate_configs(tmp_path) -> None:
    frame = _fixture_frame()
    definition = load_feature_definition(ROOT / "configs/features/feature_set_v1_full.yaml")
    features = frame.loc[:, definition.all_features]
    protocol, search_configs, baselines = _small_configs()

    first = run_nested_cv(
        features, frame["Attrition"], definition, baselines, search_configs, protocol
    )
    second = run_nested_cv(
        features, frame["Attrition"], definition, baselines, search_configs, protocol
    )

    assert set(first["oof_probabilities"]) == {
        "logistic_baseline", "logistic_tuned", "xgboost_baseline", "xgboost_tuned"
    }
    assert set(first["outer_fold"]) == {1, 2}
    for name, probabilities in first["oof_probabilities"].items():
        assert len(probabilities) == len(frame)
        assert np.isfinite(probabilities).all()
        assert ((probabilities >= 0) & (probabilities <= 1)).all()
        assert np.allclose(probabilities, second["oof_probabilities"][name])
        assert len(first["fold_records"][name]) == 2
        assert first["fold_summary"][name]["overfitting_gap"]["mean"] == np.mean(
            [row["overfitting_gap"] for row in first["fold_records"][name]]
        )
    for name in ("logistic_tuned", "xgboost_tuned"):
        assert all(row["best_parameters"] for row in first["fold_records"][name])
        assert all(row["search_time_seconds"] >= 0 for row in first["fold_records"][name])

    full = run_full_development_search(
        features, frame["Attrition"], definition, search_configs, protocol
    )
    for family in ("logistic_regression", "xgboost"):
        candidate = candidate_configuration(
            family, search_configs[family], full[family]["best_parameters"]
        )
        path = tmp_path / f"{family}_tuned_v1.yaml"
        path.write_text(yaml.safe_dump(candidate), encoding="utf-8")
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert loaded["role"] == "tuned_candidate"
        if family == "logistic_regression":
            assert loaded["parameters"]["class_weight"] is None
        else:
            assert loaded["parameters"]["scale_pos_weight"] == 1.0

    metadata = {
        "development_rows": len(frame), "negative_count": 60, "positive_count": 20,
        "feature_set_version": definition.version, "feature_count": 25,
        "outer_splits": 2, "inner_splits": 2, "outer_seed": 42, "inner_seed": 43,
        "threshold": 0.5, "development_sha256": "synthetic", "python_version": "test",
        "pandas_version": "test", "sklearn_version": "test", "xgboost_version": "test",
        "logistic_combinations": 2,
    }
    destination = tmp_path / "tuning.md"
    write_tuning_report(first, full, search_configs, metadata, destination)
    report = destination.read_text(encoding="utf-8")
    for label in ("Logistic baseline", "Logistic tuned", "XGBoost baseline", "XGBoost tuned"):
        assert label in report
    assert "Locked final test accessed: No" in report
    assert "Full-development search" in report
    assert not list(tmp_path.glob("*.joblib"))
