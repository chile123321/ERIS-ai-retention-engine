"""Small searches only: nested isolation, identity alignment and fail-closed resume."""

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.impute import SimpleImputer
from sklearn.model_selection import ParameterGrid, ParameterSampler
from sklearn.preprocessing import StandardScaler

from eris_ml.evaluation.feature_ablation import make_shared_folds
from eris_ml.evaluation.tuning_v3_reporting import append_summary
from eris_ml.features.schema_v3 import V3InputValidator, load_protocol
from eris_ml.models import baseline_v3 as baseline
from eris_ml.models import tuning_v3 as tuning

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def setup():
    config = load_protocol(ROOT / "configs/features/feature_set_v3_final_15.yaml")
    policy = tuning.load_search_policy(ROOT)
    policy["logistic"]["C"] = [0.1]
    policy["logistic"]["class_weight"] = [None]
    policy["random_forest"] = {
        "n_iter": 2,
        "distributions": {"n_estimators": [5], "max_depth": [2, None]},
    }
    policy["execution"]["forest_n_jobs"] = 1
    rng = np.random.default_rng(5)
    X = {}
    for spec in config["features"]:
        choices = spec.get("vocabulary", spec.get("values", list(range(1, 30))))
        X[spec["name"]] = rng.choice(choices, 80)
    frame = pd.DataFrame(X)
    frame["TrainingTimesLastYear"] = frame.TrainingTimesLastYear.astype(float)
    frame["YearsWithCurrManager"] = frame.YearsWithCurrManager.astype(float)
    frame.loc[::7, ["TrainingTimesLastYear", "YearsWithCurrManager"]] = np.nan
    frame["source_row"] = np.arange(100, 180)
    frame["EmployeeNumber"] = np.arange(1000, 1080)
    frame["Attrition"] = [0, 0, 0, 1] * 20
    assignments = frame[baseline.IDS].copy()
    assignments["fold"] = make_shared_folds(frame.Attrition.to_numpy())
    return config, policy, frame, assignments


def test_locked_production_budget_and_deterministic_candidates():
    policy = tuning.load_search_policy(ROOT)
    grid, forest, api = tuning.search_spaces(policy)
    assert len(list(ParameterGrid(grid))) == 20
    sampled = list(ParameterSampler(forest, n_iter=40, random_state=43))
    assert len(sampled) == 40
    assert sampled == list(ParameterSampler(forest, n_iter=40, random_state=43))
    assert "L1/L2" in api or "penalty=l1/l2" == api
    assert policy["scoring"] == "average_precision"
    assert not policy["full_development_fit"]


@pytest.mark.parametrize("model", tuning.MODELS)
def test_nested_preprocessing_isolation_and_refit(setup, monkeypatch, model):
    config, policy, frame, assignments = setup
    training_positions = np.flatnonzero(assignments.fold.to_numpy() != 1)
    expected = [
        set(frame.iloc[training_positions[t]].index)
        for t, _ in tuning.inner_splits(frame.iloc[training_positions].Attrition.to_numpy())
    ]
    expected.append(set(frame.iloc[training_positions].index))
    observed, imputed, scaled = [], [], []
    validator_fit, imputer_fit, scaler_fit = (
        V3InputValidator.fit,
        SimpleImputer.fit,
        StandardScaler.fit,
    )

    def spy_validator(self, X, y=None):
        assert len(X.columns) == 15
        assert not set(baseline.IDS + ["Gender", "MaritalStatus", "JobRole"]) & set(X.columns)
        observed.append(set(X.index))
        return validator_fit(self, X, y)

    def spy_imputer(self, X, y=None):
        imputed.append(set(X.index))
        fitted = imputer_fit(self, X, y)
        np.testing.assert_allclose(fitted.statistics_, X.median().to_numpy())
        return fitted

    def spy_scaler(self, X, y=None, sample_weight=None):
        if isinstance(X, pd.DataFrame):
            scaled.append(set(X.index))
        return scaler_fit(self, X, y, sample_weight=sample_weight)

    monkeypatch.setattr(V3InputValidator, "fit", spy_validator)
    monkeypatch.setattr(SimpleImputer, "fit", spy_imputer)
    monkeypatch.setattr(StandardScaler, "fit", spy_scaler)
    result = tuning.fit_outer(frame, assignments, config, policy, model, 1)
    assert result["candidate_count"] == 2
    assert len(observed) == 9 and len(imputed) == 9
    assert all(ids in expected for ids in observed + imputed + scaled)
    assert observed[-1] == expected[-1]
    assert result["optional_missing_smoke"] == "passed"
    assert result["outer_train_median_check"] == "passed"
    assert result["positive_class"] == 1
    heldout = set(assignments.loc[assignments.fold == 1, "source_row"])
    assert set(result["outer_train_source_rows"]).isdisjoint(heldout)
    assert all(set(ids).isdisjoint(heldout) for ids in result["inner_validation_source_rows"])
    assert set(p["source_row"] for p in result["predictions"]) == heldout


def test_end_to_end_small_searches_join_by_id_determinism_and_summary(setup, tmp_path):
    config, policy, frame, assignments = setup
    checkpoints = [
        tuning.fit_outer(frame, assignments, config, policy, m, f)
        for m in tuning.MODELS
        for f in range(1, 6)
    ]
    records = [p for c in checkpoints for p in c["predictions"]]
    oof = pd.DataFrame(records)
    aligned = tuning.aligned_oof(
        oof.sample(frac=1, random_state=9), assignments, list(tuning.MODELS)
    )
    assert aligned.groupby("model").size().to_dict() == dict.fromkeys(tuning.MODELS, 80)
    repeat = tuning.fit_outer(frame, assignments, config, policy, "logistic_regression", 1)
    np.testing.assert_allclose(
        [p["probability"] for p in repeat["predictions"]],
        [p["probability"] for p in checkpoints[0]["predictions"]],
        rtol=1e-12,
        atol=1e-14,
    )
    comparison = tuning.compare(oof.sample(frac=1, random_state=1), aligned, assignments)
    for model in tuning.MODELS:
        assert (
            comparison["comparisons"][model]["pooled_delta_tuned_minus_baseline"][
                "average_precision"
            ]
            == 0
        )
    # Report formatter expects the two historical references; use synthetic copies, no training.
    comparison["baseline"]["xgboost"] = comparison["baseline"]["random_forest"]
    comparison["baseline"]["dummy_prior"] = comparison["baseline"]["random_forest"]
    summary = tmp_path / "summary.md"
    summary.write_text("original baseline\n", encoding="utf-8")
    original_bytes = summary.read_bytes()
    append_summary(summary, "original baseline\n", "t9", comparison, checkpoints)
    first = summary.read_bytes()
    append_summary(summary, "original baseline\n", "t9", comparison, checkpoints)
    assert summary.read_bytes() == first
    assert first.startswith(original_bytes)
    with pytest.raises(ValueError, match="Summary changed"):
        append_summary(summary, "different baseline", "t9", comparison, checkpoints)


@pytest.mark.parametrize(
    "fault", ["duplicate", "missing", "target", "fold", "employee", "nan", "inf", "range"]
)
def test_oof_corruption_rejected(setup, fault):
    _, _, _, assignments = setup
    oof = assignments.rename(columns={"Attrition": "y_true"}).assign(
        feature_set="v3_final_15", model="logistic_regression", probability=0.2
    )
    if fault == "duplicate":
        oof.iloc[1] = oof.iloc[0]
    elif fault == "missing":
        oof = oof.iloc[1:]
    else:
        column, value = {
            "target": ("y_true", 1),
            "fold": ("fold", 9),
            "employee": ("EmployeeNumber", -1),
            "nan": ("probability", np.nan),
            "inf": ("probability", np.inf),
            "range": ("probability", 1.2),
        }[fault]
        oof.loc[0, column] = value
    with pytest.raises(ValueError, match="OOF"):
        tuning.aligned_oof(oof, assignments, ["logistic_regression"])


def test_checkpoint_identity_tampering_and_no_overwrite(tmp_path):
    path = tmp_path / "checkpoint.json"
    identity = tuning.digest({"data": "abc", "folds": [1, 2], "config": {"C": 1}})
    payload = {"model": "logistic_regression", "fold": 1, "probability": [0.1, 0.9]}
    tuning.save_checkpoint(path, identity, payload)
    assert tuning.load_checkpoint(path, identity) == payload
    with pytest.raises(FileExistsError):
        tuning.save_checkpoint(path, identity, payload)
    with pytest.raises(ValueError, match="identity/integrity"):
        tuning.load_checkpoint(path, tuning.digest({"data": "changed"}))
    modified = json.loads(path.read_text())
    modified["payload"]["probability"][0] = 0.8
    path.write_text(json.dumps(modified))
    with pytest.raises(ValueError, match="identity/integrity"):
        tuning.load_checkpoint(path, identity)


def test_changed_assignment_stops_before_fit(setup, monkeypatch):
    config, policy, frame, assignments = setup
    assignments = assignments.copy()
    assignments.loc[0, "Attrition"] = 1

    def forbidden(*args, **kwargs):
        pytest.fail("Model factory must not be reached for invalid alignment")

    monkeypatch.setattr(tuning, "search_pipeline", forbidden)
    with pytest.raises(ValueError, match="alignment"):
        tuning.fit_outer(frame, assignments, config, policy, "logistic_regression", 1)


def test_outer_validation_values_and_labels_cannot_change_selection(setup):
    config, policy, frame, assignments = setup
    original = tuning.fit_outer(frame, assignments, config, policy, "logistic_regression", 1)
    modified, modified_assignment = frame.copy(), assignments.copy()
    mask = assignments.fold == 1
    # Permute held-out labels within their fold; train labels and membership remain untouched.
    modified.loc[mask, "Attrition"] = np.roll(frame.loc[mask, "Attrition"].to_numpy(), 1)
    modified_assignment.loc[mask, "Attrition"] = modified.loc[mask, "Attrition"]
    modified.loc[mask, "MonthlyIncome"] = 999999
    after = tuning.fit_outer(
        modified, modified_assignment, config, policy, "logistic_regression", 1
    )
    assert original["best_params"] == after["best_params"]
    assert [r["split_test_scores"] for r in original["inner_cv_results"]] == [
        r["split_test_scores"] for r in after["inner_cv_results"]
    ]


def test_resume_artifact_preservation_and_path_guard(tmp_path):
    path = tmp_path / "metrics.json"
    tuning.preserve_json(path, {"AP": 0.5}, False)
    original = path.read_bytes()
    tuning.preserve_json(path, {"AP": 0.5}, True)
    assert path.read_bytes() == original
    with pytest.raises(ValueError, match="overwrite"):
        tuning.preserve_json(path, {"AP": 0.6}, True)
    with pytest.raises(ValueError):
        baseline.safe_path(tmp_path, "data/final_test_raw_v1.csv")
    with pytest.raises(ValueError):
        tuning.run(tmp_path, "r1")


def test_inner_cv_requires_both_classes():
    with pytest.raises(ValueError, match="four rows"):
        tuning.inner_splits(np.array([0] * 10 + [1] * 3))


def test_protocol_tampering_rejected(tmp_path):
    import yaml

    policy = tuning.load_search_policy(ROOT)
    modified = copy.deepcopy(policy)
    modified["random_forest"]["n_iter"] = 80
    target = tmp_path / tuning.PROTOCOL
    target.parent.mkdir(parents=True)
    target.write_text(yaml.safe_dump(modified))
    with pytest.raises(ValueError, match="budget/protocol"):
        tuning.load_search_policy(tmp_path)


def test_interrupted_run_resumes_only_unfinished_folds_and_completed_run_is_read_only(
    setup, tmp_path, monkeypatch
):
    config, policy, frame, assignments = setup
    reference = pd.concat(
        [
            assignments.rename(columns={"Attrition": "y_true"}).assign(
                feature_set="v3_final_15", model=m, probability=0.2
            )
            for m in ("logistic_regression", "random_forest", "xgboost", "dummy_prior")
        ],
        ignore_index=True,
    )
    identity = baseline.json_safe({"protected_sha256": {}, "config": config, "policy": policy})
    summary = tmp_path / "artifacts/reports/v3_final_15/summary.md"
    summary.parent.mkdir(parents=True)
    summary.write_text("baseline remains unchanged\n", encoding="utf-8")
    monkeypatch.setattr(tuning, "load_search_policy", lambda root: policy)
    monkeypatch.setattr(
        tuning,
        "prepare_run",
        lambda root, p: (config, frame, assignments, reference, copy.deepcopy(identity)),
    )
    monkeypatch.setattr(tuning.subprocess, "check_output", lambda *a, **kw: "test-git-state")
    calls = []

    def simulated_fit(f, a, c, p, model, fold):
        calls.append((model, fold))
        if len(calls) == 3:
            raise RuntimeError("simulated interruption")
        predictions = reference.loc[(reference.model == model) & (reference.fold == fold)]
        return {
            "model": model,
            "fold": fold,
            "elapsed_seconds": 0.0,
            "inner_fits": 8,
            "best_params": {},
            "warnings": {},
            "outer_metrics": baseline.metrics(
                predictions.y_true.to_numpy(), predictions.probability.to_numpy(), 0.5
            ),
            "predictions": predictions.to_dict("records"),
        }

    monkeypatch.setattr(tuning, "fit_outer", simulated_fit)
    with pytest.raises(RuntimeError, match="interruption"):
        tuning.run(tmp_path, "t1")
    checkpoint_dir = summary.parent / "t1/checkpoints"
    before = {p: p.read_bytes() for p in checkpoint_dir.glob("*.json")}
    assert len(before) == 2
    tuning.run(tmp_path, "t1", resume=True)
    assert all(path.read_bytes() == value for path, value in before.items())
    assert calls.count(("logistic_regression", 1)) == 1
    assert calls.count(("logistic_regression", 2)) == 1
    assert len(calls) == 11  # ten completed folds plus the interrupted attempt
    completed = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    tuning.run(tmp_path, "t1", resume=True)
    assert len(calls) == 11
    assert all(path.read_bytes() == content for path, content in completed.items())
    identity["config"] = {"changed": True}
    with pytest.raises(ValueError, match="Resume rejected"):
        tuning.run(tmp_path, "t1", resume=True)
    assert len(calls) == 11
