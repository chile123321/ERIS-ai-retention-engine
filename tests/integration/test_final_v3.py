"""One-time final evaluation risks, using synthetic data only (no real holdout reads)."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer

from eris_ml.evaluation import final_v3 as final
from eris_ml.evaluation.final_v3_reporting import evidence
from eris_ml.features.schema_v3 import V3InputValidator, build_pipeline, load_protocol
from eris_ml.models import baseline_v3 as baseline
from eris_ml.models.bundle_v3 import CandidateBundle, feature_mapping
from eris_ml.models.candidate_v3 import bundle_metadata

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def synthetic(tmp_path):
    config = load_protocol(ROOT / "configs/features/feature_set_v3_final_15.yaml")
    rng = np.random.default_rng(52)
    names = [f["name"] for f in config["features"]]
    frame = pd.DataFrame(
        {
            f["name"]: rng.choice(f.get("vocabulary", f.get("values", [1, 2, 3])), 1470)
            for f in config["features"]
        }
    )
    frame["Age"] = rng.integers(18, 61, 1470)
    frame["source_row"] = np.arange(1470)
    frame["EmployeeNumber"] = np.arange(10000, 11470)
    frame["Attrition"] = ["No"] * 986 + ["Yes"] * 190 + ["No"] * 247 + ["Yes"] * 47
    dev, holdout = frame.iloc[:1176].copy(), frame.iloc[1176:].copy()
    dev["Attrition"] = dev.Attrition.map({"No": 0, "Yes": 1})
    membership = frame[["source_row", "EmployeeNumber"]].copy()
    membership["split"] = ["development"] * 1176 + ["final_test"] * 294
    pipeline = build_pipeline(config, "random_forest")
    pipeline.set_params(model__n_estimators=3, model__max_depth=2)
    pipeline.fit(dev[names], dev.Attrition)
    meta = bundle_metadata(
        config,
        {
            "resolved_model_parameters": pipeline.named_steps["model"].get_params(),
            "best_params": {},
            "selection_cv_ap": 0.5,
        },
        {"synthetic": True},
        "b1",
        list(range(100)),
    )
    bundle = CandidateBundle(
        pipeline, meta, pipeline[:-1].transform(dev[names].iloc[:100]), feature_mapping(pipeline)
    )
    path = tmp_path / final.HOLDOUT
    path.parent.mkdir(parents=True)
    holdout.to_csv(path, index=False)
    summary = tmp_path / final.SUMMARY
    summary.parent.mkdir(parents=True)
    summary.write_bytes(b"historical summary\r\n")
    freeze = {
        "holdout_expected_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "protected_sha256": {},
        "summary_before_bytes": len(summary.read_bytes()),
        "summary_before_sha256": hashlib.sha256(summary.read_bytes()).hexdigest(),
        "prior_holdout_use": {"not_virgin_independent_v3_holdout": True},
    }
    baseline.write_json(
        tmp_path / "artifacts/reports/v3_final_15/c1/calibration.json",
        {
            "metrics": {
                "rf_tuned_raw": {"pooled_oof": dict.fromkeys(final.PROBABILITY_METRICS, 0.4)}
            }
        },
    )
    baseline.write_json(
        tmp_path / "artifacts/reports/v3_final_15/b1/selection.json", {"selection_cv_ap": 0.5}
    )
    pd.DataFrame(
        {
            "attribute": ["AgeGroup"] * 4,
            "group": final.AGE_GROUPS,
            "fold": 0,
            "alerts": [10, 20, 30, 40],
            "tp": [2, 4, 6, 8],
            "positives": [20, 40, 60, 70],
            "tpr": [0.1, 0.1, 0.1, 0.12],
        }
    ).to_csv(tmp_path / "artifacts/reports/v3_final_15/b1/subgroups.csv", index=False)
    return tmp_path, bundle, dev, membership, holdout, freeze


def test_one_read_one_prediction_no_fit_outputs_and_resume(synthetic, monkeypatch):
    root, bundle, dev, membership, holdout, freeze = synthetic
    monkeypatch.setattr(final, "preflight", lambda r: (bundle, dev, membership, freeze))
    original_read = Path.read_bytes
    reads, calls = [], []

    def read(path):
        if path == root / final.HOLDOUT:
            assert (root / final.REPORT / "pre_evaluation_freeze.json").exists()
            assert (root / final.CLAIM).exists()
            reads.append(path)
        return original_read(path)

    monkeypatch.setattr(Path, "read_bytes", read)
    predict = final.predict_batch

    def spy(b, records):
        assert list(records.columns) == bundle.metadata["feature_order"]
        calls.append(len(records))
        return predict(b, records)

    monkeypatch.setattr(final, "predict_batch", spy)

    def forbidden(*args, **kwargs):
        pytest.fail("No fit/read/predict/scoring permitted on the guarded path")

    for model in (RandomForestClassifier, SimpleImputer, V3InputValidator):
        monkeypatch.setattr(model, "fit", forbidden)
    boot = final.bootstrap
    monkeypatch.setattr(final, "bootstrap", lambda y, p: boot(y, p, iterations=12))
    final.run(root)
    assert len(reads) == 1 and calls == [294]
    pred = pd.read_csv(root / final.PREDICTIONS, float_precision="round_trip")
    expected = bundle.pipeline.predict_proba(
        bundle.inputs(holdout[bundle.metadata["feature_order"]])
    )[:, 1]
    np.testing.assert_array_equal(pred.probability, expected)
    assert pred[["top_5", "top_10", "top_15"]].sum().tolist() == [14, 29, 44]
    metrics = json.loads((root / final.REPORT / "final_metrics.json").read_text())
    assert metrics == final.summarize_predictions(pred, 190 / 1176)
    status = json.loads((root / final.REPORT / "final_evaluation_status.json").read_text())
    assert status["final_holdout_evaluated"] is True
    assert status["evaluated_row_count"] == 294 and status["evaluated_positive_count"] == 47
    assert (root / final.REPORT / "reliability.png").exists()
    hashes = {
        p: hashlib.sha256(original_read(p)).hexdigest() for p in (root / final.REPORT).iterdir()
    }
    monkeypatch.setattr(final, "preflight", forbidden)
    monkeypatch.setattr(final, "predict_batch", forbidden)
    monkeypatch.setattr(final, "summarize_predictions", forbidden)
    monkeypatch.setattr(final, "bootstrap", forbidden)
    final.run(root, verify_only=True)
    assert len(reads) == 1
    assert hashes == {p: hashlib.sha256(original_read(p)).hexdigest() for p in hashes}
    with pytest.raises(FileExistsError):
        final.run(root)
    with pytest.raises(ValueError):
        final.run(root, "f2")


def test_hash_mismatch_stops_before_holdout(tmp_path, monkeypatch):
    directory = tmp_path / final.BUNDLE
    directory.mkdir(parents=True)
    (directory / "pipeline.joblib").write_bytes(b"wrong model")

    def forbidden(*args, **kwargs):
        pytest.fail("Must reject hash before loader/holdout read")

    monkeypatch.setattr(final, "load_bundle", forbidden)
    with pytest.raises(ValueError, match="SHA mismatch"):
        final.run(tmp_path)
    assert not (tmp_path / final.CLAIM).exists()
    assert not (tmp_path / final.REPORT).exists()


def test_holdout_alignment_and_consumed_failure(synthetic, monkeypatch):
    root, bundle, dev, membership, holdout, freeze = synthetic
    numeric = holdout.copy()
    numeric["Attrition"] = numeric.Attrition.map({"No": 0, "Yes": 1})
    pd.testing.assert_frame_equal(
        final.validate_holdout(holdout, dev, membership),
        final.validate_holdout(numeric, dev, membership),
    )
    bad = holdout.copy()
    bad.iloc[0, bad.columns.get_loc("EmployeeNumber")] = dev.EmployeeNumber.iloc[0]
    with pytest.raises(ValueError, match="overlap"):
        final.validate_holdout(bad, dev, membership)
    bad = membership.copy()
    bad.loc[1176, "EmployeeNumber"] = -1
    with pytest.raises(ValueError, match="membership"):
        final.validate_holdout(holdout, dev, bad)
    freeze["holdout_expected_sha256"] = "invalid"
    monkeypatch.setattr(final, "preflight", lambda r: (bundle, dev, membership, freeze))
    with pytest.raises(ValueError, match="historical V1 hash"):
        final.run(root)
    assert (root / final.CLAIM).exists() and (root / final.REPORT / "failure.json").exists()
    assert not (root / final.PREDICTIONS).exists()
    with pytest.raises(FileExistsError):
        final.run(root)


def test_bootstrap_ties_support_and_evidence():
    y = np.array([0, 0, 1, 1, 0, 0])
    p = np.array([0.2, 0.3, 0.8, 0.7, 0.4, 0.1])
    first = final.bootstrap(y, p, iterations=20)
    assert first == final.bootstrap(y, p, iterations=20)
    for interval in first["intervals"].values():
        assert interval["lower"] <= interval["upper"]
    selected = final.top_k(np.full(100, 0.5), np.arange(100)[::-1], 10)
    assert selected.sum() == 10 and selected[-10:].sum() == 10
    frame = pd.DataFrame({"Age": [25] * 6, "y_true": y, "probability": p, "top_10": 0})
    groups = final.subgroup_table(frame)
    assert groups.n.sum() == 6 and groups.alerts.sum() == 0
    assert groups.low_support.all()
    assert groups.average_precision.isna().all() and groups.brier_score.isna().all()
    assert groups.precision.isna().all()
    assert groups.loc[groups.n.eq(0), ["tpr", "fpr"]].isna().all().all()
    metrics = final.summarize_predictions(frame, 0.2)
    label = evidence(metrics, first, pd.DataFrame({"lift": [2, 2, 2]}))
    assert label in {"weak", "moderate", "strong"}
