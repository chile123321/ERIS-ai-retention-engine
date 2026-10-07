"""Synthetic-only correctness checks for the saved-prediction fairness audit."""

import copy
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from eris_ml.evaluation.fairness_v2 import (
    CAPACITIES,
    FEATURES,
    GROUPS,
    measure,
    summarize,
    validate_inputs,
    validate_protocol,
)
from eris_ml.evaluation.fairness_v2_reporting import build_report

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def protocol():
    return yaml.safe_load((ROOT / "configs/evaluation/fairness_protocol_v2.yaml").read_text())


@pytest.fixture
def saved():
    n = 200
    rng = np.random.default_rng(72)
    rows = np.arange(n)
    development = pd.DataFrame({
        "source_row": rows, "EmployeeNumber": rows + 1000,
        "Attrition": (rows % 4 < 2).astype(int),
        "Age": np.array([25, 35, 45, 55])[rows % 4],
    })
    assignment = development[["source_row", "EmployeeNumber", "Attrition"]].copy()
    assignment["fold"] = rows // 40 + 1
    assignment["development_position"] = rows
    audit = development.copy()
    audit["Gender"] = np.where(rows % 3 == 0, "Female", "Male")
    audit["MaritalStatus"] = np.array(["Single", "Married", "Divorced"])[rows % 3]
    audit["split"] = "development"
    baseline, calibration, policy = [], [], []
    for feature in FEATURES:
        base = assignment.drop(columns="development_position").copy()
        base["feature_set"] = feature
        base["probability"] = np.round(rng.uniform(0, 1, n), 1)  # deliberate ties
        baseline.append(base.assign(model="logistic_regression"))
        calibration.append(base.assign(method="raw"))
        for capacity in CAPACITIES:
            part = base.assign(policy="top_k", capacity_percent=capacity, prediction=0)
            for _, fold in part.groupby("fold"):
                chosen = fold.sort_values(
                    ["probability", "source_row"], ascending=[False, True]
                ).head(len(fold) * capacity // 100).index
                part.loc[chosen, "prediction"] = 1
            policy.append(part)
    return dict(development=development, assignment=assignment, audit=audit,
                baseline=pd.concat(baseline, ignore_index=True),
                calibration=pd.concat(calibration, ignore_index=True),
                policy=pd.concat(policy, ignore_index=True))


def validate(saved):
    return validate_inputs(**saved, expected_rows=200, expected_counts={0: 100, 1: 100})


def load_cli():
    spec = importlib.util.spec_from_file_location(
        "audit_fairness_v2", ROOT / "scripts/audit_fairness_v2.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_protocol_and_support_are_locked(protocol):
    validate_protocol(protocol)
    for key, value in (("minimum_support", {"total_rows": 1}),
                       ("capacity_percent", [10]), ("business_approved", True)):
        changed = copy.deepcopy(protocol)
        changed[key] = value
        with pytest.raises(ValueError, match="locked"):
            validate_protocol(changed)


def test_counts_denominators_intervals_and_probability_metrics(protocol):
    labels = np.array([1] * 20 + [0] * 80)
    scores = np.full(100, .25)
    prediction = np.array([1] * 12 + [0] * 8 + [1] * 8 + [0] * 72)
    row = measure(labels, scores, protocol, prediction)
    assert (row["tp"], row["fp"], row["fn"], row["tn"]) == (12, 8, 8, 72)
    for metric, num, den in (("alert_rate", 20, 100), ("tpr", 12, 20),
                             ("fpr", 8, 80), ("precision", 12, 20)):
        assert row[metric] == num / den
        assert row[f"{metric}_numerator"] == num
        assert row[f"{metric}_denominator"] == den
        assert row[f"{metric}_ci_low"] < row[metric] < row[f"{metric}_ci_high"]
    probability = measure(labels, scores, protocol)
    assert probability["brier"] == pytest.approx(np.mean((scores - labels) ** 2))
    assert probability["roc_auc"] == .5
    assert probability["average_precision"] == .2


@pytest.mark.parametrize("labels", [[], [0] * 60, [1] * 60, [0, 1] * 5])
def test_sparse_and_one_class_are_undefined_not_zero(protocol, labels):
    y = np.array(labels)
    p = np.zeros(len(y))
    raw = measure(y, p, protocol)
    for name in ("average_precision", "roc_auc", "brier"):
        assert raw[name] is None
    row = measure(y, p, protocol, np.zeros(len(y)))
    assert row["support"] == "insufficient evidence"
    assert row["n"] == len(y) and row["positives"] == int(y.sum())
    assert all(row[name] is None for name in ("tpr", "fpr", "precision"))


def test_precision_requires_alert_support_even_when_labels_supported(protocol):
    y = np.array([0, 1] * 50)
    prediction = np.zeros(100)
    prediction[1] = 1
    row = measure(y, np.full(100, .2), protocol, prediction)
    assert row["support"] == "supported"
    assert row["precision"] is None and row["precision_denominator"] == 1
    assert row["tpr"] == .02


@pytest.mark.parametrize("kind", [
    "id", "target", "fold", "probability", "duplicate", "missing", "topk", "missing_group",
    "unknown_group", "nondevelopment", "assignment",
])
def test_input_corruption_fails_before_metrics(saved, kind):
    if kind in ("id", "target", "fold", "probability"):
        column = {"id": "EmployeeNumber", "target": "Attrition",
                  "fold": "fold", "probability": "probability"}[kind]
        saved["policy"].loc[0, column] = 999
    elif kind == "duplicate":
        saved["baseline"] = pd.concat([saved["baseline"], saved["baseline"].iloc[:1]])
    elif kind == "missing":
        saved["policy"] = saved["policy"].iloc[1:]
    elif kind == "topk":
        part = saved["policy"]
        yes = part.index[(part.fold == 1) & (part.capacity_percent == 5)
                         & (part.feature_set == FEATURES[0]) & (part.prediction == 1)][0]
        no = part.index[(part.fold == 1) & (part.capacity_percent == 5)
                        & (part.feature_set == FEATURES[0]) & (part.prediction == 0)][0]
        part.loc[yes, "prediction"], part.loc[no, "prediction"] = 0, 1
    elif kind in ("missing_group", "unknown_group"):
        saved["audit"].loc[0, "Gender"] = None if kind == "missing_group" else "unknown"
    elif kind == "nondevelopment":
        saved["audit"].loc[0, "split"] = "final_test"
    else:
        saved["assignment"].loc[0, "development_position"] = 1
    with pytest.raises(ValueError):
        validate(saved)


def test_end_to_end_no_fit_reconciliation_and_report(saved, protocol, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Fitting prohibited")
    monkeypatch.setattr(Pipeline, "fit", forbidden)
    monkeypatch.setattr(LogisticRegression, "fit", forbidden)
    raw, policy = validate(saved)
    tables = summarize(raw, policy, protocol)
    again = summarize(raw, policy, protocol)
    for name in tables:
        pd.testing.assert_frame_equal(tables[name], again[name])
    assert len(raw) == 400 and len(policy) == 1200
    overall = tables["policy"].query("fold == 0 and attribute == 'Overall'")
    assert overall.n.eq(200).all() and overall.positives.eq(100).all()
    assert list(overall.alerts) == [10, 20, 30, 10, 20, 30]
    for _, scope in tables["policy"].groupby(["feature_set", "capacity_percent", "fold"]):
        counts = ["n", "positives", "negatives", "alerts", "tp", "fp", "fn", "tn"]
        total = scope.loc[scope.attribute == "Overall", counts].iloc[0]
        for attr in GROUPS.keys() - {"Overall"}:
            assert scope.loc[scope.attribute == attr, counts].sum().equals(total)
    report = build_report(tables, {"versions": {"python": "synthetic"}})
    assert all(feature in report for feature in FEATURES)
    assert "undefined" in report and "No fit" in report
    assert "source_row," not in report  # no employee-level table


def test_fold_isolation_and_missing_fold_groups(saved, protocol):
    raw, policy = validate(saved)
    original = summarize(raw, policy, protocol)
    # Change only fold 1 attributes in the joined audit fixture: other folds must not change.
    raw.loc[raw.fold == 1, "Gender"] = "Female"
    policy.loc[policy.fold == 1, "Gender"] = "Female"
    changed = summarize(raw, policy, protocol)
    for key in ("policy", "probability"):
        pd.testing.assert_frame_equal(
            original[key].query("fold > 1"), changed[key].query("fold > 1"))
    empty = changed["policy"].query("fold == 1 and attribute == 'Gender' and group == 'Male'")
    assert empty.n.eq(0).all() and empty.tpr.isna().all()
    assert empty.support.eq("insufficient evidence").all()


@pytest.mark.parametrize("path", [
    "data/processed/final_test_raw_v1.csv", "data/quarantine/fairness_audit_mixed_v1.csv",
    "../outside.csv", "data/processed/fairness_audit_v1.csv",
])
def test_path_guard_rejects_before_stat_or_open(path, monkeypatch, tmp_path):
    cli = load_cli()
    def prohibited(*args, **kwargs):
        raise AssertionError("No stat/open allowed")
    monkeypatch.setattr(Path, "is_symlink", prohibited)
    with pytest.raises(ValueError):
        cli.safe_path(tmp_path, path)


def test_existing_outputs_are_never_overwritten(tmp_path):
    cli = load_cli()
    output = tmp_path / cli.OUTPUTS["report"]
    output.parent.mkdir(parents=True)
    output.write_text("previous immutable report")
    before = output.read_bytes()
    with pytest.raises(ValueError, match="existing"):
        cli.run(tmp_path)
    assert output.read_bytes() == before


def test_provenance_requires_exact_reviewed_pair_and_historical_bytes(tmp_path, monkeypatch):
    import hashlib
    cli = load_cli()
    old = b"historical source\n"
    historical = hashlib.sha256(old).hexdigest()
    bridge = {"src/example.py": {"historical": historical, "current": "new-digest",
                                "traces": ["baseline_ablation_v2"]}}
    monkeypatch.setattr(cli.subprocess, "check_output", lambda *a, **k: old)
    cli.verify_revision(tmp_path, "src/example.py", historical, "new-digest",
                        bridge, "baseline_ablation_v2")
    for field in ("current", "historical", "traces"):
        altered = copy.deepcopy(bridge)
        altered["src/example.py"][field] = [] if field == "traces" else "wrong"
        with pytest.raises(ValueError, match="Unreviewed"):
            cli.verify_revision(tmp_path, "src/example.py", historical, "new-digest",
                                altered, "baseline_ablation_v2")
    monkeypatch.setattr(cli.subprocess, "check_output", lambda *a, **k: b"different")
    with pytest.raises(ValueError, match="Historical HEAD"):
        cli.verify_revision(tmp_path, "src/example.py", historical, "new-digest",
                            bridge, "baseline_ablation_v2")


def test_registry_mapping_cannot_authorize_experimental_scope_change(tmp_path, monkeypatch):
    import hashlib
    cli = load_cli()
    relative = "configs/features/feature_set_v2_practical.yaml"
    old = yaml.safe_dump({
        "version": "v2", "preferred_feature_set": "sixteen",
        "comparison_feature_sets": ["sixteen", "seventeen"], "comparison_protocol": {"seed": 42},
        "status": "old",
    }).encode()
    current = yaml.safe_load(old)
    current["status"] = "new"
    destination = tmp_path / relative
    destination.parent.mkdir(parents=True)
    destination.write_text(yaml.safe_dump(current))
    digest = hashlib.sha256(old).hexdigest()
    bridge = {relative: {"historical": digest, "current": "new", "traces": ["baseline"]}}
    monkeypatch.setattr(cli.subprocess, "check_output", lambda *a, **k: old)
    cli.verify_revision(tmp_path, relative, digest, "new", bridge, "baseline")
    current["comparison_protocol"]["seed"] = 99
    destination.write_text(yaml.safe_dump(current))
    with pytest.raises(ValueError, match="experimental"):
        cli.verify_revision(tmp_path, relative, digest, "new", bridge, "baseline")


def test_json_preflight_is_exclusive_and_does_not_allow_nan(tmp_path):
    cli = load_cli()
    path = tmp_path / "preflight.json"
    cli.write_json(path, {"status": "locked"})
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        cli.write_json(path, {"status": "modified"})
    assert path.read_bytes() == before
    with pytest.raises(ValueError):
        cli.write_json(tmp_path / "invalid.json", {"value": float("nan")})
    assert not (tmp_path / "invalid.json").exists()
