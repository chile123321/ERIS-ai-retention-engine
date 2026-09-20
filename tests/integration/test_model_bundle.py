"""Synthetic-only Step 12A build/load/explain/report integration test."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from eris_ml.models import bundle_build
from eris_ml.models.persistence import load_bundle


def test_synthetic_build_creates_checked_bundle_card_and_shap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    synthetic_bundle: tuple[object, pd.DataFrame, object],
) -> None:
    synthetic, small, definition = synthetic_bundle
    root = Path(__file__).resolve().parents[2]
    manifest = yaml.safe_load((root / "configs/release/candidate_v1.yaml").read_text(
        encoding="utf-8"
    ))
    for relative in (bundle_build.FEATURE_CONFIG, bundle_build.MODEL_CONFIG,
                     bundle_build.MANIFEST_PATH):
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((root / relative).read_bytes())
    repeated = pd.concat([small] * 15, ignore_index=True).iloc[:1176].copy()
    repeated["source_row"] = np.arange(len(repeated))
    repeated["EmployeeNumber"] = np.arange(10000, 10000 + len(repeated))
    repeated["Attrition"] = np.array([0] * 986 + [1] * 190)
    development_path = tmp_path / bundle_build.DEVELOPMENT_CSV
    development_path.parent.mkdir(parents=True, exist_ok=True)
    repeated.to_csv(development_path, index=False)
    ledger_path = tmp_path / bundle_build.LEDGER_PATH
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    ledger_path.write_text("run_count: 1\nstatus: completed\n", encoding="utf-8")
    metrics = {
        "pr_auc": 0.60, "roc_auc": 0.83, "brier": 0.09,
        "precision": 0.60, "recall": 0.49, "f1": 0.54, "f2": 0.51,
        "alert_rate": 0.13, "tn": 232, "fp": 15, "fn": 24, "tp": 23,
    }
    summary = {
        "metrics": metrics,
        "bootstrap": {"intervals": {name: [0.1, 0.9] for name in
                                    ("pr_auc", "roc_auc", "brier", "precision",
                                     "recall", "f1", "f2", "alert_rate")}},
        "result": "PASS_WITH_KNOWN_LIMITATIONS",
    }
    monkeypatch.setattr(bundle_build, "preflight_bundle", lambda path: (
        manifest, summary, repeated, definition, repeated["Attrition"], "a" * 64,
    ))

    def synthetic_fit(*args: object) -> object:
        assert len(args[1]) == 1176
        return synthetic.pipeline

    monkeypatch.setattr(bundle_build, "fit_frozen", synthetic_fit)
    ledger_before = ledger_path.read_bytes()
    result = bundle_build.build_bundle(tmp_path)
    assert ledger_path.read_bytes() == ledger_before
    assert len(result["importance"]) == 25
    assert result["global_additivity_error"] < 1e-4
    assert (tmp_path / bundle_build.SHAP_PLOT_PATH).is_file()
    assert (tmp_path / bundle_build.SHAP_CSV_PATH).is_file()
    assert "AgeGroup" in (tmp_path / bundle_build.MODEL_CARD_PATH).read_text(encoding="utf-8")
    assert "final-test csv" in (tmp_path / bundle_build.BUILD_REPORT_PATH).read_text(
        encoding="utf-8"
    ).lower()
    loaded = load_bundle(tmp_path / bundle_build.BUNDLE_PATH)
    assert len(loaded.predict_proba(small.iloc[:2])) == 2
    with pytest.raises(ValueError, match="overwrite"):
        bundle_build.build_bundle(tmp_path)
