"""Build the frozen development-only XGBoost bundle without final-test access."""

from __future__ import annotations

import json
import platform
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

from eris_ml.evaluation.final_evaluation import (
    LEDGER_PATH,
    MANIFEST_PATH,
    METRICS_PATH,
    assert_runtime_candidate,
    assert_unused_ledger,
    fit_frozen,
    load_yaml,
    manifest_sha256,
    validate_data,
)
from eris_ml.features.definitions import FeatureDefinition, load_feature_definition
from eris_ml.models.bundle import ModelBundle
from eris_ml.models.explainability import global_importance
from eris_ml.models.persistence import load_bundle, save_model_bundle, sidecar_paths
from eris_ml.utils.freeze import (
    DEVELOPMENT_CSV,
    FEATURE_CONFIG,
    MODEL_CONFIG,
    MODEL_PARAMETERS,
    validate_freeze_manifest,
    verify_manifest_hashes,
)
from eris_ml.utils.hashing import sha256_file

BUNDLE_PATH = "artifacts/models/eris_xgboost_v1.joblib"
SHAP_CSV_PATH = "artifacts/explainability/shap_global_v1.csv"
SHAP_PLOT_PATH = "artifacts/explainability/shap_summary_v1.png"
MODEL_CARD_PATH = "docs/model_card_v1.md"
BUILD_REPORT_PATH = "artifacts/reports/model_bundle_build_v1.md"


def preflight_bundle(root: Path) -> tuple[dict[str, Any], dict[str, Any], pd.DataFrame,
                                          FeatureDefinition, pd.Series, str]:
    """Read only frozen config, final summary, ledger and development data."""
    manifest = load_yaml(root / MANIFEST_PATH)
    validate_freeze_manifest(manifest)
    verify_manifest_hashes(manifest, root)
    assert_runtime_candidate(manifest)
    digest = manifest_sha256(root / MANIFEST_PATH)
    ledger = load_yaml(root / LEDGER_PATH)
    if (ledger.get("candidate") != manifest["candidate_name"]
        or ledger.get("run_count") != 1 or ledger.get("status") != "completed"
        or ledger.get("result") != "PASS_WITH_KNOWN_LIMITATIONS"
        or ledger.get("candidate_manifest_sha256") != digest
        or ledger.get("threshold") != 0.345651):
        raise ValueError("Final evaluation ledger is not the completed frozen PASS record.")
    try:
        assert_unused_ledger(ledger)
    except ValueError:
        pass  # Expected: the one-time final-test rerun guard still rejects this ledger.
    else:
        raise ValueError("One-time final-test rerun guard did not reject the completed ledger.")
    summary = json.loads((root / METRICS_PATH).read_text(encoding="utf-8"))
    if (summary.get("candidate") != manifest["candidate_name"]
        or summary.get("candidate_manifest_sha256") != digest
        or summary.get("final_test_sha256") != ledger["final_test_sha256"]
        or summary.get("result") != ledger["result"]
        or summary.get("threshold") != 0.345651):
        raise ValueError("Locked final summary disagrees with the completed ledger.")
    config = load_yaml(root / MODEL_CONFIG)
    if (config.get("model") != "xgboost" or config.get("random_seed") != 42
        or config.get("parameters") != MODEL_PARAMETERS):
        raise ValueError("XGBoost configuration drifted from the freeze manifest.")
    definition = load_feature_definition(root / FEATURE_CONFIG)
    if definition.version != "v1-full" or len(definition.all_features) != 25:
        raise ValueError("Frozen 25-feature schema changed.")
    development = pd.read_csv(root / DEVELOPMENT_CSV)
    target = validate_data(development, definition, final=False)
    return manifest, summary, development, definition, target, digest


def bundle_metadata(manifest: dict[str, Any], development: pd.DataFrame,
                    definition: FeatureDefinition, *, manifest_hash: str,
                    development_hash: str, timestamp: str) -> dict[str, Any]:
    """Record schema and observed ordinal domains, never employee-level rows."""
    feature_types = {
        **{name: "nominal" for name in definition.nominal},
        **{name: "ordinal" for name in definition.ordinal},
        **{name: "numeric" for name in definition.numeric},
    }
    domains = {name: sorted(float(value) for value in development[name].dropna().unique())
               for name in definition.ordinal}
    return {
        "candidate_version": manifest["candidate_name"], "bundle_version": "bundle-v1",
        "feature_schema_version": definition.version,
        "feature_count": len(definition.all_features),
        "feature_order": list(definition.all_features), "feature_types": feature_types,
        "ordinal_domains": domains,
        "ordinal_domain_source": "observed_development_values_not_authoritative_HR_contract",
        "threshold": 0.345651, "threshold_policy": "capacity_15_percent",
        "class_mapping": {"0": "No", "1": "Yes"},
        "calibration": "none", "probability_type": "raw", "model_family": "xgboost",
        "model_config": MODEL_CONFIG, "model_parameters": MODEL_PARAMETERS.copy(),
        "preprocessing_version": manifest["preprocessing"]["version"],
        "random_seed": 42, "training_rows": 1176, "training_scope": "development_only",
        "training_timestamp_utc": timestamp, "training_data_sha256": development_hash,
        "candidate_manifest_sha256": manifest_hash,
        "fairness_status": manifest["fairness"]["status"],
        "holdout_status": manifest["holdout"]["status"],
        "allowed_uses": manifest["scope"]["allowed_uses"],
        "prohibited_uses": manifest["scope"]["prohibited_uses"],
        "business_approved": False, "production_approved": False,
        "trusted_artifact_only": True,
        "deserialization_warning": (
            "Joblib/pickle can execute code. Load only this trusted, locally produced artifact; "
            "checksums do not authenticate origin."
        ),
        "library_versions": {
            "python": platform.python_version(), "pandas": version("pandas"),
            "numpy": version("numpy"), "scikit-learn": version("scikit-learn"),
            "xgboost": version("xgboost"), "shap": version("shap"),
            "joblib": version("joblib"), "matplotlib": version("matplotlib"),
        },
    }


def development_shap_sample(development: pd.DataFrame, target: pd.Series,
                            *, maximum: int = 500) -> pd.DataFrame:
    """Select a reproducible stratified development-only explanation sample."""
    if maximum < 2:
        raise ValueError("SHAP maximum sample size must be at least two.")
    if len(development) <= maximum:
        return development.copy()
    splitter = StratifiedShuffleSplit(n_splits=1, train_size=maximum, random_state=42)
    positions, _ = next(splitter.split(development, target))
    return development.iloc[positions].copy()


def render_shap_plot(importance: pd.DataFrame, path: Path) -> None:
    """Draw a code-native original-feature SHAP importance summary."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    display = importance.iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.barh(display["feature"], display["mean_abs_shap"], color="#2878a8")
    ax.set_xlabel("Mean absolute SHAP contribution (raw log-odds)")
    ax.set_title("XGBoost v1 — development-only SHAP importance")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def build_bundle(root: Path) -> dict[str, Any]:
    """Fit once on development, persist, reload, explain and write honest records."""
    from eris_ml.models.bundle_reporting import build_model_card, build_report

    model_path = root / BUNDLE_PATH
    metadata_path, checksum_path = sidecar_paths(model_path)
    outputs = (model_path, metadata_path, checksum_path, root / SHAP_CSV_PATH,
               root / SHAP_PLOT_PATH, root / MODEL_CARD_PATH, root / BUILD_REPORT_PATH)
    if any(path.exists() for path in outputs):
        raise ValueError("Bundle output already exists; refusing to overwrite evidence.")
    ledger_before = (root / LEDGER_PATH).read_bytes()
    manifest, summary, development, definition, target, digest = preflight_bundle(root)
    timestamp = datetime.now(UTC).isoformat(timespec="seconds")
    metadata = bundle_metadata(
        manifest, development, definition, manifest_hash=digest,
        development_hash=sha256_file(root / DEVELOPMENT_CSV), timestamp=timestamp,
    )
    pipeline = fit_frozen(manifest, development, definition, target)
    bundle = ModelBundle(pipeline, metadata)
    save_model_bundle(bundle, model_path, metadata)
    loaded = load_bundle(model_path)
    smoke = development.loc[:, list(definition.all_features)].iloc[:3]
    probabilities = loaded.predict_proba(smoke)
    predictions = loaded.predict(smoke)
    if (len(probabilities) != 3 or not np.isfinite(probabilities).all()
        or not np.array_equal(predictions, (probabilities >= 0.345651).astype(int))):
        raise ValueError("Loaded bundle prediction smoke test failed.")
    sample = development_shap_sample(development, target).loc[:,
                                                              list(definition.all_features)]
    importance, global_error = global_importance(loaded, sample, sample_size=len(sample))
    local = loaded.explain(smoke.iloc[:1], top_k=5)[0]
    if len(importance) != 25 or len(local["top_factors"]) != 5:
        raise ValueError("SHAP original-feature coverage is incomplete.")
    (root / SHAP_CSV_PATH).parent.mkdir(parents=True, exist_ok=True)
    importance.to_csv(root / SHAP_CSV_PATH, index=False)
    render_shap_plot(importance, root / SHAP_PLOT_PATH)
    card = build_model_card(manifest, summary, metadata)
    (root / MODEL_CARD_PATH).parent.mkdir(parents=True, exist_ok=True)
    (root / MODEL_CARD_PATH).write_text(card, encoding="utf-8")
    hashes = {
        name: sha256_file(root / name) for name in (
            BUNDLE_PATH, metadata_path.relative_to(root).as_posix(),
            FEATURE_CONFIG, MODEL_CONFIG, MANIFEST_PATH, DEVELOPMENT_CSV,
            MODEL_CARD_PATH, SHAP_CSV_PATH, SHAP_PLOT_PATH,
        )
    }
    report = build_report(metadata, hashes, importance, global_error,
                          local["additivity_max_error"], probabilities)
    (root / BUILD_REPORT_PATH).parent.mkdir(parents=True, exist_ok=True)
    (root / BUILD_REPORT_PATH).write_text(report, encoding="utf-8")
    if (root / LEDGER_PATH).read_bytes() != ledger_before:
        raise ValueError("Final-evaluation ledger changed during bundle build.")
    return {"metadata": metadata, "hashes": hashes, "importance": importance,
            "global_additivity_error": global_error,
            "local_additivity_error": local["additivity_max_error"],
            "bundle_size_bytes": model_path.stat().st_size}
