"""Build the fixed unweighted V3 research candidate from development only."""

from __future__ import annotations

import hashlib
import json
import platform
import re
import subprocess
import sys
import time
import warnings
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from eris_ml.evaluation.calibration_v3_reporting import stress_metrics
from eris_ml.features.schema_v3 import validate_features
from eris_ml.models import baseline_v3 as baseline
from eris_ml.models import bundle_v3 as v3
from eris_ml.models import remediation_v3 as study
from eris_ml.models import tuning_v3 as tuning
from eris_ml.models.bundle_v3_b2 import (
    FIXED_PARAMETERS,
    MODEL_VERSION,
    PROBABILITY_STATUS,
    B2Bundle,
    explain_one,
    load_bundle,
    predict_batch,
    predict_one,
    save_bundle,
)
from eris_ml.utils.hashing import sha256_file

SUMMARY = "artifacts/reports/v3_final_15/summary.md"
RUN = "b2"
MODEL = "rf_unweighted_raw"
SCENARIOS = ["missing_training", "missing_manager", "missing_both"]
POLICY: dict[str, Any] = {
    "model_version": MODEL_VERSION,
    "source": "devfix1 development OOF, fixed RF unweighted raw",
    "features": "v3_final_15 ordered 15; 13 required, 2 optional",
    "parameters": FIXED_PARAMETERS,
    "calibration": "none",
    "fit": "one fit on 1176 development rows",
    "background": "100 development rows, default_rng(42).choice without replacement",
    "missingness": "saved devfix1 outer-fold validation predictions; no new fit",
    "probability_status": PROBABILITY_STATUS,
    "threshold": None,
    "final_holdout_access": False,
    "f1_predictions_access": False,
    "external_validation_required": True,
}


def protected_path(root: Path, relative: str) -> Path:
    return study.study_path(root, relative)


def preflight(
    root: Path,
) -> tuple[
    dict[str, Any], pd.DataFrame, pd.DataFrame, dict[str, Any], dict[str, Any], dict[str, str]
]:
    config, frame, assignments, _, identity = study.prepare(root)
    previous = protected_path(root, "artifacts/reports/v3_final_15/devfix1")
    prior = tuning.read_json(previous / "preflight.json")
    manifest = tuning.read_json(previous / "manifest.json")
    if (
        prior["identity"] != baseline.json_safe(identity)
        or prior["identity_sha256"] != tuning.digest(identity)
        or not manifest["protected_hashes_unchanged"]
        or manifest["base_fits"] != 70
    ):
        raise ValueError("Frozen devfix1 preflight/fit provenance differs.")
    tuning.verify_hashes(root, manifest["output_sha256"])
    protected = dict(identity["protected_sha256"])
    protected.update(manifest["output_sha256"])
    for name in ("preflight.json", "manifest.json"):
        path = previous / name
        protected[path.relative_to(root).as_posix()] = sha256_file(path)
    for relative in (
        "src/eris_ml/models/bundle_v3_b2.py",
        "src/eris_ml/models/candidate_v3_b2.py",
        "scripts/build_candidate_v3_b2.py",
        "tests/integration/test_candidate_v3_b2.py",
    ):
        protected[relative] = sha256_file(protected_path(root, relative))
    pred = protected_path(root, "artifacts/predictions/v3_final_15/devfix1")
    original = pd.read_csv(pred / "oof.csv")
    missing = pd.read_csv(pred / "optional_missing_oof.csv")
    oof = study.validate_oof(pd.concat([original, missing], ignore_index=True), assignments)
    chosen = oof.loc[oof.model == MODEL].copy()
    raw = chosen.loc[chosen.scenario == "original"]
    metrics = tuning.read_json(previous / "metrics.json")
    rules = tuning.read_json(previous / "guardrails.json")
    top = tuning.read_json(previous / "top_k.json")
    saved_missing = tuning.read_json(previous / "optional_missing.json")[MODEL]
    actual = baseline.metrics(raw.y_true.to_numpy(), raw.probability.to_numpy(), 0.5)
    for key in ("average_precision", "roc_auc", "brier_score", "log_loss"):
        np.testing.assert_allclose(actual[key], metrics[MODEL]["pooled_oof"][key], atol=1e-12)
    if (
        len(raw) != 1176
        or raw.y_true.value_counts().to_dict() != {0: 986, 1: 190}
        or not rules[MODEL]["eligible"]
        or len(top[MODEL]) != 3
        or top[MODEL][1]["pooled"]["alerts"] != 115
        or top[MODEL][1]["pooled"]["tp"] != int(raw.loc[raw.top_10.eq(1)].y_true.sum())
    ):
        raise ValueError("Devfix1 selected model, probabilities or guardrails differ.")
    params = dict(identity["rf_parameters"])
    params["class_weight"] = None
    if any(params.get(key) != value for key, value in FIXED_PARAMETERS.items()):
        raise ValueError("Unweighted b2 fixed parameters do not match devfix1.")
    optional: dict[str, Any] = {}
    for scenario in SCENARIOS:
        result = stress_metrics(chosen.loc[chosen.scenario == scenario], raw)
        reference = saved_missing[scenario]
        for key in ("average_precision", "brier_score", "log_loss"):
            np.testing.assert_allclose(
                result["pooled"]["probability_metrics"][key],
                reference["pooled"]["probability_metrics"][key],
                atol=1e-12,
            )
        for key in ("mean_absolute_probability_change", "top_10_overlap_fraction"):
            np.testing.assert_allclose(result["pooled"][key], reference["pooled"][key], atol=1e-12)
        if result["pooled"]["top_10"] != reference["pooled"]["top_10"]:
            raise ValueError("Saved optional missing top-k counts changed.")
        optional[scenario] = result
    comparison = {
        "rf_unweighted_raw": metrics[MODEL],
        "rf_balanced_raw": metrics["rf_balanced_raw"],
        "guardrails": rules[MODEL],
        "top_10": top[MODEL][1],
        "prevalence": float(raw.y_true.mean()),
        "unweighted_mean_probability": float(raw.probability.mean()),
    }
    return (
        config,
        frame,
        assignments,
        identity,
        {"comparison": comparison, "optional": optional},
        protected,
    )


def fit_fixed(
    config: dict[str, Any], frame: pd.DataFrame, params: dict[str, Any]
) -> tuple[Any, pd.DataFrame, dict[str, Any]]:
    names = [f["name"] for f in config["features"]]
    if len(names) != 15 or len(set(names)) != 15 or len(frame) != 1176:
        raise ValueError("B2 requires the locked fifteen features and development rows.")
    X = validate_features(frame[names], config)
    if set(X) & {"source_row", "EmployeeNumber", "Attrition", "Gender", "MaritalStatus"}:
        raise ValueError("Identifier, target or audit attribute reached fit.")
    pipeline = study.pipeline_for(config, params, "rf_unweighted")
    if any(
        pipeline.named_steps["model"].get_params()[key] != value
        for key, value in FIXED_PARAMETERS.items()
    ):
        raise ValueError("Fixed RF parameters changed before fit.")
    started = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught, threadpool_limits(limits=1):
        warnings.simplefilter("always")
        pipeline.fit(X, frame.Attrition.to_numpy())
    model = pipeline.named_steps["model"]
    optional = [f["name"] for f in config["features"] if not f["required"]]
    medians = (
        pipeline.named_steps["preprocessor"]
        .named_transformers_["optional_numeric"]
        .named_steps["imputer"]
        .statistics_
    )
    if (
        not np.array_equal(pipeline.classes_, [0, 1])
        or list(pipeline.feature_names_in_) != names
        or not np.array_equal(medians, X[optional].median().to_numpy())
        or model.n_features_in_ != 18
    ):
        raise ValueError("B2 fitted feature order, classes or development medians differ.")
    return (
        pipeline,
        X,
        {
            "fit_count": 1,
            "fit_rows": len(X),
            "fit_target_distribution": frame.Attrition.value_counts().to_dict(),
            "feature_order": names,
            "resolved_model_parameters": model.get_params(),
            "optional_training_medians": dict(zip(optional, medians, strict=True)),
            "fit_elapsed_seconds": time.perf_counter() - started,
            "warnings": dict(Counter(f"{w.category.__name__}: {w.message}" for w in caught)),
        },
    )


def metadata(
    config: dict[str, Any],
    params: dict[str, Any],
    provenance: dict[str, Any],
    frame: pd.DataFrame,
    positions: list[int],
    preflight_sha: str,
    config_sha: str,
) -> dict[str, Any]:
    names = [f["name"] for f in config["features"]]
    return {
        "model_version": MODEL_VERSION,
        "status": "research_demo_candidate",
        "model_family": "random_forest",
        "class_weight": None,
        "calibration": "none",
        "feature_set": "v3_final_15",
        "feature_order": names,
        "feature_config": config,
        "required_features": [f["name"] for f in config["features"] if f["required"]],
        "optional_features": [f["name"] for f in config["features"] if not f["required"]],
        "category_vocabulary": {
            f["name"]: f["vocabulary"] for f in config["features"] if f["type"] == "categorical"
        },
        "input_validation": {
            "extra_columns": "reject",
            "required_missing": "reject",
            "optional_missing_key_or_null": "training_median",
            "numeric": "finite_nonnegative_integer_no_string_or_bool",
            "categories": "exact_IBM_vocabulary",
            "key_order": "schema_order",
        },
        "target_mapping": {"No": 0, "Yes": 1},
        "positive_class": 1,
        "class_order": [0, 1],
        "resolved_model_parameters": params,
        "library_versions": v3.versions(),
        "background": {
            "count": 100,
            "random_state": 42,
            "sampling": "numpy.default_rng.choice_without_replacement_in_development_order",
            "development_positions": positions,
            "development_source_rows": frame.iloc[positions].source_row.tolist(),
        },
        "shap_tolerance": {"atol": v3.ATOL, "rtol": v3.RTOL},
        "evaluation_status": "development_remediation_only",
        "final_holdout_evaluated": False,
        "final_holdout_evaluated_for_this_version": False,
        "external_validation_required": True,
        "parent_candidate": "rf-v3-final15-b1",
        "derived_after_final_holdout_consumed": True,
        "estimated_probability_status": PROBABILITY_STATUS,
        "threshold": None,
        "production_approved": False,
        "business_approved": False,
        "provenance": {
            "development_sha256": provenance["data_provenance"]["input_sha256"][
                config["data"]["development"]
            ],
            "config_sha256": config_sha,
            "config_content_digest": tuning.digest(config),
            "devfix1_identity_sha256": tuning.digest(provenance),
            "b2_preflight_sha256": preflight_sha,
            "parent_model_binary_sha256": study.MODEL_SHA,
            "background_source_row_sha256": tuning.digest(
                frame.iloc[positions].source_row.tolist()
            ),
        },
        "limitations": "IBM-only development OOF evidence, no independent evaluation for b2. "
        "f1 applies to b1 only. No validated employer transfer, 3-6 month horizon, "
        "intervention benefit or fairness certification. Human review required. "
        "Income, survey, overtime, job grade, snapshot and missing-value mappings "
        "for new employers remain unresolved. Trusted local joblib only.",
    }


def verify_inference(
    bundle: B2Bundle, directory: Path, X: pd.DataFrame, report: Path
) -> dict[str, Any]:
    loaded = load_bundle(directory)
    indices = [0, 1, 2]
    probe = cast(list[dict[str, Any]], X.iloc[indices].to_dict("records"))
    optional = bundle.metadata["optional_features"]
    cases = [
        probe[0],
        {k: v for k, v in probe[0].items() if k != optional[0]},
        {k: v for k, v in probe[0].items() if k != optional[1]},
        {k: v for k, v in probe[0].items() if k not in optional},
    ]
    payload = probe + cases[1:]
    original = predict_batch(bundle, payload)
    current = predict_batch(loaded, payload)
    code = (
        "import json,sys;from pathlib import Path;"
        "from eris_ml.models.bundle_v3_b2 import load_bundle,predict_batch;"
        "b=load_bundle(Path(sys.argv[1]));p=json.loads(Path(sys.argv[2]).read_text());"
        "print(json.dumps(predict_batch(b,p['records'])))"
    )
    baseline.write_json(report / "reload_probe.json", {"records": payload})
    subprocess_result = subprocess.check_output(
        [sys.executable, "-c", code, str(directory), str(report / "reload_probe.json")],
        text=True,
    )
    fresh = json.loads(subprocess_result)
    maximum = max(
        abs(a["risk_score"] - b["risk_score"])
        for group in (current, fresh)
        for a, b in zip(original, group, strict=True)
    )
    if maximum > 1e-12 or any(
        a["imputed_features"] != b["imputed_features"]
        or a["model_version"] != b["model_version"]
        or a["estimated_probability_status"] != b["estimated_probability_status"]
        for group in (current, fresh)
        for a, b in zip(original, group, strict=True)
    ):
        raise ValueError("B2 save/load or fresh-process prediction regression failed.")
    examples: dict[str, Any] = {}
    for label, record in zip(
        (
            "fixed_development_position_0",
            "fixed_development_position_1",
            "fixed_development_position_2",
            "missing_training",
            "missing_manager",
            "missing_both",
        ),
        payload,
        strict=True,
    ):
        changed_order = dict(reversed(list(record.items())))
        one = predict_one(loaded, changed_order)
        explanation = explain_one(loaded, record)
        if not np.isclose(
            one["risk_score"], original[len(examples)]["risk_score"], atol=1e-12, rtol=1e-12
        ):
            raise ValueError("B2 key order or one/batch mismatch.")
        if not np.isclose(
            explanation["base_value"] + sum(explanation["contributions"].values()),
            one["risk_score"],
            atol=1e-6,
            rtol=1e-6,
        ):
            raise ValueError("B2 original-feature SHAP additivity failed.")
        examples[label] = {"input": record, "prediction": one, "explanation": explanation}
    for invalid in (
        {k: v for k, v in probe[0].items() if k != "Age"},
        {**probe[0], "Department": "Unknown"},
        {**probe[0], "Age": float("nan")},
        {**probe[0], optional[0]: float("inf")},
        {**probe[0], "source_row": 1},
    ):
        try:
            predict_one(loaded, invalid)
        except ValueError:
            continue
        raise ValueError("B2 invalid input was accepted.")
    baseline.write_json(
        report / "inference_examples.json",
        {
            "scope": "fixed development examples; no in-sample performance claim",
            "positive_class": 1,
            "examples": examples,
        },
    )
    timings: dict[str, Any] = {}
    for name, operation in (
        ("predict_one", lambda: predict_one(loaded, probe[0])),
        ("predict_batch_100", lambda: predict_batch(loaded, X.iloc[:100])),
        ("explain_one", lambda: explain_one(loaded, probe[0])),
    ):
        operation()
        times = []
        for _ in range(5):
            start = time.perf_counter()
            operation()
            times.append(time.perf_counter() - start)
        timings[name] = {"seconds": times, "median_seconds": float(np.median(times))}
    baseline.write_json(
        report / "performance.json",
        {
            "after_warmup": True,
            "repetitions": 5,
            "SLA": None,
            "environment": {
                "python": sys.version,
                "platform": platform.platform(),
                "processor": platform.processor(),
                "libraries": v3.versions(),
            },
            "measurements": timings,
        },
    )
    return {
        "fresh_process_max_absolute_difference": maximum,
        "one_batch_order_optional_invalid": "passed",
        "shap_additivity_max_absolute_error": max(
            e["explanation"]["absolute_additivity_error"] for e in examples.values()
        ),
        "shap_cases": list(examples),
        "positive_class": 1,
    }


def model_card(evidence: dict[str, Any], optional: dict[str, Any], binary_sha: str) -> str:
    metrics = evidence["comparison"]["rf_unweighted_raw"]
    score = metrics["pooled_oof"]
    lines = [
        f"# {MODEL_VERSION}: research/demo model card",
        "",
        "Purpose: help HR prioritize attrition-risk review. Humans retain responsibility for all "
        "personnel and intervention decisions; this model must not automatically decide "
        "employment.",
        "",
        "Input: 15 locked IBM-coded features (13 required, two median-imputed optional). "
        "Department and OverTime use the exact IBM vocabularies. Output is the raw positive-class "
        "score, labeled a development-validated estimated probability, not a production "
        "probability.",
        "",
        f"Devfix1 development OOF (n=1176, positives=190): "
        f"AP={score['average_precision']:.6f}, "
        f"ROC-AUC={score['roc_auc']:.6f}, Brier={score['brier_score']:.6f}, "
        f"Log loss={score['log_loss']:.6f}; mean p="
        f"{evidence['comparison']['unweighted_mean_probability']:.6f}, "
        f"prevalence={evidence['comparison']['prevalence']:.6f}. "
        "The OOF project probability guardrails passed. A full-development fit has no new "
        "OOF score.",
        "",
        "RF balanced raw overestimated probability on development. Unweighted raw improved "
        "development Brier/Log loss while retaining similar AP; this is not an independent "
        "post-selection estimate. The b1 f1 evaluation applies only to b1; b2 has no independent "
        "final or external evaluation.",
        "",
        "Optional missing stress reuses devfix1 held-out fold predictions. "
        "Manager-tenure missingness changes predictions; see optional_missing.json. "
        "Age-related alert/TPR disparity persists. Gender/MaritalStatus assessment on the "
        "consumed final evaluation was incomplete; do not infer b2 fairness from b1.",
        "",
        "IBM benchmark data are cross-sectional. Performance at a new employer, a 3–6 month "
        "forecast horizon and intervention effects are unverified. Income units, surveys, "
        "overtime, job grade, snapshot time, optional missingness and employer category mapping "
        "need a data contract. External and temporal validation are required before production.",
        "",
        "No approved HR threshold or low/medium/high labels. SHAP explains model behavior, "
        "not causality. Bundle is a trusted local research artifact; integrity hashes do not "
        "authenticate its origin.",
        "",
        f"Bundle pipeline SHA-256: `{binary_sha}`. Parent: `rf-v3-final15-b1`; "
        "devfix1 is the development evidence. No production or business approval.",
        "",
    ]
    return "\n".join(lines)


def quality_gate(root: Path, report: Path) -> dict[str, Any]:
    commands = {
        "related_pytest": [
            sys.executable, "-m", "pytest", "tests/integration/test_candidate_v3_b2.py",
            "tests/integration/test_candidate_v3.py",
            "tests/integration/test_remediation_v3.py", "-q",
        ],
        "strict_mypy_changed": [
            sys.executable, "-m", "mypy", "src/eris_ml/models/bundle_v3_b2.py",
            "src/eris_ml/models/candidate_v3_b2.py", "scripts/build_candidate_v3_b2.py",
        ],
        "ruff_changed": [
            sys.executable, "-m", "ruff", "check", "src/eris_ml/models/bundle_v3_b2.py",
            "src/eris_ml/models/candidate_v3_b2.py", "scripts/build_candidate_v3_b2.py",
            "tests/integration/test_candidate_v3_b2.py",
        ],
        "git_diff_check": ["git", "diff", "--check"],
    }
    results = {}
    for name, command in commands.items():
        print(f"Quality gate: {name}", flush=True)
        process = subprocess.run(command, cwd=root, capture_output=True, text=True, check=False)
        results[name] = {
            "command": command,
            "exit_code": process.returncode,
            "stdout": process.stdout[-10000:],
            "stderr": process.stderr[-3000:],
        }
    baseline.write_json(report / "quality_gate.json", results)
    if any(result["exit_code"] != 0 for result in results.values()):
        raise ValueError("B2 quality gate failed; inspect report/quality_gate.json.")
    return results


def run(root: Path, run_id: str = RUN) -> None:
    if run_id != RUN or not re.fullmatch(r"b2", run_id):
        raise ValueError("Only the authorized fixed b2 run is supported.")
    report = protected_path(root, f"artifacts/reports/v3_final_15/{run_id}")
    directory = protected_path(root, f"artifacts/models/v3_final_15/{run_id}")
    if report.exists() or directory.exists() or report.is_symlink() or directory.is_symlink():
        raise FileExistsError("B2 output exists; refusing fit/overwrite.")
    config, frame, _, identity, evidence, protected = preflight(root)
    summary = protected_path(root, SUMMARY)
    summary_before = summary.read_bytes()
    git = {
        key: subprocess.check_output(["git", *command], cwd=root, text=True).strip()
        for key, command in {
            "head": ["rev-parse", "HEAD"],
            "branch": ["branch", "--show-current"],
            "status": ["status", "--short"],
        }.items()
    }
    baseline.write_json(report / "protocol.json", POLICY)
    baseline.write_json(
        report / "preflight.json",
        {
            "created_at_utc": datetime.now(UTC).isoformat(),
            "git": git,
            "development_rows": len(frame),
            "target_counts": frame.Attrition.value_counts().to_dict(),
            "feature_order": [f["name"] for f in config["features"]],
            "input_sha256": identity["data_provenance"]["input_sha256"],
            "config_sha256": sha256_file(
                protected_path(root, "configs/features/feature_set_v3_final_15.yaml")
            ),
            "devfix1_identity_sha256": tuning.digest(identity),
            "protected_sha256": protected,
            "versions": v3.versions(),
            "summary_before_sha256": hashlib.sha256(summary_before).hexdigest(),
        },
    )
    print("B2 preflight locked: 1176 development rows, fixed unweighted raw RF.", flush=True)
    params = dict(identity["rf_parameters"])
    params["class_weight"] = None
    pipeline, X, fit = fit_fixed(config, frame, params)
    baseline.write_json(report / "fit.json", fit)
    positions = np.random.default_rng(42).choice(len(X), 100, replace=False).tolist()
    background = np.asarray(pipeline[:-1].transform(X.iloc[positions]), dtype=float)
    meta = metadata(
        config,
        fit["resolved_model_parameters"],
        identity,
        frame,
        positions,
        sha256_file(report / "preflight.json"),
        sha256_file(protected_path(root, "configs/features/feature_set_v3_final_15.yaml")),
    )
    bundle = B2Bundle(pipeline, meta, background, v3.feature_mapping(pipeline))
    bundle.validate()
    explain_one(bundle, cast(dict[str, Any], X.iloc[0].to_dict()))
    save_bundle(bundle, directory)
    binary_sha = sha256_file(directory / "pipeline.joblib")
    checks = verify_inference(bundle, directory, X, report)
    baseline.write_json(
        report / "optional_missing.json",
        {
            "source": "devfix1 original and validation-copy OOF; no b2 retraining",
            "model": MODEL,
            "scenarios": evidence["optional"],
        },
    )
    baseline.write_json(report / "development_evidence.json", evidence["comparison"])
    card = report / "model_card.md"
    with card.open("x", encoding="utf-8") as stream:
        stream.write(model_card(evidence, evidence["optional"], binary_sha))
    quality_gate(root, report)
    tuning.verify_hashes(root, protected)
    if summary.read_bytes() != summary_before:
        raise ValueError("Historical summary changed during B2 build.")
    suffix = (
        "\n\n## Candidate b2: fixed RF unweighted raw (development remediation)\n\n"
        f"`{MODEL_VERSION}` uses the locked 15 inputs, 600-tree depth-4 RF and "
        "`class_weight=None`; one full-development fit. Devfix1 OOF passed project "
        "probability guardrails; see `b2/development_evidence.json` and "
        "`b2/optional_missing.json`. No hyperparameter search or calibration. "
        "The previously consumed final evaluation belongs to b1 only; b2 has no independent "
        "final evaluation, HR approval or production status. Model card: "
        "`b2/model_card.md`. No threshold or API change.\n"
    ).encode()
    with summary.open("ab") as stream:
        stream.write(suffix)
    outputs = [p for path in (report, directory) for p in path.iterdir() if p.is_file()]
    baseline.write_json(
        report / "metadata.json",
        {
            "bundle_pipeline_sha256": binary_sha,
            "bundle_model_binary_sha256": binary_sha,
            "bundle_manifest_sha256": sha256_file(directory / "manifest.json"),
            "output_sha256": {p.relative_to(root).as_posix(): sha256_file(p) for p in outputs},
            "summary_prefix_preserved": summary.read_bytes().startswith(summary_before),
            "summary_after_sha256": sha256_file(summary),
            "protected_sha256_count": len(protected),
            "protected_hashes_unchanged": True,
            "fit": fit,
            "checks": checks,
            "final_holdout_accessed": False,
            "f1_predictions_accessed": False,
            "independent_final_validation_for_b2": False,
        },
    )
    print(f"B2 bundle verified: {binary_sha}", flush=True)
