"""One development-only final RF selection/refit; immutable versioned local outputs."""

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
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss
from sklearn.model_selection import GridSearchCV
from threadpoolctl import threadpool_limits

from eris_ml.features.schema_v3 import validate_features
from eris_ml.models import baseline_v3 as baseline
from eris_ml.models import calibration_v3 as calibration
from eris_ml.models import tuning_v3 as tuning
from eris_ml.models.bundle_v3 import (
    ATOL,
    RTOL,
    CandidateBundle,
    explain_one,
    feature_mapping,
    load_bundle,
    predict_batch,
    predict_one,
    save_bundle,
    versions,
)
from eris_ml.utils.hashing import sha256_file

SUMMARY = "artifacts/reports/v3_final_15/summary.md"
AUDIT = "data/processed/fairness_audit_development_v1.csv"
AGE_GROUPS = ["<30", "30-39", "40-49", ">=50"]
POLICY: dict[str, Any] = {
    "version": "rf-candidate-v3-v1",
    "candidate_source": "t1.identity.forest_candidates",
    "candidate_count": 40,
    "cv": {"n_splits": 4, "shuffle": True, "random_state": 43},
    "scoring": "average_precision",
    "tie_breaker": "first_original_candidate_exact_score_tie",
    "estimator_seed": 42,
    "search_n_jobs": 1,
    "forest_n_jobs": 4,
    "native_threads": 1,
    "cv_fits": 160,
    "refits": 1,
    "background": {
        "count": 100,
        "random_state": 42,
        "sampling": "numpy.default_rng.choice_without_replacement_in_assignment_order",
    },
    "shap": {
        "feature_perturbation": "interventional",
        "model_output": "probability",
        "atol": ATOL,
        "rtol": RTOL,
    },
    "subgroup": {
        "age_groups": AGE_GROUPS,
        "minimum_n": 50,
        "minimum_positives": 10,
        "minimum_negatives": 10,
        "support_source": "fairness_protocol_v2",
        "missing_unknown": "reject",
        "policy": "saved_c1_top_10_no_subgroup_rerank",
        "undefined": "null",
        "small_groups": "descriptive_only_insufficient_evidence",
    },
    "production_approved": False,
    "final_holdout_evaluated": False,
}


def aligned_audit(frame: pd.DataFrame, audit: pd.DataFrame) -> pd.DataFrame:
    if (
        len(audit) != len(frame)
        or audit.source_row.duplicated().any()
        or audit.EmployeeNumber.duplicated().any()
        or audit.isna().any().any()
        or set(audit.split) != {"development"}
        or set(audit.source_row) != set(frame.source_row)
    ):
        raise ValueError("Audit metadata must be complete, unique and development-only.")
    audit = audit.set_index("source_row").loc[frame.source_row].reset_index()
    labels = audit.Attrition.replace({"No": 0, "Yes": 1}).astype(int)
    if (
        not np.array_equal(audit.EmployeeNumber, frame.EmployeeNumber)
        or not np.array_equal(audit.Age, frame.Age)
        or not np.array_equal(labels, frame.Attrition)
        or not audit.Gender.isin(["Female", "Male"]).all()
        or not audit.MaritalStatus.isin(["Single", "Married", "Divorced"]).all()
    ):
        raise ValueError("Development/audit ID, target, Age or category mismatch.")
    return audit


def prepare(
    root: Path,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    config, frame, assignments, reference, _, identity = calibration.prepare(
        root, calibration.load_policy(root)
    )
    prefix = "artifacts/reports/v3_final_15/c1"
    prior = tuning.read_json(baseline.safe_path(root, f"{prefix}/preflight.json"))
    meta = tuning.read_json(baseline.safe_path(root, f"{prefix}/metadata.json"))
    if (
        prior["identity"] != identity
        or tuning.digest(identity) != meta["identity_sha256"]
        or sha256_file(root / prefix / "preflight.json") != meta["preflight_sha256"]
    ):
        raise ValueError("Historical c1 identity changed; no silent provenance bridge.")
    protected = dict(identity["protected_sha256"])
    for key in ("checkpoint_sha256", "output_sha256"):
        tuning.verify_hashes(root, meta[key])
        protected.update(meta[key])
    for name in ("metadata.json", "preflight.json"):
        path = root / prefix / name
        protected[path.relative_to(root).as_posix()] = sha256_file(path)
    original = pd.read_csv(root / "artifacts/predictions/v3_final_15/c1/oof.csv")
    stress = pd.read_csv(root / "artifacts/predictions/v3_final_15/c1/optional_missing_oof.csv")
    verified = calibration.validate_results(
        pd.concat([original, stress], ignore_index=True), assignments
    )
    raw = verified.loc[(verified.model == "rf_tuned_raw") & (verified.scenario == "original")]
    calibration.check_raw(
        raw,
        reference.loc[reference.model == "rf_tuned_raw"],
        assignments,
        {"rtol": 1e-7, "atol": 1e-9},
    )
    if len(frame) != 1176 or frame.Attrition.value_counts().to_dict() != {0: 986, 1: 190}:
        raise ValueError("Wrong development cohort.")
    audit_path = baseline.safe_path(root, AUDIT)
    audit = aligned_audit(frame, pd.read_csv(audit_path)) if audit_path.exists() else None
    if audit is not None:
        protected[AUDIT] = sha256_file(audit_path)
    cohort = frame[[*baseline.IDS, "Age"]].copy()
    if audit is not None:
        cohort[["Gender", "MaritalStatus"]] = audit[["Gender", "MaritalStatus"]].to_numpy()
    candidates = tuning.read_json(root / "artifacts/reports/v3_final_15/t1/preflight.json")[
        "identity"
    ]["forest_candidates"]
    if len(candidates) != 40 or len({tuning.digest(x) for x in candidates}) != 40:
        raise ValueError("Expected forty distinct recorded t1 candidates.")
    for relative in (
        "src/eris_ml/models/candidate_v3.py",
        "src/eris_ml/models/bundle_v3.py",
        "scripts/build_candidate_v3.py",
        "configs/evaluation/fairness_protocol_v2.yaml",
        "typings/shap/__init__.pyi",
    ):
        protected[relative] = sha256_file(baseline.safe_path(root, relative))
    provenance = {
        "policy": POLICY,
        "feature_config": config,
        "candidates": candidates,
        "protected_sha256": protected,
        "library_versions": versions(),
        "c1_identity_sha256": meta["identity_sha256"],
        "data_provenance": identity["data_provenance"],
        "audit_metadata_available": audit is not None,
        "development_ids": frame[baseline.IDS].to_dict("records"),
        "resolved_template": baseline.json_safe(
            tuning.search_pipeline(config, tuning.load_search_policy(root), "random_forest")
        ),
        "selection_validation_source_rows": [
            frame.iloc[v].source_row.tolist()
            for _, v in tuning.inner_splits(frame.Attrition.to_numpy())
        ],
    }
    return config, frame, raw.reset_index(drop=True), cohort, provenance


def select_and_fit(
    frame: pd.DataFrame,
    config: dict[str, Any],
    candidates: list[dict[str, Any]],
    forest_jobs: int = 4,
) -> tuple[Any, dict[str, Any]]:
    """Exact supplied order; CV preprocessing is fitted inside each cloned pipeline."""
    names = [f["name"] for f in config["features"]]
    X = validate_features(frame[names], config)
    y = frame.Attrition.to_numpy()
    splits = tuning.inner_splits(y)
    template = calibration.pipeline_for(config, "rf_tuned", {"model__n_jobs": forest_jobs})
    grid = [{key: [value] for key, value in candidate.items()} for candidate in candidates]
    search: Any = GridSearchCV(
        template,
        param_grid=grid,
        scoring="average_precision",
        cv=splits,
        refit=True,
        n_jobs=1,
        error_score="raise",
        return_train_score=False,
    )
    started = time.monotonic()
    with warnings.catch_warnings(record=True) as captured, threadpool_limits(limits=1):
        warnings.simplefilter("always")
        search.fit(X, y)
    results = search.cv_results_
    if (
        results["params"] != candidates
        or not np.isfinite(results["mean_test_score"]).all()
        or int(search.best_index_) != int(np.argmax(results["mean_test_score"]))
    ):
        raise ValueError("Selection candidate order, score or exact tie rule changed.")
    pipeline = search.best_estimator_
    optional = [f["name"] for f in config["features"] if not f["required"]]
    medians = (
        pipeline.named_steps["preprocessor"]
        .named_transformers_["optional_numeric"]
        .named_steps["imputer"]
        .statistics_
    )
    if not np.array_equal(medians, X[optional].median().to_numpy()):
        raise ValueError("Final refit median differs from development median.")
    return pipeline, {
        "selection_cv_ap": float(search.best_score_),
        "best_index": int(search.best_index_),
        "best_params": search.best_params_,
        "cv_results": baseline.json_safe(results),
        "resolved_model_parameters": pipeline.named_steps["model"].get_params(),
        "cv_fits": len(candidates) * len(splits),
        "refits": 1,
        "elapsed_seconds": time.monotonic() - started,
        "warnings": dict(Counter(f"{w.category.__name__}: {w.message}" for w in captured)),
        "interpretation": "Selection score; not independent or final-test evidence.",
    }


def subgroup_table(raw: pd.DataFrame, cohort: pd.DataFrame) -> pd.DataFrame:
    joined = raw.merge(cohort, on=["source_row", "EmployeeNumber"], validate="one_to_one")
    if len(joined) != len(cohort) or not np.array_equal(joined.y_true, joined.Attrition):
        raise ValueError("Subgroup OOF/cohort alignment failed.")
    joined["AgeGroup"] = pd.cut(
        joined.Age, [-np.inf, 30, 40, 50, np.inf], labels=AGE_GROUPS, right=False
    )
    records: list[dict[str, Any]] = []
    for attribute in ["AgeGroup", *[a for a in ("Gender", "MaritalStatus") if a in cohort]]:
        groups = AGE_GROUPS if attribute == "AgeGroup" else sorted(joined[attribute].unique())
        for fold in (0, 1, 2, 3, 4, 5):
            scope = joined if fold == 0 else joined.loc[joined.fold == fold]
            for group in groups:
                block = scope.loc[scope[attribute] == group]
                y, pred = block.y_true.to_numpy(), block.top_10.to_numpy()
                n, pos = len(block), int(sum(y))
                neg = n - pos
                tp, fp = int(sum((y == 1) & (pred == 1))), int(sum((y == 0) & (pred == 1)))
                alerts = tp + fp
                records.append(
                    {
                        "attribute": attribute,
                        "group": group,
                        "fold": fold,
                        "n": n,
                        "positives": pos,
                        "negatives": neg,
                        "positive_rate": pos / n if n else None,
                        "mean_probability": float(block.probability.mean()) if n else None,
                        "average_precision": float(average_precision_score(y, block.probability))
                        if pos and neg
                        else None,
                        "brier": float(brier_score_loss(y, block.probability)) if n else None,
                        "alerts": alerts,
                        "tp": tp,
                        "fp": fp,
                        "fn": pos - tp,
                        "tn": neg - fp,
                        "alert_rate": alerts / n if n else None,
                        "tpr": tp / pos if pos else None,
                        "fpr": fp / neg if neg else None,
                        "precision": tp / alerts if alerts else None,
                        "sufficient_support": n >= 50 and pos >= 10 and neg >= 10,
                    }
                )
            subset = records[-len(groups) :]
            if (
                sum(r["n"] for r in subset) != len(scope)
                or sum(r["positives"] for r in subset) != int(scope.y_true.sum())
                or sum(r["alerts"] for r in subset) != int(scope.top_10.sum())
            ):
                raise ValueError("Subgroup totals do not equal cohort totals.")
    return pd.DataFrame(records)


def bundle_metadata(
    config: dict[str, Any],
    selection: dict[str, Any],
    provenance: dict[str, Any],
    run_id: str,
    background_positions: list[int],
) -> dict[str, Any]:
    return {
        "model_version": f"rf-v3-final15-{run_id}",
        "status": "research_demo_candidate",
        "feature_set": "v3_final_15",
        "feature_order": [f["name"] for f in config["features"]],
        "feature_config": config,
        "input_validation": {
            "extra_columns": "reject",
            "required_missing": "reject",
            "optional_missing_key_null_nan": "training_median",
            "numeric": "finite_nonnegative_integer_no_string_or_bool",
            "categories": "exact_IBM_vocabulary",
            "key_order": "normalize_to_schema",
        },
        "model_family": "random_forest",
        "calibration": "none",
        "threshold": None,
        "evaluation_status": "development_only",
        "final_holdout_evaluated": False,
        "production_approved": False,
        "business_approved": False,
        "target_mapping": {"No": 0, "Yes": 1},
        "positive_class": 1,
        "class_order": [0, 1],
        "resolved_model_parameters": selection["resolved_model_parameters"],
        "best_params": selection["best_params"],
        "selection_cv_ap": selection["selection_cv_ap"],
        "library_versions": versions(),
        "provenance": provenance,
        "background": {**POLICY["background"], "development_positions": background_positions},
        "shap_tolerance": {"atol": ATOL, "rtol": RTOL},
        "limitations": "IBM research candidate, not production approval. No validated employer "
        "transfer, 3-6 month horizon, intervention benefit or fairness certification. "
        "V1 holdout previously used; no final holdout accessed here. No default HR policy. "
        "Units and enterprise category/snapshot mappings remain unconfirmed. "
        "Load only trusted local joblib; manifest is integrity, not authentication.",
    }


def verify_runtime(
    bundle: CandidateBundle, directory: Path, X: pd.DataFrame, report: Path
) -> dict[str, Any]:
    probe = cast(list[dict[str, Any]], X.iloc[[0, 1, 2]].to_dict("records"))
    optional = [
        f["name"] for f in bundle.metadata["feature_config"]["features"] if not f["required"]
    ]
    for missing in ([optional[0]], [optional[1]], optional):
        probe.append({k: v for k, v in probe[0].items() if k not in missing})
    expected = predict_batch(bundle, probe)
    baseline.write_json(report / "reload_probe.json", {"records": probe, "expected": expected})
    code = (
        "import json,sys;from pathlib import Path;"
        "from eris_ml.models.bundle_v3 import load_bundle,predict_batch;"
        "b=load_bundle(Path(sys.argv[1]));p=json.loads(Path(sys.argv[2]).read_text());"
        "print(json.dumps(predict_batch(b,p['records'])))"
    )
    output = subprocess.check_output(
        [sys.executable, "-c", code, str(directory), str(report / "reload_probe.json")], text=True
    )
    actual = json.loads(output)
    difference = max(
        abs(a["probability"] - b["probability"]) for a, b in zip(expected, actual, strict=True)
    )
    if difference > 1e-12 or any(
        a["imputed_features"] != b["imputed_features"]
        for a, b in zip(expected, actual, strict=True)
    ):
        raise ValueError("Fresh process bundle reload regression failed.")
    loaded = load_bundle(directory)
    for record, result in zip(probe, expected, strict=True):
        one = predict_one(loaded, dict(reversed(list(record.items()))))
        if not np.isclose(one["probability"], result["probability"], atol=1e-12, rtol=1e-12):
            raise ValueError("Key order / one versus batch regression.")
        explain_one(loaded, record)
    for invalid in (
        {k: v for k, v in probe[0].items() if k != "Age"},
        {**probe[0], "Department": "Unknown"},
        {**probe[0], "Age": "30"},
        {**probe[0], "source_row": 1},
        {**probe[0], optional[0]: float("inf")},
    ):
        try:
            predict_one(loaded, invalid)
        except ValueError:
            continue
        raise ValueError("Invalid inference input accepted.")
    scores = bundle.probabilities(X)
    ordered = np.argsort(scores, kind="stable")
    examples: dict[str, Any] = {}
    for label, index in zip(
        ("low_score", "middle_score", "high_score"),
        (ordered[0], ordered[len(ordered) // 2], ordered[-1]),
        strict=True,
    ):
        record = cast(dict[str, Any], X.iloc[int(index)].to_dict())
        examples[label] = {"input": record, "output": explain_one(loaded, record)}
    masked = {k: v for k, v in examples["middle_score"]["input"].items() if k not in optional}
    examples["both_optional_missing"] = {"input": masked, "output": explain_one(loaded, masked)}
    baseline.write_json(
        report / "examples.json",
        {
            "scope": "in_sample_development_illustrations",
            "not_evaluation": True,
            "examples": examples,
        },
    )
    timings = {}
    operations: dict[str, Callable[[], Any]] = {
        "predict_one": lambda: predict_one(loaded, probe[0]),
        "predict_batch_100": lambda: predict_batch(loaded, X.iloc[:100]),
        "explain_one": lambda: explain_one(loaded, probe[0]),
    }
    for name, operation in operations.items():
        operation()
        elapsed = []
        for _ in range(5):
            started = time.perf_counter()
            operation()
            elapsed.append(time.perf_counter() - started)
        timings[name] = {"seconds": elapsed, "median_seconds": float(np.median(elapsed))}
    return {
        "fresh_process_max_absolute_difference": difference,
        "one_batch_key_order_optional_invalid_checks": "passed",
        "shap_all_classes_and_original_features_additivity": "passed",
        "shap_examples_max_absolute_error": max(
            e["output"]["absolute_additivity_error"] for e in examples.values()
        ),
        "timing_after_warmup_no_SLA": timings,
        "environment": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "executable": sys.executable,
            "versions": versions(),
        },
    }


def run(root: Path, run_id: str) -> None:
    if not re.fullmatch(r"b[0-9][a-zA-Z0-9_-]*", run_id):
        raise ValueError("Candidate run ID must begin with b and a digit.")
    report = baseline.safe_path(root, f"artifacts/reports/v3_final_15/{run_id}")
    directory = baseline.safe_path(root, f"artifacts/models/v3_final_15/{run_id}")
    if report.exists() or directory.exists():
        raise FileExistsError(
            "Candidate outputs exist; investigate, do not overwrite or auto-refit."
        )
    config, frame, raw, cohort, provenance = prepare(root)
    summary = baseline.safe_path(root, SUMMARY)
    prior_summary = summary.read_bytes()
    preflight = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "run_id": run_id,
        "provenance": provenance,
        "identity_sha256": tuning.digest(provenance),
        "summary_before_sha256": hashlib.sha256(prior_summary).hexdigest(),
        "summary_before_bytes": len(prior_summary),
        "git": {
            k: subprocess.check_output(["git", *v], cwd=root, text=True).strip()
            for k, v in {
                "head": ["rev-parse", "HEAD"],
                "branch": ["branch", "--show-current"],
                "status": ["status", "--short"],
            }.items()
        },
    }
    baseline.write_json(report / "preflight.json", preflight)
    print(
        "Preflight locked: 1176 development rows; 40 original candidates; 160+1 fits.", flush=True
    )
    pipeline, selection = select_and_fit(frame, config, provenance["candidates"])
    baseline.write_json(report / "selection.json", selection)
    print(
        f"Selection complete: index={selection['best_index']} "
        f"selection_cv_ap={selection['selection_cv_ap']:.9f}",
        flush=True,
    )
    names = [f["name"] for f in config["features"]]
    X = validate_features(frame[names], config)
    positions = np.random.default_rng(42).choice(len(X), 100, replace=False).tolist()
    background = np.asarray(pipeline[:-1].transform(X.iloc[positions]), dtype=float)
    meta = bundle_metadata(
        config,
        selection,
        {
            "preflight": (report / "preflight.json").relative_to(root).as_posix(),
            "preflight_sha256": sha256_file(report / "preflight.json"),
            "identity_sha256": preflight["identity_sha256"],
            "protected_sha256": provenance["protected_sha256"],
            "git": preflight["git"],
            "training_rows": len(frame),
            "target_counts": {"0": 986, "1": 190},
            "selection_cv": POLICY["cv"],
            "estimator_seed": 42,
        },
        run_id,
        positions,
    )
    bundle = CandidateBundle(pipeline, meta, background, feature_mapping(pipeline))
    # Check probability-space SHAP before publishing the bundle manifest.
    explain_one(bundle, cast(dict[str, Any], X.iloc[0].to_dict()))
    save_bundle(bundle, directory)
    checks = verify_runtime(bundle, directory, X, report)
    table = subgroup_table(raw, cohort)
    table.to_csv(report / "subgroups.csv", index=False, na_rep="NA", mode="x")
    tuning.verify_hashes(root, provenance["protected_sha256"])
    if summary.read_bytes() != prior_summary:
        raise ValueError("Summary changed during build; refusing to append.")
    pooled = table.loc[table.fold == 0]
    text = [
        f"\n\n## Candidate {run_id}: RF raw final development fit\n",
        f"Model `{meta['model_version']}`; one pipeline, research/demo only, "
        "no production approval.",
        f"Exact t1 forty candidates; 4-fold seed 43; 160 CV fits + 1 refit (RF seed 42). "
        f"selection_cv_ap={selection['selection_cv_ap']:.9f} is selection, "
        "not independent evaluation.",
        f"Best parameters: `{json.dumps(selection['best_params'], sort_keys=True)}`.",
        f"Bundle: `artifacts/models/v3_final_15/{run_id}`; checks: fresh process, schema, "
        "optional imputation, one/batch/key order and SHAP probability additivity passed.",
        "OOF evidence remains c1 (not recomputed from the final-fit model); see tables above. "
        "Examples are in-sample illustrations only. No default threshold or top-k policy stored.",
        "Background: random 100 development rows, NumPy default_rng seed 42. "
        "Signed one-hot contributions aggregate to 15 inputs; both class outputs checked "
        "with atol=rtol=1e-6. SHAP is not causal; contributions x100 are percentage points.",
        "\n### Descriptive saved c1 RF raw OOF / shared top10 subgroups\n",
        "Support flag: n>=50, positives>=10, negatives>=10 (V2 support convention). "
        "Defined sparse-group metrics remain descriptive, not stable evidence. "
        "NA denotes an undefined denominator/AUC; no within-group reranking.",
        "\n| Attribute/group | n | positive | alerts | TP | FP | FN | AP | Brier | "
        "TPR | FPR | Support |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in pooled.to_dict("records"):

        def fmt(value: Any) -> str:
            return "NA" if pd.isna(value) else f"{value:.4f}"

        text.append(
            f"| {row['attribute']}/{row['group']} | {row['n']} | {row['positives']} | "
            f"{row['alerts']} | {row['tp']} | {row['fp']} | {row['fn']} | "
            f"{fmt(row['average_precision'])} | {fmt(row['brier'])} | {fmt(row['tpr'])} | "
            f"{fmt(row['fpr'])} | {row['sufficient_support']} |"
        )
    text += [
        "\nPer-fold counts and undefined/low-support flags are in `" + run_id + "/subgroups.csv`. "
        "Age is also a model input; Gender/MaritalStatus are audit only when available. "
        "No fairness approval, quota changes or policy selection follows from this table.",
        "No final holdout accessed. IBM cross-sectional labels do not validate a 3-6 month "
        "forecast, intervention benefit or transfer to a new employer. V1 holdout was previously "
        "used; this candidate has not undergone final evaluation. No API/UI/deployment changes.",
    ]
    suffix = ("\n\n".join(text) + "\n").encode("utf-8")
    with summary.open("ab") as stream:
        stream.write(suffix)
    outputs = [p for p in report.iterdir() if p.is_file()] + list(directory.iterdir())
    baseline.write_json(
        report / "metadata.json",
        {
            "checks": checks,
            "bundle_manifest_sha256": sha256_file(directory / "manifest.json"),
            "output_sha256": {p.relative_to(root).as_posix(): sha256_file(p) for p in outputs},
            "summary_after_sha256": sha256_file(summary),
            "summary_prefix_preserved": True,
            "protected_hashes_unchanged": True,
            "final_holdout_accessed": False,
            "cv_fits": selection["cv_fits"],
            "refits": selection["refits"],
            "warnings": selection["warnings"],
            "production_approved": False,
        },
    )
    print(f"Candidate verified: {directory}; protected inputs unchanged.", flush=True)
