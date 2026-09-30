"""Reliability tables and predeclared guardrails for V2 Logistic calibration."""

import json
from typing import Any

import numpy as np
import pandas as pd

from eris_ml.evaluation.calibration_reporting import validate_probabilities
from eris_ml.evaluation.feature_ablation import METRICS, probability_metrics
from eris_ml.evaluation.feature_ablation_reporting import markdown_table
from eris_ml.models.calibration_v2 import FEATURES, METHODS, PROTOCOL


def reliability(target: np.ndarray, scores: np.ndarray) -> pd.DataFrame:
    labels, probability = validate_probabilities(target, scores)
    edges = np.unique(np.quantile(probability, np.linspace(0, 1, 11)))
    indices = np.searchsorted(edges[1:-1], probability, side="right")
    rows = []
    for bin_id, index in enumerate(np.unique(indices), 1):
        mask = indices == index
        count, positives = int(mask.sum()), int(labels[mask].sum())
        predicted, observed = float(probability[mask].mean()), float(labels[mask].mean())
        rows.append({"bin_id": bin_id, "count": count, "positives": positives,
                     "mean_probability": predicted, "observed_positive_rate": observed,
                     "absolute_gap": abs(predicted - observed),
                     "low_support": count < 30 or positives < 10})
    return pd.DataFrame(rows)


def choose_method(summary: pd.DataFrame, folds: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    """Same-feature pooled guards plus prespecified descriptive fold stability checks."""
    indexed = summary.set_index("method")
    fold_index = folds.set_index(["method", "fold"])
    raw = indexed.loc["raw"]
    criteria = PROTOCOL["selection"]
    records = []
    eligible = []
    for method in ("sigmoid", "isotonic"):
        candidate = indexed.loc[method]
        gains = np.array([fold_index.loc[("raw", fold), "brier_score"]
                          - fold_index.loc[(method, fold), "brier_score"] for fold in range(1, 6)])
        improved = int((gains > 0).sum())
        failures = []
        if candidate.brier_score >= raw.brier_score:
            failures.append("Brier not improved")
        for metric in ("average_precision", "roc_auc"):
            if candidate[metric] < raw[metric] - criteria["max_auc_drop"]:
                failures.append(f"{metric} guardrail")
        if improved < criteria["minimum_improved_folds"][method]:
            failures.append("insufficient improved folds")
        if float(gains.min()) < -criteria["max_fold_brier_increase"]:
            failures.append("large individual-fold Brier loss")
        if any(np.delete(gains, i).mean() <= 0 for i in range(5)):
            failures.append("gain depends on one fold")
        if not failures:
            eligible.append(method)
        records.append({"method": method, "eligible": not failures,
                        "reasons": "; ".join(failures) or "all checks passed",
                        "brier_improved_folds": improved,
                        **{f"{metric}_delta": float(candidate[metric] - raw[metric])
                           for metric in METRICS},
                        **{f"fold_{i + 1}_brier_gain": float(v) for i, v in enumerate(gains)}})
    selected = "raw"
    if eligible:
        selected = min(eligible, key=lambda method: indexed.loc[method, "brier_score"])
        if ("sigmoid" in eligible and indexed.loc["sigmoid", "brier_score"]
            <= indexed.loc[selected, "brier_score"] + criteria["sigmoid_tie_tolerance"]):
            selected = "sigmoid"
    return selected, pd.DataFrame(records)


def summarize(oof: pd.DataFrame) -> dict[str, pd.DataFrame]:
    summaries, folds, bins = [], [], []
    for feature in FEATURES:
        for method in METHODS:
            rows = oof.loc[(oof.feature_set == feature) & (oof.method == method)]
            target, scores = rows.Attrition.to_numpy(), rows.probability.to_numpy()
            binned = reliability(target, scores)
            binned["feature_set"], binned["method"] = feature, method
            bins.append(binned)
            summary = {"feature_set": feature, "method": method,
                       **probability_metrics(target, scores),
                       "mean_probability": float(scores.mean()),
                       "observed_positive_rate": float(target.mean()),
                       "ece": float((binned['count'] * binned.absolute_gap).sum() / len(rows))}
            local = []
            for fold, part in rows.groupby("fold"):
                record = {"feature_set": feature, "method": method, "fold": fold,
                          **probability_metrics(
                              part.Attrition.to_numpy(), part.probability.to_numpy()),
                          "mean_probability": float(part.probability.mean()),
                          "observed_positive_rate": float(part.Attrition.mean())}
                local.append(record)
                folds.append(record)
            for metric in METRICS:
                values = [part[metric] for part in local]
                summary[f"{metric}_mean"] = float(np.mean(values))
                summary[f"{metric}_std"] = float(np.std(values, ddof=1))
            summaries.append(summary)
    summary_frame, fold_frame = pd.DataFrame(summaries), pd.DataFrame(folds)
    decisions, deltas = [], []
    for feature in FEATURES:
        method, delta = choose_method(summary_frame.loc[summary_frame.feature_set == feature],
                                      fold_frame.loc[fold_frame.feature_set == feature])
        delta["feature_set"] = feature
        deltas.append(delta)
        decisions.append({"feature_set": feature, "selected_method": method,
                          "production_approved": False})
    return {"summary": summary_frame, "folds": fold_frame,
            "bins": pd.concat(bins, ignore_index=True),
            "deltas": pd.concat(deltas, ignore_index=True), "decisions": pd.DataFrame(decisions)}


def build_report(tables: dict[str, pd.DataFrame], trace: dict[str, Any]) -> str:
    return "\n\n".join([
        "# V2 cross-fitted calibration — fixed Logistic baselines",
        "## Protocol",
        "Development: 1,176 rows, target {0:986, 1:190}, positive rate 16.156463%. "
        "Saved five outer folds reused without regeneration. Logistic C=1, liblinear, "
        "max_iter=2000, class_weight=None, random_state=42; no tuned models used.",
        "Implementation: CalibratedClassifierCV(estimator=complete_pipeline, "
        "ensemble=False, n_jobs=1). Four inner stratified folds, shuffled with seed 43, "
        "produce out-of-fold decision_function scores on outer-training only. Each inner "
        "base pipeline fits its own imputation/encoding/scaling. The calibrator learns from "
        "these inner-OOF scores; a fresh final base pipeline is fitted on ALL outer-training "
        "rows. Outer-validation is predicted once per candidate. No prefit, no calibration "
        "on pooled development OOF, and no full-development fit. Unlike V1 ensemble=True, "
        "this protocol has one full-outer-training base estimator, not four averaged estimators.",
        "All 10 raw folds passed regression against saved baseline probabilities before "
        "any calibration fit (rtol=1e-7, atol=1e-9). Exact feature projection precedes fitting: "
        "general_16 never supplies Department or JobRole; no_department_17 retains JobRole only.",
        "## Pooled OOF and mean/sample SD (ddof=1)", markdown_table(tables["summary"]),
        "AP denotes average precision; lower Brier is better. Mean fold metrics differ from "
        "pooled OOF metrics. ECE is descriptive and depends on quantile binning.",
        "## Each outer fold", markdown_table(tables["folds"]),
        "## Deltas, guardrails and stability", markdown_table(tables["deltas"]),
        "Predeclared: reject pooled AP or ROC drops >0.005; require pooled Brier improvement, "
        "at least 3/5 improving folds for sigmoid (4/5 for isotonic), no individual Brier "
        "increase >0.005, and positive mean fold Brier gain after leaving out any one fold. "
        "These conservative stability rules are development heuristics, not significance tests. "
        "Among eligible methods choose lowest Brier; prefer sigmoid within 0.0001.",
        "## Reliability counts", markdown_table(tables["bins"]),
        "Ten quantile bins requested; tied scores remain together and duplicate edges collapse. "
        "Empty bins are omitted. low_support means count<30 OR positive cases<10. "
        "Such bins, including bins with zero positives, are not precise evidence of reliability. "
        "Bins are descriptive evaluation summaries, never calibration training inputs; "
        "bin boundaries differ by candidate, and no causal interpretation is warranted.",
        "## Separate development decisions", markdown_table(tables["decisions"]),
        "A lower Brier does not override discrimination guardrails. These decisions use only "
        "development evidence, not V1 final-test scores; they do not select a production model "
        "or prove portability to Vietnamese companies or a 3–6 month horizon. "
        "V1 final test was used once previously and is not a new independent V2 holdout.",
        "Next step: business-aware cross-fitted threshold policy evaluation for the retained "
        "probability methods, with policy selection entirely inside outer-training and "
        "explicit HR capacity/cost assumptions; no final-test access.",
        "## Trace and safety",
        "Final-test CSV accessed: no. Model serialization: no. Tuning, threshold search, API "
        "changes, commit, push: no. Existing artifacts preserved; outputs exclusively created.",
        "```json\n" + json.dumps(trace, indent=2, allow_nan=False) + "\n```",
    ]) + "\n"
