"""Offline console test: 15 IBM inputs, calibrated probability and original-feature SHAP."""

import argparse
import json
from pathlib import Path

import pandas as pd

from eris_ml.explainability.adaboost_shap import load_bundle, parse_inputs

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=ROOT / "artifacts/models/adaboost_v3/s1")
    parser.add_argument("--sample", action="store_true", help="Use the fixed development example")
    parser.add_argument("--json", action="store_true", help="Machine-readable smoke test output")
    args = parser.parse_args()
    try:
        bundle = load_bundle(args.bundle)
        sample = bundle.metadata["manual_sample"]
        if args.sample:
            frame = pd.DataFrame([sample])
        else:
            print("AdaBoost baseline + sigmoid — offline research/demo candidate")
            print("IBM coding only. Enter=sample; optional: type Missing. No inputs are saved.")
            while True:
                values = {}
                for spec in bundle.metadata["feature_config"]["features"]:
                    name = spec["name"]
                    hint = spec.get("vocabulary", spec.get("values", "nonnegative integer"))
                    optional = " / Missing" if not spec["required"] else ""
                    text = input(f"{name} ({hint}{optional}) [{sample[name]}]: ").strip()
                    values[name] = text if text else str(sample[name])
                try:
                    frame = parse_inputs(values, bundle.metadata["feature_config"])
                    break
                except (ValueError, TypeError, OverflowError) as error:
                    print(f"Invalid input: {error}. Please re-enter the form.")
        result = bundle.explainer().explain(frame)
        result["model_id"] = bundle.metadata["model_id"]
        if args.json:
            print(json.dumps(result, ensure_ascii=False, allow_nan=False))
            return
        print(f"\n{result['estimated_probability']:.2%} — estimated P(Attrition=Yes)")
        print(f"Background base: {result['base_value']:.2%}; model: {result['model_id']}")
        print(f"Imputed: {result['imputed_features']}; explain time: {result['seconds']:.2f}s")
        ranked = sorted(
            zip(result["feature_order"], result["contributions_percentage_points"], strict=True),
            key=lambda item: abs(item[1]),
            reverse=True,
        )
        print("Top 5 contributions (percentage points, not relative percent):")
        for name, contribution in ranked[:5]:
            print(f"  {name}={result['feature_values'][name]}: {contribution:+.3f} pp")
        print("Experimental decision support only. Not causal; no HR cutoff or forecast horizon.")
    except (ValueError, TypeError, OverflowError, OSError, KeyError) as error:
        parser.exit(1, f"Cannot predict/explain: {error}\n")
    except (EOFError, KeyboardInterrupt):
        parser.exit(0, "\nClosed without saving input.\n")


if __name__ == "__main__":
    main()
