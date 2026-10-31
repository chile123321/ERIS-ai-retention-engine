"""Minimal offline Tkinter tester for the frozen ERIS V3 B2 bundle."""

from __future__ import annotations

import hashlib
import json
import math
import sys
import time
import tkinter as tk
from pathlib import Path
from tkinter import ttk
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from eris_ml.models.bundle_v3_b2 import (  # noqa: E402
    explain_one,
    load_bundle,
    predict_one,
)


BUNDLE_DIR = ROOT / "artifacts" / "models" / "v3_final_15" / "b2"
EXAMPLE_PATH = ROOT / "artifacts" / "reports" / "v3_final_15" / "b2" / "inference_examples.json"
EXPECTED_MODEL_SHA256 = "0c72841a8505a910d31c2a687fbcbf5259e3afb60f9eddcb7586801519b5628f"
WARNING = "Experimental model — decision support only. Not approved for automated HR decisions."
MISSING = "Missing"


class InputError(ValueError):
    """Friendly input-validation error for the local form."""


def file_sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def load_runtime() -> tuple[Any, list[dict[str, Any]], dict[str, Any], str]:
    model_hash = file_sha256(BUNDLE_DIR / "pipeline.joblib")
    if model_hash != EXPECTED_MODEL_SHA256:
        raise RuntimeError("Bundle B2 model hash does not match the expected artifact.")
    bundle = load_bundle(BUNDLE_DIR)
    features = bundle.metadata["feature_config"]["features"]
    if [item["name"] for item in features] != bundle.metadata["feature_order"]:
        raise RuntimeError("Bundle feature schema and order do not match.")
    examples = json.loads(EXAMPLE_PATH.read_text(encoding="utf-8"))["examples"]
    sample = dict(next(iter(examples.values()))["input"])
    return bundle, features, sample, model_hash


def parse_values(values: dict[str, str], features: list[dict[str, Any]]) -> dict[str, Any]:
    record: dict[str, Any] = {}
    for definition in features:
        name = definition["name"]
        raw = values.get(name, "").strip()
        if not definition["required"] and (not raw or raw == MISSING):
            continue
        if not raw:
            raise InputError(f"{name}: please enter a value.")
        if definition["type"] == "categorical":
            if raw not in definition["vocabulary"]:
                raise InputError(f"{name}: choose one of the listed values.")
            record[name] = raw
            continue
        try:
            value = int(raw)
            if str(value) != raw.lstrip("+"):
                raise ValueError
            if not math.isfinite(float(value)):
                raise ValueError
        except (ValueError, OverflowError) as exc:
            raise InputError(f"{name}: enter a valid whole number.") from exc
        if "minimum" in definition and value < definition["minimum"]:
            raise InputError(f"{name}: minimum is {definition['minimum']}.")
        if "values" in definition and value not in definition["values"]:
            allowed = ", ".join(str(item) for item in definition["values"])
            raise InputError(f"{name}: allowed values are {allowed}.")
        record[name] = value
    return record


def friendly_error(exc: Exception) -> str:
    if isinstance(exc, InputError):
        return str(exc)
    if isinstance(exc, (ValueError, TypeError, OverflowError)):
        return "Invalid input: check each value and try again."
    return "The local model could not complete this request."


class ManualApp:
    def __init__(
        self,
        root: tk.Tk,
        bundle: Any,
        features: list[dict[str, Any]],
        sample: dict[str, Any],
        model_hash: str,
    ) -> None:
        self.root = root
        self.bundle = bundle
        self.features = features
        self.sample = sample
        self.model_hash = model_hash
        self.widgets: dict[str, ttk.Entry | ttk.Combobox] = {}
        self.status = tk.StringVar(value="Bundle loaded and verified.")
        self.result = tk.StringVar(value="No prediction yet.")
        self.details = tk.StringVar(value="")
        self.explanation = tk.StringVar(value="")
        self._build()
        self.load_sample()

    def _build(self) -> None:
        self.root.title("ERIS V3 B2 — Manual Model Test")
        self.root.minsize(760, 720)
        outer = ttk.Frame(self.root, padding=12)
        outer.pack(fill="both", expand=True)

        ttk.Label(outer, text="ERIS V3 B2 — Manual Model Test", font=("Segoe UI", 16, "bold")).pack()
        ttk.Label(
            outer,
            text=(
                f"Model: {self.bundle.metadata['model_version']}  |  Features: {len(self.features)}  |  "
                f"SHA-256: {self.model_hash[:12]}…"
            ),
        ).pack(pady=(2, 8))
        ttk.Label(outer, textvariable=self.status, foreground="#176b2c").pack()

        form = ttk.LabelFrame(outer, text="Employee input (IBM coding)", padding=10)
        form.pack(fill="x", pady=8)
        for row, definition in enumerate(self.features):
            name = definition["name"]
            suffix = " (optional)" if not definition["required"] else ""
            ttk.Label(form, text=name + suffix).grid(row=row, column=0, sticky="w", padx=(0, 12), pady=2)
            choices: list[str] = []
            if definition["type"] == "categorical":
                choices = [str(item) for item in definition["vocabulary"]]
            elif "values" in definition:
                choices = [str(item) for item in definition["values"]]
            if not definition["required"]:
                choices = [MISSING]
            if choices and (definition["type"] == "categorical" or "values" in definition):
                widget: ttk.Entry | ttk.Combobox = ttk.Combobox(
                    form, values=choices, state="readonly", width=34
                )
            else:
                widget = ttk.Entry(form, width=37)
            widget.grid(row=row, column=1, sticky="ew", pady=2)
            self.widgets[name] = widget
        form.columnconfigure(1, weight=1)

        buttons = ttk.Frame(outer)
        buttons.pack(fill="x", pady=5)
        for label, command in (
            ("Load sample", self.load_sample),
            ("Predict", self.predict),
            ("Explain", self.explain),
            ("Reset", self.reset),
            ("Exit", self.root.destroy),
        ):
            ttk.Button(buttons, text=label, command=command).pack(side="left", padx=4)

        ttk.Label(outer, textvariable=self.result, font=("Segoe UI", 15, "bold")).pack(pady=(10, 2))
        ttk.Label(outer, textvariable=self.details, justify="left").pack(anchor="w")
        ttk.Label(outer, textvariable=self.explanation, justify="left").pack(anchor="w", pady=7)
        ttk.Separator(outer).pack(fill="x", pady=8)
        ttk.Label(outer, text=WARNING, foreground="#9a3b00", wraplength=720).pack(anchor="w")
        ttk.Label(
            outer,
            text="IBM-coded demonstration only; not validated for a new employer or a 3–6 month horizon.",
            wraplength=720,
        ).pack(anchor="w", pady=(3, 0))

    def _values(self) -> dict[str, str]:
        return {name: widget.get() for name, widget in self.widgets.items()}

    def _record(self) -> dict[str, Any]:
        return parse_values(self._values(), self.features)

    def _set_value(self, name: str, value: Any) -> None:
        widget = self.widgets[name]
        widget.set(str(value)) if isinstance(widget, ttk.Combobox) else (
            widget.delete(0, tk.END), widget.insert(0, str(value))
        )

    def load_sample(self) -> None:
        for definition in self.features:
            name = definition["name"]
            value = self.sample.get(name, MISSING)
            self._set_value(name, int(value) if isinstance(value, float) and value.is_integer() else value)
        self.status.set("Saved B2 sample loaded; no identifier or target displayed.")
        self.result.set("No prediction yet.")
        self.details.set("")
        self.explanation.set("")

    def reset(self) -> None:
        for definition in self.features:
            self._set_value(definition["name"], MISSING if not definition["required"] else "")
        self.status.set("Form reset.")
        self.result.set("No prediction yet.")
        self.details.set("")
        self.explanation.set("")

    def predict(self) -> None:
        try:
            started = time.perf_counter()
            output = predict_one(self.bundle, self._record())
            elapsed_ms = (time.perf_counter() - started) * 1000
            probability = float(output["estimated_probability"])
            imputed = output["imputed_features"] or ["None"]
            self.result.set(f"{probability * 100:.2f}% — estimated probability of Attrition=1")
            self.details.set(
                f"Model: {output['model_version']}\n"
                f"Imputed optional features: {', '.join(imputed)}\n"
                f"Inference time: {elapsed_ms:.1f} ms"
            )
            self.explanation.set("")
            self.status.set("Prediction completed locally.")
        except Exception as exc:  # UI boundary: never expose a traceback.
            self.status.set(friendly_error(exc))
            self.result.set("Prediction not available.")
            self.details.set("")
            self.explanation.set("")

    def explain(self) -> None:
        try:
            started = time.perf_counter()
            output = explain_one(self.bundle, self._record())
            elapsed_ms = (time.perf_counter() - started) * 1000
            ranked = sorted(output["contributions"].items(), key=lambda item: abs(item[1]), reverse=True)[:5]
            lines = ["Top model contributions (SHAP, not causal):"]
            for name, value in ranked:
                direction = "increases" if value >= 0 else "decreases"
                lines.append(f"• {name}: {direction} probability ({value * 100:+.2f} pp)")
            lines.append(f"Explanation time: {elapsed_ms:.1f} ms")
            self.explanation.set("\n".join(lines))
            self.status.set("Explanation completed locally.")
        except Exception as exc:  # UI boundary: never expose a traceback.
            self.status.set(friendly_error(exc))
            self.explanation.set("Explanation not available.")


def main() -> None:
    try:
        bundle, features, sample, model_hash = load_runtime()
        root = tk.Tk()
        ManualApp(root, bundle, features, sample, model_hash)
        root.mainloop()
    except Exception as exc:
        print(f"Unable to start manual tester: {friendly_error(exc)}", file=sys.stderr)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
