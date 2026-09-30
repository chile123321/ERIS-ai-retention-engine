"""Shared frozen-bundle inference for the API and local manual testing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from eris_ml.models.bundle import ModelBundle


@dataclass(frozen=True)
class InferenceResult:
    """One raw probability with the bundle's frozen alert decision."""

    probability: float
    threshold: float
    alert: bool
    decision_label: str


def infer_records(bundle: ModelBundle, records: list[dict[str, Any]]) -> list[InferenceResult]:
    """Run the checked pipeline once and apply its validated metadata threshold."""
    probabilities = bundle.predict_proba(records)
    threshold = float(bundle.metadata["threshold"])
    return [
        InferenceResult(
            probability=float(value), threshold=threshold,
            alert=bool(value >= threshold),
            decision_label=(
                "REVIEW_RECOMMENDED" if value >= threshold else "NO_REVIEW_ALERT"
            ),
        )
        for value in probabilities
    ]
