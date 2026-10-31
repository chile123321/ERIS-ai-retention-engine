"""Thin s1 adapter: reuse frozen schema codec, loader and calibrated SHAP implementation."""

from pathlib import Path
from typing import Any

import pandas as pd

from eris_ml.api_b2.schemas import (
    ADABOOST_ID,
    ADABOOST_SHA256,
    FEATURE_NAMES,
    AdaBoostMetadata,
    EmployeeFeatures,
    ExplainResponse,
    PredictionResult,
    ShapContribution,
)
from eris_ml.explainability.adaboost_shap import FeatureCodec, load_bundle
from eris_ml.models.adaboost_v3 import positive_probability
from eris_ml.utils.hashing import sha256_file


class AdaBoostAdapter:
    def __init__(self, directory: Path):
        if directory.is_symlink() or sha256_file(directory / "pipeline.joblib") != ADABOOST_SHA256:
            raise ValueError("Unexpected s1 checksum")
        self.bundle = load_bundle(directory)
        if (
            self.bundle.metadata["model_id"] != ADABOOST_ID
            or tuple(self.bundle.metadata["feature_order"]) != FEATURE_NAMES
        ):
            raise ValueError("Unexpected s1 identity/schema")
        self.codec = FeatureCodec(self.bundle.metadata["feature_config"])
        # This mutable explainer is used ONLY by the single explanation worker.
        self.explainer = self.bundle.explainer()

    def frame(self, items: list[EmployeeFeatures]) -> pd.DataFrame:
        frame = pd.DataFrame([item.model_dump() for item in items], columns=FEATURE_NAMES)
        # Exact original-feature validation; the fitted pipeline alone imputes missing values.
        return self.codec.decode(self.codec.encode(frame))

    def predict(
        self, items: list[EmployeeFeatures], references: list[str]
    ) -> list[PredictionResult]:
        probabilities = positive_probability(self.bundle.model, self.frame(items))
        return [
            PredictionResult.model_validate({
                "reference_id": reference,
                "estimated_probability": float(probability),
                "imputed_features": [
                    name for name, value in item.model_dump().items() if value is None
                ],
                "metadata": AdaBoostMetadata(),
            })
            for item, reference, probability in zip(items, references, probabilities, strict=True)
        ]

    def explain(self, features: EmployeeFeatures, **response_fields: Any) -> ExplainResponse:
        raw = self.explainer.explain(self.frame([features]), cycles=64, seed=42)
        contributions = [
            ShapContribution.model_validate({
                "feature": name,
                "feature_value": raw["feature_values"][name],
                "shap_value": value,
                "direction": "increase" if value > 0 else "decrease" if value < 0 else "neutral",
            })
            for name, value in zip(FEATURE_NAMES, raw["contributions"], strict=True)
        ]
        return ExplainResponse(
            **response_fields,
            estimated_probability=raw["estimated_probability"],
            base_value=raw["base_value"],
            contributions=contributions,
            top_factors=sorted(contributions, key=lambda c: -abs(c.shap_value))[:5],
            additivity_error=raw["additivity_error"],
            metadata=AdaBoostMetadata(),
        )
