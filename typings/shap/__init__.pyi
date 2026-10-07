"""Typed subset used by ERIS: binary XGBoost, raw tree-path-dependent explanations."""

from typing import Literal

import numpy as np
from numpy.typing import NDArray
from xgboost import XGBClassifier

class TreeExplainer:
    expected_value: float | NDArray[np.float64] | None
    def __init__(
        self,
        model: XGBClassifier,
        *,
        model_output: Literal["raw"],
        feature_perturbation: Literal["tree_path_dependent"],
    ) -> None: ...
    def shap_values(self, X: NDArray[np.float64]) -> NDArray[np.float64]: ...
