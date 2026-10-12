"""Installed SHAP 0.50 subset: XGB raw and RF interventional probability explanations."""

from typing import Literal, overload

import numpy as np
from numpy.typing import NDArray
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

class Explanation:
    values: NDArray[np.float64]
    base_values: NDArray[np.float64]

class TreeExplainer:
    expected_value: float | NDArray[np.float64] | None
    @overload
    def __init__(
        self,
        model: XGBClassifier,
        *,
        model_output: Literal["raw"],
        feature_perturbation: Literal["tree_path_dependent"],
    ) -> None: ...
    @overload
    def __init__(
        self,
        model: RandomForestClassifier,
        *,
        data: NDArray[np.float64],
        model_output: Literal["probability"],
        feature_perturbation: Literal["interventional"],
    ) -> None: ...
    def __call__(self, X: NDArray[np.float64], *, check_additivity: bool = True) -> Explanation: ...
    def shap_values(self, X: NDArray[np.float64]) -> NDArray[np.float64]: ...
