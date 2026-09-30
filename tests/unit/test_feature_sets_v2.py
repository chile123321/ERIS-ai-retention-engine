"""Feature-lock tests for the four v2 comparison schemas."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from sklearn.dummy import DummyClassifier

from eris_ml.features.definitions import load_feature_definition
from eris_ml.models.training import build_model_pipeline

ROOT = Path(__file__).resolve().parents[2]
FEATURE_DIR = ROOT / "configs/features"
VARIANTS = {
    "v2_full_18": 18,
    "v2_no_department_17": 17,
    "v2_no_jobrole_17": 17,
    "v2_general_16": 16,
}
DICTIONARY_FEATURES = {
    "Age", "Department", "DistanceFromHome", "Education",
    "EnvironmentSatisfaction", "JobInvolvement", "JobLevel", "JobRole",
    "JobSatisfaction", "MonthlyIncome", "OverTime", "PerformanceRating",
    "TotalWorkingYears", "TrainingTimesLastYear", "YearsAtCompany",
    "YearsInCurrentRole", "YearsSinceLastPromotion", "YearsWithCurrManager",
}


def _definition(name: str):
    return load_feature_definition(FEATURE_DIR / f"feature_set_{name}.yaml")


@pytest.mark.parametrize(("name", "count"), VARIANTS.items())
def test_v2_variant_has_exact_name_count_and_dictionary_membership(
    name: str, count: int,
) -> None:
    definition = _definition(name)
    assert definition.version == name
    assert len(definition.all_features) == count
    assert set(definition.all_features) <= DICTIONARY_FEATURES
    assert {"Attrition", "EmployeeNumber", "source_row"}.isdisjoint(
        definition.all_features
    )


def test_v2_ablation_matrix_differs_only_by_department_and_jobrole() -> None:
    full = set(_definition("v2_full_18").all_features)
    no_department = set(_definition("v2_no_department_17").all_features)
    no_jobrole = set(_definition("v2_no_jobrole_17").all_features)
    general = set(_definition("v2_general_16").all_features)
    assert full == DICTIONARY_FEATURES
    assert no_department == full - {"Department"}
    assert no_jobrole == full - {"JobRole"}
    assert general == full - {"Department", "JobRole"}


def test_preferred_pipeline_cannot_receive_department_or_jobrole() -> None:
    definition = _definition("v2_general_16")
    assert {"Department", "JobRole"}.issubset(definition.excluded)
    assert {"Department", "JobRole"}.isdisjoint(definition.all_features)
    pipeline = build_model_pipeline(definition, DummyClassifier(strategy="prior"))
    configured = {
        column
        for _, _, columns in pipeline.named_steps["preprocessor"].transformers
        for column in columns
    }
    assert configured == set(definition.all_features)
    assert {"Department", "JobRole"}.isdisjoint(configured)


def test_v2_registry_names_preferred_schema_and_shared_folds() -> None:
    registry = yaml.safe_load(
        (FEATURE_DIR / "feature_set_v2_practical.yaml").read_text(encoding="utf-8")
    )
    assert (ROOT / registry["source_dictionary"]).resolve().is_file()
    assert set(registry["comparison_feature_sets"]) == set(VARIANTS)
    assert registry["preferred_feature_set"]["name"] == "v2_general_16"
    protocol = registry["comparison_protocol"]
    assert protocol["selection_data"] == "development_only"
    assert protocol["final_test_used"] is False
    assert protocol["shared_outer_folds"] == {
        "type": "StratifiedKFold", "n_splits": 5,
        "shuffle": True, "random_state": 42,
    }
