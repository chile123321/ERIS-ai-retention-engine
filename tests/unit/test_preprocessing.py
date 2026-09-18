"""Unit tests for Feature Set v1 definitions and preprocessing."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

from eris_ml.features.definitions import (
    FeatureDefinitionError,
    load_feature_definition,
    validate_feature_columns,
)
from eris_ml.features.preprocessing import build_preprocessor

CONFIG_PATH = (
    Path(__file__).resolve().parents[2] / "configs" / "features" / "feature_set_v1_full.yaml"
)


@pytest.fixture(scope="module")
def definition():
    """Load the authoritative Feature Set v1 definition."""
    return load_feature_definition(CONFIG_PATH)


@pytest.fixture()
def sample_frame(definition) -> pd.DataFrame:
    """Create non-sensitive input covering every configured feature."""
    nominal_values = {
        "BusinessTravel": ["Travel_Rarely", "Travel_Frequently", np.nan, "Non-Travel"],
        "Department": ["Sales", "Research & Development", "Sales", "Human Resources"],
        "EducationField": ["Marketing", "Life Sciences", "Medical", "Technical Degree"],
        "JobRole": ["Sales Executive", "Research Scientist", "Manager", "Human Resources"],
        "OverTime": ["Yes", "No", np.nan, "No"],
    }
    frame = pd.DataFrame(nominal_values)
    for index, feature in enumerate(definition.ordinal):
        frame[feature] = [1.0 + index % 2, 2.0, np.nan, 4.0]
    for index, feature in enumerate(definition.numeric):
        frame[feature] = [20.0 + index, np.nan, 35.0 + index, 50.0 + index]
    frame["Attrition"] = [0, 1, 0, 0]
    frame["EmployeeNumber"] = [101, 102, 103, 104]
    frame["source_row"] = [1, 2, 3, 4]
    frame["unexpected_metadata"] = ["a", "b", "c", "d"]
    return frame


def test_feature_group_sizes_and_total(definition) -> None:
    assert len(definition.nominal) == 5
    assert len(definition.ordinal) == 9
    assert len(definition.numeric) == 11
    assert len(definition.all_features) == 25


def test_feature_groups_do_not_overlap(definition) -> None:
    groups = [set(definition.nominal), set(definition.ordinal), set(definition.numeric)]
    assert groups[0].isdisjoint(groups[1])
    assert groups[0].isdisjoint(groups[2])
    assert groups[1].isdisjoint(groups[2])


def test_non_features_are_excluded(definition) -> None:
    prohibited = {"Attrition", "EmployeeNumber", "source_row"}
    assert prohibited.isdisjoint(definition.all_features)
    assert prohibited.issubset(definition.non_features)


def test_column_validation_detects_missing_and_unexpected(definition) -> None:
    invalid_columns = [name for name in definition.all_features if name != "Age"] + [
        "UnknownFeature"
    ]
    with pytest.raises(FeatureDefinitionError, match="missing=.*Age.*unexpected=.*UnknownFeature"):
        validate_feature_columns(invalid_columns, definition)


def test_transformer_has_three_expected_branches(definition) -> None:
    transformer = build_preprocessor(definition)
    assert isinstance(transformer, ColumnTransformer)
    assert [name for name, _, _ in transformer.transformers] == ["nominal", "ordinal", "numeric"]
    assert transformer.remainder == "drop"
    with pytest.raises(NotFittedError):
        check_is_fitted(transformer)


def test_unknown_and_missing_values_produce_finite_numeric_output(
    definition, sample_frame
) -> None:
    train = sample_frame.iloc[:3].copy(deep=True)
    validation = sample_frame.iloc[[3]].copy(deep=True)
    validation.loc[:, "BusinessTravel"] = "Space_Travel"
    validation.loc[:, "Education"] = np.nan
    validation.loc[:, "Age"] = np.nan

    transformer = build_preprocessor(definition)
    train_output = transformer.fit_transform(train)
    validation_output = transformer.transform(validation)
    train_array = train_output.toarray() if hasattr(train_output, "toarray") else train_output
    validation_array = (
        validation_output.toarray() if hasattr(validation_output, "toarray") else validation_output
    )

    assert np.issubdtype(train_array.dtype, np.number)
    assert np.isfinite(train_array).all()
    assert np.isfinite(validation_array).all()
    assert train_array.shape[1] == validation_array.shape[1]


def test_output_names_are_available_and_extra_columns_are_dropped(
    definition, sample_frame
) -> None:
    transformer = build_preprocessor(definition).fit(sample_frame)
    output_names = transformer.get_feature_names_out().tolist()
    assert output_names
    assert not any(
        name in output_names
        for name in ["Attrition", "EmployeeNumber", "source_row", "unexpected_metadata"]
    )
    assert not any(name.startswith(("nominal__", "ordinal__", "numeric__")) for name in output_names)


def test_fit_transform_does_not_mutate_input(definition, sample_frame) -> None:
    original = sample_frame.copy(deep=True)
    build_preprocessor(definition).fit_transform(sample_frame)
    pd.testing.assert_frame_equal(sample_frame, original)


def test_preprocessor_can_be_nested_in_classifier_pipeline(definition) -> None:
    pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(definition)),
            ("classifier", LogisticRegression()),
        ]
    )
    assert isinstance(pipeline.named_steps["preprocessor"], ColumnTransformer)
