"""Synthetic-only input and frozen-decision tests for the internal bundle API."""

from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from eris_ml.models.bundle import ModelBundle, validate_bundle_metadata


def test_feature_order_probability_and_threshold(synthetic_bundle: tuple[object,
                                                                          pd.DataFrame,
                                                                          object]) -> None:
    bundle, frame, _ = synthetic_bundle
    shuffled = frame.loc[:, list(reversed(frame.columns))].iloc[:4]
    normalized = bundle.validate_input(shuffled)
    assert list(normalized.columns) == bundle.metadata["feature_order"]
    probabilities = bundle.predict_proba(shuffled)
    assert np.isfinite(probabilities).all() and ((probabilities >= 0) &
                                                (probabilities <= 1)).all()
    assert np.array_equal(bundle.predict(shuffled), (probabilities >= 0.345651).astype(int))


def test_missing_extra_numeric_ordinal_and_unknown_nominal(
    synthetic_bundle: tuple[object, pd.DataFrame, object],
) -> None:
    bundle, frame, definition = synthetic_bundle
    record = frame.iloc[0].to_dict()
    missing = {key: value for key, value in record.items() if key != definition.numeric[0]}
    with pytest.raises(ValueError, match=definition.numeric[0]):
        bundle.validate_input(missing)
    with pytest.raises(ValueError, match="Attrition"):
        bundle.validate_input({**record, "Attrition": 0})
    with pytest.raises(ValueError, match="EmployeeNumber"):
        bundle.validate_input({**record, "EmployeeNumber": 123})
    with pytest.raises(ValueError, match=definition.numeric[0]):
        bundle.validate_input({**record, definition.numeric[0]: "not-numeric"})
    with pytest.raises(ValueError, match=definition.ordinal[0]):
        bundle.validate_input({**record, definition.ordinal[0]: 999})
    unknown = {**record, definition.nominal[0]: "Never_seen_category"}
    assert bundle.predict_proba(unknown).shape == (1,)
    with_missing = {**record, definition.numeric[0]: None}
    assert bundle.predict_proba(with_missing).shape == (1,)


def test_metadata_and_fitted_model_drift_rejected(
    synthetic_bundle: tuple[object, pd.DataFrame, object],
) -> None:
    bundle, _, _ = synthetic_bundle
    metadata = deepcopy(bundle.metadata)
    metadata["threshold"] = 0.5
    with pytest.raises(ValueError, match="metadata differs"):
        validate_bundle_metadata(metadata)
    metadata = deepcopy(bundle.metadata)
    metadata["model_parameters"]["max_depth"] = 2
    with pytest.raises(ValueError, match="metadata differs"):
        ModelBundle(bundle.pipeline, metadata)
