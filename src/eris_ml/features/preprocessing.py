"""Leakage-safe preprocessing for the versioned feature set."""

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from eris_ml.features.definitions import FeatureDefinition


def build_preprocessor(feature_definition: FeatureDefinition) -> ColumnTransformer:
    """Build an unfitted transformer suitable for a model pipeline and CV folds."""
    nominal_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    ordinal_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("nominal", nominal_pipeline, list(feature_definition.nominal)),
            ("ordinal", ordinal_pipeline, list(feature_definition.ordinal)),
            ("numeric", numeric_pipeline, list(feature_definition.numeric)),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_preprocessing_pipeline(feature_definition: FeatureDefinition) -> ColumnTransformer:
    """Backward-compatible alias for :func:`build_preprocessor`."""
    return build_preprocessor(feature_definition)
