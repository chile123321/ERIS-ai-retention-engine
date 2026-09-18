"""Load and validate versioned feature-set definitions."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml


class FeatureDefinitionError(ValueError):
    """Raised when a feature configuration or input schema is invalid."""


@dataclass(frozen=True)
class FeatureDefinition:
    """Validated feature groups used to construct preprocessing pipelines."""

    version: str
    nominal: tuple[str, ...]
    ordinal: tuple[str, ...]
    numeric: tuple[str, ...]
    target: tuple[str, ...]
    identifiers: tuple[str, ...]
    provenance: tuple[str, ...]
    excluded: tuple[str, ...]

    @property
    def all_features(self) -> tuple[str, ...]:
        """Return model features in stable group order."""
        return self.nominal + self.ordinal + self.numeric

    @property
    def non_features(self) -> tuple[str, ...]:
        """Return columns explicitly prohibited from model inputs."""
        return self.target + self.identifiers + self.provenance + self.excluded


def _string_tuple(value: Any, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(item, str) and item for item in value):
        raise FeatureDefinitionError(f"'{field_name}' must be a list of non-empty strings.")
    return tuple(value)


def _mapping(value: Any, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise FeatureDefinitionError(f"'{field_name}' must be a mapping.")
    return value


def load_feature_definition(path: Path) -> FeatureDefinition:
    """Load and validate the authoritative YAML feature definition."""
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise FeatureDefinitionError(f"Could not read feature configuration: {path}") from exc
    except yaml.YAMLError as exc:
        raise FeatureDefinitionError(f"Invalid YAML in feature configuration: {path}") from exc

    root = _mapping(raw, "root")
    groups = _mapping(root.get("feature_groups"), "feature_groups")
    allowed_groups = {"nominal", "ordinal", "numeric"}
    unexpected_groups = set(groups) - allowed_groups
    missing_groups = allowed_groups - set(groups)
    if missing_groups or unexpected_groups:
        raise FeatureDefinitionError(
            f"Feature groups mismatch; missing={sorted(missing_groups)}, "
            f"unexpected={sorted(unexpected_groups)}."
        )

    non_features = _mapping(root.get("non_features"), "non_features")
    allowed_non_feature_groups = {"target", "identifiers", "provenance", "excluded"}
    unexpected_non_feature_groups = set(non_features) - allowed_non_feature_groups
    missing_non_feature_groups = allowed_non_feature_groups - set(non_features)
    if missing_non_feature_groups or unexpected_non_feature_groups:
        raise FeatureDefinitionError(
            "Non-feature groups mismatch; "
            f"missing={sorted(missing_non_feature_groups)}, "
            f"unexpected={sorted(unexpected_non_feature_groups)}."
        )

    definition = FeatureDefinition(
        version=str(root.get("version", "")),
        nominal=_string_tuple(groups["nominal"], "feature_groups.nominal"),
        ordinal=_string_tuple(groups["ordinal"], "feature_groups.ordinal"),
        numeric=_string_tuple(groups["numeric"], "feature_groups.numeric"),
        target=_string_tuple(non_features["target"], "non_features.target"),
        identifiers=_string_tuple(non_features["identifiers"], "non_features.identifiers"),
        provenance=_string_tuple(non_features["provenance"], "non_features.provenance"),
        excluded=_string_tuple(non_features["excluded"], "non_features.excluded"),
    )
    if not definition.version:
        raise FeatureDefinitionError("Feature definition must include a non-empty version.")

    configured_count = root.get("expected_feature_count")
    if configured_count != 25:
        raise FeatureDefinitionError(
            f"expected_feature_count must be 25, received {configured_count!r}."
        )

    features = definition.all_features
    duplicates = sorted({name for name in features if features.count(name) > 1})
    if duplicates:
        raise FeatureDefinitionError(f"Features overlap across groups: {duplicates}.")
    if len(features) != configured_count:
        raise FeatureDefinitionError(
            f"Feature definition contains {len(features)} features; expected {configured_count}."
        )

    prohibited = set(definition.non_features)
    leaked = sorted(set(features) & prohibited)
    if leaked:
        raise FeatureDefinitionError(f"Non-feature columns included as model features: {leaked}.")

    required_roles = {
        "target": "Attrition",
        "identifiers": "EmployeeNumber",
        "provenance": "source_row",
    }
    for role, required_column in required_roles.items():
        if required_column not in getattr(definition, role):
            raise FeatureDefinitionError(
                f"'{required_column}' must be declared under non_features.{role}."
            )
    return definition


def validate_feature_columns(
    columns: Iterable[str], definition: FeatureDefinition, *, allow_extra: bool = False
) -> None:
    """Validate that input columns match the configured model features."""
    supplied = tuple(columns)
    duplicates = sorted({name for name in supplied if supplied.count(name) > 1})
    if duplicates:
        raise FeatureDefinitionError(f"Duplicate input columns: {duplicates}.")

    expected = set(definition.all_features)
    actual = set(supplied)
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected) if not allow_extra else []
    if missing or unexpected:
        raise FeatureDefinitionError(
            f"Input feature mismatch; missing={missing}, unexpected={unexpected}."
        )


def load_feature_names(path: Path) -> tuple[str, ...]:
    """Return all feature names from a validated configuration."""
    return load_feature_definition(path).all_features
