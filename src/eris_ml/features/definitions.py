"""Feature-set configuration interfaces."""

from pathlib import Path


def load_feature_names(path: Path) -> tuple[str, ...]:
    """Load an approved, versioned feature list."""
    raise NotImplementedError("Feature definitions await authoritative approval.")
