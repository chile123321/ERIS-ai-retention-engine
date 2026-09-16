"""Guards for the supplied immutable data split."""

from pathlib import Path


def assert_development_path(path: Path) -> None:
    """Reject paths that appear to reference the locked final test."""
    normalized = path.as_posix().lower().replace("-", "_")
    forbidden = ("final_test", "locked_test", "test_locked")
    if any(token in normalized for token in forbidden):
        raise ValueError("Locked final test access is prohibited during development.")
