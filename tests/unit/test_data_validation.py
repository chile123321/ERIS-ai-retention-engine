"""Tests for locked-test access safeguards."""

from pathlib import Path

import pytest

from eris_ml.data.splitting import assert_development_path


def test_locked_final_test_path_is_rejected() -> None:
    """Development workflows cannot point to a locked-test path."""
    with pytest.raises(ValueError, match="prohibited"):
        assert_development_path(Path("data/splits/locked_final_test.csv"))


def test_development_path_is_allowed() -> None:
    """The supplied development split is an allowed development input."""
    assert_development_path(Path("data/splits/development.csv"))
