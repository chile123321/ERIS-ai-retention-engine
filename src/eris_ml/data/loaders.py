"""Controlled dataset-loading interfaces."""

from pathlib import Path
from typing import Any


def load_development_data(path: Path) -> Any:
    """Load only the established development split."""
    raise NotImplementedError("Development data loading has not been implemented.")
