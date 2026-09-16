"""Versioned data-contract interfaces."""

from pathlib import Path
from typing import Any


def load_data_contract(path: Path) -> dict[str, Any]:
    """Load an approved YAML data contract."""
    raise NotImplementedError("Data-contract loading is planned for the validation milestone.")
