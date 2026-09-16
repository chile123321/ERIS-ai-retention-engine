"""Evaluation report interfaces."""

from pathlib import Path
from typing import Any


def write_evaluation_report(results: dict[str, Any], destination: Path) -> None:
    """Write a traceable evaluation report for a model candidate."""
    raise NotImplementedError("Evaluation reporting has not been implemented.")
