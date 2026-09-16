"""Structured logging setup."""

import logging


def configure_logging(level: str = "INFO") -> None:
    """Configure minimal process logging."""
    logging.basicConfig(level=level.upper(), format="%(asctime)s %(levelname)s %(name)s %(message)s")
