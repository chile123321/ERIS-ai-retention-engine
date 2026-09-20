"""Controlled dataset-loading interfaces."""

from pathlib import Path

import pandas as pd

from eris_ml.data.splitting import assert_development_path


def load_development_data(path: Path) -> pd.DataFrame:
    """Load only the established development split."""
    path = Path(path)
    assert_development_path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Development CSV does not exist: {path}")

    try:
        frame = pd.read_csv(path)
    except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeError) as exc:
        raise ValueError(f"Development CSV could not be read: {path}") from exc
    except OSError as exc:
        raise OSError(f"Development CSV could not be read: {path}") from exc

    if frame.empty:
        raise ValueError(f"Development CSV is empty: {path}")
    return frame
