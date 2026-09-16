"""Package smoke test."""


def test_package_import() -> None:
    """The src-layout package exposes its scaffold version."""
    import eris_ml

    assert eris_ml.__version__ == "0.1.0"
