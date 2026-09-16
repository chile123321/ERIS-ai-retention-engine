"""Data contract tests awaiting an approved column schema."""

import pytest


@pytest.mark.skip(reason="Authoritative data contract columns have not been approved.")
def test_approved_contract_columns() -> None:
    """Validate required columns once the authoritative contract is populated."""
