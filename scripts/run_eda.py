"""Run development-only EDA (scaffold entry point)."""

import argparse


def main() -> None:
    """Parse arguments and stop before any data access."""
    parser = argparse.ArgumentParser(description="Run development-only EDA; not implemented.")
    parser.parse_args()
    raise NotImplementedError("EDA is outside the scaffold milestone.")


if __name__ == "__main__":
    main()
