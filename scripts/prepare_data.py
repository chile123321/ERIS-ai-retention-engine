"""Validate and prepare development data (scaffold entry point)."""

import argparse


def main() -> None:
    """Parse arguments and stop before any data access."""
    parser = argparse.ArgumentParser(description="Prepare development data; not yet implemented.")
    parser.parse_args()
    raise NotImplementedError("Data preparation is outside the scaffold milestone.")


if __name__ == "__main__":
    main()
