"""Build a versioned model bundle (scaffold entry point)."""

import argparse


def main() -> None:
    """Parse arguments without creating an artifact."""
    parser = argparse.ArgumentParser(description="Build an approved bundle; not implemented.")
    parser.parse_args()
    raise NotImplementedError("Bundle creation is outside the scaffold milestone.")


if __name__ == "__main__":
    main()
