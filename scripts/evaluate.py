"""Evaluate model candidates (scaffold entry point)."""

import argparse


def main() -> None:
    """Parse arguments without reading any evaluation split."""
    parser = argparse.ArgumentParser(description="Evaluate development results; not implemented.")
    parser.parse_args()
    raise NotImplementedError("Evaluation is outside the scaffold milestone.")


if __name__ == "__main__":
    main()
