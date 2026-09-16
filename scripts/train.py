"""Train model candidates (scaffold entry point)."""

import argparse


def main() -> None:
    """Parse arguments without starting training."""
    parser = argparse.ArgumentParser(description="Train on development folds; not implemented.")
    parser.parse_args()
    raise NotImplementedError("Training is outside the scaffold milestone.")


if __name__ == "__main__":
    main()
