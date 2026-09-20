"""Build the frozen Step 12A research bundle from development only."""

import argparse
from pathlib import Path

from eris_ml.models.bundle_build import build_bundle

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    """Run one fixed-path build; no final-test paths or API setup are accepted."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    result = build_bundle(ROOT)
    print("Bundle build PASS: eris-xgboost-v1, development-only, 25 features, "
          f"SHA-256 {result['hashes']['artifacts/models/eris_xgboost_v1.joblib']}.")


if __name__ == "__main__":
    main()
