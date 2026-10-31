"""Fit and verify the fixed, development-only RF unweighted b2 candidate."""

import argparse
from pathlib import Path

from eris_ml.models.candidate_v3_b2 import run

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="b2")
    args = parser.parse_args()
    run(Path(__file__).resolve().parents[1], args.run_id)
