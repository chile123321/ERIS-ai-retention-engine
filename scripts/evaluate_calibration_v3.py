"""Evaluate V3 sigmoid, fold-batch top-k and optional missingness without new tuning."""

import argparse
from pathlib import Path

from eris_ml.models.calibration_v3 import run

ROOT = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="c1")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    run(ROOT, args.run_id, args.resume)
