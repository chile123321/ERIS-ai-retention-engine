"""Authorized frozen V3 f1 evaluation once; --resume only verifies saved outputs."""

import argparse
from pathlib import Path

from eris_ml.evaluation.final_v3 import run

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="f1", choices=["f1"])
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    run(Path(__file__).resolve().parents[1], args.run_id, args.resume)
