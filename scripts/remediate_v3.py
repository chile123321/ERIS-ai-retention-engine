"""Fixed development-only V3 remediation; never reads final holdout or f1 predictions."""

import argparse
from pathlib import Path

from eris_ml.models.remediation_v3 import run

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="devfix1")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    run(Path(__file__).resolve().parents[1], args.run_id, args.resume)
