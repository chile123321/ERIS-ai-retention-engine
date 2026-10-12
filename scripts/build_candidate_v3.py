"""Select/refit one RF V3 candidate; existing run directories are never overwritten."""

import argparse
from pathlib import Path

from eris_ml.models.candidate_v3 import run

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="b1")
    args = parser.parse_args()
    run(Path(__file__).resolve().parents[1], args.run_id)
