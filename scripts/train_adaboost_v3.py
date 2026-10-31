"""Run AdaBoost baseline, nested tuning and exploratory sigmoid on saved development folds."""

import argparse
from pathlib import Path

from eris_ml.models.adaboost_v3 import run

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="a1")
    args = parser.parse_args()
    run(Path(__file__).resolve().parents[1], args.run_id)
