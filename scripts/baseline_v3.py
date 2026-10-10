"""Run the locked V3 baseline once; use a new run ID to preserve prior V3 outputs."""

import argparse
from pathlib import Path

from eris_ml.models.baseline_v3 import run

ROOT = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="r1")
    args = parser.parse_args()
    run(ROOT, ROOT / "configs/features/feature_set_v3_final_15.yaml", args.run_id)
