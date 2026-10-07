"""Build the explicitly authorized V2 IBM research/demo Logistic candidate only."""

import json
from pathlib import Path

from eris_ml.models.bundle_v2_build import BUNDLE, build

if __name__ == "__main__":
    result = build(Path(__file__).resolve().parents[1])
    print(json.dumps({"artifact": BUNDLE, "status": result["status"],
                      "checks": result["checks"], "warnings": result["warnings"]}, indent=2))
