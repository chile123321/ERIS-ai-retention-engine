"""Export the B2 contract without lifespan, inference, settings or secrets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from eris_ml.api_b2.main import app

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "docs/model-api-v1.openapi.json"


def contract_bytes() -> bytes:
    # FastAPI caches OpenAPI; restore that cache to leave the imported app unchanged.
    previous = app.openapi_schema
    try:
        return (
            json.dumps(app.openapi(), ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        ).encode("utf-8")
    finally:
        app.openapi_schema = previous


def export(output: Path = DEFAULT_OUTPUT) -> Path:
    # Optional filenames are restricted to docs, never artifacts/data or arbitrary paths.
    if not output.is_absolute():
        output = ROOT / output
    if output.parent != ROOT / "docs" or output.suffix != ".json" or output.is_symlink():
        raise ValueError("Output must be a nonsymlink JSON file directly in repository docs.")
    if output.parent.is_symlink():
        raise ValueError("Symlink output directory rejected.")
    payload = contract_bytes()
    if output.exists():
        if output.read_bytes() != payload:
            raise FileExistsError("Existing export differs; review it explicitly before replacing.")
        return output
    with output.open("xb") as stream:
        stream.write(payload)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(export(args.output))


if __name__ == "__main__":
    main()
