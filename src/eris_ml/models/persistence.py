"""Checksummed local persistence for the frozen research model bundle.

Joblib/pickle can execute code when loaded. Only load trusted, locally produced files.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import joblib

from eris_ml.models.bundle import ModelBundle, validate_bundle_metadata
from eris_ml.utils.hashing import sha256_file


def sidecar_paths(path: Path) -> tuple[Path, Path]:
    """Return metadata JSON and checksum manifest paths for a joblib artifact."""
    if path.suffix != ".joblib":
        raise ValueError("Bundle path must end in .joblib.")
    return path.with_suffix(".metadata.json"), path.with_suffix(".sha256")


def _write_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = stream.name
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and os.path.exists(temporary):
            os.unlink(temporary)


def save_model_bundle(bundle: ModelBundle, path: Path, metadata: dict[str, Any]) -> None:
    """Persist a single fitted pipeline, its non-row metadata and both checksums."""
    validate_bundle_metadata(metadata)
    if bundle.metadata != metadata:
        raise ValueError("Bundle and JSON metadata differ.")
    metadata_path, checksum_path = sidecar_paths(path)
    if any(item.exists() for item in (path, metadata_path, checksum_path)):
        raise ValueError("Refusing to overwrite an existing model bundle or sidecar.")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = stream.name
            joblib.dump(bundle, stream, compress=3)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and os.path.exists(temporary):
            os.unlink(temporary)
    encoded = (json.dumps(metadata, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    _write_atomic(metadata_path, encoded.encode("utf-8"))
    checksums = (f"{sha256_file(path)}  {path.name}\n"
                 f"{sha256_file(metadata_path)}  {metadata_path.name}\n")
    _write_atomic(checksum_path, checksums.encode("ascii"))


def load_model_bundle(path: Path) -> ModelBundle:
    """Verify both sidecars before any trusted local joblib deserialization."""
    metadata_path, checksum_path = sidecar_paths(path)
    lines = checksum_path.read_text(encoding="ascii").splitlines()
    expected_names = [path.name, metadata_path.name]
    if len(lines) != 2:
        raise ValueError("Bundle checksum manifest must contain exactly two entries.")
    for line, file_path, name in zip(lines, (path, metadata_path), expected_names,
                                      strict=True):
        parts = line.split("  ")
        if len(parts) != 2 or parts[1] != name or parts[0] != sha256_file(file_path):
            raise ValueError(f"Bundle checksum verification failed for {name}.")
    metadata: Any = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise ValueError("Bundle JSON metadata must be an object.")
    validate_bundle_metadata(metadata)
    # Hashes guard against accidental alteration, not origin authentication.
    bundle: Any = joblib.load(path)
    if not isinstance(bundle, ModelBundle) or bundle.metadata != metadata:
        raise ValueError("Deserialized bundle differs from its checked metadata.")
    return bundle


def load_bundle(path: Path) -> ModelBundle:
    """Concise alias for explicitly loading a trusted, checksummed local bundle."""
    return load_model_bundle(path)
