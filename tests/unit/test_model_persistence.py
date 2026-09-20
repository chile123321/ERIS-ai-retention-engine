"""Synthetic-only persistence and checksum tests."""

from pathlib import Path

import pytest

from eris_ml.models.persistence import load_bundle, save_model_bundle, sidecar_paths


def test_model_bundle_round_trip(tmp_path: Path, synthetic_bundle: tuple[object, object,
                                                                         object]) -> None:
    bundle, frame, _ = synthetic_bundle
    path = tmp_path / "model.joblib"
    save_model_bundle(bundle, path, bundle.metadata)
    metadata_path, checksum_path = sidecar_paths(path)
    assert path.is_file() and metadata_path.is_file() and checksum_path.is_file()
    restored = load_bundle(path)
    assert restored.metadata == bundle.metadata
    assert restored.predict(frame.iloc[:3]).tolist() == bundle.predict(frame.iloc[:3]).tolist()
    with pytest.raises(ValueError, match="overwrite"):
        save_model_bundle(bundle, path, bundle.metadata)


def test_tampered_bundle_is_rejected_before_deserialization(
    tmp_path: Path, synthetic_bundle: tuple[object, object, object],
) -> None:
    bundle, _, _ = synthetic_bundle
    path = tmp_path / "model.joblib"
    save_model_bundle(bundle, path, bundle.metadata)
    path.write_bytes(path.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="checksum verification failed"):
        load_bundle(path)


def test_tampered_metadata_is_rejected(
    tmp_path: Path, synthetic_bundle: tuple[object, object, object],
) -> None:
    bundle, _, _ = synthetic_bundle
    path = tmp_path / "model.joblib"
    save_model_bundle(bundle, path, bundle.metadata)
    metadata_path, _ = sidecar_paths(path)
    metadata_path.write_bytes(metadata_path.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="checksum verification failed"):
        load_bundle(path)
