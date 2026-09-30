"""Local manual-test validation against the real frozen bundle, never data CSVs."""

from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from xgboost import XGBClassifier

from eris_ml.models.inference import infer_records
from scripts import manual_model_test as manual


@pytest.fixture(scope="module")
def checked() -> tuple[Any, dict[str, Any], dict[str, dict[str, Any]]]:
    bundle, contract = manual.load_frozen_bundle()
    return bundle, contract, manual.load_examples()


def test_frozen_bundle_and_synthetic_examples(checked: tuple[Any, dict[str, Any],
                                                         dict[str, dict[str, Any]]]) -> None:
    bundle, contract, examples = checked
    assert manual.EXPECTED_BUNDLE_SHA256 == (
        "bb9c00307a36763f91de58d598caf952cca3f451e7928ca39d1a22b1763dac21"
    )
    assert bundle.metadata["bundle_version"] == "bundle-v1"
    assert bundle.metadata["candidate_version"] == "eris-xgboost-v1"
    assert bundle.metadata["probability_type"] == "raw"
    assert len(bundle.metadata["feature_order"]) == 25
    assert len(manual.DESCRIPTIONS) == 25
    for signal in ("low", "high"):
        example = examples[f"synthetic_{signal}_signal"]
        record, features = manual.validate_payload(example, bundle, contract)
        assert record.record_id == example["record_id"]
        assert len(features) == 25
        expected = infer_records(bundle, [features])[0]
        result = manual.predict_local(example, bundle, contract, include_explanation=False)
        assert result["probability"] == expected.probability
        assert result["threshold"] == expected.threshold == 0.345651
        assert result["alert"] == expected.alert == (result["probability"] >= 0.345651)
        assert result["decision_label"] == expected.decision_label
        assert result["production_approved"] is False
        assert 0 <= result["probability"] <= 1


@pytest.mark.parametrize("change,field", [
    ({"Age": 999}, "Age"),
    ({"Department": "Invalid Department"}, "Department"),
    ({"Education": "three"}, "Education"),
    ({"Attrition": "Yes"}, "Attrition"),
])
def test_invalid_inputs_are_rejected(
    checked: tuple[Any, dict[str, Any], dict[str, dict[str, Any]]],
    change: dict[str, Any], field: str,
) -> None:
    bundle, contract, examples = checked
    with pytest.raises(manual.ManualTestError, match=field):
        manual.predict_local({**examples["synthetic_low_signal"], **change},
                             bundle, contract, include_explanation=False)
    missing = {key: value for key, value in examples["synthetic_low_signal"].items()
               if key != "Age"}
    with pytest.raises(manual.ManualTestError, match="Age"):
        manual.validate_payload(missing, bundle, contract)


def test_shap_skip_and_failure_do_not_hide_prediction(
    checked: tuple[Any, dict[str, Any], dict[str, dict[str, Any]]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle, contract, examples = checked

    def broken(*args: object, **kwargs: object) -> object:
        raise RuntimeError("Synthetic SHAP failure")

    monkeypatch.setattr(bundle, "explain", broken)
    example = examples["synthetic_low_signal"]
    skipped = manual.predict_local(example, bundle, contract, include_explanation=False)
    assert skipped["explanation_error"] is None and skipped["top_factors"] == []
    failed = manual.predict_local(example, bundle, contract, include_explanation=True)
    assert failed["explanation_error"] is not None
    assert failed["probability"] == skipped["probability"]
    assert failed["alert"] == skipped["alert"]


def test_interactive_prompts_all_features_and_requires_confirmation(
    checked: tuple[Any, dict[str, Any], dict[str, dict[str, Any]]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle, contract, examples = checked
    example = examples["synthetic_low_signal"]
    first = bundle.metadata["feature_order"][0]
    assert first == "BusinessTravel"
    responses = iter(["1", *([""] * 24), "y"])
    monkeypatch.setattr("builtins.input", lambda _prompt: next(responses))
    selected = manual.prompt_record(bundle, contract, example)
    assert selected is not None
    assert selected[first] != example[first]
    assert len(selected) == 26  # 25 features plus a non-model record_id
    manual.validate_payload(selected, bundle, contract)


def test_json_guard_and_bundle_hash_guard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{not-json", encoding="utf-8")
    with pytest.raises(manual.ManualTestError, match="syntax"):
        manual.load_input_file(bad_json)
    with pytest.raises(manual.ManualTestError, match="regular JSON"):
        manual.load_input_file(Path("data/processed/final_test_raw_v1.csv"))
    fake_bundle = tmp_path / "wrong.joblib"
    fake_bundle.write_bytes(b"not a frozen model")
    monkeypatch.setattr(manual, "BUNDLE_PATH", fake_bundle)
    with pytest.raises(manual.ManualTestError, match="SHA-256"):
        manual.load_frozen_bundle()


def test_prediction_never_trains_or_calls_network(
    checked: tuple[Any, dict[str, Any], dict[str, dict[str, Any]]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle, contract, examples = checked

    def forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("Training or network access attempted")

    monkeypatch.setattr(XGBClassifier, "fit", forbidden)
    monkeypatch.setattr("socket.create_connection", forbidden)
    monkeypatch.setattr("socket.socket.connect", forbidden)
    assert manual.predict_local(examples["synthetic_high_signal"], bundle, contract,
                                include_explanation=False)["threshold"] == 0.345651


def test_local_loader_does_not_open_processed_or_locked_data(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_open = Path.open

    def guarded_open(path: Path, *args: Any, **kwargs: Any) -> Any:
        location = str(path).replace("\\", "/").casefold()
        assert "/data/processed/" not in location
        assert "/data/raw/" not in location
        assert "final_test" not in location
        assert "quarantine" not in location
        return original_open(path, *args, **kwargs)

    def forbidden_csv(*args: object, **kwargs: object) -> object:
        raise AssertionError("Local manual test attempted CSV access")

    monkeypatch.setattr(Path, "open", guarded_open)
    monkeypatch.setattr(pd, "read_csv", forbidden_csv)
    bundle, contract = manual.load_frozen_bundle()
    result = manual.predict_local(manual.load_examples()["synthetic_low_signal"],
                                  bundle, contract, include_explanation=False)
    assert result["threshold"] == 0.345651
