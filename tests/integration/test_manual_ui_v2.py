"""No-fit V2 UI checks; optional installed-bundle checks explicitly skip if absent."""

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

pytest.importorskip("streamlit", reason="Install the manual-ui extra for V2 UI tests")
from streamlit import config
from streamlit.testing.v1 import AppTest

from eris_ml.manual_ui import v2_streamlit as ui
from eris_ml.models.bundle_v2 import FEATURE_ORDER, FORBIDDEN, input_schema, load_bundle
from eris_ml.models.persistence import sidecar_paths
from eris_ml.utils.hashing import sha256_file


@pytest.fixture
def record():
    values = dict.fromkeys(FEATURE_ORDER, "3")
    values["OverTime"] = "No"
    return values


def test_form_features_exactly_match_bundle_and_config(record):
    names = [name for group in ui.SECTIONS.values() for name in group]
    assert len(names) == len(set(names)) == 16
    assert set(names) == set(FEATURE_ORDER) == set(ui.LABELS)
    assert set(names).isdisjoint(FORBIDDEN)
    assert list(ui.parse_manual(record).columns) == input_schema()["feature_order"]
    table = ui.feature_table(ui.parse_manual(record))
    assert len(table) == 16
    assert list(table.columns) == ["Feature IBM", "Giá trị"]


@pytest.mark.parametrize(("name", "value"), [
    ("Age", ""), ("Age", "   "), ("Age", "not-an-age"), ("Age", "-1"),
    ("Age", "3.5"), ("Age", "nan"), ("Age", "inf"), ("Age", "1e999"),
    ("OverTime", ""), ("OverTime", "Maybe"), ("Education", "6"),
    ("PerformanceRating", "2"), ("MonthlyIncome", "1,000"),
])
def test_invalid_input_has_field_error_not_default(record, name, value):
    record[name] = value
    with pytest.raises(ValueError, match=name):
        ui.parse_manual(record)


def test_extra_or_missing_feature_rejected(record):
    record["Department"] = "Sales"
    with pytest.raises(ValueError, match="16"):
        ui.parse_manual(record)
    del record["Department"]
    del record["Age"]
    with pytest.raises(ValueError, match="16"):
        ui.parse_manual(record)


def test_sample_rule_reads_only_first_three_feature_rows(tmp_path, monkeypatch, record):
    development = tmp_path / ui.DEVELOPMENT
    development.parent.mkdir(parents=True)
    frame = pd.concat([ui.parse_manual(record)] * 4, ignore_index=True)
    frame["Age"] = [20, 30, 40, 50]
    # Synthetic extra columns are deliberately NOT parsed by the UI.
    frame["Attrition"] = [1, 0, 1, 0]
    frame["EmployeeNumber"] = [5, 6, 7, 8]
    frame.to_csv(development, index=False)
    bundle = SimpleNamespace(metadata={"provenance": {"input_sha256": {
        ui.DEVELOPMENT: sha256_file(development),
    }}})
    monkeypatch.setattr(ui, "ROOT", tmp_path)
    original = pd.read_csv
    calls = []

    def read(path, **kwargs):
        assert path == development
        assert kwargs == {"usecols": FEATURE_ORDER, "nrows": 3}
        calls.append(path)
        return original(path, **kwargs)

    monkeypatch.setattr(pd, "read_csv", read)
    samples = ui.load_samples(bundle)
    assert samples["Age"].tolist() == [20, 30, 40]
    assert list(samples.columns) == FEATURE_ORDER
    assert len(calls) == 1
    bundle.metadata["provenance"]["input_sha256"][ui.DEVELOPMENT] = "wrong"
    with pytest.raises(ValueError, match="fingerprint"):
        ui.load_samples(bundle)
    assert len(calls) == 1  # Refuse mismatched data before parsing anything.


@pytest.fixture
def local_config():
    options = {"server.address": "127.0.0.1", "browser.gatherUsageStats": False}
    previous = {key: config.get_option(key) for key in options}
    for key, value in options.items():
        config.set_option(key, value)
    yield
    for key, value in previous.items():
        config.set_option(key, value)


@pytest.mark.parametrize(("option", "value"), [
    ("server.address", "0.0.0.0"), ("browser.gatherUsageStats", True),
])
def test_ui_refuses_unsafe_launch_before_loading(local_config, monkeypatch, option, value):
    config.set_option(option, value)
    monkeypatch.setattr(ui, "load_bundle", lambda _: pytest.fail("Must not load"))
    app = AppTest.from_string("from eris_ml.manual_ui.v2_streamlit import render\nrender()")
    app.run()
    assert not app.exception
    assert "localhost" in app.error[0].value
    assert not app.button


def test_bundle_failure_does_not_fallback_or_show_details(local_config, monkeypatch):
    def fail(path):
        assert path == ui.BUNDLE
        raise ValueError("private diagnostic must not appear")

    monkeypatch.setattr(ui, "load_bundle", fail)
    app = AppTest.from_string("from eris_ml.manual_ui.v2_streamlit import render\nrender()")
    app.run()
    assert not app.exception
    assert "Không nạp được bundle V2" in app.error[0].value
    assert "private diagnostic" not in app.error[0].value
    assert not app.button


def test_saved_bundle_sample_manual_invalid_and_repeat(local_config, monkeypatch, capsys):
    if not ui.BUNDLE.is_file() or not (ui.ROOT / ui.DEVELOPMENT).is_file():
        pytest.skip("Local V2 bundle/development required; no replacement artifacts are created")
    protected = [ui.BUNDLE, *sidecar_paths(ui.BUNDLE), ui.ROOT / ui.DEVELOPMENT]
    before = {path: sha256_file(path) for path in protected}
    original_open = Path.open

    def safe_open(path, *args, **kwargs):
        assert "final_test" not in str(path).lower()
        if path in protected:
            mode = args[0] if args else kwargs.get("mode", "r")
            assert not any(flag in mode for flag in "wa+x")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", safe_open)
    monkeypatch.setattr(Pipeline, "fit", lambda *a, **k: pytest.fail("UI must never fit"))
    bundle = load_bundle(ui.BUNDLE)
    samples = ui.load_samples(bundle)
    app = AppTest.from_string("from eris_ml.manual_ui.v2_streamlit import render\nrender()")
    app.run(timeout=30)
    assert not app.exception and not app.error
    assert "logistic-general16-v2-r1" in app.success[0].value
    app.button[0].click().run()
    probability = float(bundle.predict_proba(samples.iloc[[0]])[0])
    assert 0 <= probability <= 1
    assert float(app.code[0].value.split(" = ")[1]) == pytest.approx(probability, abs=1e-15)
    assert app.metric[0].value == f"{probability * 100:.2f}%".replace(".", ",")
    assert len(app.dataframe[-1].value) == 16
    first = app.code[0].value
    app.button[0].click().run()
    assert app.code[0].value == first
    app.radio[0].set_value("Nhập thủ công").run()
    assert len(app.text_input) == 16
    assert not app.metric  # Clear stale sample result on mode change.
    app.text_input(key="field_Age").set_value("45")
    app.button[0].click().run()
    manual = samples.iloc[[0]].copy()
    manual["Age"] = 45
    direct = float(bundle.predict_proba(manual)[0])
    assert float(app.code[0].value.split(" = ")[1]) == pytest.approx(direct, abs=1e-15)
    app.text_input(key="field_Age").set_value("private-invalid-age")
    app.button[0].click().run()
    assert "Age" in app.error[0].value and not app.metric
    assert "private-invalid-age" not in capsys.readouterr().out
    assert not app.exception
    assert before == {path: sha256_file(path) for path in protected}
