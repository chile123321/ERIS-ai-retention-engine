"""The local SHAP stub is narrow and agrees with the supported runtime surface."""

import inspect
import subprocess
import sys
from pathlib import Path

import shap

ROOT = Path(__file__).resolve().parents[2]


def test_shap_boundary_parameters_exist_in_runtime() -> None:
    inspect.signature(shap.TreeExplainer).bind(
        object(), model_output="raw", feature_perturbation="tree_path_dependent"
    )
    inspect.signature(shap.TreeExplainer.shap_values).bind(object(), object())


def test_shap_stub_checks_arguments_instead_of_hiding_untyped_imports(tmp_path: Path) -> None:
    fixture = tmp_path / "shap_boundary.py"
    fixture.write_text(
        "import shap\n"
        "from xgboost import XGBClassifier\n"
        "model = XGBClassifier()\n"
        "shap.TreeExplainer(model, model_output='raw', "
        "feature_perturbation='tree_path_dependent')\n"
        "shap.TreeExplainer(model, model_output='wrong', "
        "feature_perturbation='tree_path_dependent')\n",
        encoding="utf-8",
    )
    result = subprocess.run(
        [sys.executable, "-m", "mypy", "--strict", "--python-version", "3.11",
         "--follow-imports=normal", str(fixture)],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    errors = [line for line in result.stdout.splitlines() if ": error:" in line]
    assert result.returncode == 1, result.stdout + result.stderr
    assert len(errors) == 1, result.stdout
    assert ':5: error:' in errors[0] and '[arg-type]' in errors[0], result.stdout
