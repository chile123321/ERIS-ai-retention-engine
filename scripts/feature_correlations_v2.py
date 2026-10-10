"""Read-only development correlations; one self-contained aggregate HTML output."""

from __future__ import annotations

import argparse
import html
import json
import platform
import subprocess
from datetime import datetime
from importlib.metadata import version
from pathlib import Path
from typing import Literal
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from eris_ml.features.definitions import load_feature_definition
from eris_ml.utils.hashing import sha256_file

ROOT = Path(__file__).resolve().parents[1]
DEV = "data/processed/development_raw_v1.csv"
CONFIG = "configs/features/feature_set_v2_general_16.yaml"
TRACE = "artifacts/reports/baseline_feature_ablation_v2_trace.json"
OOF = "artifacts/predictions/baseline_feature_ablation_v2_oof.csv"
FOLDS = "artifacts/predictions/baseline_feature_ablation_v2_assignments.csv"
OUTPUT = "artifacts/reports/feature_correlations_general16_v2.html"
KEYS = ["source_row", "EmployeeNumber", "Attrition"]


def align(development: pd.DataFrame, assignment: pd.DataFrame,
          oof: pd.DataFrame) -> pd.DataFrame:
    """Reject wrong membership, labels or folds rather than aligning by row position."""
    for frame in (development, assignment, oof):
        if (len(frame) != len(development) or frame[KEYS].isna().any().any()
                or any(frame[key].duplicated().any() for key in KEYS[:2])):
            raise ValueError("Invalid or duplicate IDs/labels/row counts.")
    reference = development[KEYS].sort_values("source_row").reset_index(drop=True)
    for frame in (assignment, oof):
        candidate = frame[KEYS].sort_values("source_row").reset_index(drop=True)
        if not reference.equals(candidate):
            raise ValueError("Development/OOF ID or target mismatch.")
    folds = assignment.set_index("source_row")["fold"].sort_index()
    saved = oof.set_index("source_row")["fold"].sort_index()
    if not folds.equals(saved) or set(folds) != {1, 2, 3, 4, 5}:
        raise ValueError("Fold mismatch.")
    probability = oof["probability"].to_numpy(dtype=float)
    if not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
        raise ValueError("Invalid probability.")
    return development.merge(oof[KEYS + ["probability"]], on=KEYS,
                             how="left", validate="one_to_one")


def correlations(frame: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    rows = []
    methods: tuple[tuple[Literal["pearson", "spearman"], str], ...] = (
        ("pearson", "Pearson"), ("spearman", "Spearman"),
    )
    for name in features:
        x = frame[name]
        if name == "OverTime":
            if not x.isin(["No", "Yes"]).all():
                raise ValueError("Unexpected OverTime value; no automatic imputation.")
            x = x.map({"No": 0, "Yes": 1})
        x = pd.to_numeric(x, errors="raise")
        if not np.isfinite(x.to_numpy(dtype=float)).all():
            raise ValueError(f"{name}: missing/nonfinite values, no imputation.")
        record: dict[str, str | int | float] = {"Feature": name, "n": len(x)}
        for target, label in (("Attrition", "nhãn"), ("probability", "OOF")):
            y = frame[target]
            if not np.isfinite(y.to_numpy(dtype=float)).all():
                raise ValueError("Nonfinite outcome.")
            for method, heading in methods:
                # Constant columns have undefined correlation, never a fabricated zero.
                value = (float(x.corr(y, method=method))
                         if x.nunique() > 1 and y.nunique() > 1 else float("nan"))
                record[f"{heading} / {label}"] = value
        rows.append(record)
    table = pd.DataFrame(rows)
    return table.sort_values("Pearson / nhãn", key=lambda col: col.abs(), ascending=False)


def build(root: Path, output: Path) -> pd.DataFrame:
    paths = [DEV, CONFIG, TRACE, OOF, FOLDS]
    for checked_path in [root / p for p in paths] + [output]:
        if "final_test" in str(checked_path).lower() or checked_path.is_symlink():
            raise ValueError("Forbidden final-test path or symlink.")
    if output.exists():
        raise ValueError("Output exists; refusing to overwrite.")
    hashes = {p: sha256_file(root / p) for p in paths}
    trace = json.loads((root / TRACE).read_text(encoding="utf-8"))
    for path in (DEV, CONFIG):
        if hashes[path] != trace["input_sha256"][path]:
            raise ValueError(f"Historical input hash mismatch: {path}")
    for key, path in (("oof", OOF), ("assignments", FOLDS)):
        if hashes[path] != trace["output_sha256"][key]:
            raise ValueError(f"Historical output hash mismatch: {path}")
    definition = load_feature_definition(root / CONFIG)
    features = list(definition.all_features)
    if definition.version != "v2_general_16" or len(features) != 16:
        raise ValueError("Expected locked general_16 configuration.")
    development = pd.read_csv(root / DEV, usecols=KEYS + features)
    if (len(development) != 1176
            or development.Attrition.value_counts().to_dict() != {0: 986, 1: 190}):
        raise ValueError("Development size/target distribution differs from provenance.")
    assignment = pd.read_csv(root / FOLDS)
    saved = pd.read_csv(root / OOF)
    candidate = saved.loc[(saved.feature_set == definition.version)
                          & (saved.model == "logistic_regression")].copy()
    table = correlations(align(development, assignment, candidate), features)
    provenance = {
        "generated_at": datetime.now(ZoneInfo("Asia/Bangkok")).isoformat(),
        "head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root,
                                        text=True).strip(),
        "input_sha256": hashes,
        "script_sha256": sha256_file(Path(__file__)),
        "python": platform.python_version(),
        "libraries": {name: version(name) for name in ("numpy", "pandas", "scipy")},
        "final_test_accessed": False, "model_fit": False,
    }
    for path, digest in hashes.items():
        if sha256_file(root / path) != digest:
            raise ValueError("An input changed during calculation.")
    display = table.to_html(index=False, float_format=lambda n: f"{n:+.4f}",
                            na_rep="undefined", border=0)
    bars = []
    for index, (_, row) in enumerate(table.iterrows()):
        value = float(row["Pearson / nhãn"])
        if not np.isfinite(value):
            continue
        y = 35 + index * 28
        width = abs(value) * 320
        left = 570 if value >= 0 else 570 - width
        color = "#b45309" if value >= 0 else "#0369a1"
        bars.append(f'<text x="5" y="{y + 14}">{html.escape(str(row.Feature))}</text>'
                    f'<rect x="{left}" y="{y}" width="{width}" height="18" fill="{color}"/>'
                    f'<text x="915" y="{y + 14}">{value:+.4f}</text>')
    document = f'''<!doctype html><html lang="vi"><meta charset="utf-8">
<title>ERIS V2 — Tương quan feature</title>
<style>body{{font:16px system-ui;max-width:1120px;margin:40px auto;padding:0 20px;
color:#172e40}}table{{border-collapse:collapse;width:100%}}th,td{{padding:10px;
border-bottom:1px solid #dce3e9;text-align:right}}th:first-child,td:first-child{{text-align:left}}
thead{{background:#eaf1f7}}pre{{white-space:pre-wrap;overflow-wrap:anywhere}}
svg{{width:100%;font:14px system-ui}}.note{{background:#fff5dc;padding:16px}}</style>
<h1>Tương quan từng feature — V2 general_16</h1>
<p>1.176 dòng development • 190 Attrition=1 (16,16%) • 16 feature • Logistic baseline raw.</p>
<p class="note">Đây là tương quan hai biến, <strong>không phải mức đóng góp riêng,
feature importance hoặc quan hệ nhân quả</strong>. Không dùng để tự động loại feature.
IBM cắt ngang không xác nhận hiệu năng tại công ty mới hay horizon 3–6 tháng.</p>
<h2>Cách đọc</h2><ul>
<li><b>Nhãn</b>: Attrition thực tế trong IBM, No=0/Yes=1.</li>
<li><b>OOF</b>: xác suất từ Logistic baseline raw general_16 đã lưu, mỗi dòng do model
không fit trên dòng đó dự đoán. Không phải dự đoán in-sample từ bundle final fit.
Đây là mô tả liên hệ với điểm model, không phải kiểm định độc lập hay giải thích nhân quả.</li>
<li>Pearson: liên hệ tuyến tính; với nhãn nhị phân tương đương point-biserial.
Spearman: liên hệ theo thứ hạng, hữu ích khi xem các thang ordinal IBM.</li>
<li>Hệ số từ −1 đến +1. Dương: giá trị lớn thường đi cùng nhãn/điểm cao hơn;
âm: ngược chiều. Gần 0 không loại trừ quan hệ phi tuyến hoặc tương tác.</li>
<li>OverTime: No=0, Yes=1; các trường còn lại giữ mã/số IBM, không scale hoặc impute.
Pearson với thang ordinal ngầm coi khoảng cách giữa các mã là đều; nên đối chiếu Spearman.</li>
<li>Không tính p-value, không tuyên bố statistical significance. Cột hằng ghi undefined.</li>
</ul><h2>Pearson với nhãn — sắp theo độ lớn tuyệt đối</h2>
<svg viewBox="0 0 990 510" role="img" aria-label="Pearson của 16 feature với Attrition">
<text x="245" y="20">−1</text><text x="565" y="20">0</text>
<text x="885" y="20">+1</text><line x1="570" y1="28" x2="570" y2="490"
stroke="#aab8c4"/>{''.join(bars)}</svg>
<h2>Bảng đầy đủ</h2>{display}
<details><summary>Provenance và kiểm tra đầu vào</summary>
<p>ID/nhãn/fold khớp development, assignment và OOF; mỗi candidate đủ 1.176 dòng,
ID duy nhất; xác suất hữu hạn [0,1]. Hash dữ liệu/config/OOF/assignment khớp trace cũ.
Không đọc/hash final test, không train, không sửa artifact cũ. Không xuất ID nhân viên.</p>
<pre>{html.escape(json.dumps(provenance, ensure_ascii=False, indent=2))}</pre></details></html>'''
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        stream.write(document)
    return table


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / OUTPUT)
    args = parser.parse_args()
    result = build(ROOT, args.output)
    print(result.to_string(index=False, float_format=lambda n: f"{n:+.4f}"))
    print(f"Saved: {args.output}")
