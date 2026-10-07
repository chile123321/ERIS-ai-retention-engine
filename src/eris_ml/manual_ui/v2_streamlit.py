"""Offline, probability-only V2 smoke UI; no fitting, API calls or input persistence."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import streamlit as st

from eris_ml.models.bundle_v2 import (
    FEATURE_ORDER,
    LogisticResearchBundle,
    input_schema,
    load_bundle,
    validate_input,
)
from eris_ml.utils.hashing import sha256_file

ROOT = Path(__file__).resolve().parents[3]
BUNDLE = ROOT / "artifacts/models/eris_logistic_general16_v2_r1.joblib"
DEVELOPMENT = "data/processed/development_raw_v1.csv"
SAMPLE_POSITIONS = (0, 1, 2)  # Physical CSV order, fixed before observing labels/scores.
LABELS = {
    "Age": "Tuổi (năm)",
    "Education": "Học vấn (mã IBM 1–5)",
    "JobLevel": "Cấp bậc (mã IBM 1–5)",
    "MonthlyIncome": "Thu nhập tháng (giá trị IBM, không quy đổi VND)",
    "TotalWorkingYears": "Tổng số năm làm việc",
    "YearsAtCompany": "Số năm tại công ty",
    "YearsInCurrentRole": "Số năm trong vai trò hiện tại",
    "YearsSinceLastPromotion": "Số năm từ lần thăng chức gần nhất",
    "YearsWithCurrManager": "Số năm với quản lý hiện tại",
    "EnvironmentSatisfaction": "Hài lòng môi trường (mã IBM 1–4)",
    "JobInvolvement": "Gắn kết công việc (mã IBM 1–4)",
    "JobSatisfaction": "Hài lòng công việc (mã IBM 1–4)",
    "PerformanceRating": "Hiệu suất (mã quan sát IBM 3 hoặc 4)",
    "OverTime": "Làm thêm giờ (No/Yes theo IBM)",
    "TrainingTimesLastYear": "Số lần đào tạo năm trước",
    "DistanceFromHome": "Khoảng cách từ nhà (giá trị IBM)",
}
SECTIONS = {
    "Hồ sơ nhân sự": ["Age", "Education", "JobLevel", "MonthlyIncome"],
    "Kinh nghiệm và thâm niên": [
        "TotalWorkingYears", "YearsAtCompany", "YearsInCurrentRole",
        "YearsSinceLastPromotion", "YearsWithCurrManager",
    ],
    "Khảo sát và đánh giá": [
        "EnvironmentSatisfaction", "JobInvolvement", "JobSatisfaction", "PerformanceRating",
    ],
    "Chấm công, đào tạo và khoảng cách": [
        "OverTime", "TrainingTimesLastYear", "DistanceFromHome",
    ],
}


def load_samples(bundle: LogisticResearchBundle) -> pd.DataFrame:
    """Only the pinned development file; parse exactly 16 columns, never IDs/labels."""
    path = ROOT / DEVELOPMENT
    if path.is_symlink():
        raise ValueError("Development symlink rejected.")
    expected = bundle.metadata["provenance"]["input_sha256"][DEVELOPMENT]
    if sha256_file(path) != expected:
        raise ValueError("Development fingerprint differs from V2 bundle provenance.")
    frame = pd.read_csv(path, usecols=FEATURE_ORDER, nrows=max(SAMPLE_POSITIONS) + 1)
    if len(frame) != 3:
        raise ValueError("Expected three fixed development samples.")
    return validate_input(frame.iloc[list(SAMPLE_POSITIONS)])


def parse_manual(values: dict[str, str]) -> pd.DataFrame:
    """Parse form text, then delegate all IBM domain checks to the bundle schema."""
    if set(values) != set(FEATURE_ORDER):
        raise ValueError("Biểu mẫu phải có đúng 16 trường của general_16.")
    parsed: dict[str, str | float] = {}
    for name in FEATURE_ORDER:
        text = values[name].strip()
        if not text:
            raise ValueError(f"{name}: không được để trống; không tự điền dữ liệu thiếu.")
        if name == "OverTime":
            parsed[name] = text
        else:
            try:
                parsed[name] = float(text)
            except ValueError:
                raise ValueError(f"{name}: hãy nhập số nguyên không âm.") from None
    try:
        return validate_input(parsed)
    except ValueError as exc:
        name = str(exc).split(":", 1)[0]
        schema = input_schema()
        domain = {**schema["ordinal_domains"], **schema["nominal_domains"]}.get(name)
        rule = f"chỉ nhận {domain}" if domain else "cần số nguyên hữu hạn, không âm"
        raise ValueError(f"{name}: {rule}; không tự thay thế giá trị sai.") from None


def feature_table(frame: pd.DataFrame) -> pd.DataFrame:
    """String values avoid Arrow coercion between the No/Yes and numeric fields."""
    return pd.DataFrame({
        "Feature IBM": FEATURE_ORDER,
        "Giá trị": [str(frame.iloc[0][name]) for name in FEATURE_ORDER],
    })


def render() -> None:
    st.set_page_config(page_title="ERIS V2 • Kiểm thử offline", layout="wide")
    st.title("ERIS V2 — Kiểm thử dự đoán offline")
    st.caption("Logistic baseline raw · general_16 · 16 feature · research_demo_candidate")
    if (st.get_option("server.address") != "127.0.0.1"
            or st.get_option("browser.gatherUsageStats") is not False):
        st.error("Hãy mở bằng scripts/manual_model_ui_v2.py: chỉ localhost, tắt telemetry.")
        st.stop()
    st.info(
        "Minh họa theo cách mã hóa IBM; xác suất chưa được xác thực tại doanh nghiệp mới. "
        "Không phải kết luận nhân viên sẽ nghỉ việc. Không áp dụng cutoff hoặc top-k cho một người."
    )
    try:
        # Small local bundle; reload/check sidecars each run rather than cache stale artifacts.
        bundle = load_bundle(BUNDLE)
        samples = load_samples(bundle)
        digest = sha256_file(BUNDLE)
    except Exception:
        # Do not expose raw exception details or employee values to console/logs.
        st.error(
            "Không nạp được bundle V2/hồ sơ mẫu. Kiểm tra file và sidecar V2, "
            "hash development và đúng môi trường Python 3.12.10 đã đóng gói. "
            "Không chuyển sang V1 hoặc tự fit lại."
        )
        st.stop()
    st.success(f"Đã nạp bundle V2 · {bundle.metadata['candidate_version']}")
    st.caption(f"Artifact: {BUNDLE.name} · SHA256: {digest[:16]}… · class dương: Attrition=1")
    st.caption("Chỉ xử lý trong bộ nhớ máy local; không ghi input vào log/file, không gọi API.")
    mode = st.radio("Cách thử", ["Hồ sơ mẫu IBM", "Nhập thủ công"], horizontal=True)
    sample_index = 0
    if mode == "Hồ sơ mẫu IBM":
        st.caption(
            "Ba dòng đầu development theo thứ tự file (vị trí 0, 1, 2), "
            "không chọn theo nhãn/xác suất. Chỉ đọc 16 feature, không hiển thị ID/nhãn."
        )
        sample_index = st.selectbox("Chọn hồ sơ mẫu", [0, 1, 2],
                                    format_func=lambda n: f"Mẫu {n + 1}")
    if st.session_state.get("view") != (mode, sample_index):
        st.session_state.pop("prediction", None)
        st.session_state["view"] = (mode, sample_index)
    values: dict[str, str] = {}
    with st.form("v2_prediction", clear_on_submit=False):
        if mode == "Hồ sơ mẫu IBM":
            st.dataframe(feature_table(samples.iloc[[sample_index]]), hide_index=True,
                         height=220, width="stretch")
        else:
            st.caption(
                "Khởi tạo từ Mẫu 1; có thể sửa từng trường. Số nguyên không âm; "
                "không coi các mã/đơn vị IBM là data contract doanh nghiệp."
            )
            for title, names in SECTIONS.items():
                st.subheader(title)
                columns = st.columns(2)
                for index, name in enumerate(names):
                    with columns[index % 2]:
                        # Text fields allow missing/invalid input to be reported, not filled with 0.
                        values[name] = st.text_input(
                            f"{LABELS[name]} — {name}",
                            value=str(samples.iloc[0][name]), key=f"field_{name}",
                        )
        submitted = st.form_submit_button("Dự đoán", type="primary")
    if submitted:
        st.session_state.pop("prediction", None)
        try:
            frame = (parse_manual(values) if mode == "Nhập thủ công"
                     else samples.iloc[[sample_index]])
            probability = float(bundle.predict_proba(frame)[0])
            timestamp = datetime.now(timezone(timedelta(hours=7))).isoformat(timespec="seconds")
            st.session_state["prediction"] = (probability, frame.copy(), timestamp)
        except ValueError as exc:
            st.error(str(exc))
        except Exception:
            st.error("Không thể dự đoán với bundle V2; không lưu input hoặc dùng model thay thế.")
    if "prediction" in st.session_state:
        probability, frame, timestamp = st.session_state["prediction"]
        st.divider()
        st.subheader("Kết quả lần gửi gần nhất")
        st.metric("Xác suất Attrition=1 theo model V2",
                  f"{probability * 100:.2f}%".replace(".", ","))
        st.progress(probability, text="Thang xác suất 0–100% (không phải mức phân loại)")
        st.caption(f"Thời điểm dự đoán (UTC+07): {timestamp}")
        st.code(f"p(Attrition=1) = {probability:.17g}", language=None)
        st.caption("16 giá trị thực sự đã gửi vào bundle ở lần dự đoán này:")
        st.dataframe(feature_table(frame), hide_index=True, height=600, width="stretch")


if __name__ == "__main__":
    render()
