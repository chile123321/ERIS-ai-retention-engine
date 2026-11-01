# ERIS Retention Engine

ERIS đang chuyển sang dataset mới. Active tree hiện chỉ là nền tảng trung lập cho
data intake và đánh giá; chưa có dataset production, target, feature schema, model,
bundle hoặc API mới được phê duyệt.

IBM/B2 đã được đóng băng làm reference lịch sử tại tag
`archive/ibm-b2-final`. Không mặc định sao chép feature, vocabulary, preprocessing,
target hay business assumptions của IBM sang dataset mới.

## Cấu trúc

```text
configs/             Quy tắc cho config tương lai, chưa có config model giả
data/                Dataset local-only, chỉ commit README
artifacts/           Output local-only, chỉ commit README
docs/architecture.md Kiến trúc và ranh giới trách nhiệm
docs/archive/        Hồ sơ frozen reference
src/eris_ml/         Hashing, metric và split-boundary utilities trung lập
tests/unit/          Unit tests không phụ thuộc IBM
```

## Workflow cho dataset mới

1. Data intake và provenance.
2. Data audit.
3. Chốt target và feature contract.
4. Tạo immutable split.
5. Chạy baselines.
6. So sánh candidate và calibration khi cần.
7. Đánh giá final test đúng protocol.
8. Đóng gói model.
9. Tích hợp API/backend sau phê duyệt.

## Cài đặt và quality gate

Yêu cầu Python 3.12 và [uv](https://docs.astral.sh/uv/):

```bash
uv sync --frozen --extra dev
uv run pytest -q
uv run ruff check .
uv run mypy src
git diff --check
```

Hiện không có `.env.example` vì active code không đọc environment variable. Khi có
runtime thật, người nhận chỉ được yêu cầu đặt file `.env` được chuyển qua kênh riêng
vào repo root rồi chạy `docker compose up --build`; repository phải tự chứa hoặc tự tải
và xác minh mọi artifact còn lại.

## Xem frozen IBM/B2 reference

```bash
git fetch --tags
git switch --detach archive/ibm-b2-final
git switch main
```

Không merge ngược code IBM/B2 vào pipeline dataset mới.
