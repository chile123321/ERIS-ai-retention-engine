# ERIS Random Forest B2 Model Service

Repository tối giản để nạp và vận hành duy nhất research candidate:

- Model ID: `rf-v3-final15-unweighted-b2`
- Schema: `v3_final_15` (15 feature, 13 bắt buộc, 2 optional)
- Bundle: `artifacts/models/v3_final_15/b2/`
- SHA-256 `pipeline.joblib`:
  `0c72841a8505a910d31c2a687fbcbf5259e3afb60f9eddcb7586801519b5628f`

Model trả xác suất raw cho `Attrition=1`; không áp threshold, risk level hay quyết
định nhân sự tự động. Đây là model nghiên cứu trên IBM HR Analytics, chưa được xác
thực cho doanh nghiệp mới, fairness production hoặc horizon 3–6 tháng.

## Cấu trúc còn lại

```text
artifacts/models/v3_final_15/b2/              Bundle bất biến
artifacts/reports/v3_final_15/{b2,devfix1,audit_b2_1}/
artifacts/predictions/v3_final_15/devfix1/    Evidence trực tiếp của B2
configs/features/feature_set_v3_final_15.yaml Schema feature
src/eris_ml/api_b2/                           FastAPI service
src/eris_ml/models/bundle_v3_b2.py            Public load/predict/explain interface
deploy/api_b2/                                 Docker runtime
tests/                                        B2 API/schema/contract tests
docs/                                         API contract và OpenAPI
```

Một số module legacy-named dưới `src/eris_ml/` vẫn được giữ vì object đã pickle trong
bundle tham chiếu trực tiếp đến các import path đó. Xóa hoặc đổi tên chúng sẽ làm bundle
không nạp được.

## Cài đặt local

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

Bundle và evidence trong `artifacts/` bị Git ignore; chúng phải được chuyển giao nội bộ
và kiểm tra checksum, không được tái tạo ở startup.

## Chạy API local

```powershell
$env:ERIS_SERVICE_API_KEY = & .\.venv\Scripts\python.exe -B -c "import secrets; print(secrets.token_urlsafe(40))"
.\.venv\Scripts\python.exe -m uvicorn eris_ml.api_b2.main:app --host 127.0.0.1 --port 8000 --workers 1 --no-access-log
```

Endpoints: `/health`, `/ready`, `/v1/model/info`, `/v1/predict`,
`/v1/predict/batch`, `/v1/explain`. Các endpoint `/v1/*` yêu cầu header
`X-ERIS-Service-Key`. Xem [API contract](docs/api_b2_v1_contract.md).

## Docker

```powershell
$env:ERIS_SERVICE_API_KEY = & .\.venv\Scripts\python.exe -B -c "import secrets; print(secrets.token_urlsafe(40))"
docker compose -f compose.model-api.yml up --build -d
docker compose -f compose.model-api.yml down
Remove-Item Env:ERIS_SERVICE_API_KEY
```

## Quality gate

```powershell
.\.venv\Scripts\python.exe -B -m pytest -q -p no:cacheprovider
.\.venv\Scripts\python.exe -B -m mypy --no-incremental --cache-dir=.mypy_b2_check src
.\.venv\Scripts\python.exe -B -m ruff check --no-cache src scripts tests
.\.venv\Scripts\python.exe -m pip check
git diff --check
```

Không còn data/training/tuning/calibration/AdaBoost/UI example trong repository này.
Muốn train dataset mới, hãy tạo repository hoặc namespace nghiên cứu riêng; không ghi
đè bundle và evidence B2.
