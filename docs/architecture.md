# ERIS architecture

ERIS tách bốn layer:

1. **Data contract** — provenance, target, schema, units, missingness và immutable split.
2. **Model research** — leakage-safe baselines, comparison, calibration và audit.
3. **Model package** — frozen pipeline, schema, metadata, checksums và model card.
4. **Serving/API** — authenticated runtime, health/readiness và integration contract.

Active project hiện chỉ ở layer 1. Chưa có dataset, contract hay model mới được chốt.

Prediction probability, business threshold, outreach capacity và HR action là các
layer quyết định riêng. Không nhúng policy nghiệp vụ vào xác suất model và không coi
output model là quyết định nhân sự tự động.

IBM/B2 chỉ là frozen historical reference; pipeline mới không được phụ thuộc vào schema
hoặc module runtime của reference đó.
