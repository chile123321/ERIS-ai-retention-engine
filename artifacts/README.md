# Artifacts

Artifacts là local-only; Git chỉ giữ file README này. Không commit model bundle,
report, prediction, plot hoặc output cấp nhân viên.

Mỗi accepted run phải ghi:

- dataset hash;
- split version;
- feature-set version;
- model config;
- random seed;
- dependency versions;
- metrics;
- model checksum.

Không ghi đè artifact của run trước; dùng namespace/version riêng.
