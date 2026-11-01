# B2 artifacts

Chỉ giữ artifact trực tiếp phục vụ hoặc chứng minh Random Forest B2:

- `models/v3_final_15/b2/`: bundle immutable dùng bởi API.
- `reports/v3_final_15/b2/`: trace đóng gói và smoke checks.
- `reports/v3_final_15/devfix1/`: development-only selection evidence mà B2 dẫn chiếu.
- `reports/v3_final_15/audit_b2_1/`: pre-API audit.
- `predictions/v3_final_15/devfix1/`: OOF evidence tương ứng.

Toàn bộ thư mục này ngoài file README bị Git ignore. Không commit bundle, row-level
predictions hoặc employee data. Không đổi tên hay sửa artifact vì metadata/trace chứa
path và checksum lịch sử.
