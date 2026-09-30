# ERIS V2 — Tài liệu trao đổi quyết định với HR

**Bản chuẩn bị thảo luận, chưa được HR/doanh nghiệp phê duyệt.** Không chốt policy,
feature set hay ngưỡng production. Tài liệu chỉ chứa số liệu tổng hợp, không có
định danh hoặc danh sách nhân viên.

## 1. Hai lựa chọn đầu vào

| Lựa chọn | Ý nghĩa với HR | Điểm cần cân nhắc |
|---|---|---|
| `general_16` (bộ 16) | Không dùng phòng ban hoặc chức danh công việc làm đầu vào mô hình. | Không cần ánh xạ `JobRole`; các trường còn lại vẫn phải được chuẩn hóa. |
| `no_department_17` (bộ 17) | Không dùng phòng ban, nhưng có dùng chức danh `JobRole`. | Cần quy tắc ánh xạ chức danh của từng công ty sang danh mục dùng khi huấn luyện; phải xử lý chức danh mới/không rõ và duy trì phiên bản ánh xạ. |

Cả hai là Logistic baseline raw, không phải model tuned hoặc calibrated.
Backend có thể lưu chức danh phục vụ quản lý mà không đưa vào mô hình bộ 16.
Kết quả IBM không chứng minh bộ nào sẽ tốt hơn ở doanh nghiệp khác.

## 2. HR cần xem bao nhiêu người?

Dữ liệu đánh giá có **1.176 dòng, 190 nhãn nghỉ việc và 986 nhãn không nghỉ việc**.
TP = được chọn và mang nhãn nghỉ việc IBM; FP = được chọn nhưng mang nhãn không
nghỉ việc; FN = nhãn nghỉ việc bị bỏ sót. Đây **không phải** số người được xác nhận
sẽ nghỉ trong 3–6 tháng, hoặc có thể can thiệp trước một tháng.

Top-k chọn những người có điểm cao nhất trong mỗi nhóm đánh giá:

| Năng lực giả định | Bộ | Người cần xem | TP | FP | Bỏ sót (FN) |
|---|---|---:|---:|---:|---:|
| 5% | 16 | 55 | 35 | 20 | 155 |
| 5% | 17 | 55 | 41 | 14 | 149 |
| 10% | 16 | 115 | 67 | 48 | 123 |
| 10% | 17 | 115 | 67 | 48 | 123 |
| 15% | 16 | 175 | 84 | 91 | 106 |
| 15% | 17 | 175 | 90 | 85 | 100 |

Các tổng trên cộng từ năm nhóm kiểm tra kỹ thuật, mỗi nhóm có 235 hoặc 236 dòng.
Số cần xem được làm tròn xuống riêng từng nhóm: `k=floor(số người × tỷ lệ)`.
Vì vậy tổng 55/115/175 không phải lấy tỷ lệ rồi làm tròn trên toàn bộ 1.176 dòng.
**Các outer CV folds không tự động đại diện cho quy mô, lịch hay thành phần một
đợt làm việc thực tế của HR.** Không lấy 115 làm năng lực HR đã được xác nhận.

Ở cùng số lượt xem, bộ 17 phát hiện thêm ròng 6/0/6 nhãn dương tính tại mức
5%/10%/15%. Lợi ích này phải cân nhắc với chi phí và độ tin cậy của ánh xạ chức danh.

## 3. Cùng 115 cảnh báo và 67 TP, nhưng không cùng danh sách

Bảng dưới tính trực tiếp từ OOF top-k 10%, ghép đúng cùng người và fold. “Tỷ lệ
chung” = số người chung / số người trong một danh sách; “TP riêng” là ca được một
bộ phát hiện nhưng bộ kia không chọn, không phải người nằm ngoài dữ liệu đánh giá.

| Nhóm kỹ thuật | Mỗi danh sách | Người chung | Tỷ lệ chung | Chỉ bộ 16 | Chỉ bộ 17 | TP chung | TP riêng 16 | TP riêng 17 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Fold 1 | 23 | 15 | 65,22% | 8 | 8 | 12 | 2 | 2 |
| Fold 2 | 23 | 17 | 73,91% | 6 | 6 | 12 | 4 | 4 |
| Fold 3 | 23 | 18 | 78,26% | 5 | 5 | 12 | 2 | 1 |
| Fold 4 | 23 | 16 | 69,57% | 7 | 7 | 9 | 3 | 3 |
| Fold 5 | 23 | 16 | 69,57% | 7 | 7 | 10 | 1 | 2 |
| Gộp | 115 | 82 | 71,30% | 33 | 33 | 55 | 12 | 12 |

Hai danh sách có tổng cộng 148 người khác nhau; giao/hợp (Jaccard) = 82/148 =
55,41%. Mỗi danh sách có 67 TP = 55 chung + 12 riêng. **Tổng hiệu năng bằng nhau
không có nghĩa quyết định xem ai là tương đương.** Không đề xuất gộp hai danh sách
để tăng phát hiện: việc đó thay đổi ngân sách, không còn là policy 10% đã đánh giá.

## 4. Top-k hay ngưỡng xác suất?

| Policy | Cách dùng | Hệ quả với HR |
|---|---|---|
| Top-k | Xếp toàn bộ nhân viên đủ điều kiện trong một đợt, chọn đúng k người. | Giữ số lượt xem theo quy tắc làm tròn; cần biết đầy đủ đợt. Không có một ngưỡng cố định áp dụng độc lập cho từng người; vẫn chọn k ngay cả khi điểm thấp. |
| Ngưỡng xác suất | Cảnh báo khi điểm lớn hơn hoặc bằng một ngưỡng đã xác định trước. | Có thể xét riêng từng người, nhưng số cảnh báo ở đợt mới có thể vượt hoặc thấp hơn ngân sách. |

Trong kiểm tra kỹ thuật, top-k không vượt ngân sách ở bất kỳ fold nào. Ngưỡng học
từ inner CV vượt ngân sách ở 3/1/2 folds của bộ 16 và 3/2/2 folds của bộ 17,
tương ứng 5%/10%/15%; không chỉnh lại ngưỡng theo kết quả nhóm kiểm tra.

Quy tắc hòa hiện tại: top-k ưu tiên mã dòng kỹ thuật tăng dần khi điểm bằng nhau;
ngưỡng chọn tất cả người có điểm bằng ngưỡng. Mã dòng chỉ giúp tái lập thử nghiệm,
không phải lý do nghiệp vụ để ưu tiên ai. HR cần xác nhận hoặc yêu cầu quy tắc khác;
nếu thay đổi phải đánh giá lại trước khi triển khai. Mốc 0,5 chỉ là tham chiếu.

## 5. Phiếu cần HR xác nhận

Các ô dưới để trống có chủ ý; không được suy diễn là đã đồng ý.

| Nội dung cần xác nhận | Câu trả lời/lý do | Trạng thái quyết định | Người xác nhận | Ngày xác nhận |
|---|---|---|---|---|
| Một đợt diễn ra bao lâu; gồm nhân viên nào, tại thời điểm dữ liệu nào; xử lý người đã được cảnh báo trước đó ra sao? | | | | |
| HR thực sự xem được tối đa bao nhiêu người mỗi đợt; giới hạn là số tuyệt đối hay tỷ lệ? | | | | |
| Ưu tiên top-k hay ngưỡng xác suất, vì sao; chấp nhận bỏ sót và cảnh báo nhầm đến mức nào? | | | | |
| Có chấp nhận cách làm tròn và quy tắc hòa điểm; nếu không, thay bằng quy tắc nào? | | | | |
| Có thể chuẩn hóa JobRole đáng tin cậy tại các công ty mục tiêu không; ai quản lý ánh xạ; chọn bộ 16 hay 17? | | | | |
| Sau cảnh báo, ai tiếp nhận, sẽ làm gì, trong bao lâu; ghi nhận hành động, phản hồi và kết quả can thiệp ra sao? | | | | |
| Đã chốt đơn vị thu nhập, cách xác định làm thêm, thang khảo sát, thời điểm chụp dữ liệu và cách xử lý thiếu dữ liệu chưa? | | | | |

## 6. Nếu chưa có HR thực tế: kịch bản mô phỏng cho đồ án

Lựa chọn chưa được chọn: ☐ **Kịch bản mô phỏng cho đồ án**.

Một đề xuất để thảo luận là mô phỏng top-k 10%, bộ 16 là phương án ưu tiên về
đầu vào và bộ 17 là đối chứng. Giả định phải ghi rõ: dữ liệu IBM, năm nhóm kỹ thuật
đã lưu, làm tròn xuống từng nhóm, quy tắc hòa hiện tại, không có năng lực HR hay
chi phí FP/FN được xác nhận, không khẳng định hiệu quả can thiệp hoặc chân trời
nghỉ việc. Tỷ lệ 10% chỉ là điểm bắt đầu trao đổi, **không phải mức tối ưu kinh doanh**.

Người phụ trách đồ án có thể xác nhận phạm vi mô phỏng sau này, nhưng không được
chuyển xác nhận học thuật thành “HR/doanh nghiệp đã phê duyệt”. Hiện chưa ghi nhận
xác nhận nào trong tài liệu này.

## 7. Decision log

### Kết quả kỹ thuật đã quan sát

| Mục | Bằng chứng |
|---|---|
| Kết quả top-k theo ba mức | Bảng mục 2, tính lại từ OOF và khớp report/metrics đã lưu. |
| Hai danh sách không tương đương ở 10% | Bảng mục 3: 82 người chung, 55 TP chung, 12 TP riêng mỗi bộ. |
| Ngưỡng không bảo đảm năng lực đợt mới | Số fold vượt ngân sách ở mục 4; đây là quan sát kỹ thuật, không phải phê duyệt. |

### Giả định chờ xác nhận

| Mục | Điều còn thiếu |
|---|---|
| Năng lực 5%/10%/15% | Phạm vi đợt và năng lực HR thực tế. |
| Chọn policy, bộ feature và quy tắc hòa | Ý kiến HR, khả năng ánh xạ chức danh, chuẩn dữ liệu và quy trình tiếp nhận cảnh báo. |
| Kịch bản đồ án thay cho thử nghiệm doanh nghiệp | Xác nhận phạm vi học thuật riêng; không thay thế business approval. |

### Quyết định đã xác nhận

| Quyết định | Trạng thái | Người xác nhận | Ngày xác nhận | Bằng chứng xác nhận |
|---|---|---|---|---|
| | | | | |

## 8. Điều kiện chuyển bước

Chỉ sau khi HR xác nhận phạm vi đợt và năng lực xử lý, chọn policy và feature set,
chốt quy tắc dữ liệu đầu vào, hòa điểm và quy trình sau cảnh báo, mới thiết kế
contract API/model artifact và kế hoạch đánh giá trên dữ liệu doanh nghiệp mới.
Kế hoạch mới cần định nghĩa nhãn, thời điểm đo, chân trời dự đoán và tiêu chí đánh
giá trước; không coi kết quả IBM là sự xác thực cho doanh nghiệp mục tiêu.
Nếu chỉ làm đồ án, phải ghi rõ contract là mô phỏng, chưa được chấp thuận triển khai.

## Nguồn và kiểm tra

Nguồn nội bộ: `docs/threshold_policy_v2.md`, report và metrics/folds/paired thuộc
`artifacts/reports/threshold_policy_v2*`; các bảng được đối chiếu với
`artifacts/predictions/threshold_policy_v2_oof.csv`. **Không gửi CSV cấp nhân viên
kèm tài liệu chia sẻ này.** SHA-256 của OOF:
`7bd9c5017ea70b548b82918b11f0181441129804c03dc98431be9c38cee5c221`.

Đã kiểm tra fingerprint, ghép đúng ID/target/fold và tính lại số liệu tổng hợp.
Không đọc final-test CSV, train/tune/calibrate, chọn ngưỡng production hoặc sửa API.
Final test V1 đã được dùng trước đây, không phải holdout độc lập mới cho V2 và
không được dùng trong package này. Artifacts cũ giữ nguyên; không commit/push.
