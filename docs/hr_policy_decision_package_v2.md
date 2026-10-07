# ERIS V2 — Tài liệu trao đổi quyết định với HR

**Bản chuẩn bị thảo luận, chưa được HR/doanh nghiệp phê duyệt.** Không chốt policy,
feature set hay ngưỡng production. Tài liệu chỉ chứa số liệu tổng hợp, không có
định danh hoặc danh sách nhân viên.

AI chỉ hỗ trợ HR xem xét nguy cơ nghỉ việc và ưu tiên hỗ trợ/can thiệp; quyết định
nhân sự do con người chịu trách nhiệm. Cảnh báo không phải kết luận về một cá nhân,
không tự động kích hoạt quyết định bất lợi hoặc thay thế việc kiểm tra bối cảnh.
V2 đã hoàn tất các thử nghiệm development và fairness audit. Theo yêu cầu riêng,
đã fit/đóng gói Logistic raw general_16 thành **research/demo candidate theo mã hóa
IBM**; xem [trạng thái](v2_status.md#researchdemo-final-fit--2026-10-07). Đây không
phải xác nhận policy HR, data contract doanh nghiệp hay production. Chưa có API V2;
các ô quyết định dưới đây vẫn để trống.

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

| Năng lực giả định | Bộ | Cảnh báo/lượt xem | TP | FP | FN | Precision | Recall | Lượt xem/TP |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 5% | 16 | 55 | 35 | 20 | 155 | 63,64% | 18,42% | 1,571 |
| 5% | 17 | 55 | 41 | 14 | 149 | 74,55% | 21,58% | 1,341 |
| 10% | 16 | 115 | 67 | 48 | 123 | 58,26% | 35,26% | 1,716 |
| 10% | 17 | 115 | 67 | 48 | 123 | 58,26% | 35,26% | 1,716 |
| 15% | 16 | 175 | 84 | 91 | 106 | 48,00% | 44,21% | 2,083 |
| 15% | 17 | 175 | 90 | 85 | 100 | 51,43% | 47,37% | 1,944 |

Precision = TP/cảnh báo; recall = TP/190; lượt xem/TP = cảnh báo/TP
(nếu TP=0 thì `undefined`, không phải 0). Đây là tỷ số tổng lượt xem trên tổng
nhãn dương được phát hiện, không phải thời gian xử lý hoặc lợi ích can thiệp.

Các tổng trên cộng từ năm nhóm kiểm tra kỹ thuật, mỗi nhóm có 235 hoặc 236 dòng.
Số cần xem được làm tròn xuống riêng từng nhóm: `k=floor(số người × tỷ lệ)`.
Vì vậy tổng 55/115/175 không phải lấy tỷ lệ rồi làm tròn trên toàn bộ 1.176 dòng.
**Các outer CV folds không tự động đại diện cho quy mô, lịch hay thành phần một
đợt làm việc thực tế của HR.** Không lấy 115 làm năng lực HR đã được xác nhận.
**115 cảnh báo ở 10% là tổng của năm outer-validation folds**, không phải dự báo
số nhân viên HR sẽ xem trong một tháng tại doanh nghiệp mới. Mỗi dòng được xem
ở đúng một fold; một đợt doanh nghiệp mới phải được định nghĩa riêng trước khi xếp hạng.

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

Quy tắc hòa hiện tại: top-k ưu tiên `source_row` tăng dần khi điểm bằng nhau;
ngưỡng chọn tất cả người có điểm bằng ngưỡng. Mã dòng chỉ giúp tái lập thử nghiệm,
không phải lý do nghiệp vụ để ưu tiên ai. HR cần xác nhận hoặc yêu cầu quy tắc khác;
nếu thay đổi phải đánh giá lại trước khi triển khai. Mốc 0,5 chỉ là tham chiếu.
`source_row` không phải ID nhân viên cho hệ thống thật. Cần xác nhận một ID ổn định,
duy nhất trong phạm vi đợt, nguồn cấp ID và thứ tự so sánh khi hòa điểm; ID không
là feature. **Chưa giả định backend có sẵn ID hoặc cấu trúc batch phù hợp.**

## 5. Fairness ảnh hưởng thế nào đến quyết định?

Nguồn: [tóm tắt fairness](fairness_v2.md) và
[report kỹ thuật](../artifacts/reports/fairness_v2.md). Đây là kết quả trên nhãn IBM,
không phải bằng chứng fairness ở doanh nghiệp Việt Nam hay hiệu quả can thiệp.
Age đã là feature của cả hai bộ; AgeGroup là nhóm audit. Gender và MaritalStatus
chỉ phục vụ audit, không bổ sung vào model. MaritalStatus là phân tích thăm dò.

Minimum support đã khóa: n≥50, dương≥10, âm≥10; precision cần thêm ≥10 cảnh báo.
Metric thiếu hỗ trợ ghi `undefined`, không thay bằng 0 và không coi là “không có
chênh lệch”. Đủ minimum support cũng không phải chứng nhận fairness.

**Tuổi, tại top-k 15%:** ở cả hai bộ, nhóm 18–29 có n=258, dương=74, âm=184,
TP=49 (TPR 66,2%); nhóm 50+ có n=139, dương=15, âm=124, TP=2 (TPR 13,3%).
Gap TPR là **52,9 điểm phần trăm**. Nhóm 50+ chỉ có 2–5 ca dương mỗi fold và
**0/5 folds đủ minimum support**; nhóm 18–29 đủ ở 2/5 folds. Không diễn giải
gap pooled này là ước lượng ổn định qua folds hoặc kết luận nguyên nhân.

**MaritalStatus, tại top-k 10%:** cùng tổng 67 TP nhưng nhóm được phát hiện thay đổi.

| Nhóm | n | Dương | Âm | TP bộ 16 | TP bộ 17 | TPR 16 / 17 | Fold đủ hỗ trợ |
|---|---:|---:|---:|---:|---:|---:|---:|
| Divorced | 263 | 29 | 234 | 11 | 13 | 37,9% / 44,8% | 0/5 |
| Married | 540 | 66 | 474 | 21 | 15 | 31,8% / 22,7% | 5/5 |
| Single | 373 | 95 | 278 | 35 | 39 | 36,8% / 41,1% | 5/5 |

Bộ 17 thêm 2 TP Divorced và 4 TP Single nhưng mất 6 TP Married. Divorced chỉ
có 2–9 ca dương mỗi fold; gap TPR/FPR đầy đủ giữa các nhóm MaritalStatus theo
fold không đủ hỗ trợ để tổng hợp. Không suy diễn tổng TP bằng nhau thành tác động
giống nhau lên các nhóm, hoặc suy diễn chênh lệch pooled thành bằng chứng chắc chắn.

**Gender:** tại 15%, gap TPR pooled giảm từ 10,4 xuống 3,1 điểm phần trăm ở
bộ 17, nhưng gap theo fold vẫn biến động; việc này không xóa các vấn đề theo tuổi
hoặc tình trạng hôn nhân. Chi tiết số hỗ trợ và biến động nằm trong report fairness.

Không tuyên bố bộ 16 hay 17 “đạt fairness”; không tự thêm quota theo nhóm, đổi
ngân sách, hòa điểm, feature hoặc policy từ kết quả audit. Người xác nhận cần
ghi nhận các hạn chế này cùng lựa chọn, không chỉ chọn theo tổng TP hoặc một gap.

## 6. Phiếu quyết định cho HR/chủ dự án

Các ô dưới để trống có chủ ý; không được suy diễn là đã đồng ý.

| Nội dung cần xác nhận | Câu trả lời/lý do | Trạng thái quyết định | Người xác nhận | Ngày xác nhận |
|---|---|---|---|---|
| Phạm vi quyết định: sử dụng với HR thực tế hay chỉ mô phỏng đồ án; ai có thẩm quyền xác nhận phạm vi đó? | | | | |
| Một đợt gồm những nhân viên nào, tần suất bao lâu; điều kiện tham gia/loại trừ và xử lý cảnh báo lặp qua nhiều đợt? | | | | |
| Thời điểm chốt danh sách và snapshot dữ liệu mỗi đợt; xử lý bản ghi đến muộn hoặc sửa sau thời điểm chốt? | | | | |
| Năng lực HR thực tế: tối đa bao nhiêu lượt/người xem mỗi đợt, ai xử lý và trong bao lâu; nếu chỉ mô phỏng, giả định là gì? | | | | |
| Chọn top-k hay ngưỡng xác suất và lý do; chấp nhận bỏ sót/cảnh báo nhầm ra sao? | | | | |
| Capacity được chọn là số tuyệt đối hay tỷ lệ, giá trị bao nhiêu; xử lý quá tải hoặc đợt ít người thế nào? | | | | |
| Quy tắc làm tròn k: có chấp nhận floor(n×capacity), kể cả k=0 ở đợt nhỏ, hay yêu cầu quy tắc khác? | | | | |
| Hòa điểm: ID ổn định nào, ai cấp/kiểm tra tính duy nhất và thứ tự so sánh; có chấp nhận thứ tự đó không? Không mặc định dùng source_row ngoài IBM. | | | | |
| Có chuẩn hóa JobRole đáng tin cậy không; ai quản lý ánh xạ, chức danh mới/không rõ và phiên bản mapping? | | | | |
| Chọn general_16 hay no_department_17 trong phạm vi đã xác nhận; lý do và đánh đổi fairness/tính áp dụng được chấp nhận? | | | | |
| Ai được xem điểm/cảnh báo; phạm vi truy cập, mục đích sử dụng, bảo mật và thời hạn lưu? | | | | |
| Ai kiểm tra bối cảnh, quyết định và thực hiện hành động hỗ trợ sau cảnh báo; khi nào cần dừng hoặc chuyển người phụ trách? | | | | |
| Ghi nhận hành động, người chịu trách nhiệm, thời điểm, kết quả và phản hồi/sửa dữ liệu thế nào; đánh giá lại khi nào? Không coi phản hồi là nhãn nghỉ việc đã xác thực. | | | | |
| Đã xem xét hạn chế fairness, nhóm thiếu hỗ trợ và giới hạn nhãn IBM; xử lý thắc mắc của người bị ảnh hưởng thế nào? | | | | |
| Đã chốt đơn vị thu nhập, cách xác định làm thêm, thang khảo sát, thời điểm chụp dữ liệu và cách xử lý thiếu dữ liệu chưa? | | | | |
| Quyết định tổng thể, phạm vi hiệu lực và các điều kiện còn treo; người quyết định cuối cùng và ngày xác nhận? | | | | |

Không có ô nào được điền sẵn bằng tên/ngày hoặc suy ra phê duyệt từ việc đã đọc
tài liệu. Xác nhận đồ án không tương đương chấp thuận sử dụng cho nhân viên thật.

## 7. Nếu chưa có HR thực tế: đề xuất mô phỏng cho đồ án

**Trạng thái phương án: đề xuất, chưa phê duyệt.**
Lựa chọn chưa được xác nhận: ☐ **Kịch bản mô phỏng cho đồ án**.

Đề xuất minh họa để chủ dự án xem xét: **Logistic baseline raw general_16 + top-k
10%**; bộ 17 tiếp tục là đối chứng IBM. Lý do thảo luận: không phụ thuộc mapping
JobRole và giới hạn lượt xem theo từng đợt đã chốt. Đây **không phải policy tối ưu,
không phải quyết định HR và chưa được phép áp dụng cho nhân viên thật**.

Các giả định vẫn cần người thật xác nhận:

- Chỉ mô phỏng học thuật trên bằng chứng IBM đã lưu hoặc dữ liệu giả lập được
  cho phép; không biến năm folds thành năm tháng hoặc năm đợt doanh nghiệp thật.
- Bộ 16 và capacity 10% là lựa chọn minh họa, không được xác nhận bởi ưu thế
  fairness; chấp nhận đưa các cờ review và thiếu hỗ trợ vào demo/báo cáo.
- Phải định nghĩa đợt, snapshot, đối tượng tham gia, capacity, floor và trường hợp
  k=0 trước khi dùng top-k. Không chọn một người chỉ vì cần “ít nhất một cảnh báo”.
- source_row chỉ dùng để tái hiện thứ tự hòa của IBM. ID ổn định cho bối cảnh
  khác và khả năng quản lý đợt chưa được xác minh, cần chốt trong data contract.
- Năng lực HR, chi phí FP/FN, quyền xem, hành động/ghi nhận sau cảnh báo và hiệu
  quả can thiệp chưa được xác nhận. Không tối ưu chi phí tự đặt hoặc hứa horizon
  nghỉ việc 3–6 tháng. Con người kiểm tra và chịu trách nhiệm quyết định nhân sự.

Người phụ trách đồ án có thể xác nhận phạm vi mô phỏng sau này, nhưng không được
chuyển xác nhận học thuật thành “HR/doanh nghiệp đã phê duyệt”. Hiện chưa ghi nhận
xác nhận nào trong tài liệu này.

## 8. Decision log

### Kết quả kỹ thuật đã quan sát

| Mục | Bằng chứng |
|---|---|
| Kết quả top-k theo ba mức | Bảng mục 2, tính lại từ OOF và khớp report/metrics đã lưu. |
| Hai danh sách không tương đương ở 10% | Bảng mục 3: 82 người chung, 55 TP chung, 12 TP riêng mỗi bộ. |
| Ngưỡng không bảo đảm năng lực đợt mới | Số fold vượt ngân sách ở mục 4; đây là quan sát kỹ thuật, không phải phê duyệt. |
| Fairness có các đánh đổi và thiếu hỗ trợ | Mục 5: gap tuổi, chuyển dịch TP giữa các nhóm MaritalStatus, biến động theo fold; không có chứng nhận fairness. |

### Giả định chờ xác nhận

| Mục | Điều còn thiếu |
|---|---|
| Năng lực 5%/10%/15% | Phạm vi đợt và năng lực HR thực tế. |
| Chọn policy, bộ feature và quy tắc hòa | Ý kiến HR, khả năng ánh xạ chức danh, chuẩn dữ liệu và quy trình tiếp nhận cảnh báo. |
| Kịch bản đồ án thay cho thử nghiệm doanh nghiệp | Xác nhận phạm vi học thuật riêng; không thay thế business approval. |
| Đề xuất general_16 + top-k 10% | Chưa có xác nhận policy HR/mô phỏng; bundle demo raw đã được cho phép riêng, không áp dụng cho nhân viên thật. |
| ID/batch, quyền xem và quy trình phản hồi | Cần xác nhận yêu cầu và kiểm tra nguồn dữ liệu; không giả định backend đã hỗ trợ. |

### Quyết định đã xác nhận

| Quyết định | Trạng thái | Người xác nhận | Ngày xác nhận | Bằng chứng xác nhận |
|---|---|---|---|---|
| | | | | |

## 9. Điều kiện chuyển bước và điểm phải dừng

**Hồ sơ đủ để ra quyết định, nhưng chưa đủ điều kiện chốt data contract doanh nghiệp
hoặc chọn candidate áp dụng thực tế:** còn thiếu xác nhận ở mục 6. Việc fit/bundle
research/demo theo IBM được người dùng cho phép riêng không thay thế xác nhận này.
Các nội dung cần xác nhận không được tự điền từ đề xuất ở mục 7.

1. Người có thẩm quyền xác nhận phạm vi HR/mô phỏng, lựa chọn feature/policy,
   đợt, capacity, hòa điểm, quyền xem và quy trình sau cảnh báo; ghi rõ điều còn treo.
2. Sau đó mới hoàn thiện và phê duyệt **data contract**: tên/kiểu/đơn vị/thang đo,
   snapshot, thiếu dữ liệu, ID/batch và mapping nếu có. Đây là quy tắc dữ liệu,
   **không phải thiết kế schema API/backend**. Contract cho đồ án phải ghi phạm vi
   mô phỏng, không phải chấp thuận triển khai doanh nghiệp.
3. Candidate áp dụng theo phạm vi HR/đồ án đã xác nhận vẫn cần quyết định và data
   contract tương ứng. Bundle general_16 hiện có chỉ là ngoại lệ được cho phép
   riêng để nghiên cứu/demo trên mã hóa IBM, không phải bundle đã được HR phê
   duyệt. Không dùng final test V1 để chọn hoặc chứng nhận V2.
4. **Dừng và thông báo chủ dự án trước mọi thiết kế hoặc thay đổi API/backend**,
   kể cả contract API hoặc tích hợp web. Không suy ra quyền triển khai từ việc
   phê duyệt mô phỏng, data contract hoặc bundle. Chờ chỉ dẫn riêng tại mốc này.

Lượt chuẩn bị hồ sơ ban đầu không thực hiện các bước trên; lần cập nhật trạng thái
bundle chỉ ghi nhận phạm vi research/demo riêng, không điền quyết định thay HR.
Đánh giá doanh nghiệp mới sau này cần nhãn và mốc thời gian rõ ràng, không coi
kết quả IBM là xác thực hiệu năng, fairness hoặc horizon tại doanh nghiệp mục tiêu.

## Nguồn và kiểm tra

Nguồn nội bộ: `docs/threshold_policy_v2.md`, report và metrics/folds/paired thuộc
`artifacts/reports/threshold_policy_v2*`; các bảng được đối chiếu với
`artifacts/predictions/threshold_policy_v2_oof.csv`. **Không gửi CSV cấp nhân viên
kèm tài liệu chia sẻ này.** SHA-256 của OOF:
`7bd9c5017ea70b548b82918b11f0181441129804c03dc98431be9c38cee5c221`.

Bổ sung fairness từ `docs/fairness_v2.md`, report `artifacts/reports/fairness_v2.md`,
các bảng `fairness_v2_policy.csv`, `fairness_v2_paired_policy.csv`,
`fairness_v2_probability.csv`, `fairness_v2_probability_stability.csv` và trace
`fairness_v2_trace.json`. Các bảng tổng hợp lưu sẵn được đối chiếu lại; không chạy
lại audit hoặc tái tạo OOF. Số phần trăm/lượt xem đã làm tròn để trình bày, không
thay đổi số liệu gốc. Lượt xem/TP chỉ mô tả nhãn IBM, không phải hiệu quả can thiệp.

Đã kiểm tra fingerprint, ghép đúng ID/target/fold và tính lại số liệu tổng hợp.
Không đọc final-test CSV, train/tune/calibrate, chọn ngưỡng production hoặc sửa API.
Final test V1 đã được dùng trước đây, không phải holdout độc lập mới cho V2 và
không được dùng trong package này. Artifacts cũ giữ nguyên; không commit/push.
