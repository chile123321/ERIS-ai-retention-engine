# Fairness/subgroup V2 — kết quả development, 2026-10-01

**Trạng thái: audit kỹ thuật hoàn tất; HR, fairness và production chưa được phê duyệt.**
Bộ 16 = Logistic baseline raw `v2_general_16`; bộ 17 = Logistic baseline raw
`v2_no_department_17`. Bộ 16 vẫn là ứng viên ưu tiên về tính áp dụng, bộ 17 là
đối chứng cần ánh xạ JobRole. Audit này không chọn lại model hoặc policy.

## Preflight và protocol

- Đúng 1.176 development rows, 190 dương/986 âm. ID duy nhất; ID/target/fold
  khớp assignment và OOF. Hai raw candidate và sáu top-k candidate đều đủ
  1.176 dòng; raw khớp baseline/calibration với rtol=1e-7, atol=1e-9.
- Chỉ đọc development-only audit đã recovery. Không đọc file hỗn hợp hoặc
  mở/hash final-test CSV. Không fit lại model, chọn threshold hay thay dự đoán.
- [Protocol](../configs/evaluation/fairness_protocol_v2.yaml) được khóa và
  preflight được ghi trước khi tính subgroup. Nhóm tuổi kế thừa V1: 18–29,
  30–39, 40–49, 50+. Age đã là input của cả hai model; AgeGroup chỉ là phép
  nhóm để audit. Gender/MaritalStatus chỉ phục vụ audit, không thêm vào feature.
  MaritalStatus là phân tích thăm dò.
- Missing/unknown bị từ chối trước metric, không impute hoặc bỏ dòng.
  Minimum support: n≥50, dương≥10, âm≥10; precision cần thêm ≥10 cảnh báo.
  Performance metric thiếu hỗ trợ ghi `undefined`, không phải 0; không tính
  AUC một lớp. Vẫn giữ số đếm, phân bố điểm và tỷ lệ cảnh báo quan sát được.
- Top-k là xếp hạng **toàn bộ từng fold**, không xếp hạng lại trong subgroup:
  k=floor(n×capacity), điểm giảm dần, hòa điểm theo source_row tăng dần.
  Mỗi fold có 11/23/35 cảnh báo, tổng 55/115/175. Không biến top-k thành
  một ngưỡng chung; fold kỹ thuật không tự xác định đợt làm việc của HR.
- Wilson 95% chỉ là khoảng mô tả có điều kiện trên các dự đoán đã lưu, không
  bao quát bất định do training/CV/đợt mới, không phải kiểm định ý nghĩa thống kê.
  Fold summaries dùng sample SD và ghi số fold có metric hợp lệ.

## Tổng policy: nhãn IBM, không phải số ca sẽ nghỉ trong 3–6 tháng

| Bộ | Top-k giả định | Cảnh báo | TP | FP | FN | Precision | Recall |
|---|---|---|---|---|---|---|---|
| 16 | 5% | 55 | 35 | 20 | 155 | 63.6% | 18.4% |
| 16 | 10% | 115 | 67 | 48 | 123 | 58.3% | 35.3% |
| 16 | 15% | 175 | 84 | 91 | 106 | 48.0% | 44.2% |
| 17 | 5% | 55 | 41 | 14 | 149 | 74.5% | 21.6% |
| 17 | 10% | 115 | 67 | 48 | 123 | 58.3% | 35.3% |
| 17 | 15% | 175 | 90 | 85 | 100 | 51.4% | 47.4% |

## Hỗ trợ và TPR theo nhóm

Mẫu số TPR là cột Dương; FPR là Âm; alert rate là n; precision là số cảnh báo.
Mỗi thuộc tính là một phân hoạch riêng, **không cộng chồng các thuộc tính**.
Tất cả nhóm pooled đạt minimum support, nhưng không đồng nghĩa đủ bằng chứng
để kết luận công bằng hoặc ổn định. Bảng ghi TPR pooled, không thay metric fold
thiếu hỗ trợ bằng TPR pooled.

| Nhóm | n | Dương | Âm | Fold đủ hỗ trợ | TPR 5% (16 / 17) | TPR 10% (16 / 17) | TPR 15% (16 / 17) |
|---|---|---|---|---|---|---|---|
| 18-29 | 258 | 74 | 184 | 2/5 | 27.0% / 33.8% | 52.7% / 52.7% | 66.2% / 66.2% |
| 30-39 | 497 | 75 | 422 | 5/5 | 20.0% / 18.7% | 30.7% / 26.7% | 37.3% / 40.0% |
| 40-49 | 282 | 26 | 256 | 1/5 | 0.0% / 3.8% | 15.4% / 23.1% | 19.2% / 34.6% |
| 50+ | 139 | 15 | 124 | 0/5 | 0.0% / 6.7% | 6.7% / 13.3% | 13.3% / 13.3% |
| Female | 472 | 71 | 401 | 5/5 | 21.1% / 25.4% | 42.3% / 42.3% | 50.7% / 49.3% |
| Male | 704 | 119 | 585 | 5/5 | 16.8% / 19.3% | 31.1% / 31.1% | 40.3% / 46.2% |
| Divorced | 263 | 29 | 234 | 0/5 | 17.2% / 20.7% | 37.9% / 44.8% | 51.7% / 58.6% |
| Married | 540 | 66 | 474 | 5/5 | 19.7% / 15.2% | 31.8% / 22.7% | 37.9% / 37.9% |
| Single | 373 | 95 | 278 | 5/5 | 17.9% / 26.3% | 36.8% / 41.1% | 46.3% / 50.5% |

Có **10 ô precision pooled undefined** do ít hơn 10 cảnh báo.
Theo fold, Age 50+ chỉ có 21–35 người và 2–5 dương; Divorced có 48–61 người
và 2–9 dương: không fold nào đủ hỗ trợ. Age 40–49 chỉ 1/5 fold đủ hỗ trợ;
18–29 chỉ 2/5 vì một số fold n=49. Vì vậy gap TPR/FPR giữa **toàn bộ**
nhóm tuổi hoặc toàn bộ nhóm MaritalStatus không có fold nào hợp lệ để tổng
hợp; không thể khẳng định chênh lệch pooled ổn định qua năm folds.

### Ví dụ 10%: cùng tổng TP, khác phân bố

| Nhóm | Bộ 16: alerts / TP / FP / FN | Bộ 17: alerts / TP / FP / FN |
|---|---|---|
| 18-29 | 61 / 39 / 22 / 35 | 60 / 39 / 21 / 35 |
| 30-39 | 43 / 23 / 20 / 52 | 37 / 20 / 17 / 55 |
| 40-49 | 8 / 4 / 4 / 22 | 14 / 6 / 8 / 20 |
| 50+ | 3 / 1 / 2 / 14 | 4 / 2 / 2 / 13 |
| Female | 52 / 30 / 22 / 41 | 51 / 30 / 21 / 41 |
| Male | 63 / 37 / 26 / 82 | 64 / 37 / 27 / 82 |
| Divorced | 26 / 11 / 15 / 18 | 26 / 13 / 13 / 16 |
| Married | 46 / 21 / 25 / 45 | 43 / 15 / 28 / 51 |
| Single | 43 / 35 / 8 / 60 | 46 / 39 / 7 / 56 |

Ở 10%, hai bộ có 82 cảnh báo chung, 55 TP chung và mỗi bộ 12 TP riêng:
cùng 115 cảnh báo/67 TP không có nghĩa cùng danh sách. Không công bố ID cá nhân.

## Xác suất raw theo nhóm

AP = average precision, không phải diện tích PR tính bằng quy tắc khác.
Quantiles 0/10/25/50/75/90/100, sample SD và metric từng fold có trong CSV kỹ thuật.

| Nhóm | Tỷ lệ dương | Mean score 16 / 17 | AP 16 / 17 | ROC-AUC 16 / 17 | Brier 16 / 17 |
|---|---|---|---|---|---|
| 18-29 | 28.7% | 0.2749 / 0.2750 | 0.6181 / 0.6770 | 0.8045 / 0.8242 | 0.1569 / 0.1461 |
| 30-39 | 15.1% | 0.1611 / 0.1588 | 0.4500 / 0.4603 | 0.7548 / 0.7722 | 0.1086 / 0.1058 |
| 40-49 | 9.2% | 0.1123 / 0.1153 | 0.2650 / 0.3478 | 0.7157 / 0.7428 | 0.0788 / 0.0761 |
| 50+ | 10.8% | 0.0747 / 0.0766 | 0.2393 / 0.2811 | 0.6086 / 0.6317 | 0.0973 / 0.0924 |
| Female | 15.0% | 0.1629 / 0.1609 | 0.4670 / 0.5077 | 0.7986 / 0.8043 | 0.1034 / 0.0995 |
| Male | 16.9% | 0.1650 / 0.1663 | 0.4902 / 0.5321 | 0.7604 / 0.7855 | 0.1156 / 0.1103 |
| Divorced | 11.0% | 0.1639 / 0.1605 | 0.4297 / 0.5137 | 0.8034 / 0.8323 | 0.0859 / 0.0785 |
| Married | 12.2% | 0.1551 / 0.1535 | 0.3218 / 0.3243 | 0.7006 / 0.7154 | 0.1017 / 0.1022 |
| Single | 25.5% | 0.1774 / 0.1821 | 0.6880 / 0.7251 | 0.8328 / 0.8458 | 0.1414 / 0.1306 |

## Những điểm cần human review

- **AgeGroup:** cả hai bộ đều có cờ review tại cả ba capacity. Ở 15%, TPR
  18–29 là 49/74 (66,2%) nhưng 50+ chỉ 2/15 (13,3%) ở cả hai bộ; gap
  52,9 điểm phần trăm. FPR 18–29 là 37/184 (20,1%) ở bộ 16 và 34/184
  (18,5%) ở bộ 17; 50+ là 5/124 (4,0%) ở cả hai. Số dương 50+ ít và
  metric fold thiếu hỗ trợ; đây là lý do review, không phải kết luận nguyên nhân.
- **Gender:** 10% có cùng gap TPR 11,2 điểm phần trăm (Female 30/71,
  Male 37/119). 15% gap pooled giảm từ 10,4 xuống 3,1 điểm khi dùng bộ 17,
  nhưng delta TP là Female −1, Male +7. Gap Gender theo fold ở 15% có mean
  ± sample SD 20,1±12,5 điểm (16), 20,2±14,4 điểm (17), range lần lượt
  7,1–36,1 và 1,0–35,0 điểm. Gap pooled nhỏ **không chứng minh ổn định**.
  Bộ 17/Gender/15% không có cờ screening pooled, không phải chứng nhận fairness.
- **MaritalStatus (thăm dò):** tại 10%, bộ 17 thêm 2 TP Divorced và 4 TP
  Single nhưng mất 6 TP Married; mất ở 4/5 folds, fold còn lại hòa.
  Gap TPR tăng 6,1→22,1 điểm. Mean score Single 17,7%/18,2% thấp hơn tỷ lệ
  dương 25,5%; Divorced 16,4%/16,0% cao hơn 11,0%. Không suy ra nguyên nhân
  hoặc tự chạy lại calibration. Minimum support theo fold không đủ để tổng
  hợp gap đầy đủ của thuộc tính này.
- **Đánh đổi 16/17:** bộ 17 thêm 6 TP ở 5% và 15%, không thêm ở 10%;
  delta TP theo folds lần lượt [2,0,2,−1,3], [0,0,−1,0,1], [1,2,1,2,0].
  Lợi ích không đồng đều theo nhóm; không có “người thắng fairness” từ một
  metric. JobRole vẫn đòi hỏi ánh xạ đáng tin cậy tại doanh nghiệp đích.

Cờ mô tả kế thừa V1: TPR/FPR max−min>0,10; selection-rate min/max<0,80;
|mean score−prevalence|>0,05. Đây không phải ngưỡng pháp lý/phê duyệt.
Cờ AgeGroup V1 chỉ là lý do kiểm tra; khác model/policy nên không chuyển kết luận V1.

## Bằng chứng, kiểm chứng và giới hạn

[Report kỹ thuật](../artifacts/reports/fairness_v2.md),
[preflight](../artifacts/reports/fairness_v2_preflight.json),
[trace](../artifacts/reports/fairness_v2_trace.json),
[kiểm chứng độc lập](../artifacts/reports/fairness_v2_verification.json).
Chín CSV `artifacts/reports/fairness_v2_*.csv` chứa probability, policy,
disparities, paired probability/policy và bốn bảng stability. Tất cả output
mới chỉ chứa số tổng hợp, không có employee-level output; artifacts bị Git ignore.

Preflight xác minh 20 historical output-hash entries; 15 source-drift entries
được đối chiếu đúng historical/current hash và HEAD qua hồ sơ audit môi trường
(không phải 15 file riêng biệt). Không sửa trace/guard cũ.
Kiểm chứng độc lập bằng crosstab và tính metric trực tiếp khớp 120 dòng probability,
360 dòng policy, 180 dòng delta, mẫu số và mọi tổng phân hoạch, mean/sample SD.
126 file được bảo vệ và toàn bộ 26 file tracked đang sửa từ trước giữ nguyên hash.
62 input hashes và 10 output hashes trong trace mới đã được kiểm tra lại.

Đúng interpreter của môi trường sạch đã kiểm thử:
`C:\Users\Chile\AppData\Local\Temp\eris-env-audit-e01b331ff3c6437c83a882b24bb920c2\venv\Scripts\python.exe`.
Python 3.12.10; NumPy 2.4.6, pandas 3.0.6, sklearn 1.9.1, SciPy 1.16.3.
28 test fairness mới pass; full pytest 252 passed, 1 existing skip, 54 warnings.
Strict Mypy 71 files, Ruff toàn repo, pip check và diff check pass.
Skip do data contract chưa được phê duyệt; warnings hiện hữu gồm 50 cảnh báo
Logistic, 1 test-client, 3 SHAP plotting. Synthetic tests không fit lại model
nghiên cứu. Python 3.11 mới resolve dependency, **chưa runtime test**.

IBM cắt ngang không chứng minh fairness tại Việt Nam/doanh nghiệp mới, hiệu quả
can thiệp hoặc horizon 3–6 tháng. Final test V1 đã được dùng một lần, không phải
holdout độc lập mới của V2; audit này không sử dụng final-test data/score.

## Điều kiện chuyển bước và trạng thái quyết định

**Đủ thông tin để trao đổi với HR hoặc xác nhận riêng kịch bản mô phỏng đồ án;
chưa đủ để phê duyệt fairness/production.** Không còn blocker alignment/provenance.
Thiếu hỗ trợ subgroup theo fold phải giữ trong hồ sơ, không được diễn giải thành
“không có disparity”.

Một bước tiếp theo duy nhất: dùng tài liệu này cùng
[gói trao đổi HR](hr_policy_decision_package_v2.md) để xác nhận phạm vi đợt,
năng lực xử lý, top-k hay ngưỡng xác suất, hòa điểm, khả năng ánh xạ JobRole,
quy trình can thiệp và cách ghi nhận kết quả. Nếu không có HR thật, ghi rõ
kịch bản mô phỏng và người chịu trách nhiệm, không gọi là doanh nghiệp phê duyệt.

- Trạng thái quyết định:
- Người xác nhận:
- Ngày xác nhận:

Không tự chuyển sang đóng băng candidate, bundle/API hoặc đánh giá dữ liệu mới.

