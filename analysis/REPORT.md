# Phân tích pipeline đánh giá LLM debugging — dataset v2

---

## 0. Bản đồ dữ liệu (đọc trước khi phân tích)

| Tầng | File | Nội dung | Cột quyết định |
|---|---|---|---|
| 0.1 | `stage0/step1/step1_{variant}.xlsx` | Ground truth sau khi gắn nhiễu | `gt_status`, `gt_input`, `gt_expected_output`, `gt_actual_output`, `gt_reason` |
| 0.2 | `stage0/step2/step2_{variant}_{model}_{run}.xlsx` | Zero-shot Qwen & Llama, 3 run | `pred_status`, `comp.` |
| 1.1 | `stage1/step1/step1_classify1_{variant}.xlsx` → `step1_filter1_{variant}.xlsx` | Thẩm định + lọc ra **D1** | `match`, `label`, `filtered_label` |
| 1.2 | `stage1/step2/step2_filter1_{variant}_qwen3.7-max_{1..3}.xlsx` | Qwen few-shot trên D1 | `pred_status` |
| 1.3 | `stage1/step3/step3_filter1_{variant}_{claude,gpt}_{1..2}.xlsx` | Claude/GPT zero-shot trên D1 | `pred_status` |
| 2.1 | `stage2/step1/step1_classify2_*` → `step1_filter2_{variant}.csv` | Thẩm định + lọc ra **D2** | `filtered_label` |
| 2.2 | `stage2/step2/step2_filter2_{variant}_{model}_{1..2}.csv` | Few-shot Claude/GPT trên D2 | `pred_status` |
| 3 | `stage3/classify3_{variant}_{model}.csv` | Thẩm định cuối | `label` |

**Khoá join:** `id` (1..179), duy nhất, không trùng, nhất quán xuyên suốt. D2 ⊂ D1 ⊂ D0 đã kiểm chứng.

**Bảng mã nhãn** (suy ra từ crosstab `label × match × (gt_status==pred_status)`):

- `match/comp.` = `sim`/`diff` → verdict trùng hay không.
- `label` 1 = đúng verdict **và** đúng lập luận · 2 = đúng một phần · 3 = sai.
- `filtered_label` = nhãn sau thẩm định lần hai (chặt hơn `label`). **Đây mới là nhãn dùng để lọc.**
- Quy tắc lọc: `D_{k+1} = { case : filtered_label ∈ {2,3} }`.

---

## 1. Inventory & chất lượng gắn nhiễu (Stage 0.1)

| variant | N | WA | RE | Accepted | TLE | MLE | Compile Error |
|---|---|---|---|---|---|---|---|
| logic1 | 179 | 150 | 23 | 4 | 2 | 0 | 0 |
| logic2 | 179 | 92 | 30 | 55 | 1 | 1 | 0 |
| reference1 | 179 | 86 | 56 | 34 | 3 | 0 | 0 |
| **reference2** | 179 | 1 | 0 | 0 | 0 | 0 | **178** |

Topic: String 50 · Bit manipulation 50 · Tree 50 · Graph 29 (mỗi nhánh).
Level: Medium 76 · Easy 57 · Hard 46.

**Insight 1 — `reference2` hỏng, phải loại khỏi luận văn.** 178/179 case cho Compile Error: loại nhiễu này phá cú pháp chứ không phá ngữ nghĩa, nên bài toán trở thành "đọc thông báo compiler", không còn đo được năng lực suy luận. Dữ liệu xác nhận điều này: `reference2` không có bất kỳ file downstream nào. ⇒ **N thực tế = 3 × 179 = 537**, không phải 4 × 179.

**Insight 2 — Tỷ lệ "nhiễu vô hiệu" chênh lệch lớn giữa các nhánh.** `gt_status = Accepted` nghĩa là gắn nhiễu xong chương trình vẫn chạy đúng:

- logic1: 4/179 = **2.2%** (gắn nhiễu tốt nhất)
- reference1: 34/179 = **19.0%**
- logic2: 55/179 = **30.7%** (gần 1/3 mẫu là nhiễu chết)

Con số này phải được báo cáo — nó là thước đo chất lượng của chính bộ sinh nhiễu, và nó bóp méo mọi accuracy tính trên logic2.

**Insight 3 — Mỗi loại nhiễu tạo một "profile lỗi" riêng.** logic1 gần như thuần WA (84%); reference1 mới là nhánh sinh nhiều Runtime Error nhất (56, gấp 2.4× logic1). Nếu luận văn kết luận gì về "LLM debug loại lỗi nào", kết luận đó phải tách theo nhánh nhiễu, vì phân bố verdict không đồng nhất (χ² sẽ rất lớn).

---

## 2. Baseline zero-shot (Stage 0.2)

Accuracy status-exact, trung bình 3 run:

| variant | Qwen3.7-max | Llama-3.3-70B |
|---|---|---|
| logic1 | 0.9255 ± 0.0086 | 0.5847 ± 0.0086 |
| logic2 | 0.8790 ± 0.0032 | 0.5251 ± 0.0000 |
| reference1 | 0.8529 ± 0.0085 | 0.2998 ± 0.0116 |

Self-consistency (3 run):

| variant | model | luôn đúng | luôn sai | dao động |
|---|---|---|---|---|
| logic1 | Qwen | 0.888 | 0.039 | 0.073 |
| logic1 | Llama | 0.570 | 0.402 | 0.028 |
| logic2 | Qwen | 0.838 | 0.095 | 0.067 |
| logic2 | Llama | 0.520 | 0.469 | 0.011 |
| reference1 | Qwen | 0.821 | 0.117 | 0.062 |
| reference1 | Llama | 0.285 | 0.676 | 0.039 |

**Insight 4 — Llama không đủ năng lực để làm một nửa của bộ lọc.** Chênh 27–55 điểm % so với Qwen. Trên reference1, Llama đúng 30% — dưới cả mức đoán theo class đa số (WA = 48%). Kiểm tra phân bố dự đoán của Llama trên logic1: 54 lần đoán `Accepted` trong khi ground truth chỉ có 4 → Llama **bias mạnh về "code không có lỗi"**. Hệ quả: phép giao "sai ở cả Qwen VÀ Llama" trên thực tế bị Qwen chi phối hoàn toàn, vì tập sai của Llama gần như bao trùm tập sai của Qwen.

**Insight 5 — Sai số 3-run rất nhỏ (σ ≤ 0.012).** Nghĩa là chênh lệch accuracy < 2 điểm % giữa hai cấu hình **không** có ý nghĩa. Với N = 179, sai số chuẩn nhị thức ở p ≈ 0.85 là ~2.7 điểm %; đừng diễn giải chênh lệch nhỏ hơn khoảng đó.

---

## 3. Thẩm định nhãn — insight mạnh nhất của bộ dữ liệu

| variant | acc theo verdict (`label==1`) | acc sau thẩm định (`filtered_label==1`) | chênh |
|---|---|---|---|
| logic1 | 0.8883 | 0.7877 | **−10.1 điểm** |
| logic2 | 0.8380 | 0.7709 | −6.7 điểm |
| reference1 | 0.8212 | 0.7151 | **−10.6 điểm** |

Crosstab `label × filtered_label` (logic1):

```
filtered_label    1   2   3
label
1               141  18   0     <- 18 case bị hạ bậc
2                 0  13   0
3                 0   0   7
```

**Insight 6 — "Đúng vì lý do sai" chiếm ~11% số case được coi là đúng.** 18/159 case ở logic1 đoán trúng verdict nhưng test case / lập luận sai. Không case nào được nâng bậc — thẩm định chỉ đi một chiều. ⇒ **Accuracy tính theo verdict thổi phồng năng lực model 7–11 điểm %.** Đây là luận điểm trung tâm nên đưa vào abstract: benchmark chỉ so `pred_status` với `gt_status` là không đủ; phải chấm cả `pred_input` / `pred_expected_output` / `pred_reason`.

---

## 4. Escalation: Stage 1.2 và 1.3 (tất cả trên D1 — nơi Stage 0 sai 100%)

Recovery rate (tỷ lệ cứu được):

| variant | Qwen few-shot (3 run) | Claude-opus-4.8 zero-shot (2 run) | GPT-5.6 zero-shot (2 run) |
|---|---|---|---|
| logic1 | **0.658 ± 0.000** | 0.526 ± 0.037 | 0.684 ± 0.000 |
| logic2 | **0.565 ± 0.022** | 0.543 ± 0.031 | 0.609 ± 0.031 |
| reference1 | **0.510 ± 0.052** | 0.392 ± 0.000 | 0.510 ± 0.028 |

**Insight 7 — Few-shot cho model nhỏ ≥ zero-shot cho model lớn.** Qwen + few-shot đánh bại Claude-opus-4.8 zero-shot ở **cả 3 nhánh** (+13.2, +2.2, +11.8 điểm) và hoà GPT-5.6 zero-shot ở reference1. Kết luận cho luận văn: trên tập ca khó, **đầu tư vào prompt rẻ hơn và hiệu quả hơn đầu tư vào model**. Đây là kết quả có giá trị thực tiễn (chi phí/token của Qwen thấp hơn Claude/GPT nhiều lần).

**Insight 8 — GPT-5.6 > Claude-opus-4.8 ở zero-shot, nhưng đảo chiều ở few-shot.** Xem mục 5. Không có model nào thống trị mọi tầng — nên báo cáo theo cặp (model × strategy × độ khó tập), đừng xếp hạng model bằng một con số.

---

## 5. Stage 2 — few-shot trên D2

| variant | Claude few-shot | GPT few-shot |
|---|---|---|
| logic1 | 0.250 (0.313 / 0.188) | **0.500** (0.563 / 0.438) |
| logic2 | **0.432** (0.500 / 0.364) | 0.273 (0.273 / 0.273) |
| reference1 | **0.429** (0.400 / 0.457) | 0.329 (0.371 / 0.286) |

**Insight 9 — Lợi ích escalation giảm mạnh theo tầng.** Recovery rơi từ ~55% (D1) xuống ~35% (D2), dù dùng model mạnh hơn **và** few-shot. Mỗi vòng lọc chưng cất ra một tập khó hơn hẳn — đúng như thiết kế mong đợi, và đó chính là bằng chứng cho thấy D2 là một benchmark "hard core" có giá trị.

**Insight 10 — Phương sai giữa 2 run rất lớn ở tầng này** (Claude/logic1: 0.313 vs 0.188 — chênh 12.5 điểm trên n=16). Với n = 16–35, khoảng tin cậy 95% rộng ±20 điểm %. **Không được so sánh Claude vs GPT ở Stage 2 bằng point estimate.** Cần ≥ 5 run và báo cáo bootstrap CI, hoặc chỉ kết luận định tính.

---

## 6. Attrition & accuracy toàn pipeline

| variant | D0 | D1 | D2 | giải được ở S1 | giải được ở S2 | còn lại |
|---|---|---|---|---|---|---|
| logic1 | 179 | 38 (21.2%) | 16 (8.9%) | 141 | 22 | **10** |
| logic2 | 179 | 46 (25.7%) | 22 (12.3%) | 133 | 24 | **13** |
| reference1 | 179 | 51 (28.5%) | 35 (19.6%) | 128 | 16 | **19** |

Accuracy cộng dồn:

| variant | sau Stage 0 | sau Stage 1 | cuối pipeline |
|---|---|---|---|
| logic1 | 0.788 | 0.911 | **0.944** |
| logic2 | 0.743 | 0.877 | **0.927** |
| reference1 | 0.715 | 0.805 | **0.894** |

**Insight 11 — Pipeline phân tầng thu được +10.6 đến +17.9 điểm % so với zero-shot đơn model**, và làm được điều đó trong khi chỉ gửi 21–29% dữ liệu lên model đắt tiền. Đây là luận điểm hiệu quả chi phí, nên có một bảng ước lượng token/chi phí đi kèm.

**Insight 12 — Lõi cứng 42/537 = 7.8% case không model/chiến lược nào giải được.** Đây là đóng góp dataset của luận văn: một tập con đã được chứng minh là khó với 4 model và 2 chiến lược prompt. Nên xuất tập này ra thành artifact riêng và phân tích định tính từng case.

---

## 7. Phân tích theo verdict / topic / level (gộp 3 nhánh, N = 537)

### Theo verdict

| gt_status | n | acc Stage 0 | vào D1 | vào D2 | residual |
|---|---|---|---|---|---|
| Wrong Answer | 328 | 0.784 | 21.7% | 9.8% | **3.4%** |
| Accepted | 93 | 0.763 | 23.7% | 17.2% | 12.9% |
| Runtime Error | 109 | 0.679 | 32.1% | 17.4% | 12.8% |
| Time Limit Exceeded | 6 | **0.000** | 100% | 83.3% | 66.7% |
| Memory Limit Exceeded | 1 | **0.000** | 100% | 100% | 100% |

**Insight 13 — TLE/MLE là điểm mù tuyệt đối: 0/7 case đúng ở Stage 0, 5/7 vẫn sai sau toàn bộ pipeline.** Lý do hợp lý: dự đoán TLE đòi hỏi lập luận về độ phức tạp tiệm cận và ngưỡng thời gian thực thi — thứ mà mô hình không thể suy ra từ việc đọc code. **Cảnh báo thống kê: n = 7 quá nhỏ để claim định lượng.** Hãy báo cáo định tính ("không một case TLE/MLE nào được giải ở baseline") và, nếu muốn claim mạnh, phải bổ sung mẫu TLE/MLE lên ≥ 30/nhánh.

**Insight 14 — Model "ảo giác bug": 12.9% case `Accepted` vẫn sai sau toàn pipeline.** Confusion matrix Stage 0 (Qwen zero-shot run 1) cho thấy trong 93 case ground-truth `Accepted`, model đoán nhầm 10 lần thành `Runtime Error` và 5 lần thành `Wrong Answer`. Đây là **false positive trong phát hiện lỗi** — chiều lỗi nguy hiểm nhất cho ứng dụng thực tế (báo động giả). Nên tách riêng thành một metric: *false-alarm rate*.

Confusion matrix Stage 0 (Qwen zero-shot run 1, gộp 3 nhánh):

```
pred →        Accepted  CompErr  MLE   RE   TLE   WA
gt ↓
Accepted            75        1     1   10     1    5
MLE                  0        0     0    1     0    0
Runtime Error        1        0     0   87     1   20
TLE                  1        0     0    1     3    1
Wrong Answer         1        0     0   13     1  313
```

Nhầm lẫn chủ đạo: **RE → WA (20 case)** và **WA → RE (13 case)**. Model phân biệt được "có lỗi / không lỗi" tốt hơn nhiều so với phân biệt *loại* lỗi.

### Theo topic

| topic | n | acc Stage 0 | residual |
|---|---|---|---|
| Graph | 87 | **0.851** | **1.1%** |
| String | 150 | 0.787 | 4.7% |
| Bit manipulation | 150 | 0.733 | 11.3% |
| Tree | 150 | **0.667** | 11.3% |

### Theo level

| level | n | acc Stage 0 | residual |
|---|---|---|---|
| Easy | 171 | 0.772 | **9.4%** |
| Medium | 228 | 0.715 | 7.9% |
| Hard | 138 | 0.775 | **5.8%** |

**Insight 15 (phản trực giác, đáng viết nhất) — Độ khó LeetCode KHÔNG dự đoán độ khó cho LLM.** Accuracy: Easy 0.772, Medium 0.715, Hard 0.775 — không đơn điệu. Residual thậm chí **đảo chiều**: Easy 9.4% > Hard 5.8%.

Đã kiểm tra confounding: Graph (topic dễ nhất cho LLM) lệch hẳn về phía khó với con người — 30 Hard / 54 Medium / chỉ 3 Easy — nên kết luận này *không* phải do Graph kéo. ⇒ Cái quyết định độ khó debug với LLM là **loại lỗi và cấu trúc dữ liệu**, không phải nhãn độ khó thuật toán. Tree (thao tác con trỏ, đệ quy, null-handling) khó nhất; Graph (thuật toán mẫu, khuôn mẫu quen) dễ nhất.

---

## 8. Vấn đề dữ liệu cần xử lý trước khi công bố

| # | Vấn đề | Bằng chứng | Đề xuất |
|---|---|---|---|
| **A** | **Filter Stage 2.1 không đúng như mô tả phương pháp.** Thiết kế nói D2 = (Qwen few-shot sai) ∩ (Claude ZS sai). Dữ liệu cho thấy D2 = (Claude ZS sai) ∩ (GPT ZS sai), khớp 100% cả 3 nhánh. Hơn nữa D2 ⊄ tập Qwen-few-shot-sai. | `b5_stage2_filter.csv`: `D2_eq_claude_AND_gpt = True`, `D2_subset_qwenFSwrong = False` | Sửa mô tả phương pháp thành "giao của hai model escalation zero-shot", **hoặc** chạy lại filter theo đúng thiết kế. Không được để lệch. |
| **B** | **Filter Stage 1.1 cũng không phải phép giao Qwen ∩ Llama theo verdict.** D1 = `filtered_label ∈ {2,3}` từ file classify1 (chỉ chứa Qwen zero-shot run 1). Ở logic1, giao status-wrong Qwen∩Llama chỉ có 17 case, còn D1 có 38. | `probe`: `q_any&l_any = 17`, `D1 = 38` | Mô tả lại: "D1 = case bị thẩm định là chưa đạt sau khi đối chiếu output của cả hai model" — và nói rõ thẩm định này có yếu tố thủ công. |
| **C** | **Rò rỉ 5 case ở D1/logic2.** `step1_filter1_logic2.xlsx` có 46 dòng nhưng classify1 chỉ gắn `filtered_label ∈ {2,3}` cho 41. Các `id` 9, 19, 30, 32, 52 có `filtered_label = 1`, `label = 1`, `match = sim` mà vẫn lọt vào D1. | `b3_stage1_classify_filter.csv`: `ro_ri = 5` | Xác định lại: hoặc sửa `filtered_label`, hoặc bỏ 5 case khỏi D1. Hiện tại D2/logic2 và mọi số downstream của logic2 đều bị ảnh hưởng. |
| **D** | **Cột `prompt_strategy` không đồng nhất** trong `stage1/step2/step2_filter1_logic1_qwen3.7-max_1.xlsx`: 25 dòng `few-shot`, 13 dòng `zero-shot` trong cùng một file lẽ ra là few-shot. | `b4_*`, cảnh báo in ra khi chạy script | Kiểm tra lại log gọi API cho 13 dòng đó; nếu thật sự là zero-shot thì phải chạy lại. |
| **E** | Số run không đồng đều: Stage 0 & 1.2 có 3 run, Stage 1.3 & 2.2 chỉ 2 run. Cột `comp.` / `match` / `label` xuất hiện không nhất quán giữa các file. | schema scan | Chuẩn hoá về cùng số run (khuyến nghị ≥ 3, lý tưởng 5 cho Stage 2 vì n nhỏ). |
| **F** | Còn cột rác: `Unnamed: 20` trong `step3_filter1_logic2_claude-opus-4-8_2.xlsx`. | schema scan | Dọn trước khi đóng gói dataset. |

---

## 9. Việc nên làm tiếp

1. Xử lý xong mục 8 (A–D là chặn công bố).
2. Chạy thêm run cho Stage 2 (n = 16–35 với 2 run là không đủ kết luận).
3. Xuất tập residual 42 case ra file riêng + phân tích định tính (đây là đóng góp dataset).
4. Thêm metric *false-alarm rate* trên nhóm `gt_status = Accepted`.
5. Bổ sung mẫu TLE/MLE nếu muốn claim về "điểm mù độ phức tạp".
6. Bổ sung bảng chi phí (token in/out × đơn giá) để định lượng luận điểm "few-shot model nhỏ rẻ hơn zero-shot model lớn".
7. Cân nhắc bỏ hẳn Llama khỏi thiết kế lọc, hoặc thay bằng model tương đương Qwen — hiện nó không đóng góp thông tin.
