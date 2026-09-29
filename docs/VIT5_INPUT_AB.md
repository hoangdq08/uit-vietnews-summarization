# Ảnh hưởng của dấu gạch dưới trong đầu vào ViT5

## Thiết kế và tái lập

So sánh hai cách biểu diễn cùng thân bài: A giữ dấu `_` của corpus đã tách từ; B chỉ thay `_` bằng khoảng trắng. Cả hai thêm `</s>`, giới hạn input 1.024 token và output `max_length=256`, dùng greedy (`num_beams=1`, `do_sample=False`), CPU float32, seed 0. `early_stopping=True` được giữ để khớp cấu hình baseline, nhưng không có tác dụng với greedy.

Checkpoint revision: `a54febd011c0a39ce49ceacd9225e63ef3a73ad3`. Phiên bản thư viện, cấu hình và ID lưu trong `metadata.json` của từng thư mục kết quả. `predictions.json` lưu output, độ dài token, hash đầu vào và ROUGE từng bài. Scorer giống pipeline chính: thay `_` bằng khoảng trắng ở prediction và gold, rồi dùng whitespace ROUGE F1.

Ba mẫu gồm 20 bài thăm dò (ID 1, 26, …, 476), 100 bài xác nhận (ID 2, 7, …, 497, không trùng mẫu thăm dò) và toàn bộ 500 bài đầu. ID chọn theo quy tắc định trước, không theo điểm output. Mẫu 500 chứa cả hai mẫu nhỏ, không phải tập kiểm chứng độc lập. Đây là phân tích khám phá trên test, không dùng để tối ưu tham số.

```bash
python experiments/vit5_input_ab.py --limit 500 --output results/vit5_input_ab_new
```

Cần môi trường có `torch`, `transformers==4.44.2`, `sentencepiece` và `protobuf`, ngoài các thư viện đánh giá của project. Runner yêu cầu thư mục output mới để giữ nguyên kết quả đã có.

## Kết quả

| Mẫu | Nhánh | ROUGE-1 | ROUGE-2 | ROUGE-L | Token input trung bình trước cắt | Bài bị cắt |
|---|---|---:|---:|---:|---:|---:|
| 20 | A giữ `_` | 0.1766 | 0.0830 | 0.1388 | 790.4 | 6 |
| 20 | B khoảng trắng | 0.2944 | 0.1527 | 0.2294 | 566.9 | 2 |
| 100 | A giữ `_` | 0.3007 | 0.1708 | 0.2466 | 836.4 | 28 |
| 100 | B khoảng trắng | 0.3243 | 0.1983 | 0.2719 | 599.3 | 7 |
| 500 | A giữ `_` | 0.2784 | 0.1474 | 0.2233 | 785.1 | 120 |
| 500 | B khoảng trắng | 0.3124 | 0.1796 | 0.2560 | 563.3 | 38 |

Nguồn: `results/vit5_input_ab_20`, `results/vit5_input_ab_confirm100`, `results/vit5_input_ab_500`. Không so trực tiếp điểm của những mẫu có ID khác nhau. Nhánh A khớp prediction baseline trong `results/preds_500.json` trên các ID tương ứng.

Trong mẫu 20 bài, ROUGE-1 của B tăng trên 18 bài và giảm trên 2 bài. Trong mẫu xác nhận 100 bài, B tăng trên 54 bài, hòa trên 2 và giảm trên 44 bài. Với 72 bài xác nhận không bị cắt ở cả hai nhánh, ROUGE-1 vẫn tăng từ 0.2681 lên 0.2926. Hiệu ứng quan sát được không chỉ xuất hiện ở nhóm giảm truncation, nhưng chưa tách riêng đóng góp của tokenization và nội dung được giữ lại.

## Đối chiếu nội dung

Ở bài `000001`, ROUGE-1 tăng từ 0.1515 lên 0.5352 khi thay `_`. Nhánh B vẫn sinh tên, ngày và mức án không được thân bài hỗ trợ. Gold có một số chi tiết không xuất hiện đầy đủ trong body, nên phải phân biệt trùng gold với được body hỗ trợ.

Trong mẫu 100 bài xác nhận, rà soát sơ bộ 20 ID bằng AI phát hiện lỗi ở cả hai nhánh, gồm nhánh B trộn tin ở `000077`, đổi vai người và ngày ở `000377`, đảo người tát ở `000402`. Bảng nhận xét và giới hạn phương pháp nằm trong [VIT5_CONFIRM100.md](VIT5_CONFIRM100.md). Đây chưa phải bộ nhãn do hai người chấm độc lập.

## Kết luận

Thay `_` bằng khoảng trắng tăng ROUGE trung bình và giảm số bài bị cắt input trên các mẫu đã đo. Chưa kiểm định thống kê, chưa đánh giá factuality độc lập trên toàn mẫu, và chưa đủ căn cứ kết luận bản tóm tắt đúng sự thật hơn. Pipeline chính giữ đầu vào có dấu gạch dưới; demo hiển thị cả hai nhánh để đối chiếu.
