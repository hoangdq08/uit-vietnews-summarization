# Xác nhận A/B ViT5 trên 100 bài mới

## Thiết kế và kết quả

Ngày 21/09/2026. Thí nghiệm gồm 200 lượt inference CPU bằng `experiments/vit5_input_ab.py` trên 100 bài. ID 2,7,...,497 không trùng 20 ID thăm dò. Chốt trước 20 ID review 2,27,...,477 trong metadata. Cùng checkpoint revision và cấu hình với mẫu 20 bài, khác duy nhất input `_` so với space. Đây vẫn là mẫu theo quy tắc trong test Vietnews, không đại diện ngẫu nhiên toàn corpus, không dùng để tuning.

| Metric | A giữ `_` | B space |
|---|---:|---:|
| ROUGE-1 F1 | 0.3007 | 0.3243 |
| ROUGE-2 F1 | 0.1708 | 0.1983 |
| ROUGE-L F1 | 0.2466 | 0.2719 |
| Input token trung bình | 836.40 | 599.26 |
| Bài bị cắt input | 28 | 7 |

R1 B tăng trên 54 bài, hòa trên 2 bài, giảm trên 44 bài. Mức tăng nhỏ hơn mẫu 20 bài, không suy rộng mức tăng mẫu 20 bài. Đã kiểm tra đủ 200 output, ID không trùng mẫu 20 bài, hashes input, gold exact, prediction không rỗng và tính lại ba độ đo và điểm trung bình bằng scorer project.

Artifact: `results/vit5_input_ab_confirm100/{metadata,predictions,summary}.json`. Không ghi đè baseline.

Kiểm tra tái lập: prediction A khớp chính xác lịch sử `preds_500.json` 100/100 bài. Trong 72 bài không bị cắt ở cả hai nhánh, R1 0.2681 → 0.2926, R2 0.1335 → 0.1606, RL 0.2191 → 0.2367. Vì vậy mức tăng overlap không chỉ xuất hiện ở nhóm giảm truncation. Không suy từ kết quả này rằng factuality tốt hơn.

## Review factuality: 20 bài chốt trước

Đánh giá sơ bộ bởi một AI reviewer, không blind, không thay thế hai người chấm độc lập. Đọc toàn body/gold và hai output. Rubric: **F** = có chi tiết sai hoặc không được body hỗ trợ rõ, **U** = câu hỏng/mơ hồ hoặc nguồn không nhất quán cần người đọc xác nhận, **S** = chưa thấy lỗi thực tế rõ trong review này. S không có nghĩa đầy đủ ý hoặc đã xác minh thực tế ngoài bài báo. Bỏ sót ý ghi riêng, không tự coi omission là hallucination. Mỗi ID dẫn tới `data/test_tokenized/<ID>.txt.seg` và hai record cùng ID trong predictions.json.

| ID | A | B | Bằng chứng và hạn chế |
|---|---|---|---|
| 000002 | S | S | Body nói không có dâm ô và gia đình bảo lãnh. B diễn đạt đề xuất bảo lãnh rõ hơn; cả hai khác trọng tâm gold về hành khách giữ người. |
| 000027 | F | U | A thêm “tử vong” dù body nói cấp cứu/điều trị. B bỏ tử vong nhưng gọi cả hai “đi nhận vó bò”, trong body H nhận còn T giao; “nguy kịch” mạnh hơn “bị thương rất nặng”. |
| 000052 | F | F | A gọi thiếu nữ là người đâm chết nạn nhân. B đảo vai thành thiếu niên nhắc vượt đèn và gia đình tới nhà thiếu nữ. Body nói thiếu nữ tới nhà gia đình nạn nhân. |
| 000077 | S | F | A nói vụ cướp và thông tin thủ phạm, phù hợp nhưng ít ý. B trộn vụ chồng giết vợ ở cuối bài với vụ cướp, thêm tự tử và nói đã bắt được trong khi đoạn liên quan vẫn truy tìm. |
| 000102 | U | S | A viết “truy tố” thay “đề nghị truy tố”, sai khác trạng thái tố tụng cần lưu ý. B giữ đúng đề nghị. Cả hai bỏ trọng tâm gold về Lê Hoàng Quân, body có thông tin đó ở cuối. |
| 000127 | U | S | A lặp tin nhảy lầu, “Truy vấn nguyên nhân” không tự nhiên và bỏ vụ truy tố xâm hại. B liệt kê đủ bốn tin như body/gold. |
| 000152 | F | S | A biến “tập văn nghệ” thành “tập văn dùng bút”, sai hoạt động. B giữ ba tiêu đề đúng. |
| 000177 | U | U | A có thể bị hiểu thành ba nghi phạm là người phát hiện. B gán cụ thể hành vi vứt/giấu cho nghi phạm, nguồn mới nói bắt để điều tra liên quan. Cần giữ mức bất định tố tụng. |
| 000202 | F | F | A bóp méo phủ nhận giao dịch100 triệu thành câu “sẵn sàng chi”. B nói đòi tiền tài xế đưa bệnh nhân vào, nguồn nói đón bệnh nhân về/ra. Gold có1 triệu nhưng body không nêu số đó. |
| 000227 | S | U | A đúng việc bỏ trốn nhưng thiếu tin bắt lại. B thêm “hơn4 ngày” trong khi chỉ biết bỏ trốn26/5, bắt2h30/5, không có giờ bỏ trốn để khẳng định hơn4 ngày. |
| 000252 | S | S | 4.280 bánh kẹo và120 chân gà được body hỗ trợ, gold tách3.080 bánh+1.200 kẹo. Hai cách tổng hợp không mâu thuẫn. |
| 000277 | F | S | A sinh Trần Minh/Nguyễn Thị Ánh Tuyết thay Hồ Thị Ánh Tuyết. B chỉ dùng Tuyết đúng người, làm tròn “trên1,3 tỷ” thành1,3 tỷ nên mất sắc thái cận dưới. |
| 000302 | U | S | A lặp “Thị” trong tên, cần sửa tên nhưng không đủ căn cứ kết luận đổi người. B đúng diễn biến đâm chết chủ quán, bỏ chi tiết cướp. Gold nói dây chuyền vàng trong khi body chỉ nói nhiều tài sản. |
| 000327 | S | S | A nhận tội trộm đúng nhưng bỏ cái chết. B mô tả được phát hiện trên sàn ôm dây điện, không bịa nguyên nhân chết, nhưng chưa nói tử vong. |
| 000352 | S | S | Cả hai đúng việc bị đâm ở quán cà phê, đều bỏ kết quả tử vong có trong body/gold. |
| 000377 | F | F | A thêm không phải trường hợp đầu tiên ở Mỹ, body không hỗ trợ. B đổi3 người thành3 phụ nữ, gán46 tuổi của nghi phạm cho nạn nhân và biến ngày tìm thi thể thành ngày ra tòa. |
| 000402 | U | F | A thêm xôn xao dư luận và chủ thể hành động tát mơ hồ. B nói cô giáo tát lại, body quy định học sinh N tát lại người tát nhẹ/thiếu. |
| 000427 | S | S | A đúng bắt tạm giam người bị tố ép tình nhân, bỏ lý do đánh bạc. B đúng lý do đánh bạc nhưng lặp bắt/tạm giam và bỏ bối cảnh tố cáo. |
| 000452 | U | S | A nói chết tại chỗ, body có mở đầu phát hiện tử vong nhưng đoạn sau nói chết sau cấp cứu. Nguồn tự thiếu nhất quán, không dùng case này chứng nhận A đúng. B không thêm “tại chỗ”. |
| 000477 | S | S | Cả hai bám câu cuối body về bị đe dọa nên không báo gia đình, nhưng bỏ nội dung xâm hại và4.000 đồng trong gold. |

Không công bố tỷ lệ factuality như ground truth: rubric đã dùng sau khi generation bắt đầu, chấm một lượt bằng AI, vài case U cần đối chiếu người thật. Danh sách 20 ID được chốt trước output, nhưng review không blind. Các lỗi nghiêm trọng B ở 000077,000377,000402 đủ để bác bỏ kết luận “B luôn tốt hơn”.

## Kết luận và hạn chế

Nhánh B tăng ROUGE trung bình và giảm số bài bị cắt đầu vào. Một số output vẫn sai hoặc gán nhầm sự kiện, nên kết quả không chứng minh factuality tốt hơn. Pipeline chính giữ input có dấu gạch dưới; demo hiển thị cả hai nhánh để đối chiếu.

Chưa kiểm định thống kê và chưa có đánh giá độc lập bởi hai người trên toàn bộ mẫu. Danh sách lỗi sơ bộ cần được người chấm xác nhận trước khi dùng như nhãn chuẩn.
