# Hạn Chế Hiện Tại Và Hướng Phát Triển

Pipeline hiện tại đã tốt hơn đáng kể ở 3 điểm: chống lỗi JSON của LLM, giảm `NULL` oan bằng retrieval-grounded fallback, và tính điểm tổng đúng theo raw workbook. Tuy nhiên vẫn còn các giới hạn cần nhìn thẳng.

## 1. Chất Lượng Retrieval Vẫn Là Nút Thắt Chính

### Hiện trạng

- Khi retrieval không kéo đúng section, cả LLM lẫn fallback đều yếu.
- Các câu Governance và policy chi tiết vẫn dễ bị thiếu bằng chứng nếu dossier tài liệu chưa đủ.
- Câu hỏi có wording đặc thù ngành vẫn có thể bị bỏ sót dù đã có metadata routing.

### Hướng phát triển

1. Nâng chất lượng semantic retrieval và reranking.
2. Tăng coverage cho metadata question taxonomy.
3. Theo dõi chặt các bucket có `retrieval_weak` trong diagnostics sau mỗi lần chạy lớn.
4. Thiết lập benchmark retrieval theo từng nhóm câu để tránh tối ưu mù.

## 2. Citation Ở Markdown Chưa Phản Ánh Hết Dữ Liệu Gốc

### Hiện trạng

- `esg_report.json` và `scoring_audit.csv` có thể đã có `file + page`.
- Nhưng markdown clean report hiện ưu tiên `evidence_items`; nếu evidence item rỗng, người đọc có thể tưởng là không có nguồn dù `top_source_refs` hoặc `source_sections` vẫn tồn tại.

### Hướng phát triển

1. Cải thiện `clean_reporter.py` để luôn hiển thị nguồn tham chiếu ngắn khi có `top_source_refs`.
2. Bổ sung chế độ xuất report “audit-first” cho người rà soát thủ công.
3. Tách rõ “quote evidence” và “reference-only support” trong report cuối.

## 3. Numeric/Table Questions Vẫn Cần Tăng Độ Bền

### Hiện trạng

- OCR bảng phức tạp vẫn có thể làm vỡ cấu trúc.
- `NumericExtractor` đã giúp được một phần, nhưng chưa thay thế hoàn toàn table parsing chuẩn.

### Hướng phát triển

1. Tăng khả năng trích số liệu có cấu trúc.
2. Tách riêng pipeline cho bảng biểu và ratio questions.
3. So khớp đơn vị, niên độ và ngữ cảnh số liệu chặt hơn.
4. Tạo regression set riêng cho numeric disclosure và ratio calculation.

## 4. Fallback Retrieval Cần Tiếp Tục Được Kiểm Soát

### Hiện trạng

- Retrieval fallback hiện nay hữu ích để cứu các case LLM rỗng/think-only.
- Nhưng với câu ambiguous, negative option, hoặc policy rất tinh vi, fallback vẫn phải bảo thủ để tránh chấm oan.

### Hướng phát triển

1. Mở rộng taxonomy test cho từng loại câu hỏi.
2. Gắn thêm confidence/audit marker để tách câu “điểm đáng tin” và câu “cần rà thủ công”.
3. Bổ sung replay set từ các ca fail thực tế.
4. Tách policy an toàn khác nhau cho:
   - positive single-select
   - negative single-select
   - multi-select
   - historical-allowed
   - current-year-required

## 5. Khả Năng Vận Hành Dài Hơi

### Hiện trạng

- Hệ thống đã có progress autosave và resume.
- Nhưng report tổng vẫn chỉ ghi ở cuối job; nếu dừng giữa chừng thì người dùng phải đọc progress cache hoặc audit trung gian.

### Hướng phát triển

1. Sinh `partial_report.json` trong lúc chạy.
2. Thêm dashboard hoặc CLI audit summary cho mỗi lần resume.
3. Chuẩn hóa workflow kiểm thử full dossier sau mỗi thay đổi lớn.
4. Bổ sung cơ chế checkpoint cho cả report layer, không chỉ scoring progress.

## 6. Kiểm Thử Và Quan Sát

### Hiện trạng

- Đã có regression test cho parse/fallback/scoring quan trọng.
- Nhưng độ tin cậy production còn phụ thuộc vào replay trên nhiều dossier thực.

### Hướng phát triển

1. Xây thêm bộ replay test đại diện theo bucket câu hỏi.
2. So sánh score delta giữa các phiên bản code.
3. Theo dõi các chỉ số:
   - tỷ lệ `NULL`,
   - tỷ lệ `llm_empty_failure`,
   - tỷ lệ `retrieval_fallback_*`,
   - số câu `weakly_supported`,
   - tỷ lệ câu có citation đầy đủ.
4. Thiết lập acceptance gate tối thiểu trước khi coi một phiên bản là đủ tốt để chạy diện rộng.

## 7. Những Điều Không Nên Giả Định

Khi vận hành dự án này, không nên mặc định rằng:

- LLM trả answer là đủ để chấm điểm.
- `reason` đẹp nghĩa là evidence đúng.
- markdown clean report luôn phản ánh hết citation gốc.
- preflight cache hit nghĩa là scoring sẽ tốt.
- historical evidence luôn hợp lệ; nhiều câu vẫn bắt buộc current year.

Hệ thống hiện tại đã giảm đáng kể các lỗi này, nhưng chưa loại bỏ hoàn toàn.

## 8. Mục Tiêu Kỹ Thuật Hợp Lý Trong Giai Đoạn Tiếp Theo

Thay vì đặt mục tiêu mơ hồ như “không còn lỗi”, nên theo đuổi các mục tiêu vận hành đo được:

1. Giảm rõ rệt số câu `NULL` oan.
2. Tăng tỷ lệ câu có citation file/trang rõ ràng trong report JSON.
3. Giảm số câu `weakly_supported` nhưng vẫn đang được người dùng kỳ vọng có điểm.
4. Làm cho markdown report phản ánh citation gần hơn với JSON report.
5. Rút ngắn thời gian điều tra khi có fail nhờ debug artifact nhất quán.
