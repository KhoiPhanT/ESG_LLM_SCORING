# Hạn Chế Của Hệ Thống Và Định Hướng Phát Triển (Future Works)

Dự án hiện tại đã xây dựng thành công một pipeline hoàn chỉnh, tự động hóa từ khâu đọc PDF (OCR), trích xuất thông tin bằng LLM (Qwen3:30b) cho đến phân tích nhân quả (PLS-SEM). Điểm sáng của hệ thống là khả năng bám sát 100% logic khắt khe của bộ quy chuẩn VNSI 2025, phạt điểm (-1) một cách sòng phẳng và khách quan khi doanh nghiệp thiếu sót thông tin công bố.

Tuy nhiên, để đưa hệ thống này vào môi trường sản xuất thực tế (Production) với độ chính xác tuyệt đối, tránh hiện tượng "phạt oan" do lỗi kỹ thuật hoặc hạn chế của dữ liệu đầu vào, chúng ta cần nhìn nhận các hạn chế và hướng đi tiếp theo.

## 1. Mở Rộng Nguồn Dữ Liệu Đầu Vào (Data Collection)

### Vấn đề hiện tại
Bộ quy chuẩn VNSI yêu cầu đánh giá rất sâu về quản trị (Governance) và xã hội (Social) (ví dụ: giao dịch bên liên quan, tính độc lập của HĐQT, cơ chế khiếu nại của người lao động). Nếu chỉ nạp Báo cáo Thường niên và Báo cáo Tài chính, hệ thống sẽ thiếu hụt dữ liệu trầm trọng do các doanh nghiệp Việt Nam thường không công bố đủ các chính sách này trong BCTN. Theo quy chuẩn hiện hành, không tìm thấy bằng chứng công khai đồng nghĩa với việc bị trừ điểm (-1).

### Giải pháp đề xuất
Cần xây dựng pipeline tự động thu thập thêm các tài liệu đại chúng sau từ website doanh nghiệp hoặc Ủy ban Chứng khoán:
- **Báo cáo Phát triển Bền vững (Sustainability/ESG Report):** Nguồn dữ liệu cốt lõi cho Pillar E và S.
- **Điều lệ công ty (Company Charter):** Trả lời các câu hỏi về cơ cấu và quyền hạn HĐQT, quyền cổ đông thiểu số.
- **Quy chế Quản trị Công ty (Corporate Governance Regulations):** Giải quyết triệt để các câu về giao dịch với bên liên quan.
- **Bộ Quy tắc Ứng xử / Đạo đức kinh doanh (Code of Conduct):** Bằng chứng cho các chính sách nhân quyền, lao động, chống tham nhũng.
- **Báo cáo Tình hình Quản trị Công ty (6 tháng/1 năm):** Minh bạch về thù lao và các cuộc họp HĐQT.

## 2. Nâng Cấp Hệ Thống RAG (Retrieval-Augmented Generation)

### Vấn đề hiện tại
Hệ thống hiện tại (Dù đã có cải tiến với top_k=6 và 15,000 ký tự) vẫn gặp rủi ro "Điểm mù" (Blind Spots) và hiệu ứng "Phạt kép" (Double Penalty):
- **Từ đồng nghĩa & Đặc thù ngành:** Câu hỏi yêu cầu "đánh giá tác động môi trường", nhưng ngân hàng dùng từ "quản lý rủi ro tín dụng xanh". Keyword search sẽ bỏ sót. RAG sót thông tin dẫn đến LLM mặc định là doanh nghiệp vi phạm và chấm -1.
- **Gộp chung tài liệu:** RAG tìm kiếm trên toàn bộ corpus, dễ dẫn đến việc lấy nhầm ngữ cảnh không liên quan.

### Giải pháp đề xuất
1. **Semantic Search & Vector Database:** Sử dụng các mô hình nhúng (ví dụ: `BGE-m3`) kết hợp `ChromaDB`. Quan trọng nhất là xây dựng **Từ điển đồng nghĩa ESG (Semantic Dictionary)** riêng cho từng ngành (ví dụ ngành Tài chính: "môi trường" = "tín dụng xanh", "phát thải").
2. **Metadata & Router Filtering (Phân luồng tìm kiếm):** 
   - Khi LLM chấm câu Quản trị (G) → Tăng trọng số tìm kiếm trong Điều lệ và Quy chế.
   - Khi chấm câu Môi trường (E) → Ưu tiên Báo cáo PTBV.
3. **Mở rộng Context Window:** Nâng giới hạn trích xuất lên top_k=15 hoặc 40,000 ký tự để tận dụng sức mạnh xử lý context dài của Qwen3:30b, giảm thiểu tối đa hiện tượng chặt đứt đoạn văn (chunking fragmentation).

## 3. Xử lý Bảng biểu (Table Extraction)

**Vấn đề hiện tại:** Mặc dù `pytesseract` đọc chữ từ ảnh scan cực tốt, nhưng khi gặp bảng biểu cấu trúc phức tạp (cơ cấu nhân sự, lượng phát thải qua các năm), OCR thường làm vỡ hàng/cột, khiến LLM mất khả năng đối chiếu số liệu.

**Giải pháp đề xuất:**
Tích hợp thêm các mô hình chuyên đọc bảng biểu như `LlamaParse`, `Table Transformer` hoặc `Camelot` để bóc tách cấu trúc bảng thành định dạng JSON/Markdown chuẩn trước khi đưa vào RAG.

## 4. Đánh giá Đa tác tử (Multi-Agent System)

**Vấn đề hiện tại:** Hệ thống đang phụ thuộc vào 1 lượt prompt duy nhất của Qwen3 để vừa đọc ngữ cảnh vừa ra quyết định. Nếu LLM thiếu kiến thức ngành, nó sẽ đưa ra lý lẽ từ chối cứng nhắc (Ví dụ: từ chối công nhận tín dụng xanh là hoạt động môi trường cốt lõi của ngân hàng).

**Giải pháp đề xuất:** 
Thiết kế hệ thống đa tác tử (Agentic RAG):
- **Agent 1 (Researcher):** Lập luận nhiều bước (Multi-hop). "Tôi cần tìm chính sách nhân sự. Tôi thấy quy chế lương, tôi sẽ tìm tiếp quy chế thai sản...".
- **Agent 2 (Scorer):** Chấm điểm dựa trên bằng chứng của Agent 1.
- **Agent 3 (Reviewer):** Đóng vai trò "Kiểm toán viên ngành". Được cung cấp riêng prompt về đặc thù ngành (Domain Knowledge) để phản biện: *"Đối với ngân hàng, tín dụng xanh chính là quản lý môi trường. Hãy công nhận điểm này"*. Kiến trúc này sẽ giúp điểm số công bằng và chống ảo giác (hallucination).
