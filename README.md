# Hệ Thống Đánh Giá & Phân Tích ESG (VNSI 2025)

Hệ thống tự động hóa trích xuất thông tin từ Báo cáo Thường niên, Báo cáo Tài chính và Báo cáo Phát triển Bền vững của các doanh nghiệp Việt Nam để chấm điểm ESG dựa trên **bộ quy tắc VNSI 2025** của Sở Giao dịch Chứng khoán TP.HCM (HOSE).

Hệ thống sử dụng **RAG (Retrieval-Augmented Generation)** kết hợp sức mạnh suy luận của LLM cục bộ (ví dụ: `Qwen3:30b`) để phân tích ngữ cảnh, đảm bảo tính khách quan và tự động hóa cao. Hệ thống được tinh chỉnh (fine-tuned) logic cực kỳ chặt chẽ, đảm bảo đối chiếu khắt khe với các quy định "phạt thiếu thông tin công bố" (-1 điểm) của VNSI.

---

## 🚀 Tính năng nổi bật
1. **Xử lý PDF toàn diện (Hybrid OCR):** Tự động phát hiện file PDF chứa mã hóa font lỗi (Garbage text) và chuyển sang dùng `EasyOCR` hoặc `pytesseract` để quét lại dữ liệu.
2. **Cơ chế Scoring Engine bám sát VNSI:** 
   - Không còn tình trạng "nuốt điểm âm" (shielding penalties).
   - Tự động nhận diện và tính toán các câu có luật "cộng dồn điểm ẩn" hoặc "chọn nhiều đáp án".
3. **Phân tích ngữ cảnh (Semantic Extraction):** Khả năng lọc từ khóa rủi ro (risk keywords), phát hiện chứng chỉ ISO, và đánh giá tính phù hợp của dữ liệu trước khi đưa vào LLM.
4. **Caching siêu tốc:** Tự động cache kết quả OCR và trích xuất PDF ở mức đoạn văn để tiết kiệm tài nguyên cho các lần chạy sau.

---

## 🛠 Hướng dẫn Cài đặt & Chạy dự án (Dành cho Team)

### 1. Yêu cầu hệ thống
- Python 3.9+
- Ollama (hoặc nền tảng LLM cục bộ khác) đang chạy mô hình (mặc định: `qwen2.5:32b` hoặc `qwen3:30b`).
- Hệ điều hành: Windows/macOS/Linux.

### 2. Cài đặt thư viện (Dependencies)
Clone kho lưu trữ và cài đặt môi trường ảo:

```bash
# Clone dự án
git clone https://github.com/KhoiPhanT/ESG_LLM_SCORING.git
cd ESG_LLM_SCORING

# Tạo và kích hoạt môi trường ảo (venv)
python3 -m venv venv
source venv/bin/activate  # macOS/Linux
# venv\Scripts\activate   # Windows

# Cài đặt toàn bộ thư viện cần thiết
pip install -r requirements.txt
```

*(Lưu ý: Thư viện `easyocr` có thể yêu cầu cài đặt thêm PyTorch phiên bản hỗ trợ GPU nếu bạn muốn tăng tốc quá trình quét ảnh).*

### 3. Chuẩn bị Dữ liệu (Inputs)
Cấu trúc thư mục dữ liệu đầu vào bắt buộc:
```text
inputs/
├── 20250506 - VNSI - Bo cau hoi PTBV 2025.xlsx   <-- File chứa bộ rule
└── ACB/
    └── reports/
        ├── ACB_Baocaothuongnien_2024.pdf         <-- Báo cáo của doanh nghiệp
        ├── ACB_Baocaotaichinh_2024.pdf
        └── ACB_Nghiquyet_DHDCD.pdf
```
*Bạn có thể tạo các thư mục doanh nghiệp khác (như `VNM/reports/`, `FPT/reports/`) tương tự.*

### 4. Cách chạy (Execution)
Chạy pipeline chính bằng lệnh sau:
```bash
# Cú pháp: python main.py <đường_dẫn_chứa_pdf> <tên_công_ty> <ngành_nghề> <năm>
python main.py inputs/ACB/reports/ ACB Financials 2024
```

Sau khi chạy xong, kết quả sẽ được lưu vào:
- `outputs/reports/ACB_2024_esg_report.json` (Báo cáo JSON chi tiết điểm số)
- `outputs/vnsi_rules.json` (Bản phân tích rule VNSI đã parse từ Excel)
- `outputs/cache/` (Dữ liệu text được lưu lại)

---

## 📚 Cấu trúc Tài liệu
Các tài liệu phân tích kỹ thuật và định hướng nâng cấp hệ thống nằm ở mục `docs/manual`:
- `1_cau_truc_du_an.md`: Kiến trúc các module cốt lõi (Core packages).
- `2_luong_hoat_dong.md`: Workflow xử lý dữ liệu.
- `4_huong_phat_trien.md`: Roadmap tương lai (Semantic Search, Agentic RAG, Table Extraction).

---

## 🤝 Lưu ý dành cho lập trình viên (Developer Notes)
- Code logic chấm điểm nằm tại `core/scoring/scoring_engine.py`. Bất kỳ thay đổi nào liên quan đến cách quy đổi điểm số từ A/B/C/D phải sửa tại đây.
- Prompt điều khiển LLM nằm tại `core/llm_client.py`. Hiện tại LLM bị ép buộc **phải chọn đáp án** dựa trên ngữ cảnh, không được phép trả `NULL`.
- Nếu có lỗi liên quan đến quá trình đọc PDF, vui lòng kiểm tra bộ bắt lỗi Garbage Text tại `core/ingestion/pdf_parser.py`.