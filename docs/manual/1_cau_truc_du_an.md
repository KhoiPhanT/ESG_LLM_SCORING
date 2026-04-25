# Cấu Trúc Dự Án Đánh Giá ESG (v2)

Dự án được tổ chức theo mô hình module hóa (modular), giúp phân tách rõ ràng trách nhiệm của từng phần, dễ dàng bảo trì và mở rộng trong tương lai.

## Cây Thư Mục Tổng Quan

```text
esg/
│
├── core/                       # 🧠 Lõi xử lý chính của hệ thống
│   ├── ingestion/              # Thu thập và tiền xử lý dữ liệu đầu vào
│   │   ├── excel_parser.py     # Đọc bộ rule VNSI, trọng số ngành từ Excel
│   │   ├── pdf_parser.py       # Trích xuất text từ PDF, tự động OCR nếu là ảnh scan
│   │   └── text_cleaner.py     # Làm sạch văn bản, loại bỏ stopwords với NLTK
│   │
│   ├── extraction/             # Trích xuất thông tin
│   │   └── keyword_matcher.py  # Đếm từ khóa rủi ro và nhận diện chứng chỉ ISO
│   │
│   ├── scoring/                # Chấm điểm ESG
│   │   ├── screening.py        # Đánh giá 5 câu hỏi sàng lọc (Total Kill Switch)
│   │   └── vnsi_scorer.py      # Chấm điểm 82 câu VNSI với LLM, áp dụng trọng số ngành
│   │
│   ├── analytics/              # Phân tích dữ liệu & Chỉ số rủi ro
│   │   ├── risk_index.py       # Tính toán chỉ số ESGUI
│   │   ├── wui_loader.py       # Đọc dữ liệu World Uncertainty Index
│   │   └── causal_inference.py # Phân tích nhân quả PLS-SEM (ESG -> Hiệu quả tài chính)
│   │
│   └── llm_client.py           # Client giao tiếp với Qwen3:30b qua Ollama REST API
│
├── inputs/                     # 📂 Chứa dữ liệu đầu vào
│   ├── ACB/reports/            # Các file PDF báo cáo thường niên
│   ├── 20250506 - VNSI...xlsx  # Bộ câu hỏi và trọng số VNSI gốc
│   └── ACB/chỉ số.../Data.xlsx # Dữ liệu tài chính cho mô hình PLS-SEM
│
├── docs/                       # 📚 Tài liệu dự án
│   ├── manual/                 # Hướng dẫn sử dụng & Luồng hoạt động (bạn đang ở đây)
│   └── WUI.xlsx                # Dữ liệu chỉ số bất ổn (World Uncertainty Index)
│
├── outputs/                    # 💾 Dữ liệu sinh ra trong quá trình chạy
│   ├── cache/                  # Cache kết quả OCR để tăng tốc độ chạy lại
│   ├── reports/                # File JSON chứa báo cáo kết quả đánh giá (Đầu ra cuối cùng)
│   └── *.json                  # Các file rules, weights được parse từ Excel
│
└── main.py                     # 🚀 Script khởi chạy toàn bộ pipeline (Orchestrator)
```

## Giải Thích Các Module Cốt Lõi

1. **`llm_client.py`**: Trái tim của hệ thống. Nó gửi prompt tiếng Việt (kèm theo context từ báo cáo) cho mô hình `Qwen3:30b` đang chạy cục bộ qua Ollama và ép mô hình trả về định dạng JSON thuần túy (`{"answer": "...", "reason": "...", "evidence": "..."}`). Có cơ chế tự động thử lại (auto-retry) nếu JSON bị lỗi.
2. **`pdf_parser.py`**: Xử lý vấn đề hóc búa nhất của dữ liệu Việt Nam là báo cáo file scan. Nó dùng `PyMuPDF` để quét text trước. Nếu phát hiện số trang chứa text dưới 30%, nó tự động kích hoạt `pytesseract` để chuyển đổi toàn bộ PDF thành văn bản và lưu lại vào thư mục `cache/`.
3. **`vnsi_scorer.py`**: Động cơ chấm điểm. Không chỉ hỏi LLM, nó còn dịch ngược đáp án (A, B, C) thành điểm số (vd: +1, 0, -1) dựa theo đúng logic được quy định trong file Excel VNSI, sau đó nhóm điểm lại theo các trụ cột E, S, G và nhân với trọng số của từng ngành (ví dụ Ngân hàng thì Quản trị - G chiếm đến 60% tổng điểm).
