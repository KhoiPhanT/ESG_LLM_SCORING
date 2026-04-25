# Hướng Dẫn Sử Dụng (User Guide)

Tài liệu này hướng dẫn bạn cách khởi chạy hệ thống, xử lý các lỗi thường gặp và cách đọc hiểu báo cáo đầu ra.

## 1. Yêu cầu Hệ Thống (Prerequisites)

Trước khi chạy, đảm bảo máy bạn (Macbook) đã được cài đặt các phần mềm sau:
- **Ollama**: Đang mở và đã tải model `qwen3:30b`. (Kiểm tra bằng lệnh `ollama list` ở terminal).
- **Tesseract OCR & Poppler**: Dùng để xử lý các file PDF dạng hình ảnh.
- Môi trường Python (venv) đã được kích hoạt và cài đủ thư viện (pandas, fitz, pytesseract, pdf2image, semopy, nltk).

## 2. Cách Khởi Chạy Pipeline

Để chấm điểm ESG cho một báo cáo bất kỳ, bạn mở Terminal, trỏ vào thư mục dự án và gõ lệnh sau:

```bash
# 1. Kích hoạt môi trường ảo (nếu chưa kích hoạt)
source venv/bin/activate

# 2. Chạy file main.py với các tham số: 
# python3 main.py <đường_dẫn_file_pdf> <tên_công_ty> <ngành_gics> <năm>
python3 main.py inputs/ACB/reports/ACB_Baocaothuongnien_2024.pdf ACB Financials 2024
```

### Giải thích các Tham số (Arguments):
- `inputs/.../ACB_Baocaothuongnien_2024.pdf`: Đường dẫn tới file PDF báo cáo thường niên bạn muốn chấm.
- `ACB`: Tên viết tắt của công ty (dùng để đặt tên file báo cáo đầu ra).
- `Financials`: Ngành nghề theo chuẩn GICS (VD: Financials, Real Estate, Consumer Staples...). **Rất quan trọng**, vì nó quyết định việc áp dụng trọng số ESG (VD Financials thì Quản trị chiếm 60% tổng điểm).
- `2024`: Năm của báo cáo. Dùng để hệ thống mapping với chỉ số WUI của năm tương ứng.

## 3. Quá trình Chạy (Runtime)

- **Lần đầu tiên với file scan**: Nếu báo cáo của bạn là file hình ảnh (như ACB 2024), lần chạy đầu tiên hệ thống sẽ mất khoảng 5-10 phút để chạy OCR toàn bộ 80-100 trang.
- **Từ lần chạy thứ 2**: Hệ thống sẽ tự động sử dụng file text đã lưu trong `outputs/cache/`, bỏ qua bước OCR nên sẽ cực kỳ nhanh.
- **Thời gian chấm của LLM**: Việc gửi 87 câu hỏi (5 câu sàng lọc + 82 câu VNSI) tới LLM sẽ mất khoảng 15-20 phút tùy thuộc vào tốc độ sinh token của model trên máy bạn. Bạn cứ để máy chạy và có thể theo dõi tiến độ (`[1/113]`, `[2/113]`) trực tiếp trên màn hình.

## 4. Đọc Báo Cáo Kết Quả

Sau khi hoàn thành, hệ thống sinh ra một file JSON tại thư mục `outputs/reports/` (Ví dụ: `ACB_2024_esg_report.json`).

Bạn có thể mở file này bằng VS Code. Trong file có các block thông tin chính:

### a) Điểm Tổng & Trọng số (`scores`)
Cho biết điểm tổng hợp trên thang 100 và điểm chi tiết từng trụ cột E, S, G.
```json
"scores": {
  "E": 78.47,
  "S": 77.12,
  "G": 99.32,
  "total": 85.29,
  ...
}
```

### b) Chỉ số Rủi ro (`esgui`)
Cho biết chỉ số WUI được dùng và ESGUI Index cuối cùng. Kèm theo dòng chữ diễn giải mức độ rủi ro dễ hiểu.

### c) Chi tiết Chấm Điểm (`scoring_details`)
Đây là phần giá trị nhất. Nó liệt kê đầy đủ 82 câu hỏi VNSI, ví dụ:
```json
{
  "id": "E.1.1.1",
  "question": "Công ty có chính sách liên quan tới quản lý các tác động môi trường còn hiệu lực hay không?",
  "answer": "C",
  "score": 1.0,
  "reason": "ACB đã xây dựng và công bố chính sách bảo vệ môi trường công khai...",
  "evidence": "Ngân hàng đang thúc đẩy chuyển đổi xanh, tuân thủ các cam kết..."
}
```
Thông qua phần `evidence` (Dẫn chứng) và `reason` (Lý do), bạn có thể kiểm tra chéo (cross-check) xem LLM có bị "ảo giác" (hallucination) hay không, hoặc dùng đoạn evidence đó để copy trực tiếp vào bản báo cáo cuối khóa của mình.

## 5. Các Lỗi Thường Gặp (Troubleshooting)

| Lỗi (Error) | Nguyên nhân | Cách khắc phục |
|-------------|------------|----------------|
| `ConnectionError: Max retries exceeded with url: /api/chat` | Ollama chưa được bật | Mở ứng dụng Ollama trên Mac hoặc gõ `ollama serve` ở terminal khác. |
| `FileNotFoundError: tesseract is not installed` | Thiếu công cụ OCR | Chạy lệnh `brew install tesseract` |
| Điểm ESGUI báo `NaN` hoặc lỗi | File WUI.xlsx bị sai cấu trúc hoặc không có dữ liệu năm đó | Kiểm tra lại file `docs/WUI.xlsx`, đảm bảo năm bạn nhập vào khi chạy lệnh có tồn tại trong dữ liệu WUI. |
| Lỗi thiếu thư viện Python | Chưa active môi trường ảo | Chạy `source venv/bin/activate` và thử lại. |
