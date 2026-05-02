# Hướng Dẫn Sử Dụng

Tài liệu này mô tả cách chạy pipeline hiện tại, cách hiểu cache, và cách đọc đúng các file report/audit.

## 1. Yêu Cầu Hệ Thống

Trước khi chạy, cần có:

- **Ollama** đang hoạt động và đã có model phù hợp.
  - Dự án hiện thường dùng `qwen3:30b`.
- **Tesseract OCR** và các dependency đọc PDF scan.
- **Python environment** đã cài đủ các package của dự án.

## 1A. Ràng Buộc Dữ Liệu Đầu Vào

### 1. PDF và dossier tài liệu

- Đầu vào thực tế của pipeline là một file entry nằm trong dossier tài liệu của doanh nghiệp.
- PDF vẫn là định dạng mặc định cho:
  - báo cáo thường niên,
  - báo cáo phát triển bền vững,
  - báo cáo quản trị,
  - báo cáo tài chính,
  - điều lệ, quy chế, nghị quyết, tài liệu đại hội nếu có.
- Hệ thống hoạt động tốt nhất khi PDF có text layer sẵn.
- Với PDF scan, hệ thống vẫn xử lý được bằng OCR, nhưng:
  - chậm hơn,
  - dễ giảm chất lượng trích xuất bảng,
  - dễ làm retrieval và numeric extraction kém ổn định hơn.

### 2. Cách tổ chức folder tài liệu

- Nên gom tài liệu theo dossier công ty trong `inputs/`.
- Có thể giữ nhiều tài liệu nhiều năm trong cùng dossier nếu mục tiêu là chấm bộ câu hỏi có cho phép historical evidence.
- Không nên trộn tài liệu của nhiều công ty trong cùng dossier.
- Nếu muốn một run “sạch” cho một bộ tài liệu cụ thể, nên đảm bảo folder đó chỉ chứa tài liệu đúng doanh nghiệp mục tiêu.

### 3. Workbook/rules

- Workbook VNSI là nguồn rule gốc cho screening, scoring và metadata retrieval.
- Không nên tự ý đổi cấu trúc sheet/cột nếu chưa kiểm tra lại parser.
- Điểm tổng hiện tại không còn dùng trọng số pillar ngành, nhưng workbook vẫn là nguồn chân lý cho:
  - câu hỏi,
  - logic đáp án,
  - max score,
  - question type,
  - time policy và metadata liên quan.

### 4. Dữ liệu bảng và dữ liệu số

- Các câu numeric/ratio nhạy với chất lượng OCR hơn các câu policy text.
- Nếu có tài liệu scan nhiều bảng số liệu, nên kỳ vọng:
  - thời gian chạy tăng,
  - số câu numeric cần rà thủ công cao hơn,
  - citation vẫn có thể đúng file/trang nhưng answer numeric chưa chắc mạnh bằng policy questions.

## 2. Cách Chạy

Ví dụ chạy dossier `VNM - 2024 test` với context window `24576`:

```bash
source venv/bin/activate
ESG_OLLAMA_NUM_CTX=24576 ESG_OLLAMA_TIMEOUT=300 python3 main.py "inputs/VNM - 2024 test/VNM_Baocaothuongnien_2024.pdf" VNM "Consumer Staples" 2024
```

### Giải thích tham số

- `pdf_path`: file entry nằm trong dossier tài liệu.
- `company_name`: mã công ty, ví dụ `VNM`.
- `industry_sector`: tên ngành dùng trong rules và reporting.
- `year`: năm đánh giá mục tiêu.

### Lưu ý quan trọng về dossier tài liệu

Nếu file đầu vào nằm trong một folder dossier theo công ty, hệ thống có thể tự nạp toàn bộ tài liệu liên quan trong folder đó, không chỉ đúng một file PDF. Đây là hành vi chủ ý để hỗ trợ các câu hỏi cho phép dùng historical evidence.

## 3. Cache Và Resume

Hệ thống có nhiều lớp cache khác nhau:

- `outputs/cache/*_scoring_progress.json`
  - Lưu tiến độ chấm sau mỗi câu.
  - Dùng để resume nếu tiến trình bị dừng.

- `outputs/audit/*_retrieval_preflight.json`
  - Cache kết quả preflight retrieval.
  - Sẽ tự invalid nếu fingerprint bộ tài liệu thay đổi.

- OCR/text cache
  - Giúp tránh OCR lại ở các lần chạy sau.

### Khi nào cần chạy sạch hoàn toàn

Nếu muốn chạy lại từ đầu, bỏ qua cả preflight cache và progress resume:

```bash
ESG_OLLAMA_NUM_CTX=24576 ESG_OLLAMA_TIMEOUT=300 ESG_DISABLE_PREFLIGHT_CACHE=1 ESG_NO_RESUME_SCORING=1 python3 main.py "inputs/VNM - 2024 test/VNM_Baocaothuongnien_2024.pdf" VNM "Consumer Staples" 2024
```

## 4. Các File Đầu Ra

Sau khi chạy xong, các file quan trọng nhất là:

- `outputs/reports/<COMPANY>_<YEAR>_esg_report.json`
  - Báo cáo đầy đủ nhất.

- `outputs/reports/<COMPANY>_<YEAR>_esg_report_clean.md`
  - Bản markdown dễ đọc.

- `outputs/audit/<COMPANY>_<YEAR>_scoring_audit.csv`
  - Bảng audit nhanh theo từng câu.

- `outputs/audit/<COMPANY>_<YEAR>_scoring_audit.json`
  - Audit dạng JSON.

- `outputs/audit/<COMPANY>_<YEAR>_retrieval_preflight.json`
  - Kiểm tra trước về chất lượng retrieval.

- `outputs/debug/llm_json_errors/*.json`
  - Debug khi LLM trả rỗng, parse hỏng, hoặc JSON phải repair.

## 5. Cách Đọc Báo Cáo

### a) `scores`

Đây là phần điểm tổng hợp:

- `raw_total`: tổng điểm raw sau penalty
- `raw_max`: tổng điểm tối đa
- `raw_percentage`: `raw_total / raw_max * 100`
- `score_100`: bằng đúng `raw_percentage`

E/S/G chỉ còn là breakdown để hiển thị, không còn là trọng số dùng để tính điểm tổng.

Các field thường gặp:

- `E`, `S`, `G`: phần trăm theo từng pillar
- `total`: tổng điểm raw sau penalty
- `raw_total`: tương đương `total`
- `raw_max`: tổng điểm tối đa
- `raw_percentage`: tỷ lệ raw trên raw_max
- `score_100`: điểm tổng cuối cùng
- `pillar_scores`: breakdown chi tiết theo pillar
- `factor_scores`: breakdown theo factor
- `diagnostics`: thống kê lỗi/fallback/coverage

### b) `scoring_details`

Đây là phần quan trọng nhất. Mỗi câu thường có:

- `answer`
- `score`
- `reason`
- `answer_origin`
- `parse_status`
- `evidence_items`
- `evidence_source_ref`
- `top_source_refs`
- `source_sections`

Ngoài ra còn có thể có:

- `resolution_status`
- `confidence`
- `conflict_detected`
- `retry_used`
- `retry_attempts`
- `retry_profiles`
- `query_plan`
- `retrieval_meta`
- `numeric_extraction`
- `llm_call_info`

### c) Citation file và trang

Citation được format theo kiểu:

- `TênFile.pdf p.12`
- `TênFile.pdf pp.12-14`

Nguồn citation được lấy từ metadata của retrieval/evidence, không chỉ từ câu chữ tự do do LLM sinh ra.

Thứ tự nên đọc để kiểm tra nguồn:

1. `evidence_items`
   - đầy đủ quote + file + page
2. `evidence_source_ref`
   - citation ngắn đại diện
3. `top_source_refs`
   - top nguồn retrieval cho câu đó
4. `source_sections`
   - ngữ cảnh retrieval đầy đủ hơn

Nếu markdown clean report trông như không có nguồn, hãy kiểm tra lại `esg_report.json` trước vì JSON thường đầy đủ hơn.

### d) Fallback do LLM lỗi

Nếu LLM không sinh JSON hợp lệ, report vẫn có thể ghi:

- `answer_origin = retrieval_fallback_single`
- `answer_origin = retrieval_fallback_multi`

Các câu này vẫn có thể được chấm điểm nếu retrieval fallback dựng được `evidence_items` có grounding.

### e) `answer_origin`

`answer_origin` cho biết câu trả lời cuối cùng đến từ đâu. Một số giá trị thường gặp:

- `llm_valid_json`: LLM trả JSON hợp lệ ngay từ đầu
- `llm_repaired_json`: JSON phải được sửa cấu trúc nhẹ rồi mới dùng
- `llm_regex_salvaged`: phải salvage answer từ output không hoàn hảo
- `retrieval_fallback_single`: single-select được cứu từ retrieval
- `retrieval_fallback_multi`: multi-select được cứu từ retrieval
- `numeric_override`: numeric extractor override answer

### f) `parse_status`

`parse_status` giúp biết lỗi nằm ở đâu:

- `valid_json`
- `repaired_json`
- `answer_regex_only`
- `empty_raw_response`
- `empty_after_think_strip`
- `llm_transport_error`
- `json_malformed`

Nếu thấy nhiều `empty_after_think_strip`, đó thường là dấu hiệu model chỉ sinh phần `<think>` mà không ra JSON cuối.

## 6. Những Điều Cần Biết Về Partial Run

- Nếu pipeline dừng giữa chừng:
  - progress cache vẫn còn,
  - nhưng thường chưa có report JSON tổng hoàn chỉnh.
- Có thể resume ở lần chạy sau nếu không bật `ESG_NO_RESUME_SCORING=1`.

### Khi resume, điều gì được giữ lại

- các câu đã chấm xong trong progress cache,
- factor scores tạm thời,
- diagnostics tạm thời,
- answer registry.

### Khi resume, điều gì chưa chắc đã có

- report JSON cuối cùng,
- scoring audit cuối cùng,
- clean markdown hoàn chỉnh,
- advisor/report phụ trợ cuối pipeline.

## 7. Các Lỗi Thường Gặp

| Lỗi | Nguyên nhân | Cách xử lý |
|---|---|---|
| `ConnectionError` hoặc lỗi gọi Ollama | Ollama chưa chạy | Mở Ollama hoặc chạy `ollama serve` |
| `FileNotFoundError: tesseract is not installed` | Thiếu OCR runtime | Cài `tesseract` |
| LLM hay trả `NULL` hoặc parse fail | Context quá dài, model sinh think-only | Tăng timeout/context hợp lý và kiểm tra `outputs/debug/llm_json_errors/` |
| Preflight chạy lại từ đầu | Cache preflight không còn match fingerprint hiện tại | Bình thường nếu dossier đã đổi file/hash/timestamp |
| Không thấy citation trong markdown | `evidence_items` của câu đó rỗng hoặc clean report chưa phản ánh hết top source refs | Kiểm tra `esg_report.json` và `scoring_audit.csv` trước |

## 8. Khuyến Nghị Khi Kiểm Tra Chất Lượng Kết Quả

Nếu muốn rà xem một lần chạy có đáng tin không, nên kiểm tra theo thứ tự:

1. `scores.diagnostics`
   - xem tỷ lệ `NULL`, fallback, weakly supported
2. `scoring_audit.csv`
   - lọc các câu `loss_reason`
3. `scoring_details`
   - kiểm tra các câu có điểm dương nhưng citation yếu
4. `outputs/debug/llm_json_errors/`
   - xem model đang fail theo kiểu gì

Các dấu hiệu tốt:

- tỷ lệ `llm_empty_failure` thấp
- ít câu `weakly_supported`
- câu có điểm dương thường có `evidence_items`
- citation file/trang xuất hiện ổn định

Các dấu hiệu xấu:

- nhiều `NULL` từ cùng một nhóm câu
- nhiều `empty_after_think_strip`
- nhiều câu `answer` dương nhưng `evidence_items` rỗng
- markdown không có nguồn và JSON cũng không có `evidence_source_ref`
