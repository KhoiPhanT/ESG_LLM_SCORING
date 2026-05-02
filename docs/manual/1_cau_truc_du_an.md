# Cấu Trúc Dự Án Đánh Giá ESG

Dự án được tổ chức theo hướng module hóa để tách riêng các lớp: ingestion, retrieval, extraction, scoring, reporting và analytics. Phiên bản hiện tại không còn tính điểm tổng theo trọng số ngành E/S/G; điểm tổng được tính trực tiếp từ điểm raw theo workbook.

## Cây Thư Mục Tổng Quan

```text
esg/
│
├── core/                           # Lõi xử lý chính
│   ├── analytics/                  # ESGUI, PLS-SEM, clean report
│   ├── audit/                      # Audit review list và advisor
│   ├── cache/                      # Fingerprint, cache manager
│   ├── evidence/                   # Evidence extractor, verifier, numeric extractor
│   ├── extraction/                 # Keyword analytics
│   ├── ingestion/                  # PDF parser, corpus builder, Excel/rule parser
│   ├── reporting/                  # Format nguồn, hiển thị citation
│   ├── resolution/                 # Resolve answer + evidence state
│   ├── retrieval/                  # Retrieval engine, rerank, metadata routing
│   ├── scoring/                    # Screening, scoring engine, scorer, scoring contract
│   └── llm_client.py               # Giao tiếp Ollama/Qwen, parse JSON, retry metadata
│
├── docs/                           # Tài liệu dự án
│   └── manual/                     # Bộ hướng dẫn sử dụng và vận hành
│
├── inputs/                         # Dữ liệu đầu vào
│   ├── <COMPANY>/                  # Hồ sơ tài liệu theo doanh nghiệp
│   └── *.xlsx                      # Workbook VNSI, dữ liệu WUI, dữ liệu SEM
│
├── outputs/                        # Dữ liệu sinh ra khi chạy
│   ├── audit/                      # Preflight, scoring audit, coverage audit
│   ├── cache/                      # OCR cache, scoring progress cache
│   ├── debug/                      # Debug LLM JSON / parse failures
│   ├── reports/                    # JSON report và markdown report
│   └── *.json                      # Rules/metadata sinh từ workbook
│
├── tests/                          # Regression tests
└── main.py                         # Entry point điều phối toàn pipeline
```

## Các Module Cốt Lõi

1. **`core/llm_client.py`**
   - Gửi prompt tiếng Việt tới Ollama.
   - Ép mô hình trả JSON.
   - Phân loại rõ các trạng thái như `valid_json`, `repaired_json`, `empty_raw_response`, `empty_after_think_strip`.
   - Lưu metadata cuộc gọi để debug khi LLM không ra JSON cuối.
   - Hỗ trợ nhiều chế độ prompt/response mode để phục vụ retry ladder:
     - `full`
     - `minimal`
     - `answer_only`

2. **`core/ingestion/`**
   - Parse workbook VNSI thành rules JSON.
   - Đọc PDF và OCR khi cần.
   - Xây dựng dossier tài liệu theo công ty; một folder có thể chứa cả tài liệu nhiều năm nếu câu hỏi cho phép dùng historical evidence.
   - Ghi metadata tài liệu như loại tài liệu, năm dự đoán, đường dẫn gốc, fingerprint file.

3. **`core/retrieval/`**
   - Chọn các đoạn tài liệu liên quan nhất cho từng câu hỏi.
   - Có metadata routing theo loại câu hỏi, tài liệu, năm và tín hiệu option-level.
   - Đây là lớp nền để LLM không phải đọc toàn bộ báo cáo.
   - Trả về `source_sections` có đầy đủ thông tin để report/audit có thể trích dẫn theo file và trang.

4. **`core/evidence/evidence_extractor.py`**
   - Điều phối LLM + retrieval để tạo answer, reason, evidence.
   - Có retry ladder khi LLM trả rỗng hoặc chỉ sinh `<think>`.
   - Có fallback retrieval-grounded cho multi-select và single-select trong các trường hợp phù hợp.
   - Có `NumericExtractor` cho các câu numeric/ratio.
   - Dựng `evidence_items` theo `source_id`, `source_file`, `page_start`, `page_end`.

5. **`core/resolution/answer_resolver.py`**
   - Quyết định trạng thái cuối của câu trả lời: `supported`, `weakly_supported`, `insufficient`.
   - Không cho điểm dương nếu thiếu bằng chứng hoặc bằng chứng không được grounding xác nhận.
   - Gắn `confidence` ở mức question-level dựa trên evidence items.

6. **`core/scoring/`**
   - `screening.py`: áp dụng các câu screening và penalty.
   - `scoring_engine.py`: dịch đáp án sang điểm theo đúng logic workbook.
   - `vnsi_scorer.py`: chạy toàn bộ bộ câu hỏi, lưu progress từng câu, tổng hợp diagnostics.
   - `scoring_contract.py`: tính `raw_total`, `raw_percentage`, `score_100`, và ghi audit CSV/JSON.
   - `vnsi_scorer.py` cũng là nơi gắn citation ngắn như `evidence_source_ref`, `top_source_refs`, và `source_sections[*].source_ref`.

7. **`core/reporting/` và `core/analytics/clean_reporter.py`**
   - Format citation theo `TênFile.pdf p.X` hoặc `pp.X-Y`.
   - Sinh report JSON đầy đủ và markdown clean report.
   - Audit riêng giúp rà soát các câu bị mất điểm, thiếu bằng chứng hoặc fallback.

## Các Loại Dữ Liệu Quan Trọng Trong Runtime

### 1. Rules và metadata

- `screening rules`
- `scoring rules`
- `question metadata`
- `query plan metadata`

Đây là lớp quyết định:
- loại câu hỏi là gì,
- option nào là positive/negative,
- có được dùng historical evidence không,
- có cần current year hay không,
- câu đó nên ưu tiên loại tài liệu nào.

### 2. `source_sections`

Đây là các đoạn retrieval top-k cho từng câu hỏi. Mỗi section thường mang các trường:

- `source_id`
- `source_file`
- `source_path`
- `document_type`
- `year_guess`
- `page_start`
- `page_end`
- `score` hoặc `rerank_score`
- `matched_keywords`
- `matched_terms`
- `matched_options`

`source_sections` là lớp dữ liệu gốc để:
- LLM đọc context,
- evidence verifier đối chiếu,
- report tạo citation.

### 3. `evidence_items`

Đây là lớp bằng chứng đã được gắn với một câu trả lời cụ thể. Mỗi item thường có:

- `quote`
- `source_id`
- `source_file`
- `source_path`
- `document_type`
- `page_start`
- `page_end`
- `retrieval_score`
- `llm_confidence`
- `confidence`
- `verification_status`

Nếu một câu được chấm điểm dương, về nguyên tắc nó nên đi kèm `evidence_items` hữu dụng.

### 4. `diagnostics`

Diagnostics là lớp theo dõi vận hành, dùng để biết hệ thống đang fail ở retrieval, LLM, hay scoring.

Ví dụ:
- tỷ lệ `llm_empty_failure`
- tỷ lệ `retrieval_fallback_single`
- số câu `weakly_supported`
- số câu `NULL`
- số câu có positive answer nhưng thiếu evidence

## Cơ Chế Citation

Citation trong dự án không được tạo từ “văn phong” của LLM, mà đi theo chuỗi kỹ thuật như sau:

1. Retrieval chọn `source_sections`.
2. LLM hoặc fallback chỉ ra `source_id` và/hoặc `quote`.
3. `EvidenceExtractor` dựng `evidence_items` gắn với `source_file`, `page_start`, `page_end`.
4. `vnsi_scorer.py` tạo:
   - `evidence_source_ref`
   - `top_source_refs`
   - `source_sections[*].source_ref`
5. `core/reporting/source_refs.py` format về dạng ngắn:
   - `TênFile.pdf p.5`
   - `TênFile.pdf pp.5-7`

Điểm quan trọng là:
- `reason` chỉ là diễn giải,
- `evidence` là quote thô,
- citation thực sự nằm ở các field dẫn xuất từ metadata file/trang.

## Điểm Quan Trọng Của Phiên Bản Hiện Tại

- **Không còn trọng số pillar E/S/G trong điểm tổng.**
  - `score_100 = raw_percentage`
  - E/S/G percentage chỉ còn là breakdown để hiển thị.

- **Có autosave tiến độ từng câu.**
  - Nếu dừng giữa chừng, hệ thống có thể resume từ `outputs/cache/*_scoring_progress.json`.

- **Có phân biệt report và audit.**
  - `outputs/reports/*_esg_report.json`: báo cáo đầy đủ toàn pipeline.
  - `outputs/audit/*_scoring_audit.csv|json`: bảng rà soát câu hỏi và citation ngắn.

- **Citation được gắn theo file và trang.**
  - Citation lấy từ metadata của retrieval sections và evidence items, không phải chỉ từ văn bản tự do do LLM sinh ra.

- **Có phân biệt nguồn gốc câu trả lời.**
  - `llm_valid_json`
  - `llm_repaired_json`
  - `llm_regex_salvaged`
  - `retrieval_fallback_single`
  - `retrieval_fallback_multi`
  - `numeric_override`

- **Có phân biệt nguyên nhân lỗi parse/LLM.**
  - `empty_raw_response`
  - `empty_after_think_strip`
  - `llm_transport_error`
  - `json_malformed`
  - `repaired_json`

- **Report cuối và progress cache là hai thứ khác nhau.**
  - Progress cache dùng để resume.
  - Report cuối chỉ sinh khi pipeline hoàn tất.
