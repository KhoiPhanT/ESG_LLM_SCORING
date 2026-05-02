"""
PDF Parser — Trích xuất text từ PDF với tự động OCR cho các trang scan.
Kết quả được cache vào outputs/cache/ để không phải chạy lại.
"""
from __future__ import annotations

import hashlib
import json
import os

import fitz  # PyMuPDF
from pdf2image import convert_from_path

from core.cache import CacheManager


class PDFParser:
    CACHE_DIR = "outputs/cache"
    CACHE_SCHEMA = "pdf_parser_v2"

    def __init__(self, file_path, use_ocr=True):
        self.file_path = file_path
        self.use_ocr = use_ocr
        os.makedirs(self.CACHE_DIR, exist_ok=True)
        self._reader = None
        self.cache_manager = CacheManager(run_key="document_cache")

    def _get_reader(self):
        if self._reader is None:
            import easyocr
            print("  [OCR] Đang khởi tạo EasyOCR mô hình (lần đầu sẽ hơi lâu)...")
            self._reader = easyocr.Reader(['vi', 'en'])
        return self._reader

    def _file_hash(self):
        digest = hashlib.sha256()
        with open(self.file_path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _cache_key(self):
        basename = os.path.basename(self.file_path)
        h = self._file_hash()[:12]
        mode = "ocr" if self.use_ocr else "native"
        return os.path.join(self.CACHE_DIR, f"{basename}_{h}_{mode}.json")

    def _cache_fingerprint(self) -> str:
        return CacheManager.file_fingerprint(
            self.file_path,
            extra={
                "schema": self.CACHE_SCHEMA,
                "mode": "ocr" if self.use_ocr else "native",
            },
        )

    def extract_text(self):
        """
        Trích xuất toàn bộ text từ PDF. Tự động dùng OCR nếu trang là ảnh scan.
        Kết quả được cache lại.
        """
        cache_path = self._cache_key()
        fingerprint = self._cache_fingerprint()
        forced = CacheManager.is_forced("ocr")
        if os.path.exists(cache_path) and not forced:
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    cached = json.load(f)
                if isinstance(cached, dict):
                    if (
                        cached.get("schema_version") != self.CACHE_SCHEMA
                        or cached.get("input_fingerprint") != fingerprint
                    ):
                        pages = None
                    else:
                        pages = cached.get("pages")
                else:
                    # Backward compatibility for old list-shaped caches.  The
                    # file hash is still embedded in the cache filename.
                    pages = cached
                if isinstance(pages, list) and pages:
                    print(f"  [CACHE HIT] Đọc từ cache: {cache_path}")
                    self.cache_manager.record(
                        "ocr",
                        "hit",
                        self.CACHE_SCHEMA,
                        fingerprint,
                        path=cache_path,
                    )
                    return pages
            except Exception as e:
                self.cache_manager.record(
                    "ocr",
                    "failed",
                    self.CACHE_SCHEMA,
                    fingerprint,
                    path=cache_path,
                    reason="cache_parse_error",
                    error=str(e),
                )
        elif forced:
            print(f"  [CACHE] OCR forced rebuild: {os.path.basename(self.file_path)}")

        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"Không tìm thấy file PDF: {self.file_path}")

        doc = fitz.open(self.file_path)
        total_pages = len(doc)
        native_pages = []
        native_text_pages = 0
        
        import re

        for page_num in range(total_pages):
            text = doc[page_num].get_text("text").strip()
            
            # Detect corrupted font encoding (garbage text)
            bad_chars = sum(1 for c in text if c in '~§¢£¤¥¦¨©ª«¬®¯°±²³´µ¶·¸¹º»¼½¾¿')
            digit_in_words = len(re.findall(r'[a-zA-Z][0-9][a-zA-Z]', text)) + len(re.findall(r'[a-zA-Z][0-9]\s', text)) + len(re.findall(r'\s[0-9][a-zA-Z]', text))
            is_garbage = bad_chars > 5 or digit_in_words > 10
            
            if len(text) > 50 and not is_garbage:
                native_text_pages += 1
            
            native_pages.append(
                self._build_page_record(
                    page_number=page_num + 1,
                    text=text,
                    extraction_method="native",
                )
            )
        doc.close()

        text_ratio = native_text_pages / total_pages if total_pages > 0 else 0
        print(
            f"  [PDF] {os.path.basename(self.file_path)}: "
            f"{native_text_pages}/{total_pages} pages có text ({text_ratio:.0%})"
        )

        extracted_data = native_pages
        if self.use_ocr:
            if text_ratio < 0.3:
                print(f"  [OCR] File gần như scan, đang OCR toàn bộ {total_pages} trang...")
                extracted_data = self._ocr_document(range(1, total_pages + 1), total_pages=total_pages)
            else:
                sparse_pages = [
                    page["page"] for page in native_pages if page["char_count"] < 50
                ]
                if sparse_pages:
                    print(f"  [OCR] OCR bổ sung {len(sparse_pages)} trang ít text...")
                    ocr_pages = self._ocr_document(sparse_pages, total_pages=total_pages)
                    ocr_map = {page["page"]: page for page in ocr_pages}
                    extracted_data = [ocr_map.get(page["page"], page) for page in native_pages]

        CacheManager.atomic_write_json(
            cache_path,
            {
                "schema_version": self.CACHE_SCHEMA,
                "input_fingerprint": fingerprint,
                "file_path": os.path.abspath(self.file_path),
                "mode": "ocr" if self.use_ocr else "native",
                "pages": extracted_data,
            },
            indent=2,
        )
        self.cache_manager.record(
            "ocr",
            "rebuilt",
            self.CACHE_SCHEMA,
            fingerprint,
            path=cache_path,
            reason="forced_rebuild" if forced else "missing_or_invalid_cache",
        )
        print(f"  [CACHE] Đã lưu cache: {cache_path}")

        return extracted_data

    def _ocr_document(self, page_numbers, total_pages):
        """OCR document theo danh sách trang bằng pdf2image + EasyOCR."""
        results = []
        try:
            import numpy as np
            from PIL import Image
            reader = self._get_reader()
            is_image = self.file_path.lower().endswith(('.png', '.jpg', '.jpeg'))

            for page_num in page_numbers:
                print(f"    OCR trang {page_num}/{total_pages}...")
                if is_image:
                    img = Image.open(self.file_path)
                    img_np = np.array(img)
                else:
                    images = convert_from_path(
                        self.file_path,
                        first_page=page_num,
                        last_page=page_num,
                        dpi=200,
                    )
                    if not images:
                        continue
                    img_np = np.array(images[0])
                
                text_list = reader.readtext(img_np, detail=0, paragraph=True)
                text = "\n".join(text_list)
                
                results.append(
                    self._build_page_record(
                        page_number=page_num,
                        text=text.strip(),
                        extraction_method="ocr",
                    )
                )
        except Exception as e:
            print(f"  [OCR ERROR] {e}")
        return results

    def _build_page_record(self, page_number, text, extraction_method):
        normalized = (text or "").strip()
        return {
            "page": page_number,
            "text": normalized,
            "extraction_method": extraction_method,
            "ocr_quality_score": self._estimate_text_quality(normalized),
            "char_count": len(normalized),
            "word_count": len(normalized.split()) if normalized else 0,
        }

    def _estimate_text_quality(self, text):
        if not text:
            return 0.0
        characters = [char for char in text if not char.isspace()]
        if not characters:
            return 0.0

        readable_chars = sum(
            1
            for char in characters
            if char.isalnum() or char in ".,:;!?%/-()[]{}\"'áàảãạăắằẳẵặâấầẩẫậđéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵ"
        )
        ratio = readable_chars / len(characters)
        garbage_penalty = 0.15 if "�" in text else 0.0
        return round(max(0.0, min(1.0, ratio - garbage_penalty)), 3)

    def get_full_text(self):
        """Trả về toàn bộ text gộp lại."""
        pages = self.extract_text()
        return "\n\n".join(p["text"] for p in pages if p["text"])

    def get_sections_by_keyword(self, keywords, window_chars=4000):
        """
        Tìm các đoạn text liên quan đến keywords.
        Trả về danh sách các đoạn text (context windows) chứa keyword.
        """
        pages = self.extract_text()
        full_text = "\n\n".join(p["text"] for p in pages if p["text"])
        full_text_lower = full_text.lower()
        sections = []
        seen_positions = set()

        for kw in keywords:
            kw_lower = kw.lower()
            start = 0
            while True:
                pos = full_text_lower.find(kw_lower, start)
                if pos == -1:
                    break
                # Tránh trùng lặp
                bucket = pos // (window_chars // 2)
                if bucket not in seen_positions:
                    seen_positions.add(bucket)
                    context_start = max(0, pos - window_chars // 2)
                    context_end = min(len(full_text), pos + window_chars // 2)
                    sections.append(full_text[context_start:context_end])
                start = pos + len(kw_lower)

        return sections if sections else [full_text[:6000]]


if __name__ == "__main__":
    # Test: dùng file có text (2021)
    parser = PDFParser("inputs/ACB/reports/ACB_Baocaothuongnien_2021.pdf", use_ocr=True)
    pages = parser.extract_text()
    print(f"\nĐã trích xuất {len(pages)} trang.")
    total_chars = sum(len(p["text"]) for p in pages)
    print(f"Tổng ký tự: {total_chars:,}")
