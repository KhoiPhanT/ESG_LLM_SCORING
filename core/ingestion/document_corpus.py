"""
Document Corpus - Gom nhiều tài liệu PDF của cùng doanh nghiệp/năm để tra cứu chung.
"""
from __future__ import annotations

import os
import re
import json
import hashlib
from dataclasses import dataclass
from typing import Iterable

from core.cache import CacheManager
from core.ingestion.document_classifier import DocumentClassifier, DocumentMetadata
from core.ingestion.pdf_parser import PDFParser
from core.retrieval.chunk_labeler import ChunkLabeler
from core.structure_builder import SectionBuilder
from core.structure_builder.semantic_chunker import SemanticChunker
from core.table_extraction import TableExtractor


@dataclass
class DocumentRecord:
    path: str
    label: str
    doc_type: str
    parser: PDFParser
    metadata: DocumentMetadata


class DocumentCorpus:
    CACHE_DIR = "outputs/cache/corpus"
    SECTION_CACHE_SCHEMA = "v3"
    TABLE_CACHE_SCHEMA = "v3"
    DOC_TYPE_SCHEMA = "document_classifier_v1"

    def __init__(self, document_paths: Iterable[str], target_year: int | None = None):
        self.classifier = DocumentClassifier()
        self.target_year = target_year
        unique_paths = []
        seen = set()
        for path in document_paths:
            norm = os.path.abspath(path)
            if norm not in seen and os.path.exists(norm):
                seen.add(norm)
                unique_paths.append(norm)

        self.documents = []
        self.skipped_duplicates = []
        self._pages_cache = {}
        self._full_text_cache = {}
        self._section_cache = {}
        self._table_cache = {}
        self.cache_manager = CacheManager(run_key="document_cache")
        os.makedirs(self.CACHE_DIR, exist_ok=True)
        self.section_builder = SectionBuilder()
        self.semantic_chunker = SemanticChunker()
        self.table_extractor = TableExtractor()
        self.chunk_labeler = ChunkLabeler()
        seen_hashes = {}
        for path in unique_paths:
            file_hash = self._hash_file(path)
            if file_hash in seen_hashes:
                self.skipped_duplicates.append({
                    "source_path": path,
                    "file_name": os.path.basename(path),
                    "duplicate_of": seen_hashes[file_hash],
                    "file_hash": file_hash,
                })
                continue
            seen_hashes[file_hash] = path
            parser = PDFParser(path, use_ocr=True)
            pages = self._get_pages_from_parser(path, parser)
            metadata = self.classifier.classify(path, pages=pages)
            self.documents.append(
                DocumentRecord(
                    path=path,
                    label=os.path.basename(path),
                    doc_type=metadata.document_type,
                    parser=parser,
                    metadata=metadata,
                )
            )

    def _infer_doc_type(self, path: str) -> str:
        name = os.path.basename(path).lower()
        if "nghiquyet" in name or "dhdcd" in name:
            return "resolution"
        if "taichinh" in name or "kiemtoan" in name:
            return "financial_report"
        if "ptbv" in name or "sustainability" in name:
            return "sustainability_report"
        if "thuongnien" in name or "annual" in name:
            return "annual_report"
        return "other"

    def extract_all(self):
        extracted = []
        for doc in self.documents:
            pages = self._get_pages(doc)
            extracted.append({"document": doc, "pages": pages, "metadata": doc.metadata.to_dict()})
        return extracted

    def build_registry(self):
        manifest = []
        for doc in self.documents:
            item = doc.metadata.to_dict()
            item["duplicate_group"] = doc.metadata.file_hash[:16]
            item["effective_status"] = self._infer_effective_status(item)
            item["skipped_duplicate"] = False
            manifest.append(item)
        for duplicate in self.skipped_duplicates:
            manifest.append({
                "source_path": duplicate["source_path"],
                "file_name": duplicate["file_name"],
                "file_extension": os.path.splitext(duplicate["file_name"])[1].lower(),
                "file_hash": duplicate["file_hash"],
                "page_count": 0,
                "company_guess": "",
                "year_guess": None,
                "document_type": "duplicate",
                "classification_confidence": 1.0,
                "classification_reasons": [f"duplicate of {os.path.basename(duplicate['duplicate_of'])}"],
                "text_extraction_method": "skipped",
                "average_ocr_quality": 0.0,
                "needs_review": False,
                "duplicate_group": duplicate["file_hash"][:16],
                "effective_status": "duplicate_skipped",
                "skipped_duplicate": True,
            })
        return manifest

    def get_document_pages(self, path: str) -> list[dict]:
        for doc in self.documents:
            if doc.path == os.path.abspath(path):
                return self._get_pages(doc)
        return self._pages_cache.get(os.path.abspath(path), [])

    def get_document_sections(self, path: str) -> list[dict]:
        abs_path = os.path.abspath(path)
        for doc in self.documents:
            if doc.path == abs_path:
                return self._get_sections(doc)
        return self._section_cache.get(abs_path, [])

    def get_document_tables(self, path: str) -> list[dict]:
        abs_path = os.path.abspath(path)
        for doc in self.documents:
            if doc.path == abs_path:
                return self._get_tables(doc)
        return self._table_cache.get(abs_path, [])

    def get_full_text(self, preferred_doc_types: list[str] | None = None) -> str:
        chunks = []
        for doc in self._iter_documents(preferred_doc_types):
            text = self._get_full_text(doc)
            if text.strip():
                chunks.append(
                    f"[DOC: {doc.label} | TYPE: {doc.doc_type} | PAGES: 1-{doc.metadata.page_count}]\n{text}"
                )
        return "\n\n".join(chunks)

    def get_sections_by_keyword(
        self,
        keywords: list[str],
        preferred_doc_types: list[str] | None = None,
        max_sections: int = 6,
    ) -> list[str]:
        sections = self.get_section_records_by_keyword(
            keywords=keywords,
            preferred_doc_types=preferred_doc_types,
            max_sections=max_sections,
        )
        return [self._format_section(section) for section in sections]

    def get_section_records_by_keyword(
        self,
        keywords: list[str],
        preferred_doc_types: list[str] | None = None,
        max_sections: int = 6,
    ) -> list[dict]:
        sections = []
        for doc in self._iter_documents(preferred_doc_types):
            doc_sections = self._extract_sections_from_sections(doc, keywords)
            if not doc_sections:
                doc_sections = self._extract_sections_from_pages(doc, keywords)
            if doc_sections:
                sections.extend(doc_sections)
            if len(sections) >= max_sections:
                break

        if sections:
            sections.sort(key=lambda item: item["quality_score"], reverse=True)
            return sections[:max_sections]

        fallback_docs = list(self._iter_documents(preferred_doc_types)) or self.documents
        fallback_sections = []
        for doc in fallback_docs[:3]:
            pages = self._get_pages(doc)
            useful_pages = [page for page in pages if not self._is_low_value_page(page)]
            if not useful_pages:
                useful_pages = pages[:2]
            if not useful_pages:
                continue
            combined = "\n\n".join(page.get("text", "") for page in useful_pages[:2]).strip()[:8000]
            if combined:
                fallback_sections.append(
                    {
                        "source_file": doc.label,
                        "source_path": doc.path,
                        "document_type": doc.doc_type,
                        "year_guess": doc.metadata.year_guess,
                        "page_start": useful_pages[0]["page"],
                        "page_end": useful_pages[min(len(useful_pages), 2) - 1]["page"],
                        "content": combined,
                        "coverage_source": "keyword_fallback",
                        "quality_score": self._section_quality(combined),
                    }
                )
        return fallback_sections

    def choose_preferred_doc_types(self, q_id: str = "", question: str = "") -> list[str] | None:
        qid = (q_id or "").upper()
        text = (question or "").lower()

        if qid == "SL5" or "kiểm toán ngoại trừ" in text or "báo cáo tài chính" in text:
            return ["financial_report", "annual_report"]

        if qid.startswith("G.") or "đhđcđ" in text or "hội đồng quản trị" in text or "cổ đông" in text:
            return ["resolution", "annual_report", "financial_report"]

        is_policy_question = (
            "chính sách" in text
            or "chinh sach" in text
            or "policy" in text
            or "cam kết" in text
            or "cam ket" in text
        )
        asks_revenue = "doanh thu" in text or "revenue" in text

        if qid.startswith("E.") and ("assurance" in text or "kiểm toán" in text):
            return ["sustainability_report", "annual_report", "financial_report"]

        if qid.startswith("E."):
            if is_policy_question:
                return ["policy_document", "sustainability_report", "annual_report", "financial_report"]
            if asks_revenue:
                return ["sustainability_report", "annual_report", "financial_report"]
            return ["sustainability_report", "annual_report", "financial_report"]

        if qid.startswith("S."):
            if is_policy_question:
                return ["policy_document", "annual_report", "sustainability_report", "financial_report"]
            if asks_revenue:
                return ["annual_report", "financial_report", "sustainability_report"]
            return ["annual_report", "sustainability_report", "financial_report"]

        return None

    def _iter_documents(self, preferred_doc_types: list[str] | None = None):
        if not preferred_doc_types:
            yield from self.documents
            return

        preferred = []
        fallback = []
        preferred_set = set(preferred_doc_types)
        for doc in self.documents:
            if doc.doc_type in preferred_set:
                preferred.append(doc)
            else:
                fallback.append(doc)

        for doc in preferred + fallback:
            yield doc

    def _get_pages(self, doc: DocumentRecord):
        if doc.path not in self._pages_cache:
            self._pages_cache[doc.path] = doc.parser.extract_text()
        return self._pages_cache[doc.path]

    def _get_pages_from_parser(self, path: str, parser: PDFParser):
        if path not in self._pages_cache:
            self._pages_cache[path] = parser.extract_text()
        return self._pages_cache[path]

    def _hash_file(self, path: str) -> str:
        digest = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _infer_effective_status(self, item: dict) -> str:
        doc_type = item.get("document_type")
        year = item.get("year_guess")
        target_year = self.target_year or 2024
        if doc_type in {"sustainability_report", "annual_report", "financial_report", "resolution", "policy_document"}:
            if year == target_year:
                return "current_periodic_disclosure"
            if year and year > target_year:
                return "future_periodic_disclosure"
            if year and year < target_year:
                return "historical_periodic_disclosure"
            return "periodic_disclosure_unknown_year"
        if year and year >= target_year:
            return "latest_or_current"
        return "historical_reference"

    def _get_full_text(self, doc: DocumentRecord):
        if doc.path not in self._full_text_cache:
            pages = self._get_pages(doc)
            self._full_text_cache[doc.path] = "\n\n".join(
                page["text"] for page in pages if page.get("text")
            )
        return self._full_text_cache[doc.path]

    def _get_sections(self, doc: DocumentRecord):
        if doc.path not in self._section_cache:
            cached_sections = self._load_cached_section_records(doc)
            if cached_sections is not None:
                self._section_cache[doc.path] = cached_sections
                return cached_sections
            pages = self._get_pages(doc)
            sections = self.section_builder.build(
                doc=doc,
                pages=pages,
                is_low_value_page=self._is_low_value_page,
            )
            # Split oversized sections into smaller coherent chunks
            sections = self.semantic_chunker.chunk_sections(sections)
            sections.extend(self._get_tables(doc))
            sections.extend(self._build_page_fallback_sections(doc, pages))
            sections = [self.chunk_labeler.annotate(section, doc=doc) for section in sections]
            self._section_cache[doc.path] = sections
            self._save_cached_section_records(doc, sections)
        return self._section_cache[doc.path]

    def _build_page_fallback_sections(self, doc: DocumentRecord, pages: list[dict], window_pages: int = 2) -> list[dict]:
        """
        Add page-level windows so retrieval can search all evidence even when
        automatic heading detection creates a bad coarse section.
        """
        useful_pages = [page for page in pages if (page.get("text") or "").strip()]
        if not useful_pages:
            return []

        windows = []
        seen_ranges = set()
        for idx, page in enumerate(useful_pages):
            if self._is_low_value_page(page):
                continue
            end = min(len(useful_pages), idx + window_pages)
            chunk_pages = useful_pages[idx:end]
            if not chunk_pages:
                continue
            page_range = (chunk_pages[0]["page"], chunk_pages[-1]["page"])
            if page_range in seen_ranges:
                continue
            seen_ranges.add(page_range)
            content = "\n\n".join(
                candidate.get("text", "")
                for candidate in chunk_pages
                if candidate.get("text")
            ).strip()
            if len(content.split()) < 40:
                continue
            windows.append({
                "chunk_id": f"{doc.label}:page_fallback:{page_range[0]}-{page_range[1]}",
                "document_id": doc.metadata.file_hash[:16],
                "source_file": doc.label,
                "source_path": doc.path,
                "document_type": doc.doc_type,
                "year_guess": doc.metadata.year_guess,
                "section_title": f"page fallback {page_range[0]}-{page_range[1]}",
                "page_start": page_range[0],
                "page_end": page_range[1],
                "chunk_type": "page_window",
                "table_family": None,
                "content": content[:7000],
                "coverage_source": "page_fallback",
                "quality_score": self._section_quality(content),
            })
        return windows

    def _get_tables(self, doc: DocumentRecord):
        if doc.path not in self._table_cache:
            cached_tables = self._load_cached_table_records(doc)
            if cached_tables is not None:
                self._table_cache[doc.path] = cached_tables
                return cached_tables
            pages = self._get_pages(doc)
            self._table_cache[doc.path] = self.table_extractor.extract(doc=doc, pages=pages)
            self._save_cached_table_records(doc, self._table_cache[doc.path])
        return self._table_cache[doc.path]

    def _cache_bucket_dir(self, cache_kind: str) -> str:
        bucket = os.path.join(self.CACHE_DIR, cache_kind)
        os.makedirs(bucket, exist_ok=True)
        return bucket

    def _cache_signature(self, doc: DocumentRecord, cache_schema: str) -> str:
        year_part = doc.metadata.year_guess if doc.metadata.year_guess is not None else "any"
        fingerprint = self._document_cache_fingerprint(doc, cache_schema)
        return f"{doc.metadata.file_hash[:16]}_{doc.doc_type}_{year_part}_{self.target_year or 'any'}_{fingerprint[:16]}"

    def _document_cache_fingerprint(self, doc: DocumentRecord, cache_schema: str) -> str:
        return CacheManager.hash_json({
            "schema_version": cache_schema,
            "file_hash": doc.metadata.file_hash,
            "source_path": os.path.abspath(doc.path),
            "doc_type": doc.doc_type,
            "year_guess": doc.metadata.year_guess,
            "target_year": self.target_year,
            "document_type_schema": self.DOC_TYPE_SCHEMA,
        })

    def _section_cache_path(self, doc: DocumentRecord) -> str:
        return os.path.join(self._cache_bucket_dir("sections"), f"{self._cache_signature(doc, self.SECTION_CACHE_SCHEMA)}.json")

    def _table_cache_path(self, doc: DocumentRecord) -> str:
        return os.path.join(self._cache_bucket_dir("tables"), f"{self._cache_signature(doc, self.TABLE_CACHE_SCHEMA)}.json")

    def _load_json_cache(self, path: str, expected_schema: str) -> dict | None:
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception:
            return None
        if not isinstance(payload, dict):
            return None
        if payload.get("schema_version") != expected_schema:
            return None
        return payload

    def _save_json_cache(self, path: str, payload: dict) -> None:
        CacheManager.atomic_write_json(path, payload, indent=None)

    def _record_cache(
        self,
        stage: str,
        status: str,
        schema_version: str,
        input_fingerprint: str,
        path: str,
        reason: str | None = None,
    ) -> None:
        manager = getattr(self, "cache_manager", None)
        if manager is None:
            manager = CacheManager(run_key="document_cache")
            self.cache_manager = manager
        manager.record(
            stage,
            status,
            schema_version,
            input_fingerprint,
            path=path,
            reason=reason,
        )

    def _load_cached_section_records(self, doc: DocumentRecord) -> list[dict] | None:
        if CacheManager.is_forced("corpus") or CacheManager.is_forced("sections"):
            self._record_cache(
                "corpus_sections",
                "rebuilt",
                self.SECTION_CACHE_SCHEMA,
                self._document_cache_fingerprint(doc, self.SECTION_CACHE_SCHEMA),
                path=self._section_cache_path(doc),
                reason="forced_rebuild",
            )
            return None
        payload = self._load_json_cache(self._section_cache_path(doc), self.SECTION_CACHE_SCHEMA)
        if not payload:
            return None
        expected_fingerprint = self._document_cache_fingerprint(doc, self.SECTION_CACHE_SCHEMA)
        if payload.get("input_fingerprint") != expected_fingerprint:
            return None
        if payload.get("file_hash") != doc.metadata.file_hash:
            return None
        if payload.get("target_year") != self.target_year:
            return None
        sections = payload.get("sections")
        if isinstance(sections, list):
            self._record_cache(
                "corpus_sections",
                "hit",
                self.SECTION_CACHE_SCHEMA,
                expected_fingerprint,
                path=self._section_cache_path(doc),
            )
            return sections
        return None

    def _save_cached_section_records(self, doc: DocumentRecord, sections: list[dict]) -> None:
        fingerprint = self._document_cache_fingerprint(doc, self.SECTION_CACHE_SCHEMA)
        payload = {
            "schema_version": self.SECTION_CACHE_SCHEMA,
            "input_fingerprint": fingerprint,
            "file_hash": doc.metadata.file_hash,
            "target_year": self.target_year,
            "doc_type": doc.doc_type,
            "year_guess": doc.metadata.year_guess,
            "sections": sections,
        }
        self._save_json_cache(self._section_cache_path(doc), payload)
        self._record_cache(
            "corpus_sections",
            "rebuilt",
            self.SECTION_CACHE_SCHEMA,
            fingerprint,
            path=self._section_cache_path(doc),
            reason="missing_or_stale_cache",
        )

    def _load_cached_table_records(self, doc: DocumentRecord) -> list[dict] | None:
        if CacheManager.is_forced("corpus") or CacheManager.is_forced("tables"):
            self._record_cache(
                "corpus_tables",
                "rebuilt",
                self.TABLE_CACHE_SCHEMA,
                self._document_cache_fingerprint(doc, self.TABLE_CACHE_SCHEMA),
                path=self._table_cache_path(doc),
                reason="forced_rebuild",
            )
            return None
        payload = self._load_json_cache(self._table_cache_path(doc), self.TABLE_CACHE_SCHEMA)
        if not payload:
            return None
        expected_fingerprint = self._document_cache_fingerprint(doc, self.TABLE_CACHE_SCHEMA)
        if payload.get("input_fingerprint") != expected_fingerprint:
            return None
        if payload.get("file_hash") != doc.metadata.file_hash:
            return None
        if payload.get("target_year") != self.target_year:
            return None
        tables = payload.get("tables")
        if isinstance(tables, list):
            self._record_cache(
                "corpus_tables",
                "hit",
                self.TABLE_CACHE_SCHEMA,
                expected_fingerprint,
                path=self._table_cache_path(doc),
            )
            return tables
        return None

    def _save_cached_table_records(self, doc: DocumentRecord, tables: list[dict]) -> None:
        fingerprint = self._document_cache_fingerprint(doc, self.TABLE_CACHE_SCHEMA)
        payload = {
            "schema_version": self.TABLE_CACHE_SCHEMA,
            "input_fingerprint": fingerprint,
            "file_hash": doc.metadata.file_hash,
            "target_year": self.target_year,
            "doc_type": doc.doc_type,
            "year_guess": doc.metadata.year_guess,
            "tables": tables,
        }
        self._save_json_cache(self._table_cache_path(doc), payload)
        self._record_cache(
            "corpus_tables",
            "rebuilt",
            self.TABLE_CACHE_SCHEMA,
            fingerprint,
            path=self._table_cache_path(doc),
            reason="missing_or_stale_cache",
        )

    def _extract_sections_from_sections(self, doc: DocumentRecord, keywords: list[str]):
        sections = []
        normalized_keywords = [keyword.lower() for keyword in keywords if keyword]
        for section in self._get_sections(doc):
            content = section.get("content", "")
            lowered = content.lower()
            if not content.strip():
                continue
            matched = [keyword for keyword in normalized_keywords if keyword in lowered]
            if not matched:
                continue
            section_record = dict(section)
            section_record["matched_keywords"] = matched
            sections.append(section_record)
        return sections

    def _extract_sections_from_pages(self, doc: DocumentRecord, keywords: list[str], window_pages: int = 1):
        pages = self._get_pages(doc)
        sections = []
        seen_ranges = set()
        normalized_keywords = [keyword.lower() for keyword in keywords if keyword]
        for idx, page in enumerate(pages):
            text = page.get("text", "")
            lowered = text.lower()
            if not text.strip() or self._is_low_value_page(page):
                continue
            matched = [keyword for keyword in normalized_keywords if keyword in lowered]
            if not matched:
                continue
            start_idx = max(0, idx - window_pages)
            end_idx = min(len(pages) - 1, idx + window_pages)
            page_range = (pages[start_idx]["page"], pages[end_idx]["page"])
            if page_range in seen_ranges:
                continue
            seen_ranges.add(page_range)
            content = "\n\n".join(
                candidate.get("text", "")
                for candidate in pages[start_idx:end_idx + 1]
                if candidate.get("text")
            ).strip()
            if not content:
                continue
            quality_score = self._section_quality(content)
            if quality_score <= 0.15:
                continue
            sections.append(
                {
                    "source_file": doc.label,
                    "source_path": doc.path,
                    "document_type": doc.doc_type,
                    "year_guess": doc.metadata.year_guess,
                    "page_start": page_range[0],
                    "page_end": page_range[1],
                    "matched_keywords": matched,
                    "content": content[:6000],
                    "coverage_source": "keyword_page_window",
                    "quality_score": quality_score,
                }
            )
        return sections

    def _extract_sections_from_text(self, full_text: str, keywords: list[str], window_chars: int = 4000):
        lowered = full_text.lower()
        sections = []
        seen_positions = set()
        for keyword in keywords:
            start = 0
            keyword_lower = keyword.lower()
            while True:
                pos = lowered.find(keyword_lower, start)
                if pos == -1:
                    break
                bucket = pos // max(1, (window_chars // 2))
                if bucket not in seen_positions:
                    seen_positions.add(bucket)
                    context_start = max(0, pos - window_chars // 2)
                    context_end = min(len(full_text), pos + window_chars // 2)
                    sections.append(full_text[context_start:context_end])
                start = pos + len(keyword_lower)
        return sections if sections else [full_text[:6000]]

    def _is_low_value_page(self, page: dict) -> bool:
        text = (page.get("text") or "").strip().lower()
        if len(text) < 60:
            return True
        page_no = int(page.get("page", 0) or 0)
        first_lines = " ".join(text.splitlines()[:6])
        toc_markers = ["mục lục", "table of contents", "nội dung", "contents"]
        cover_markers = ["báo cáo thường niên", "annual report", "phát triển bền vững", "financial statements"]
        dotted_lines = first_lines.count("....") >= 2 or first_lines.count("...") >= 4
        if dotted_lines or any(marker in first_lines for marker in toc_markers):
            return True
        if page_no <= 2 and any(marker in first_lines for marker in cover_markers) and len(text.split()) < 120:
            return True
        return False

    def is_low_value_page(self, page: dict) -> bool:
        return self._is_low_value_page(page)

    def _section_quality(self, text: str) -> float:
        stripped = (text or "").strip()
        if not stripped:
            return 0.0
        words = stripped.split()
        unique_ratio = len(set(word.lower() for word in words)) / max(len(words), 1)
        digit_bonus = 0.05 if any(char.isdigit() for char in stripped) else 0.0
        return round(min(1.0, unique_ratio + digit_bonus), 3)

    def section_quality(self, text: str) -> float:
        return self._section_quality(text)

    def _format_section(self, section: dict) -> str:
        return (
            f"[DOC: {section['source_file']} | TYPE: {section['document_type']} | "
            f"PAGES: {section['page_start']}-{section['page_end']}]\n"
            f"{section['content']}"
        )


def discover_related_pdf_paths(primary_pdf_path: str, company_name: str = "", year: int | None = None) -> list[str]:
    primary_abs = os.path.abspath(primary_pdf_path)
    if not os.path.exists(primary_abs):
        return [primary_abs]

    if os.path.isdir(primary_abs):
        directory = primary_abs
        include_primary = False
    else:
        directory = os.path.dirname(primary_abs)
        include_primary = True

    company = (company_name or "").lower().strip()
    year_text = str(year) if year else ""

    # If the folder name matches the company, include ALL relevant files in it.
    # This is intentional: many VNSI questions allow historical evidence
    # (`historical_allowed` / `latest_valid_allowed`), so a company dossier may
    # legitimately contain documents across multiple years.
    folder_name = os.path.basename(directory).lower()
    folder_is_company = company and company in folder_name

    candidates = []
    for entry in sorted(os.listdir(directory)):
        if not entry.lower().endswith((".pdf", ".png", ".jpg", ".jpeg")):
            continue
        full_path = os.path.abspath(os.path.join(directory, entry))
        lowered = entry.lower()

        if folder_is_company:
            # A company dossier is treated as a curated evidence folder, not a
            # single-year-only drop. Current-year-only questions are filtered
            # later by retrieval/scoring time policy rather than here.
            candidates.append(full_path)
        else:
            same_company = not company or company in lowered
            same_year = not year_text or re.search(rf"(?<!\d){re.escape(year_text)}(?!\d)", lowered)
            if (include_primary and full_path == primary_abs) or (same_company and same_year):
                candidates.append(full_path)

    if include_primary and primary_abs not in candidates:
        candidates.insert(0, primary_abs)
    return candidates
