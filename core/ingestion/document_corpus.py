"""
Document Corpus - Gom nhiều tài liệu PDF của cùng doanh nghiệp/năm để tra cứu chung.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Iterable

from core.ingestion.document_classifier import DocumentClassifier, DocumentMetadata
from core.ingestion.pdf_parser import PDFParser
from core.retrieval.chunk_labeler import ChunkLabeler
from core.structure_builder import SectionBuilder
from core.table_extraction import TableExtractor


@dataclass
class DocumentRecord:
    path: str
    label: str
    doc_type: str
    parser: PDFParser
    metadata: DocumentMetadata


class DocumentCorpus:
    def __init__(self, document_paths: Iterable[str]):
        self.classifier = DocumentClassifier()
        unique_paths = []
        seen = set()
        for path in document_paths:
            norm = os.path.abspath(path)
            if norm not in seen and os.path.exists(norm):
                seen.add(norm)
                unique_paths.append(norm)

        self.documents = []
        self._pages_cache = {}
        self._full_text_cache = {}
        self._section_cache = {}
        self._table_cache = {}
        self.section_builder = SectionBuilder()
        self.table_extractor = TableExtractor()
        self.chunk_labeler = ChunkLabeler()
        for path in unique_paths:
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
        return [doc.metadata.to_dict() for doc in self.documents]

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
                        "page_start": useful_pages[0]["page"],
                        "page_end": useful_pages[min(len(useful_pages), 2) - 1]["page"],
                        "content": combined,
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

        if qid.startswith("E.") and ("assurance" in text or "kiểm toán" in text):
            return ["sustainability_report", "annual_report", "financial_report"]

        if qid.startswith("E."):
            return ["sustainability_report", "annual_report", "financial_report"]

        if qid.startswith("S."):
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

    def _get_full_text(self, doc: DocumentRecord):
        if doc.path not in self._full_text_cache:
            pages = self._get_pages(doc)
            self._full_text_cache[doc.path] = "\n\n".join(
                page["text"] for page in pages if page.get("text")
            )
        return self._full_text_cache[doc.path]

    def _get_sections(self, doc: DocumentRecord):
        if doc.path not in self._section_cache:
            pages = self._get_pages(doc)
            sections = self.section_builder.build(
                doc=doc,
                pages=pages,
                is_low_value_page=self._is_low_value_page,
            )
            sections.extend(self._get_tables(doc))
            sections = [self.chunk_labeler.annotate(section, doc=doc) for section in sections]
            self._section_cache[doc.path] = sections
        return self._section_cache[doc.path]

    def _get_tables(self, doc: DocumentRecord):
        if doc.path not in self._table_cache:
            pages = self._get_pages(doc)
            self._table_cache[doc.path] = self.table_extractor.extract(doc=doc, pages=pages)
        return self._table_cache[doc.path]

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
                    "page_start": page_range[0],
                    "page_end": page_range[1],
                    "matched_keywords": matched,
                    "content": content[:6000],
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

    candidates = []
    for entry in sorted(os.listdir(directory)):
        if not entry.lower().endswith(".pdf"):
            continue
        full_path = os.path.abspath(os.path.join(directory, entry))
        lowered = entry.lower()

        same_company = not company or company in lowered
        same_year = not year_text or re.search(rf"(?<!\d){re.escape(year_text)}(?!\d)", lowered)
        if (include_primary and full_path == primary_abs) or (same_company and same_year):
            candidates.append(full_path)

    if include_primary and primary_abs not in candidates:
        candidates.insert(0, primary_abs)
    return candidates
