"""
Build coarse document sections from extracted PDF pages.
"""
from __future__ import annotations

import re


class SectionBuilder:
    HEADING_PATTERNS = [
        re.compile(r"^\s*(phần|chuong|chương|muc|mục)\s+[ivxlcdm\d]+[\.\-:]?\s+.+$", re.IGNORECASE),
        re.compile(r"^\s*\d+(?:\.\d+){0,3}[\.\-:]?\s+.+$"),
        re.compile(r"^\s*[A-ZĐ][A-ZĐ0-9\s,:()/%\-]{10,}$"),
    ]

    GENERIC_TITLES = {
        "mục lục",
        "table of contents",
        "nội dung",
        "contents",
    }

    def build(self, doc, pages: list[dict], is_low_value_page) -> list[dict]:
        if not pages:
            return []

        sections = []
        current = None

        for page in pages:
            text = (page.get("text") or "").strip()
            if not text:
                continue

            title = self._detect_heading(text, page.get("page", 0))
            low_value = is_low_value_page(page)

            if title and not low_value:
                if current:
                    sections.append(self._finalize_section(doc, current))
                current = {
                    "section_title": title,
                    "page_start": page["page"],
                    "page_end": page["page"],
                    "pages": [page],
                    "chunk_type": self._infer_chunk_type(title, text),
                }
                continue

            if current is None:
                current = {
                    "section_title": self._fallback_title(doc, page),
                    "page_start": page["page"],
                    "page_end": page["page"],
                    "pages": [page],
                    "chunk_type": "section",
                }
            else:
                current["page_end"] = page["page"]
                current["pages"].append(page)

        if current:
            sections.append(self._finalize_section(doc, current))

        return self._merge_short_sections(doc, sections)

    def _detect_heading(self, text: str, page_number: int) -> str | None:
        lines = [line.strip() for line in text.splitlines()[:10] if line.strip()]
        if not lines:
            return None

        for line in lines[:4]:
            normalized = self._normalize_title(line)
            if normalized in self.GENERIC_TITLES:
                return None
            if len(line.split()) <= 1:
                continue
            if any(pattern.match(line) for pattern in self.HEADING_PATTERNS):
                return self._clean_title(line)

        if page_number <= 3:
            return None
        return None

    def _merge_short_sections(self, doc, sections: list[dict]) -> list[dict]:
        if not sections:
            return []

        merged = []
        for section in sections:
            content_words = len((section.get("content") or "").split())
            if merged and content_words < 80 and section["page_start"] == merged[-1]["page_end"] + 1:
                merged[-1]["page_end"] = section["page_end"]
                merged[-1]["content"] = f"{merged[-1]['content']}\n\n{section['content']}".strip()
                merged[-1]["quality_score"] = max(merged[-1]["quality_score"], section["quality_score"])
                continue
            merged.append(section)
        return merged

    def _finalize_section(self, doc, current: dict) -> dict:
        content = "\n\n".join(page.get("text", "") for page in current["pages"] if page.get("text")).strip()
        return {
            "chunk_id": f"{doc.label}:{current['page_start']}-{current['page_end']}",
            "document_id": doc.metadata.file_hash[:16],
            "source_file": doc.label,
            "source_path": doc.path,
            "document_type": doc.doc_type,
            "year_guess": doc.metadata.year_guess,
            "section_title": current["section_title"],
            "page_start": current["page_start"],
            "page_end": current["page_end"],
            "chunk_type": current["chunk_type"],
            "content": content,
            "coverage_source": "section_builder",
            "quality_score": self._quality_score(content),
        }

    def _fallback_title(self, doc, page: dict) -> str:
        return f"{doc.doc_type} page {page.get('page', 0)}"

    def _infer_chunk_type(self, title: str, text: str) -> str:
        lowered = f"{title}\n{text}".lower()
        if "phụ lục" in lowered or "appendix" in lowered:
            return "appendix"
        if "bảng" in lowered or "table" in lowered:
            return "table_section"
        return "section"

    def _quality_score(self, text: str) -> float:
        stripped = (text or "").strip()
        if not stripped:
            return 0.0
        words = stripped.split()
        unique_ratio = len(set(word.lower() for word in words)) / max(len(words), 1)
        digit_bonus = 0.08 if any(char.isdigit() for char in stripped) else 0.0
        title_bonus = 0.07 if len(words) >= 60 else 0.0
        return round(min(1.0, unique_ratio + digit_bonus + title_bonus), 3)

    def _clean_title(self, value: str) -> str:
        return re.sub(r"\s+", " ", value.strip(" -:\t"))

    def _normalize_title(self, value: str) -> str:
        return self._clean_title(value).lower()
