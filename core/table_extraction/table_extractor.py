"""
Heuristic table extraction from OCR/native page text.
"""
from __future__ import annotations

import re


class TableExtractor:
    TABLE_TITLE_PATTERNS = [
        re.compile(r"^\s*(bang|bảng|table)\b", re.IGNORECASE),
        re.compile(r"^\s*\d+(?:\.\d+){0,2}\s+.+$"),
    ]

    def extract(self, doc, pages: list[dict]) -> list[dict]:
        tables = []
        for page in pages:
            page_tables = self._extract_page_tables(doc, page)
            tables.extend(page_tables)
        return self._merge_adjacent_tables(doc, tables)

    def _extract_page_tables(self, doc, page: dict) -> list[dict]:
        text = (page.get("text") or "").strip()
        if not text:
            return []

        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if len(lines) < 4:
            return []

        tables = []
        current_lines = []
        current_title = None

        for index, line in enumerate(lines):
            if self._is_table_title(line):
                current_title = line

            if self._looks_like_table_row(line):
                current_lines.append(line)
                continue

            if len(current_lines) >= 3:
                tables.append(self._build_table_chunk(doc, page, current_lines, current_title, index))
            current_lines = []
            current_title = None

        if len(current_lines) >= 3:
            tables.append(self._build_table_chunk(doc, page, current_lines, current_title, len(lines)))

        return [table for table in tables if table]

    def _merge_adjacent_tables(self, doc, tables: list[dict]) -> list[dict]:
        if not tables:
            return []

        merged = []
        for table in tables:
            if (
                merged
                and table["source_file"] == merged[-1]["source_file"]
                and table["page_start"] == merged[-1]["page_end"] + 1
                and table["table_family"] == merged[-1]["table_family"]
            ):
                merged[-1]["page_end"] = table["page_end"]
                merged[-1]["content"] = f"{merged[-1]['content']}\n{table['content']}".strip()
                merged[-1]["quality_score"] = max(merged[-1]["quality_score"], table["quality_score"])
                continue
            merged.append(table)
        return merged

    def _build_table_chunk(self, doc, page: dict, table_lines: list[str], title: str | None, line_index: int) -> dict | None:
        content = "\n".join(table_lines).strip()
        if len(content.split()) < 12:
            return None

        family = self._infer_table_family(title, content)
        if not self._is_useful_table(content, family):
            return None

        section_title = title or self._fallback_title(doc, family, page["page"])
        return {
            "chunk_id": f"{doc.label}:table:{page['page']}:{line_index}",
            "document_id": doc.metadata.file_hash[:16],
            "source_file": doc.label,
            "source_path": doc.path,
            "document_type": doc.doc_type,
            "section_title": section_title,
            "page_start": page["page"],
            "page_end": page["page"],
            "chunk_type": "table_section",
            "table_family": family,
            "content": content[:9000],
            "quality_score": self._quality_score(content, title=section_title),
        }

    def _is_table_title(self, line: str) -> bool:
        return any(pattern.match(line) for pattern in self.TABLE_TITLE_PATTERNS)

    def _looks_like_table_row(self, line: str) -> bool:
        if len(line) < 4:
            return False
        digit_count = sum(1 for char in line if char.isdigit())
        word_count = len(line.split())
        separators = line.count("|") + line.count(":") + line.count("%")
        has_money = any(token in line.lower() for token in ["ty", "triệu", "trieu", "đồng", "dong"])
        long_numeric = len(re.findall(r"\d[\d\.,]{2,}", line)) >= 1
        multiple_numeric_cells = len(re.findall(r"\d[\d\.,]{1,}", line)) >= 2
        contact_like = any(token in line.lower() for token in ["điện thoại", "dien thoai", "fax", "dia chi", "địa chỉ"])
        return (
            not contact_like and (
            (digit_count >= 6 and word_count <= 18)
            or (multiple_numeric_cells and separators >= 1)
            or (long_numeric and has_money)
            )
        )

    def _infer_table_family(self, title: str | None, content: str) -> str:
        haystack = f"{title or ''}\n{content}".lower()
        if any(token in haystack for token in ["co2", "phát thải", "phat thai", "năng lượng", "nang luong", "nước", "nuoc", "chất thải", "chat thai"]):
            return "environmental_metrics"
        if any(token in haystack for token in ["nhân viên", "nhan vien", "lao động", "lao dong", "đào tạo", "dao tao"]):
            return "workforce_metrics"
        if any(token in haystack for token in ["lợi nhuận", "loi nhuan", "tài sản", "tai san", "doanh thu"]):
            return "financial_metrics"
        if any(token in haystack for token in ["biểu quyết", "bieu quyet", "tán thành", "tan thanh", "cổ phần", "co phan"]):
            return "voting_results"
        return "general_table"

    def _quality_score(self, content: str, title: str | None = None) -> float:
        lines = [line for line in content.splitlines() if line.strip()]
        numeric_lines = sum(1 for line in lines if self._looks_like_table_row(line))
        density = numeric_lines / max(len(lines), 1)
        title_bonus = 0.12 if title else 0.0
        return round(min(1.0, 0.4 + density * 0.5 + title_bonus), 3)

    def _fallback_title(self, doc, family: str, page_number: int) -> str:
        return f"{doc.doc_type} {family} page {page_number}"

    def _is_useful_table(self, content: str, family: str) -> bool:
        lowered = content.lower()
        noisy_tokens = [
            "điện thoại", "dien thoai", "fax", "địa chỉ", "dia chi",
            "website", "email", "doc lap", "ty do", "hanh phuc",
        ]
        if any(token in lowered for token in noisy_tokens):
            return False

        numeric_cells = len(re.findall(r"\d[\d\.,]{1,}", content))
        if family == "general_table" and numeric_cells < 4:
            return False

        lines = [line for line in content.splitlines() if line.strip()]
        if family == "general_table" and len(lines) < 3:
            return False

        return True
