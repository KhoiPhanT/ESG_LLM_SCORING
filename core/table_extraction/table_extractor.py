"""
Heuristic table extraction from OCR/native page text.
"""
from __future__ import annotations

import re
import os


class TableExtractor:
    TABLE_TITLE_PATTERNS = [
        re.compile(r"^\s*(bang|bảng|table)\b", re.IGNORECASE),
        re.compile(r"^\s*\d+(?:\.\d+){0,2}\s+.+$"),
    ]

    def extract(self, doc, pages: list[dict]) -> list[dict]:
        tables = []
        tables.extend(self._extract_with_pdfplumber(doc))
        for page in pages:
            page_tables = self._extract_page_tables(doc, page)
            tables.extend(page_tables)
            tables.extend(self._extract_metric_key_values(doc, page))
        return self._merge_adjacent_tables(doc, tables)

    def _extract_with_pdfplumber(self, doc) -> list[dict]:
        try:
            import pdfplumber
        except Exception:
            return []

        if not str(doc.path).lower().endswith(".pdf") or not os.path.exists(doc.path):
            return []

        tables = []
        try:
            with pdfplumber.open(doc.path) as pdf:
                for page_index, page in enumerate(pdf.pages, start=1):
                    for table_index, table in enumerate(page.extract_tables() or []):
                        normalized = self._normalize_pdfplumber_table(table)
                        if not normalized:
                            continue
                        title = self._nearby_table_title(page.extract_text() or "")
                        content = self._table_to_text(normalized)
                        family = self._infer_table_family(title, content)
                        if not self._is_useful_table(content, family):
                            continue
                        tables.append({
                            "chunk_id": f"{doc.label}:pdfplumber_table:{page_index}:{table_index}",
                            "document_id": doc.metadata.file_hash[:16],
                            "source_file": doc.label,
                            "source_path": doc.path,
                            "document_type": doc.doc_type,
                            "year_guess": doc.metadata.year_guess,
                            "section_title": title or self._fallback_title(doc, family, page_index),
                            "page_start": page_index,
                            "page_end": page_index,
                            "chunk_type": "table_section",
                            "table_family": family,
                            "content": content[:12000],
                            "coverage_source": "pdfplumber_table",
                            "quality_score": min(1.0, self._quality_score(content, title=title) + 0.15),
                            "table_columns": normalized[0] if normalized else [],
                            "table_rows": normalized[1:] if len(normalized) > 1 else [],
                            "extraction_method": "pdfplumber",
                        })
        except Exception:
            return []
        return tables

    def _normalize_pdfplumber_table(self, table) -> list[list[str]]:
        rows = []
        for row in table or []:
            cells = [re.sub(r"\s+", " ", str(cell or "").strip()) for cell in row or []]
            if any(cells):
                rows.append(cells)
        if len(rows) < 2:
            return []
        return rows

    def _table_to_text(self, rows: list[list[str]]) -> str:
        widths = [max(len(row[i]) if i < len(row) else 0 for row in rows) for i in range(max(len(r) for r in rows))]
        lines = []
        for row in rows:
            padded = [(row[i] if i < len(row) else "").ljust(widths[i]) for i in range(len(widths))]
            lines.append(" | ".join(padded).strip())
        return "\n".join(lines)

    def _nearby_table_title(self, text: str) -> str | None:
        lines = [line.strip() for line in (text or "").splitlines() if line.strip()]
        for line in lines[:12]:
            if self._is_table_title(line):
                return line
        for line in lines[:8]:
            lowered = line.lower()
            if any(token in lowered for token in ["năng lượng", "nước", "phát thải", "chất thải", "nhân sự", "đào tạo", "doanh thu"]):
                return line[:160]
        return None

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
            "year_guess": doc.metadata.year_guess,
            "section_title": section_title,
            "page_start": page["page"],
            "page_end": page["page"],
            "chunk_type": "table_section",
            "table_family": family,
            "content": content[:9000],
            "coverage_source": "text_table",
            "quality_score": self._quality_score(content, title=section_title),
            "extraction_method": "text_heuristic",
        }

    def _extract_metric_key_values(self, doc, page: dict) -> list[dict]:
        text = (page.get("text") or "").strip()
        if not text:
            return []

        lowered = text.lower()
        metric_tokens = [
            "tỷ đồng", "ty dong", "triệu đồng", "trieu dong", "sản phẩm", "san pham",
            "hộp sữa", "hop sua", "người", "nguoi", "tấn", "tan", "m3", "kwh",
            "mj", "co2", "nước", "nuoc", "phát thải", "phat thai", "chất thải",
            "chat thai", "cộng đồng", "cong dong", "thiện nguyện", "thien nguyen",
            "quỹ sữa", "quy sua", "bão yagi", "hộp sữa",
        ]
        if not any(token in lowered for token in metric_tokens):
            return []

        lines = [line.strip() for line in text.splitlines() if line.strip()]
        metric_lines = []
        for index, line in enumerate(lines):
            line_lower = line.lower()
            has_number = bool(re.search(r"\d[\d\.,]*", line))
            has_unit = any(token in line_lower for token in metric_tokens)
            if has_number or has_unit:
                start = max(0, index - 2)
                end = min(len(lines), index + 3)
                metric_lines.extend(lines[start:end])

        deduped = list(dict.fromkeys(metric_lines))
        if len(deduped) < 4:
            return []

        content = "\n".join(deduped).strip()
        if len(re.findall(r"\d[\d\.,]*", content)) < 2:
            return []

        family = self._infer_table_family(None, content)
        csr_context = any(token in lowered for token in [
            "cộng đồng", "cong dong", "thiện nguyện", "thien nguyen", "quỹ sữa",
            "quy sua", "bão yagi", "trẻ em", "tre em", "lan tỏa", "lan toa",
        ])
        if any(token in lowered for token in ["thù lao", "thu lao", "lương thưởng", "luong thuong", "hội đồng quản trị", "hoi dong quan tri"]):
            family = "governance_compensation_metrics"
        elif csr_context:
            family = "csr_impact_metrics"

        return [{
            "chunk_id": f"{doc.label}:metric_kv:{page['page']}",
            "document_id": doc.metadata.file_hash[:16],
            "source_file": doc.label,
            "source_path": doc.path,
            "document_type": doc.doc_type,
            "year_guess": doc.metadata.year_guess,
            "section_title": f"{doc.doc_type} metric highlights page {page['page']}",
            "page_start": page["page"],
            "page_end": page["page"],
            "chunk_type": "metric_kv_section",
            "table_family": family,
            "content": content[:9000],
            "coverage_source": "metric_key_value",
            "quality_score": min(1.0, self._quality_score(content, title="metric highlights") + 0.1),
            "extraction_method": "metric_key_value",
        }]

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
        if any(token in haystack for token in ["thù lao", "thu lao", "lương thưởng", "luong thuong", "hội đồng quản trị", "hoi dong quan tri"]):
            return "governance_compensation_metrics"
        if any(token in haystack for token in ["cộng đồng", "cong dong", "thiện nguyện", "thien nguyen", "quỹ sữa", "quy sua", "bão yagi", "trẻ em", "tre em", "lan tỏa", "lan toa"]):
            return "csr_impact_metrics"
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
