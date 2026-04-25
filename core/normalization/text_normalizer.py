"""
Text normalization helpers used before indexing and retrieval.
"""
from __future__ import annotations

import re
import unicodedata


class TextNormalizer:
    SYNONYM_GROUPS = {
        "đhđcđ": ["đại hội đồng cổ đông", "dhdcd", "đhđcđ"],
        "hđqt": ["hội đồng quản trị", "hdqt", "hđqt"],
        "ptbv": ["phát triển bền vững", "ptbv", "sustainability", "esg"],
        "kiểm toán": ["kiểm toán", "kiem toan", "audit", "ý kiến kiểm toán", "opinion"],
        "cổ đông": ["cổ đông", "co dong", "shareholder"],
        "môi trường": ["môi trường", "moi truong", "environmental", "ems"],
        "người lao động": ["người lao động", "nguoi lao dong", "nhân viên", "nhan vien", "employee"],
    }

    LOW_VALUE_PATTERNS = [
        r"\bmục lục\b",
        r"\btable of contents\b",
        r"\bnội dung\b",
        r"\bcopyright\b",
        r"\bmiễn trừ trách nhiệm\b",
        r"\bthông báo\b",
    ]

    def normalize(self, text: str) -> str:
        if not text:
            return ""
        normalized = unicodedata.normalize("NFC", text)
        normalized = normalized.replace("\ufeff", " ").replace("\u200b", " ")
        normalized = re.sub(r"[ \t]+", " ", normalized)
        normalized = re.sub(r"\n{3,}", "\n\n", normalized)
        return normalized.strip()

    def normalize_for_search(self, text: str) -> str:
        normalized = self.normalize(text).lower()
        normalized = self._ascii_fold(normalized)
        normalized = re.sub(r"[^0-9a-zA-Zà-ỹđ\s]", " ", normalized)
        normalized = re.sub(r"\s+", " ", normalized)
        return normalized.strip()

    def expand_term(self, term: str) -> list[str]:
        normalized_term = self.normalize_for_search(term)
        expanded = {normalized_term}
        for canonical, variants in self.SYNONYM_GROUPS.items():
            normalized_variants = {self.normalize_for_search(item) for item in variants}
            if normalized_term == self.normalize_for_search(canonical) or normalized_term in normalized_variants:
                expanded.update(normalized_variants)
                expanded.add(self.normalize_for_search(canonical))
        return [term for term in expanded if term]

    def is_low_value_text(self, text: str) -> bool:
        normalized = self.normalize_for_search(text)
        if len(normalized) < 40:
            return True
        if normalized.count(" ... ") >= 2 or normalized.count(" . . . ") >= 2:
            return True
        return any(re.search(pattern, normalized) for pattern in self.LOW_VALUE_PATTERNS)

    def _ascii_fold(self, text: str) -> str:
        folded = text.replace("đ", "d")
        folded = unicodedata.normalize("NFD", folded)
        return "".join(char for char in folded if unicodedata.category(char) != "Mn")
