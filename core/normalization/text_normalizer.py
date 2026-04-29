"""
Text normalization helpers used before indexing and retrieval.
Supports industry-specific synonym expansion for domain-aware search.
"""
from __future__ import annotations

import json
import os
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

    INDUSTRY_SYNONYMS_PATH = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "industry_synonyms.json"
    )

    def __init__(self, industry_sector: str = ""):
        self._industry_sector = industry_sector
        self._industry_synonyms: dict[str, list[str]] = {}
        if industry_sector:
            self._load_industry_synonyms(industry_sector)

    def set_industry(self, industry_sector: str) -> None:
        """Set or change the industry sector for domain-aware synonym expansion."""
        if industry_sector != self._industry_sector:
            self._industry_sector = industry_sector
            self._load_industry_synonyms(industry_sector)

    def _load_industry_synonyms(self, industry_sector: str) -> None:
        """Load industry-specific synonyms from JSON file."""
        if not os.path.exists(self.INDUSTRY_SYNONYMS_PATH):
            return

        try:
            with open(self.INDUSTRY_SYNONYMS_PATH, "r", encoding="utf-8") as f:
                all_synonyms = json.load(f)
        except (json.JSONDecodeError, OSError):
            return

        # Load sector-specific + default synonyms
        sector_synonyms = all_synonyms.get(industry_sector, {})
        default_synonyms = all_synonyms.get("_default", {})

        # Merge: sector-specific takes priority
        merged = dict(default_synonyms)
        for key, values in sector_synonyms.items():
            existing = merged.get(key, [])
            merged[key] = list(dict.fromkeys(existing + values))

        self._industry_synonyms = merged

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

        # Core synonym groups
        for canonical, variants in self.SYNONYM_GROUPS.items():
            normalized_variants = {self.normalize_for_search(item) for item in variants}
            if normalized_term == self.normalize_for_search(canonical) or normalized_term in normalized_variants:
                expanded.update(normalized_variants)
                expanded.add(self.normalize_for_search(canonical))

        # Industry-specific synonyms
        for canonical, variants in self._industry_synonyms.items():
            normalized_canonical = self.normalize_for_search(canonical)
            normalized_variants = {self.normalize_for_search(v) for v in variants}
            if normalized_term == normalized_canonical or normalized_term in normalized_variants:
                expanded.update(normalized_variants)
                expanded.add(normalized_canonical)

        return [term for term in expanded if term]

    def get_industry_expansions(self, term: str) -> list[str]:
        """Get only industry-specific expansions for a term (used for query enrichment)."""
        normalized_term = self.normalize_for_search(term)
        expansions = []
        for canonical, variants in self._industry_synonyms.items():
            normalized_canonical = self.normalize_for_search(canonical)
            normalized_variants = {self.normalize_for_search(v) for v in variants}
            if normalized_term == normalized_canonical or normalized_term in normalized_variants:
                expansions.extend(variants)
        return list(dict.fromkeys(expansions))

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
