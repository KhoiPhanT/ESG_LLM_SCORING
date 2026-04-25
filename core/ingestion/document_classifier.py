"""
Document classification and metadata inference utilities.
"""
from __future__ import annotations

import hashlib
import os
import re
from dataclasses import asdict, dataclass, field
from typing import Iterable


@dataclass
class DocumentMetadata:
    source_path: str
    file_name: str
    file_extension: str
    file_hash: str
    page_count: int
    company_guess: str = ""
    year_guess: int | None = None
    document_type: str = "other"
    classification_confidence: float = 0.0
    classification_reasons: list[str] = field(default_factory=list)
    text_extraction_method: str = "unknown"
    average_ocr_quality: float = 0.0
    needs_review: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


class DocumentClassifier:
    DOC_TYPE_RULES = {
        "resolution": [
            "nghị quyết",
            "nghi quyet",
            "nghiquyet",
            "đhđcđ",
            "dhdcd",
            "đại hội đồng cổ đông",
            "bien ban",
            "biên bản",
        ],
        "financial_report": [
            "báo cáo tài chính",
            "bao cao tai chinh",
            "baocaotaichinh",
            "kiểm toán",
            "kiem toan",
            "kiemtoan",
            "hợp nhất",
            "hop nhat",
            "financial statement",
        ],
        "sustainability_report": [
            "phát triển bền vững",
            "phat trien ben vung",
            "ptbv",
            "sustainability",
            "esg",
        ],
        "annual_report": [
            "báo cáo thường niên",
            "bao cao thuong nien",
            "thuong nien",
            "baocaothuongnien",
            "thuongnien",
            "annual report",
        ],
    }

    def classify(
        self,
        file_path: str,
        pages: Iterable[dict] | None = None,
        company_hint: str = "",
        year_hint: int | None = None,
    ) -> DocumentMetadata:
        abs_path = os.path.abspath(file_path)
        file_name = os.path.basename(abs_path)
        page_list = list(pages or [])
        page_count = len(page_list)
        first_pages_text = "\n".join((page.get("text") or "") for page in page_list[:3])
        combined_text = f"{file_name}\n{first_pages_text}".lower()

        file_hash = self._hash_file(abs_path)
        year_guess = year_hint or self._extract_year(file_name, first_pages_text)
        company_guess = company_hint or self._extract_company(file_name, first_pages_text)
        document_type, confidence, reasons = self._infer_document_type(file_name, combined_text)
        extraction_method = self._summarize_extraction_method(page_list)
        average_ocr_quality = self._average_ocr_quality(page_list)
        needs_review = confidence < 0.75 or document_type == "other"

        return DocumentMetadata(
            source_path=abs_path,
            file_name=file_name,
            file_extension=os.path.splitext(file_name)[1].lower(),
            file_hash=file_hash,
            page_count=page_count,
            company_guess=company_guess,
            year_guess=year_guess,
            document_type=document_type,
            classification_confidence=round(confidence, 2),
            classification_reasons=reasons,
            text_extraction_method=extraction_method,
            average_ocr_quality=round(average_ocr_quality, 3),
            needs_review=needs_review,
        )

    def _hash_file(self, file_path: str) -> str:
        digest = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _extract_year(self, file_name: str, text: str) -> int | None:
        candidates = re.findall(r"(?<!\d)(20\d{2})(?!\d)", f"{file_name}\n{text}")
        return int(candidates[0]) if candidates else None

    def _extract_company(self, file_name: str, text: str) -> str:
        name_tokens = re.findall(r"\b[A-Z]{2,5}\b", file_name.upper())
        if name_tokens:
            return name_tokens[0]

        content_tokens = re.findall(r"\b[A-Z]{2,5}\b", text.upper())
        if content_tokens:
            return content_tokens[0]
        return ""

    def _infer_document_type(self, file_name: str, combined_text: str) -> tuple[str, float, list[str]]:
        lowered_name = file_name.lower()
        scores = {}
        reasons_by_type = {}

        for doc_type, keywords in self.DOC_TYPE_RULES.items():
            score = 0.0
            reasons = []
            for keyword in keywords:
                if keyword in lowered_name:
                    score += 0.7
                    reasons.append(f"filename contains '{keyword}'")
                elif keyword in combined_text:
                    score += 0.4
                    reasons.append(f"first pages mention '{keyword}'")
            if score > 0:
                scores[doc_type] = score
                reasons_by_type[doc_type] = reasons

        if not scores:
            return "other", 0.3, ["No strong filename/content signal found"]

        best_type = max(scores, key=scores.get)
        confidence = min(0.98, 0.45 + scores[best_type] / 2)
        return best_type, confidence, reasons_by_type[best_type]

    def _summarize_extraction_method(self, pages: list[dict]) -> str:
        methods = {page.get("extraction_method", "unknown") for page in pages}
        if not methods:
            return "unknown"
        if methods == {"native"}:
            return "native"
        if methods == {"ocr"}:
            return "ocr"
        if "native" in methods and "ocr" in methods:
            return "mixed"
        return methods.pop()

    def _average_ocr_quality(self, pages: list[dict]) -> float:
        scores = [
            float(page.get("ocr_quality_score", 0.0) or 0.0)
            for page in pages
            if page.get("text")
        ]
        return sum(scores) / len(scores) if scores else 0.0
