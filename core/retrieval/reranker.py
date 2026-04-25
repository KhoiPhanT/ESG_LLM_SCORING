"""
Lightweight rule-aware reranker for lexical retrieval candidates.
"""
from __future__ import annotations


class RetrievalReranker:
    def rerank(self, candidates: list[dict], query, rule: dict) -> list[dict]:
        reranked = []
        for candidate in candidates:
            item = dict(candidate)
            rerank_delta, reasons = self._score_candidate(item, query, rule)
            item["rerank_score"] = round(float(item.get("score", 0.0)) + rerank_delta, 4)
            item["rerank_reasons"] = reasons
            reranked.append(item)

        reranked.sort(
            key=lambda item: (
                item.get("rerank_score", item.get("score", 0.0)),
                item.get("score", 0.0),
                item.get("quality_score", 0.0),
            ),
            reverse=True,
        )
        return reranked

    def _score_candidate(self, candidate: dict, query, rule: dict) -> tuple[float, list[str]]:
        delta = 0.0
        reasons = []
        title = str(candidate.get("section_title", "") or "").lower()
        content = str(candidate.get("content", "") or "").lower()
        chunk_type = candidate.get("chunk_type", "")
        document_type = candidate.get("document_type", "")
        table_family = candidate.get("table_family", "")
        question_text = str(rule.get("question", "") or "").lower()
        sub_category = str(rule.get("sub_category", "") or "")

        title_hits = 0
        for term in getattr(query, "intent_terms", []):
            normalized = term.lower()
            if normalized and normalized in title:
                title_hits += 1
        if title_hits:
            delta += min(1.2, 0.45 * title_hits)
            reasons.append(f"intent_title_hits={title_hits}")

        if sub_category in {"Quyền cổ đông", "Công bố thông tin", "Trách nhiệm HĐQT", "Kiểm soát"}:
            if document_type == "resolution":
                delta += 0.35
                reasons.append("governance_resolution_bonus")

        if sub_category == "Công bố thông tin":
            if any(token in title + "\n" + content for token in [
                "thành viên độc lập",
                "cân bằng giới",
                "đa dạng",
                "ủy ban",
                "lương thưởng",
            ]):
                delta += 0.45
                reasons.append("disclosure_governance_keyword_bonus")

        if sub_category == "Bên liên quan":
            if any(token in title + "\n" + content for token in [
                "quan hệ cổ đông",
                "website",
                "công bố thông tin",
                "tiếng anh",
                "độc lập",
            ]):
                delta += 0.4
                reasons.append("related_party_disclosure_bonus")

        if sub_category == "Trách nhiệm HĐQT":
            if any(token in title + "\n" + content for token in [
                "ủy ban kiểm toán",
                "ủy ban quản lý rủi ro",
                "chủ tịch ủy ban",
                "quản lý rủi ro",
            ]):
                delta += 0.55
                reasons.append("board_committee_bonus")

        if sub_category == "Kiểm soát":
            if any(token in title + "\n" + content for token in [
                "kiểm toán nội bộ",
                "kênh tương tác",
                "rủi ro",
                "phát triển bền vững",
                "tổng giám đốc",
            ]):
                delta += 0.4
                reasons.append("control_framework_bonus")

        if sub_category in {"Chính sách", "Quản lý"} and document_type in {"annual_report", "sustainability_report"}:
            delta += 0.2
            reasons.append("policy_management_doc_bonus")

        if sub_category == "Hiệu quả" and (chunk_type == "table_section" or self._looks_tabular(content)):
            delta += 0.55
            reasons.append("performance_table_bonus")

        if sub_category == "Hiệu quả" and table_family in {"environmental_metrics", "workforce_metrics", "financial_metrics"}:
            delta += 0.35
            reasons.append(f"table_family_match={table_family}")

        if sub_category == "Hiệu quả" and document_type == "resolution":
            delta -= 0.55
            reasons.append("performance_resolution_penalty")

        if sub_category == "Hiệu quả" and table_family == "financial_metrics":
            if not any(
                token in content
                for token in [
                    "nang luong",
                    "nuoc",
                    "nuoc thai",
                    "phat thai",
                    "co2",
                    "chat thai",
                    "nguyen vat lieu",
                    "nhan vien",
                    "dao tao",
                    "tai nan",
                    "tuyen dung",
                    "nghi viec",
                ]
            ):
                delta -= 0.45
                reasons.append("generic_financial_table_penalty")

        if sub_category == "Quyền cổ đông" and table_family == "voting_results":
            delta += 0.45
            reasons.append("voting_results_bonus")

        if "kiểm toán" in question_text and document_type == "financial_report":
            delta += 0.45
            reasons.append("audit_financial_report_bonus")

        if any(term in content for term in ["không", "không có", "chưa"]) and self._looks_negative_rule(rule):
            delta += 0.2
            reasons.append("negative_evidence_alignment")

        if candidate.get("low_value"):
            delta -= 0.4
            reasons.append("rerank_low_value_penalty")

        return round(delta, 4), reasons

    def _looks_negative_rule(self, rule: dict) -> bool:
        logic = str(rule.get("logic", "") or "")
        return "-1" in logic or "âm" in logic.lower()

    def _looks_tabular(self, content: str) -> bool:
        lines = [line for line in content.splitlines()[:25] if line.strip()]
        if not lines:
            return False
        dense_lines = 0
        for line in lines:
            digit_count = sum(1 for char in line if char.isdigit())
            if digit_count >= 4:
                dense_lines += 1
        return dense_lines >= 3
