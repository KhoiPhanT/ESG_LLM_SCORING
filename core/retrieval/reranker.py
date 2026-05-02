"""
Lightweight rule-aware reranker for lexical retrieval candidates.
"""
from __future__ import annotations

import re


class RetrievalReranker:
    def __init__(self, target_year: int | None = None):
        self.target_year = target_year or 2024

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
        year_guess = self._effective_year_guess(candidate)
        question_text = str(rule.get("question", "") or "").lower()
        sub_category = str(rule.get("sub_category", "") or "")
        qid = str(rule.get("id", "") or "")
        time_policy = str(rule.get("time_policy", "") or "")
        plan = rule.get("retrieval_plan") if isinstance(rule.get("retrieval_plan"), dict) else {}
        plan_text = f"{title}\n{content}"
        evidence_profile = str(plan.get("evidence_profile") or "")

        if year_guess:
            year_diff = int(year_guess) - int(self.target_year)
            if year_diff == 0:
                delta += 1.3
                reasons.append("current_year_boost")
            elif year_diff == 1:
                delta += 0.25
                reasons.append("adjacent_future_year_boost")
            elif year_diff < 0:
                historical_allowed = time_policy in {"historical_allowed", "latest_valid_allowed"}
                penalty = 0.15 if historical_allowed else min(2.4, abs(year_diff) * 0.8)
                delta -= penalty
                reasons.append(f"old_year_penalty={abs(year_diff)}yr")
            elif year_diff > 1:
                delta -= min(3.0, year_diff * 0.8)
                reasons.append(f"future_year_penalty={year_diff}yr")

        preferred_types = getattr(query, "preferred_document_types", []) or []
        if preferred_types and document_type not in preferred_types:
            delta -= 1.0
            reasons.append("non_preferred_doc_type_penalty")

        required_types = self._plan_list(plan, "required_doc_types")
        if required_types:
            if document_type in required_types:
                delta += 0.9
                reasons.append("plan_required_doc_type_match")
            else:
                delta -= 1.2
                reasons.append("plan_required_doc_type_penalty")

        plan_year_policy = str(plan.get("year_policy") or "").strip()
        if year_guess and plan_year_policy:
            year_diff = int(year_guess) - int(self.target_year)
            if plan_year_policy == "current_year_required":
                if year_diff == 0:
                    delta += 0.7
                    reasons.append("plan_current_year_match")
                elif year_diff < 0:
                    delta -= min(2.0, abs(year_diff) * 0.7)
                    reasons.append("plan_current_year_old_penalty")
                elif year_diff > 0:
                    delta -= min(2.6, year_diff * 0.8)
                    reasons.append("plan_current_year_future_penalty")
            elif plan_year_policy in {"historical_allowed", "latest_valid_allowed"} and year_diff < 0:
                delta += 0.2
                reasons.append("plan_historical_allowed")
            elif plan_year_policy != "future_target_allowed" and year_diff > 1:
                delta -= 1.0
                reasons.append("plan_future_doc_penalty")

        must_have_terms = self._plan_list(plan, "must_have_terms")
        if must_have_terms:
            hits = self._count_contains(plan_text, must_have_terms)
            if hits:
                delta += min(2.0, 0.55 * hits)
                reasons.append(f"plan_must_have_hits={hits}")
            else:
                delta -= 0.65
                reasons.append("plan_must_have_missing")

        avoid_terms = self._plan_list(plan, "avoid_terms")
        if avoid_terms:
            hits = self._count_contains(plan_text, avoid_terms)
            if hits:
                delta -= min(3.0, 1.25 * hits)
                reasons.append(f"plan_avoid_hits={hits}")

        evidence_shape = str(plan.get("evidence_shape") or "")
        if evidence_shape in {"metric_table", "financial_table", "numeric_value"}:
            if chunk_type in {"table_section", "metric_kv_section"}:
                delta += 0.8
                reasons.append(f"plan_{evidence_shape}_chunk_bonus")
            elif table_family:
                delta += 0.45
                reasons.append(f"plan_{evidence_shape}_family_bonus")
            else:
                delta -= 0.35
                reasons.append(f"plan_{evidence_shape}_non_table_penalty")
        elif evidence_shape == "governance_profile":
            if any(token in plan_text for token in ["trình độ", "học vấn", "chuyên ngành", "kinh nghiệm", "thành viên hội đồng quản trị"]):
                delta += 0.8
                reasons.append("plan_governance_profile_bonus")
        elif evidence_shape == "meeting_resolution":
            if document_type == "resolution":
                delta += 0.75
                reasons.append("plan_meeting_resolution_bonus")
        elif evidence_shape == "certificate":
            if any(token in plan_text for token in ["iso", "chứng nhận", "chung nhan", "certification"]):
                delta += 0.7
                reasons.append("plan_certificate_bonus")

        if plan_year_policy != "future_target_allowed" and evidence_profile not in {"policy_public", "metric_disclosure", "ratio_with_revenue"}:
            if any(token in plan_text for token in ["2050", "2060", "trung hòa carbon", "net zero", "net-zero"]):
                delta -= 0.75
                reasons.append("future_target_noise_penalty")

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
                "thù lao",
                "trình độ",
                "kinh nghiệm",
                "chuyên môn",
            ]):
                delta += 0.45
                reasons.append("disclosure_governance_keyword_bonus")

        if qid.startswith("G.") and any(token in question_text for token in ["thù lao", "lương thưởng"]):
            if any(token in title + "\n" + content for token in ["thù lao", "lương thưởng", "hội đồng quản trị", "triệu đồng"]):
                delta += 1.0
                reasons.append("board_compensation_bonus")

        if qid.startswith("G.") and any(token in question_text for token in ["đa dạng", "kiến thức", "kinh nghiệm"]):
            if any(token in title + "\n" + content for token in ["trình độ", "học vấn", "chuyên ngành", "kinh nghiệm", "thành viên hội đồng quản trị"]):
                delta += 1.0
                reasons.append("board_profile_bonus")

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

        if sub_category in {"Chính sách", "Quản lý"} and document_type in {"annual_report", "sustainability_report", "policy_document"}:
            delta += 0.2
            reasons.append("policy_management_doc_bonus")

        if evidence_profile == "policy_public":
            if document_type == "policy_document":
                delta += 1.6
                reasons.append("policy_document_bonus")
            if any(token in plan_text for token in ["chính sách", "chinh sach", "cam kết", "cam ket"]):
                delta += 0.7
                reasons.append("policy_text_bonus")
            if any(token in plan_text for token in ["công khai", "cong khai", "minh bạch", "minh bach", "công bố", "cong bo"]):
                delta += 0.7
                reasons.append("public_disclosure_bonus")

        if evidence_profile == "ratio_with_revenue":
            if table_family == "environmental_metrics":
                delta += 0.75
                reasons.append("ratio_numerator_table_bonus")
            if table_family == "financial_metrics" or any(token in plan_text for token in ["doanh thu", "tong doanh thu", "revenue"]):
                delta += 0.75
                reasons.append("ratio_denominator_revenue_bonus")

        if sub_category == "Hiệu quả" and (chunk_type == "table_section" or self._looks_tabular(content)):
            delta += 0.55
            reasons.append("performance_table_bonus")

        if sub_category == "Hiệu quả" and table_family in {"environmental_metrics", "workforce_metrics", "financial_metrics"}:
            delta += 0.35
            reasons.append(f"table_family_match={table_family}")

        if table_family == "csr_impact_metrics":
            delta += 0.75
            reasons.append("csr_impact_metric_bonus")

        if any(token in question_text for token in ["cộng đồng", "thiện nguyện", "trách nhiệm xã hội", "truyền thông"]):
            if any(token in title + "\n" + content for token in [
                "quỹ sữa", "vươn cao", "thiện nguyện", "cứu trợ", "đóng góp cộng đồng",
                "cộng đồng", "lan tỏa", "gắn kết yêu thương", "bão yagi", "hỗ trợ",
            ]):
                delta += 1.0
                reasons.append("csr_content_match_bonus")

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
                    "cộng đồng",
                    "thiện nguyện",
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

        if self._has_explicit_negative_evidence(content) and self._looks_negative_rule(rule):
            delta += 0.2
            reasons.append("negative_evidence_alignment")

        if candidate.get("low_value"):
            delta -= 0.4
            reasons.append("rerank_low_value_penalty")

        return round(delta, 4), reasons

    def _plan_list(self, plan: dict, field: str) -> list[str]:
        if not isinstance(plan, dict):
            return []
        value = plan.get(field)
        if isinstance(value, str):
            raw_items = [value]
        elif isinstance(value, (list, tuple, set)):
            raw_items = list(value)
        else:
            return []
        items = []
        for item in raw_items:
            text = str(item).strip().lower()
            if text and text not in {"null", "none", "n/a"}:
                items.append(text)
        return list(dict.fromkeys(items))

    def _effective_year_guess(self, candidate: dict) -> int | None:
        year_guess = candidate.get("year_guess")
        if year_guess:
            try:
                return int(year_guess)
            except (TypeError, ValueError):
                pass

        text = "\n".join([
            str(candidate.get("source_file", "") or ""),
            str(candidate.get("section_title", "") or ""),
            str(candidate.get("content", "") or "")[:1200],
        ])
        years = []
        for match in re.finditer(r"\b(20[0-4]\d)\b", text):
            try:
                year = int(match.group(1))
            except ValueError:
                continue
            if 2010 <= year <= 2049:
                years.append(year)
        if not years:
            return None
        if self.target_year in years:
            return self.target_year
        past_or_current = [year for year in years if year <= self.target_year]
        if past_or_current:
            return max(past_or_current)
        return min(years)

    def _count_contains(self, haystack: str, terms: list[str]) -> int:
        return sum(1 for term in terms if term and term in haystack)

    def _looks_negative_rule(self, rule: dict) -> bool:
        logic = str(rule.get("logic", "") or "")
        return "-1" in logic or "âm" in logic.lower()

    def _has_explicit_negative_evidence(self, content: str) -> bool:
        negative_patterns = [
            "không có chính sách",
            "khong co chinh sach",
            "không công bố",
            "khong cong bo",
            "chưa công bố",
            "chua cong bo",
            "không thực hiện",
            "khong thuc hien",
        ]
        return any(pattern in content for pattern in negative_patterns)

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
