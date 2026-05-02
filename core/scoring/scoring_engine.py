"""
Rule-based scoring engine operating on resolved answers and evidence states.
Bám sát logic bộ câu hỏi VNSI 2025 — không nhân 2, không NULL heuristic.
Enhanced: handles evidence verification states (weakly_supported, contested).
"""
from __future__ import annotations

import re

from core.scoring.score_utils import parse_score_value


class ScoringEngine:
    def score_rule(self, rule: dict, resolution: dict) -> dict:
        answer_letter = resolution.get("resolved_answer", "NULL")
        selected_options = resolution.get("selected_options", [])
        evidence_items = resolution.get("evidence_items", [])
        evidence_present = bool(evidence_items)
        logic = rule.get("logic", "")
        resolution_status = resolution.get("resolution_status", "")
        numeric_data = resolution.get("numeric_extraction")

        numeric_disclosure = self._is_numeric_disclosure_logic(logic)
        numeric_question = rule.get("question_type") in {"numeric_disclosure", "ratio_calculation"}

        # If NumericExtractor found valid data, treat as strong evidence
        has_numeric_evidence = (
            numeric_data is not None
            and numeric_data.get("value") is not None
            and numeric_data.get("confidence", 0) >= 0.5
        )

        # Tính base_score theo loại câu
        if numeric_disclosure and (evidence_present or has_numeric_evidence) and resolution_status != "weakly_supported":
            base_score = 1.0
        elif rule.get("is_multi_select"):
            base_score = self._score_multi_select(rule, selected_options)
        else:
            base_score = self._single_answer_score(logic, answer_letter)

        # Nguyên tắc bám sát VNSI:
        # - Điểm dương cần evidence để công nhận
        # - Điểm âm/0 tính thẳng theo logic
        # - Numeric extraction counts as evidence for numeric questions once
        #   the metric family/unit has passed NumericExtractor guardrails.
        effective_evidence = evidence_present or (has_numeric_evidence and numeric_question)
        if base_score > 0 and not effective_evidence:
            final_score = 0.0
        elif base_score > 0 and resolution_status == "weakly_supported" and not has_numeric_evidence:
            # Evidence exists but is ungrounded (likely hallucinated)
            # → Don't award positive points for hallucinated evidence
            # → But if NumericExtractor found data, trust structured extraction
            final_score = 0.0
        elif base_score > 0 and resolution_status == "contested":
            # Backward-compatible path for old cached contested answers.
            # Recalculate with the resolved answer
            if rule.get("is_multi_select"):
                final_score = self._score_multi_select(rule, selected_options)
            else:
                final_score = self._single_answer_score(logic, answer_letter)
        else:
            final_score = base_score

        return {
            "answer": answer_letter,
            "selected_options": selected_options,
            "base_score": round(base_score, 4),
            "score": round(final_score, 4),
            "evidence_present": evidence_present,
            "resolution_status": self._resolve_status(
                answer_letter,
                effective_evidence,
                final_score,
                resolution_status,
            ),
        }

    def _single_answer_score(self, logic, answer_letter):
        if not logic or logic == "nan" or not answer_letter:
            return 0.0
        answer_letter = answer_letter.strip().upper()

        for line in str(logic).splitlines():
            match = re.match(
                rf"\s*{re.escape(answer_letter)}[.\)]\s*([+-]?\d+(?:[.,]\d+)?)",
                line.strip(),
            )
            if match:
                return parse_score_value(match.group(1))

        if self._is_numeric_disclosure_logic(logic):
            return 1.0 if answer_letter == "A" else 0.0
        return 0.0

    def _is_numeric_disclosure_logic(self, logic) -> bool:
        text = str(logic or "").lower()
        return (
            "đề cập số liệu" in text
            or "đề cập đến số liệu" in text
            or "có đề cập số liệu" in text
            or "có đề cập đến số liệu" in text
        )

    def _score_multi_select(self, rule, selected_options):
        logic = str(rule.get("logic", ""))
        if not logic or not selected_options:
            return 0.0

        # Pattern: "+X trên 1 yêu cầu đáp ứng"
        frac_match = re.search(r"([+-]?\d+(?:[.,]\d+)?)\s*trên\s*1\s*(?:yêu cầu|điều kiện)", logic, re.IGNORECASE)
        if frac_match:
            raw_value = frac_match.group(1).replace(",", ".")
            per_item = parse_score_value(raw_value)

            # Sanity check: VNSI per-item scores are always ≤ 1.0
            # Values like "0125" (missing decimal) should be "0.125"
            if per_item > 1.0 and raw_value.startswith("0"):
                per_item = float("0." + raw_value.lstrip("0"))
            elif per_item > 1.0:
                per_item = per_item / (10 ** len(raw_value.split(".")[-1])) if "." in raw_value else per_item / 1000

            options_text = str(rule.get("options", ""))
            valid_letters = set(re.findall(r"(^|\n)([A-Z])[.\)]", options_text, re.MULTILINE))
            letters = [letter for _, letter in valid_letters]
            # If options text is missing, accept all selected options
            effective = [opt for opt in selected_options if opt in letters] if letters else selected_options
            return round(per_item * len(effective), 4)

        # Cộng dồn ẩn: sum score of each selected option
        return round(sum(self._single_answer_score(logic, opt) for opt in selected_options), 4)



    def _resolve_status(self, answer_letter, evidence_present, final_score, resolution_status=""):
        if resolution_status in {"weakly_supported", "contested"}:
            return resolution_status
        if answer_letter in {"", "NULL", "SKIP"} or not evidence_present:
            return "insufficient"
        if final_score < 0:
            return "contradicted"
        return "supported"
