"""
Rule-based scoring engine operating on resolved answers and evidence states.
Bám sát logic bộ câu hỏi VNSI 2025 — không nhân 2, không NULL heuristic.
Enhanced: handles evidence verification states (weakly_supported, contested).
"""
from __future__ import annotations

import re


class ScoringEngine:
    def score_rule(self, rule: dict, resolution: dict) -> dict:
        answer_letter = resolution.get("resolved_answer", "NULL")
        selected_options = resolution.get("selected_options", [])
        evidence_items = resolution.get("evidence_items", [])
        evidence_present = bool(evidence_items)
        logic = rule.get("logic", "")
        resolution_status = resolution.get("resolution_status", "")

        # Tính base_score theo loại câu
        if rule.get("is_multi_select"):
            base_score = self._score_multi_select(rule, selected_options)
        else:
            base_score = self._single_answer_score(logic, answer_letter)

        # Nguyên tắc bám sát VNSI:
        # - Điểm dương cần evidence để công nhận
        # - Điểm âm/0 tính thẳng theo logic
        if base_score > 0 and not evidence_present:
            final_score = 0.0
        elif base_score > 0 and resolution_status == "weakly_supported":
            # Evidence exists but is ungrounded (likely hallucinated)
            # → Don't award positive points for hallucinated evidence
            final_score = 0.0
        elif base_score > 0 and resolution_status == "contested":
            # Self-consistency check disagreed — use the resolved (conservative) answer
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
            "resolution_status": self._resolve_status(answer_letter, evidence_present, final_score, resolution_status),
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
                return float(match.group(1).replace(",", "."))

        if "nếu có đề cập số liệu" in str(logic).lower():
            return 1.0 if answer_letter == "A" else 0.0
        return 0.0

    def _score_multi_select(self, rule, selected_options):
        logic = str(rule.get("logic", ""))
        if not logic or not selected_options:
            return 0.0

        # Pattern: "+X trên 1 yêu cầu đáp ứng"
        frac_match = re.search(r"([+-]?\d+(?:[.,]\d+)?)\s*trên\s*1\s*(?:yêu cầu|điều kiện)", logic, re.IGNORECASE)
        if frac_match:
            raw_value = frac_match.group(1).replace(",", ".")
            per_item = float(raw_value)

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
