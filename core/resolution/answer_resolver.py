"""
Resolve extracted evidence into a question-level answer state.
Enhanced with evidence verification awareness.
"""
from __future__ import annotations


class AnswerResolver:
    def resolve(self, rule: dict, extraction_result: dict) -> dict:
        answer = extraction_result.get("answer", "NULL")
        selected_options = extraction_result.get("selected_options", [])
        evidence_items = extraction_result.get("evidence_items", [])
        source_sections = extraction_result.get("source_sections", [])
        conflict_detected = False if rule.get("is_multi_select") else self._detect_conflict(selected_options, source_sections)

        # Check evidence verification status
        verification = extraction_result.get("evidence_verification")
        evidence_ungrounded = verification and not verification.get("grounded", True)

        if answer in {"", "NULL", "SKIP"} or not evidence_items:
            resolution_status = "insufficient"
            confidence = 0.2 if answer == "SKIP" else 0.3
            resolved_answer = "NULL" if answer in {"", "NULL"} else answer
        elif evidence_ungrounded:
            # Evidence not found in source documents — likely hallucinated
            resolution_status = "weakly_supported"
            confidence = max(0.15, self._confidence_from_evidence(evidence_items) * 0.4)
            resolved_answer = ",".join(selected_options) if rule.get("is_multi_select") and selected_options else (selected_options[0] if selected_options else answer)
        else:
            resolution_status = "supported"
            confidence = self._confidence_from_evidence(evidence_items)
            resolved_answer = ",".join(selected_options) if rule.get("is_multi_select") and selected_options else (selected_options[0] if selected_options else answer)

        if (
            self._score_for_answer(rule, resolved_answer) < 0
            and confidence < 0.65
        ):
            resolved_answer = "NULL"
            selected_options = []
            resolution_status = "insufficient"
            confidence = min(confidence, 0.3)

        return {
            "question_id": rule.get("id", ""),
            "resolved_answer": resolved_answer,
            "selected_options": selected_options,
            "resolution_status": resolution_status,
            "confidence": confidence,
            "conflict_detected": conflict_detected,
            "reason": extraction_result.get("reason", ""),
            "evidence_items": evidence_items,
            "source_sections": source_sections,
            "numeric_extraction": extraction_result.get("numeric_extraction"),
            "numeric_override": extraction_result.get("numeric_override", False),
            "answer_origin": extraction_result.get("answer_origin", ""),
            "parse_status": extraction_result.get("parse_status"),
            "parse_error": extraction_result.get("parse_error"),
            "retry_used": extraction_result.get("retry_used", False),
            "retry_attempts": extraction_result.get("retry_attempts", 0),
            "retry_profiles": extraction_result.get("retry_profiles", []),
        }

    def _score_for_answer(self, rule: dict, answer: str) -> float:
        import re

        logic = str(rule.get("logic", "") or "")
        if not logic or not answer:
            return 0.0
        first_letter = str(answer).split(",")[0].strip().upper()
        for line in logic.splitlines():
            match = re.match(
                rf"\s*{re.escape(first_letter)}[\.\)]\s*([+-]?\d+(?:[.,]\d+)?)",
                line.strip(),
            )
            if match:
                return float(match.group(1).replace(",", "."))
        return 0.0

    def _detect_conflict(self, selected_options: list[str], source_sections: list[dict]) -> bool:
        if len(set(selected_options)) > 1:
            return True
        distinct_docs = {
            (section.get("source_file"), section.get("page_start"), section.get("page_end"))
            for section in source_sections
        }
        return len(distinct_docs) >= 3 and bool(selected_options)

    def _confidence_from_evidence(self, evidence_items: list[dict]) -> float:
        if not evidence_items:
            return 0.0
        scores = [
            float(item.get("confidence", 0.0) or 0.0)
            for item in evidence_items
        ]
        return round(sum(scores) / len(scores), 3)
