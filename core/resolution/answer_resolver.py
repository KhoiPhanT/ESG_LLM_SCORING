"""
Resolve extracted evidence into a question-level answer state.
"""
from __future__ import annotations


class AnswerResolver:
    def resolve(self, rule: dict, extraction_result: dict) -> dict:
        answer = extraction_result.get("answer", "NULL")
        selected_options = extraction_result.get("selected_options", [])
        evidence_items = extraction_result.get("evidence_items", [])
        source_sections = extraction_result.get("source_sections", [])
        conflict_detected = self._detect_conflict(selected_options, source_sections)

        if answer in {"", "NULL", "SKIP"} or not evidence_items:
            resolution_status = "insufficient"
            confidence = 0.2 if answer == "SKIP" else 0.3
            resolved_answer = "NULL" if answer in {"", "NULL"} else answer
        else:
            resolution_status = "supported"
            confidence = self._confidence_from_evidence(evidence_items)
            resolved_answer = selected_options[0] if selected_options else answer

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
        }

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
