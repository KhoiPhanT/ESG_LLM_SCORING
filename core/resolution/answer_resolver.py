"""
Resolve extracted evidence into a question-level answer state.
Enhanced with evidence verification and self-consistency awareness.
"""
from __future__ import annotations


class AnswerResolver:
    def resolve(self, rule: dict, extraction_result: dict) -> dict:
        answer = extraction_result.get("answer", "NULL")
        selected_options = extraction_result.get("selected_options", [])
        evidence_items = extraction_result.get("evidence_items", [])
        source_sections = extraction_result.get("source_sections", [])
        conflict_detected = self._detect_conflict(selected_options, source_sections)

        # Check evidence verification status
        verification = extraction_result.get("evidence_verification")
        evidence_ungrounded = verification and not verification.get("grounded", True)

        # Check self-consistency
        consistency = extraction_result.get("consistency_check")
        consistency_conflict = consistency and consistency.get("conflict", False)

        if answer in {"", "NULL", "SKIP"} or not evidence_items:
            resolution_status = "insufficient"
            confidence = 0.2 if answer == "SKIP" else 0.3
            resolved_answer = "NULL" if answer in {"", "NULL"} else answer
        elif evidence_ungrounded:
            # Evidence not found in source documents — likely hallucinated
            resolution_status = "weakly_supported"
            confidence = max(0.15, self._confidence_from_evidence(evidence_items) * 0.4)
            resolved_answer = selected_options[0] if selected_options else answer
        elif consistency_conflict:
            # Self-consistency check found a different answer
            verified_answer = consistency.get("verified_answer", answer)
            # Use the more conservative (lower-scoring) answer
            resolved_answer = self._pick_conservative_answer(
                rule, answer, verified_answer, selected_options
            )
            selected_options = [resolved_answer] if resolved_answer != answer else selected_options
            resolution_status = "contested"
            confidence = max(0.25, self._confidence_from_evidence(evidence_items) * 0.6)
            conflict_detected = True
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

    def _pick_conservative_answer(
        self, rule: dict, first_answer: str, verified_answer: str, selected_options: list[str]
    ) -> str:
        """
        When first and verified answers conflict, pick the more conservative one.
        Conservative = lower score or more negative.
        """
        import re

        logic = str(rule.get("logic", "") or "")
        if not logic:
            return verified_answer  # Default to the stricter review answer

        def score_for_letter(letter: str) -> float:
            for line in logic.splitlines():
                match = re.match(
                    rf"\s*{re.escape(letter.upper())}[.\)]\s*([+-]?\d+(?:[.,]\d+)?)",
                    line.strip(),
                )
                if match:
                    return float(match.group(1).replace(",", "."))
            return 0.0

        first_score = score_for_letter(first_answer)
        verified_score = score_for_letter(verified_answer)

        # Pick the lower-scoring (more conservative) answer
        if verified_score <= first_score:
            return verified_answer
        return first_answer

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
