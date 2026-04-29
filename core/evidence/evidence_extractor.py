"""
Evidence-first extraction layer for VNSI scoring questions.
Enhanced with evidence verification and optional self-consistency checking.
"""
from __future__ import annotations

from core.evidence.evidence_verifier import EvidenceVerifier


class EvidenceExtractor:
    def __init__(self, llm_client=None, enable_self_consistency: bool = True):
        self.llm_client = llm_client
        self.verifier = EvidenceVerifier()
        self.enable_self_consistency = enable_self_consistency

    def extract(self, rule: dict, context_bundle: dict) -> dict:
        question_id = rule.get("id", "")
        source_sections = context_bundle.get("sections", [])
        context = context_bundle.get("context", "")

        if self.llm_client:
            llm_result = self.llm_client.ask_vnsi_question(
                context=context,
                question=rule.get("question", ""),
                options=rule.get("options", ""),
                q_id=question_id,
                is_multi_select=rule.get("is_multi_select", False),
            )
        else:
            llm_result = {
                "answer": "NULL",
                "selected_options": [],
                "reason": "LLM không khả dụng",
                "evidence": None,
            }

        answer = str(llm_result.get("answer", "NULL")).strip().upper() or "NULL"
        selected_options = self._normalize_selected_options(llm_result, answer)
        evidence_items = self._build_evidence_items(
            raw_evidence=llm_result.get("evidence"),
            source_sections=source_sections,
        )

        # Verify evidence grounding
        verification = None
        if evidence_items and source_sections:
            raw_evidence = llm_result.get("evidence", "")
            verification = self.verifier.verify(str(raw_evidence), source_sections)
            if verification and not verification["grounded"]:
                # Hallucinated evidence — demote confidence
                for item in evidence_items:
                    item["confidence"] = max(0.1, float(item.get("confidence", 0.5)) * 0.35)
                    item["verification_status"] = "ungrounded"
            elif verification and verification["grounded"]:
                for item in evidence_items:
                    item["verification_status"] = "grounded"

        # Self-consistency check for ALL non-NULL answers
        # Both false positives AND false negatives are harmful for fair scoring
        consistency_result = None
        if (
            self.enable_self_consistency
            and self.llm_client
            and answer not in {"", "NULL", "SKIP"}
            and evidence_items
        ):
            consistency_result = self._self_consistency_check(
                rule, context, answer, selected_options, llm_result.get("evidence")
            )

        if answer in {"", "NULL", "SKIP"} or not evidence_items:
            extraction_status = "insufficient"
        elif verification and not verification["grounded"]:
            extraction_status = "weakly_supported"
        else:
            extraction_status = "supported"

        result = {
            "question_id": question_id,
            "answer": answer,
            "selected_options": selected_options,
            "reason": llm_result.get("reason", ""),
            "raw_evidence": llm_result.get("evidence"),
            "evidence_items": evidence_items,
            "source_sections": source_sections,
            "status": extraction_status,
        }

        if verification:
            result["evidence_verification"] = verification
        if consistency_result:
            result["consistency_check"] = consistency_result
            if consistency_result.get("conflict"):
                result["status"] = "conflicted"

        return result

    def _self_consistency_check(
        self,
        rule: dict,
        context: str,
        first_answer: str,
        first_selected: list[str],
        first_evidence: str | None,
    ) -> dict:
        """
        Run a verification pass: ask the LLM to critically evaluate the first answer.
        Returns conflict status and second-pass details.
        """
        question_id = rule.get("id", "")
        question = rule.get("question", "")
        options = rule.get("options", "")

        verify_prompt = f"""
You are a STRICT ESG auditor performing a verification check.
A previous analyst answered the following VNSI question. Your job is to verify whether this answer is correct.

QUESTION [{question_id}]: {question}

OPTIONS:
{options}

PREVIOUS ANSWER: {first_answer}
PREVIOUS EVIDENCE: {first_evidence or "No evidence provided"}

REPORT CONTEXT:
{context[:12000]}

CRITICAL VERIFICATION RULES:
- In VNSI scoring, an official corporate statement, policy mention, system implementation, or certification (e.g. ISO) is considered VALID evidence for compliance.
- Confirm the positive answer if the evidence shows the company actively manages, tracks, or has a policy for the topic, even if exact quantitative numbers are missing.
- Only override to a negative option if the evidence is completely irrelevant, or the report is absolutely silent on the specific topic.
- A general "Phát triển bền vững" is not enough, but a statement like "chúng tôi có quy trình kiểm soát..." or "đã triển khai hệ thống..." IS sufficient evidence for a positive answer.
- Think step-by-step concisely before deciding, keeping your thinking block short.

Respond with JSON only:
{{"verified_answer": "<your answer letter>", "agree": true/false, "reason": "<1 sentence explanation>"}}"""

        messages = [{"role": "user", "content": verify_prompt}]
        raw = self.llm_client._call(messages, temperature=0.1, max_tokens=2048)
        parsed = self.llm_client._parse_json(raw)

        if not parsed:
            return {"conflict": False, "reason": "Verification parse failed"}

        verified_answer = str(parsed.get("verified_answer", first_answer)).strip().upper()
        agrees = parsed.get("agree", True)

        return {
            "conflict": not agrees and verified_answer != first_answer,
            "first_answer": first_answer,
            "verified_answer": verified_answer,
            "agrees": agrees,
            "reason": parsed.get("reason", ""),
        }

    def _normalize_selected_options(self, llm_result: dict, answer: str) -> list[str]:
        selected_options = [
            str(opt).strip().upper()
            for opt in llm_result.get("selected_options", [])
            if str(opt).strip()
        ]
        if not selected_options and answer not in {"", "NULL"}:
            selected_options = [part.strip().upper() for part in answer.split(",") if part.strip()]
        return selected_options

    def _build_evidence_items(self, raw_evidence, source_sections: list[dict]) -> list[dict]:
        if not raw_evidence or str(raw_evidence).strip().lower() == "null":
            return []

        evidence_text = str(raw_evidence).strip()
        items = []

        for section in source_sections[:3]:
            items.append(
                {
                    "quote": evidence_text,
                    "source_file": section.get("source_file"),
                    "source_path": section.get("source_path"),
                    "document_type": section.get("document_type"),
                    "page_start": section.get("page_start"),
                    "page_end": section.get("page_end"),
                    "retrieval_score": section.get("score"),
                    "confidence": self._estimate_confidence(section),
                }
            )

        if items:
            return items

        return [
            {
                "quote": evidence_text,
                "source_file": None,
                "source_path": None,
                "document_type": None,
                "page_start": None,
                "page_end": None,
                "retrieval_score": None,
                "confidence": 0.3,
            }
        ]

    def _estimate_confidence(self, section: dict) -> float:
        retrieval_score = float(section.get("score", 0.0) or 0.0)
        quality_score = float(section.get("quality_score", 0.0) or 0.0)
        confidence = 0.35 + min(0.35, retrieval_score / 20) + min(0.2, quality_score / 4)
        return round(min(0.95, confidence), 3)

    def _is_positive_answer(self, rule: dict, answer: str) -> bool:
        """Check if the answer would score positive points (> 0)."""
        import re

        logic = str(rule.get("logic", "") or "")
        if not logic or logic == "nan":
            return True  # When in doubt, check it

        answer = answer.strip().upper()
        for line in logic.splitlines():
            match = re.match(
                rf"\s*{re.escape(answer)}[.\)]\s*([+-]?\d+(?:[.,]\d+)?)",
                line.strip(),
            )
            if match:
                score = float(match.group(1).replace(",", "."))
                return score > 0

        return True  # When in doubt, check it
