"""
Evidence-first extraction layer for VNSI scoring questions.
"""
from __future__ import annotations


class EvidenceExtractor:
    def __init__(self, llm_client=None):
        self.llm_client = llm_client

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

        if answer in {"", "NULL", "SKIP"} or not evidence_items:
            extraction_status = "insufficient"
        else:
            extraction_status = "supported"

        return {
            "question_id": question_id,
            "answer": answer,
            "selected_options": selected_options,
            "reason": llm_result.get("reason", ""),
            "raw_evidence": llm_result.get("evidence"),
            "evidence_items": evidence_items,
            "source_sections": source_sections,
            "status": extraction_status,
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
