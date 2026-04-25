"""
Generate review queues from retrieval/scoring audit artifacts.
"""
from __future__ import annotations


class ReviewConsole:
    def build_retrieval_review_list(self, audit_result: dict) -> list[dict]:
        review_items = []
        for entry in audit_result.get("entries", []):
            reasons = []
            if entry.get("candidate_count", 0) == 0:
                reasons.append("no_candidates")
            if entry.get("low_value_ratio", 0) > 0.34:
                reasons.append("too_much_low_value_content")
            if entry.get("preferred_hit_ratio", 0) < 0.5:
                reasons.append("preferred_doc_type_not_dominant")
            if entry.get("exact_phrase_hit_ratio", 0) == 0:
                reasons.append("no_exact_phrase_match")

            top_candidate = entry.get("top_candidate") or {}
            if top_candidate and top_candidate.get("score", 0) < 4:
                reasons.append("top_score_is_weak")
            if top_candidate and not top_candidate.get("rerank_reasons") and entry.get("exact_phrase_hit_ratio", 0) < 0.5:
                reasons.append("reranker_found_no_extra_signal")
            if entry.get("sub_category") == "Hiệu quả" and entry.get("table_candidate_ratio", 0) < 0.3:
                reasons.append("too_few_table_candidates_for_performance_question")

            if reasons:
                review_items.append(
                    {
                        "question_id": entry.get("question_id"),
                        "question": entry.get("question"),
                        "sub_category": entry.get("sub_category"),
                        "reasons": reasons,
                        "top_candidate": {
                            "source_file": top_candidate.get("source_file"),
                            "document_type": top_candidate.get("document_type"),
                            "page_start": top_candidate.get("page_start"),
                            "page_end": top_candidate.get("page_end"),
                            "score": top_candidate.get("score"),
                            "rerank_score": top_candidate.get("rerank_score"),
                            "rerank_reasons": top_candidate.get("rerank_reasons"),
                        } if top_candidate else None,
                    }
                )
        return review_items

    def build_scoring_review_list(self, scoring_details: list[dict]) -> list[dict]:
        review_items = []
        for detail in scoring_details:
            reasons = []
            if detail.get("resolution_status") == "insufficient":
                reasons.append("insufficient_evidence")
            if detail.get("conflict_detected"):
                reasons.append("conflicting_sources")
            if detail.get("confidence", 1.0) < 0.55:
                reasons.append("low_confidence")
            if detail.get("score", 0) != 0 and not detail.get("evidence_present"):
                reasons.append("scored_without_evidence")
            if detail.get("score", 0) > 0 and len(detail.get("evidence_items", [])) == 1:
                reasons.append("thin_positive_evidence")
            if detail.get("score", 0) < 0:
                reasons.append("negative_score_requires_review")

            if reasons:
                top_evidence = (detail.get("evidence_items") or [None])[0]
                review_items.append(
                    {
                        "question_id": detail.get("id"),
                        "question": detail.get("question"),
                        "answer": detail.get("answer"),
                        "score": detail.get("score"),
                        "resolution_status": detail.get("resolution_status"),
                        "confidence": detail.get("confidence"),
                        "reasons": reasons,
                        "top_evidence": {
                            "source_file": top_evidence.get("source_file"),
                            "page_start": top_evidence.get("page_start"),
                            "page_end": top_evidence.get("page_end"),
                            "confidence": top_evidence.get("confidence"),
                        } if top_evidence else None,
                    }
                )
        return review_items
