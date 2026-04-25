"""
Audit retrieval quality against current document set.
"""
from __future__ import annotations

from statistics import mean

from core.retrieval.retrieval_engine import RetrievalEngine
from core.audit.retrieval_benchmark import RetrievalBenchmark


class RetrievalAudit:
    def __init__(self, corpus):
        self.corpus = corpus
        self.engine = RetrievalEngine(corpus)

    def audit_rules(
        self,
        rules: list[dict],
        sample_size: int | None = None,
        top_k: int = 5,
        benchmark_path: str | None = None,
    ) -> dict:
        selected_rules = rules[:sample_size] if sample_size else rules
        entries = []
        for rule in selected_rules:
            retrieval = self.engine.retrieve_for_rule(rule, top_k=top_k)
            candidates = retrieval["candidates"]
            low_value_ratio = (
                sum(1 for candidate in candidates if candidate.get("low_value")) / len(candidates)
                if candidates else 1.0
            )
            preferred_hits = 0
            exact_phrase_hit = 0
            reranked_improvement = 0
            table_candidate_ratio = 0
            semantic_support_ratio = 0
            labeled_noise_ratio = 0
            preferred_types = retrieval["query"].get("preferred_document_types", [])
            if preferred_types:
                preferred_hits = sum(
                    1 for candidate in candidates if candidate["document_type"] in preferred_types
                )
            exact_phrase_hit = sum(1 for candidate in candidates if candidate.get("exact_phrase_hits"))
            reranked_improvement = sum(
                1
                for candidate in candidates
                if candidate.get("rerank_score", candidate.get("score", 0.0)) > candidate.get("score", 0.0)
            )
            table_candidate_ratio = sum(
                1 for candidate in candidates if candidate.get("chunk_type") == "table_section"
            )
            semantic_support_ratio = sum(
                1 for candidate in candidates if float(candidate.get("semantic_score", 0.0) or 0.0) >= 0.15
            )
            labeled_noise_ratio = sum(
                1 for candidate in candidates if candidate.get("labels") and any(
                    label in candidate.get("labels", [])
                    for label in ["low_value", "front_matter", "toc_like", "contact_boilerplate", "no_evidence_hint"]
                )
            )
            entries.append(
                {
                    "question_id": rule.get("id"),
                    "question": rule.get("question", "")[:180],
                    "sub_category": rule.get("sub_category"),
                    "preferred_document_types": preferred_types,
                    "candidate_count": len(candidates),
                    "low_value_ratio": round(low_value_ratio, 3),
                    "preferred_hit_ratio": round(preferred_hits / len(candidates), 3) if candidates else 0.0,
                    "exact_phrase_hit_ratio": round(exact_phrase_hit / len(candidates), 3) if candidates else 0.0,
                    "reranked_improvement_ratio": round(reranked_improvement / len(candidates), 3) if candidates else 0.0,
                    "table_candidate_ratio": round(table_candidate_ratio / len(candidates), 3) if candidates else 0.0,
                    "semantic_support_ratio": round(semantic_support_ratio / len(candidates), 3) if candidates else 0.0,
                    "labeled_noise_ratio": round(labeled_noise_ratio / len(candidates), 3) if candidates else 0.0,
                    "top_candidate": candidates[0] if candidates else None,
                    "all_candidates": candidates,
                }
            )

        result = {
            "question_count": len(entries),
            "questions_with_results": sum(1 for entry in entries if entry["candidate_count"] > 0),
            "average_low_value_ratio": round(mean(entry["low_value_ratio"] for entry in entries), 3) if entries else 0.0,
            "average_preferred_hit_ratio": round(
                mean(entry["preferred_hit_ratio"] for entry in entries), 3
            ) if entries else 0.0,
            "average_exact_phrase_hit_ratio": round(
                mean(entry["exact_phrase_hit_ratio"] for entry in entries), 3
            ) if entries else 0.0,
            "average_reranked_improvement_ratio": round(
                mean(entry["reranked_improvement_ratio"] for entry in entries), 3
            ) if entries else 0.0,
            "average_table_candidate_ratio": round(
                mean(entry["table_candidate_ratio"] for entry in entries), 3
            ) if entries else 0.0,
            "average_semantic_support_ratio": round(
                mean(entry["semantic_support_ratio"] for entry in entries), 3
            ) if entries else 0.0,
            "average_labeled_noise_ratio": round(
                mean(entry["labeled_noise_ratio"] for entry in entries), 3
            ) if entries else 0.0,
            "entries": entries,
        }
        if benchmark_path:
            result["benchmark"] = RetrievalBenchmark(benchmark_path).evaluate(
                self.engine,
                selected_rules,
                top_k=top_k,
            )
        return result
