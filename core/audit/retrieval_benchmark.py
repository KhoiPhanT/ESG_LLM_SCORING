"""
Benchmark retrieval quality against a small curated gold set.
"""
from __future__ import annotations

import json
import os
from statistics import mean


class RetrievalBenchmark:
    def __init__(self, goldset_path: str):
        self.goldset_path = goldset_path
        self.goldset = self._load_goldset()

    def evaluate(self, engine, rules: list[dict], top_k: int = 3) -> dict:
        indexed_rules = {rule.get("id"): rule for rule in rules}
        entries = []
        for item in self.goldset.get("entries", []):
            qid = item.get("question_id")
            rule = indexed_rules.get(qid)
            if not rule:
                continue
            result = engine.retrieve_for_rule(rule, top_k=top_k)
            candidates = result.get("candidates", [])
            candidate_chunk_ids = [
                self._candidate_chunk_id(candidate) for candidate in candidates
            ]
            expected_chunk_ids = item.get("expected_chunk_ids", [])
            expected_none = bool(item.get("expected_none", False))
            negative_type = item.get("negative_type", "hard")
            hit_rank = None
            if not expected_none:
                for idx, chunk_id in enumerate(candidate_chunk_ids, start=1):
                    if chunk_id in expected_chunk_ids:
                        hit_rank = idx
                        break
            entry = {
                "question_id": qid,
                "expected_none": expected_none,
                "negative_type": negative_type,
                "expected_chunk_ids": expected_chunk_ids,
                "retrieved_chunk_ids": candidate_chunk_ids,
                "hit": bool(hit_rank),
                "hit_rank": hit_rank,
                "top_candidate": candidates[0] if candidates else None,
                "candidate_count": len(candidates),
                "no_evidence_guardrail_pass": expected_none and len(candidates) == 0,
            }
            if expected_none and candidates:
                top = candidates[0]
                entry["no_evidence_guardrail_pass"] = bool(
                    top.get("semantic_score", 0.0) < 0.2 and top.get("score", 0.0) < 5.5
                )
            entries.append(entry)

        positive_entries = [entry for entry in entries if not entry["expected_none"]]
        negative_entries = [entry for entry in entries if entry["expected_none"]]
        hard_negative_entries = [entry for entry in negative_entries if entry.get("negative_type") != "soft"]
        recall_at_k = (
            sum(1 for entry in positive_entries if entry["hit"]) / len(positive_entries)
            if positive_entries else 0.0
        )
        mrr = (
            mean(1 / entry["hit_rank"] for entry in positive_entries if entry["hit_rank"])
            if any(entry["hit_rank"] for entry in positive_entries) else 0.0
        )
        no_evidence_guardrail = (
            sum(1 for entry in hard_negative_entries if entry["no_evidence_guardrail_pass"]) / len(hard_negative_entries)
            if hard_negative_entries else 0.0
        )
        return {
            "goldset_path": self.goldset_path,
            "entry_count": len(entries),
            "positive_entry_count": len(positive_entries),
            "negative_entry_count": len(negative_entries),
            "hard_negative_entry_count": len(hard_negative_entries),
            "recall_at_k": round(recall_at_k, 3),
            "mrr": round(mrr, 3),
            "no_evidence_guardrail": round(no_evidence_guardrail, 3),
            "entries": entries,
        }

    def _load_goldset(self) -> dict:
        if not os.path.exists(self.goldset_path):
            return {"entries": []}
        with open(self.goldset_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _candidate_chunk_id(self, candidate: dict) -> str:
        return candidate.get("chunk_id", "")
