"""
Compare scoring output against a manual goldset CSV.
"""
from __future__ import annotations

import csv
import os


class GoldsetBenchmark:
    def load(self, path: str) -> list[dict]:
        if not os.path.exists(path):
            return []
        with open(path, "r", encoding="utf-8") as f:
            return [row for row in csv.DictReader(f) if row.get("question_id")]

    def compare(self, details: list[dict], goldset_path: str) -> dict:
        gold_rows = self.load(goldset_path)
        by_id = {detail.get("id"): detail for detail in details or []}
        compared = []
        answer_matches = 0
        score_gap_total = 0.0
        evidence_hits = 0
        for row in gold_rows:
            detail = by_id.get(row["question_id"], {})
            expected_score = self._to_float(row.get("expected_score"))
            actual_score = self._to_float(detail.get("score"))
            expected_answer = str(row.get("expected_answer", "")).strip().upper()
            actual_answer = str(detail.get("answer", "")).strip().upper()
            answer_match = bool(expected_answer) and expected_answer == actual_answer
            evidence_hit = self._evidence_matches(detail, row)
            if answer_match:
                answer_matches += 1
            if evidence_hit:
                evidence_hits += 1
            gap = expected_score - actual_score
            score_gap_total += abs(gap)
            compared.append({
                "question_id": row["question_id"],
                "expected_answer": expected_answer,
                "actual_answer": actual_answer,
                "expected_score": expected_score,
                "actual_score": actual_score,
                "score_gap": round(gap, 4),
                "answer_match": answer_match,
                "evidence_hit": evidence_hit,
            })
        total = len(compared)
        return {
            "goldset_path": goldset_path,
            "count": total,
            "answer_accuracy": round(answer_matches / total, 4) if total else None,
            "evidence_hit_rate": round(evidence_hits / total, 4) if total else None,
            "absolute_score_gap": round(score_gap_total, 4),
            "items": compared,
        }

    def _to_float(self, value) -> float:
        try:
            return float(value or 0.0)
        except ValueError:
            return 0.0

    def _evidence_matches(self, detail: dict, gold_row: dict) -> bool:
        expected_file = str(gold_row.get("evidence_file", "") or "").strip()
        expected_page = str(gold_row.get("page", "") or "").strip()
        if not expected_file and not expected_page:
            return False
        for item in detail.get("evidence_items", []) or []:
            file_match = not expected_file or expected_file in str(item.get("source_file", ""))
            page_match = not expected_page or expected_page in {
                str(item.get("page_start", "")),
                str(item.get("page_end", "")),
            }
            if file_match and page_match:
                return True
        return False
