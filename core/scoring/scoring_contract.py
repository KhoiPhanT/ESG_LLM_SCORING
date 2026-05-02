"""
Workbook-first scoring contract for VNSI rules.

This module keeps aggregation and audit math tied to the parsed workbook logic.
It does not multiply scores or normalize them through factor max rows.
"""
from __future__ import annotations

from collections import defaultdict
import csv
import json
import os


class ScoringContract:
    def __init__(self, rules: list[dict], factor_max_scores: dict | None = None):
        self.rules = rules or []
        self.factor_max_scores = factor_max_scores or {}
        self.rules_by_id = {rule.get("id"): rule for rule in self.rules}
        self.max_by_question = {
            rule.get("id"): float(rule.get("max_score", 0.0) or 0.0)
            for rule in self.rules
        }
        self.max_by_factor = self._sum_by("factor", self.max_by_question)
        self.max_by_pillar = self._sum_by("pillar", self.max_by_question)
        self.raw_max = round(sum(self.max_by_question.values()), 4)

    def summarize(self, details: list[dict], weights: dict | None = None) -> dict:
        weights = weights or {"E": 0.0, "S": 0.0, "G": 0.0}
        raw_by_factor = defaultdict(float)
        raw_by_pillar = defaultdict(float)
        for detail in details or []:
            rule = self.rules_by_id.get(detail.get("id"), {})
            factor = detail.get("factor") or rule.get("factor") or ""
            pillar = detail.get("pillar") or rule.get("pillar") or factor[:1]
            score = float(detail.get("score", 0.0) or 0.0)
            if factor:
                raw_by_factor[factor] += score
            if pillar:
                raw_by_pillar[pillar] += score

        factor_scores = {}
        for factor in sorted(set(self.max_by_factor) | set(raw_by_factor)):
            max_points = float(self.max_by_factor.get(factor, 0.0) or 0.0)
            raw_score = float(raw_by_factor.get(factor, 0.0) or 0.0)
            max_info = self.factor_max_scores.get(factor, {})
            factor_scores[factor] = {
                "raw_score": round(raw_score, 4),
                "max_points": round(max_points, 4),
                "percentage": round(max(0.0, raw_score) / max_points * 100, 2) if max_points > 0 else 0.0,
                "pillar": max_info.get("pillar", factor[:1]),
                "label": max_info.get("content", factor),
                "workbook_factor_max_points": float(max_info.get("max_points", 0.0) or 0.0),
                "max_mismatch": round(float(max_info.get("max_points", 0.0) or 0.0) - max_points, 4),
            }

        pillar_factor_percentages = defaultdict(list)
        for factor, data in factor_scores.items():
            pillar = data.get("pillar") or factor[:1]
            if data.get("max_points", 0.0) > 0:
                pillar_factor_percentages[pillar].append(float(data.get("percentage", 0.0) or 0.0))

        pillar_scores = {}
        for pillar in ["E", "S", "G"]:
            raw_score = float(raw_by_pillar.get(pillar, 0.0) or 0.0)
            max_points = float(self.max_by_pillar.get(pillar, 0.0) or 0.0)
            raw_percentage = round(max(0.0, raw_score) / max_points * 100, 2) if max_points > 0 else 0.0
            factor_percentages = pillar_factor_percentages.get(pillar, [])
            pillar_percentage = round(sum(factor_percentages) / len(factor_percentages), 2) if factor_percentages else 0.0
            weight = float(weights.get(pillar, 0.0) or 0.0)
            pillar_scores[pillar] = {
                "raw_score": round(raw_score, 4),
                "max_points": round(max_points, 4),
                "raw_percentage": raw_percentage,
                "percentage": pillar_percentage,
                "weight": weight,
                "weighted_score": round(pillar_percentage * weight, 2),
            }

        raw_total = round(sum(float(d.get("score", 0.0) or 0.0) for d in details or []), 4)
        raw_percentage = round(max(0.0, raw_total) / self.raw_max * 100, 2) if self.raw_max > 0 else 0.0
        score_100 = round(sum(data["weighted_score"] for data in pillar_scores.values()), 2)
        return {
            "raw_total": raw_total,
            "raw_max": self.raw_max,
            "raw_percentage": raw_percentage,
            "percentage": score_100,
            "score_100": score_100,
            "pillar_scores": pillar_scores,
            "factor_scores": factor_scores,
            "weights": weights,
            "factor_max_mismatches": [
                {
                    "factor": factor,
                    "rule_sum_max": data["max_points"],
                    "workbook_factor_max_points": data["workbook_factor_max_points"],
                    "difference": data["max_mismatch"],
                }
                for factor, data in factor_scores.items()
                if abs(data["max_mismatch"]) > 0.0001
            ],
        }

    def build_audit_rows(self, details: list[dict]) -> list[dict]:
        rows = []
        for detail in details or []:
            rule = self.rules_by_id.get(detail.get("id"), {})
            max_score = float(rule.get("max_score", 0.0) or 0.0)
            score = float(detail.get("score", 0.0) or 0.0)
            rows.append({
                "question_id": detail.get("id"),
                "factor": detail.get("factor") or rule.get("factor"),
                "pillar": detail.get("pillar") or rule.get("pillar"),
                "question_type": rule.get("question_type", "default"),
                "time_policy": rule.get("time_policy", "unspecified"),
                "answer": detail.get("answer"),
                "selected_options": ",".join(detail.get("selected_options", []) or []),
                "score": round(score, 4),
                "max_score": round(max_score, 4),
                "lost_points": round(max(0.0, max_score - score), 4),
                "logic": rule.get("logic", ""),
                "resolution_status": detail.get("resolution_status", ""),
                "evidence_present": bool(detail.get("evidence_present")),
                "evidence_source_ref": detail.get("evidence_source_ref", ""),
                "top_source_refs": "; ".join(detail.get("top_source_refs", []) or []),
                "loss_reason": self.loss_reason(detail, rule, max_score=max_score, score=score),
            })
        return rows

    def write_audit(self, details: list[dict], output_dir: str, company: str, year: int) -> dict:
        os.makedirs(output_dir, exist_ok=True)
        rows = self.build_audit_rows(details)
        base = os.path.join(output_dir, f"{company}_{year}_scoring_audit")
        json_path = f"{base}.json"
        csv_path = f"{base}.csv"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)
        if rows:
            with open(csv_path, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)
        return {"json_path": json_path, "csv_path": csv_path, "rows": rows}

    def loss_reason(self, detail: dict, rule: dict, max_score: float | None = None, score: float | None = None) -> str:
        max_score = float(max_score if max_score is not None else rule.get("max_score", 0.0) or 0.0)
        score = float(score if score is not None else detail.get("score", 0.0) or 0.0)
        if detail.get("answer") in {"NULL", "", None}:
            return "llm_null_or_no_answer"
        if score < 0:
            return "negative_disclosure_or_non_compliance"
        if max_score > 0 and score == 0 and not detail.get("evidence_present"):
            qtype = rule.get("question_type", "")
            if qtype in {"numeric_disclosure", "ratio_calculation"}:
                return "numeric_or_table_evidence_missing"
            return "evidence_missing"
        if detail.get("resolution_status") == "weakly_supported":
            return "weak_evidence"
        if max_score > 0 and score < max_score:
            if rule.get("question_type") == "multi_select":
                return "multi_select_incomplete"
            return "partial_score"
        return ""

    def _sum_by(self, key: str, max_by_question: dict[str, float]) -> dict[str, float]:
        totals = defaultdict(float)
        for rule in self.rules:
            group = rule.get(key)
            if not group and key == "pillar":
                group = str(rule.get("factor", ""))[:1]
            if group:
                totals[group] += max_by_question.get(rule.get("id"), 0.0)
        return {k: round(v, 4) for k, v in totals.items()}
