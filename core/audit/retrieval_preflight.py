"""
Preflight gate for retrieval readiness.

Runs before LLM scoring.  The goal is to catch missing metadata or obviously bad
retrieval coverage while failures are still cheap to fix.
"""
from __future__ import annotations

import json
import os

from core.query_builder.question_retrieval_metadata import QuestionRetrievalMetadataBuilder


class RetrievalPreflight:
    CRITICAL_IDS = {
        "S.3.2.1",
        "S.3.2.2",
        "G.15",
        "G.19",
        "E.3.2.1",
        "E.3.3.1",
        "E.3.5.2",
    }

    def __init__(self, rules: list[dict], metadata_payload: dict, target_year: int = 2024):
        self.rules = rules or []
        self.rules_by_id = {rule.get("id"): rule for rule in self.rules}
        self.metadata_payload = metadata_payload or {}
        self.metadata = self.metadata_payload.get("metadata", {})
        self.target_year = target_year
        self.builder = QuestionRetrievalMetadataBuilder(target_year=target_year)

    def run(self, retrieval_engine, full: bool = True, verbose: bool = True) -> dict:
        metadata_validation = self.builder.validate(self.metadata_payload, self.rules)
        selected_rules = self.rules if full else [
            self.rules_by_id[qid]
            for qid in sorted(self.CRITICAL_IDS)
            if qid in self.rules_by_id
        ]
        retrieval_checks = []
        for index, rule in enumerate(selected_rules, start=1):
            qid = rule.get("id")
            if verbose:
                print(f"  [PREFLIGHT] {index}/{len(selected_rules)} {qid}", flush=True)
            plan = self.metadata.get(qid) or {}
            retrieval = retrieval_engine.retrieve_for_plan(rule, plan, top_k=10)
            candidates = retrieval.get("candidates", [])
            check = self._check_rule(rule, plan, retrieval, candidates)
            retrieval_checks.append(check)

        failures = [item for item in retrieval_checks if not item.get("passed")]
        critical_failures = [
            item for item in failures
            if item.get("question_id") in self.CRITICAL_IDS
        ]
        blocking_failures = failures if full else critical_failures
        passed = metadata_validation.get("passed") and not blocking_failures
        return {
            "passed": passed,
            "metadata_validation": metadata_validation,
            "checked_count": len(retrieval_checks),
            "full": full,
            "failure_count": len(failures),
            "critical_failure_count": len(critical_failures),
            "failures": failures[:80],
            "critical_failures": critical_failures,
            "checks": retrieval_checks,
        }

    def write(self, result: dict, output_dir: str, company: str, year: int) -> dict:
        os.makedirs(output_dir, exist_ok=True)
        json_path = os.path.join(output_dir, f"{company}_{year}_retrieval_preflight.json")
        md_path = os.path.join(output_dir, f"{company}_{year}_retrieval_preflight.md")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(self.to_markdown(result))
        return {"json_path": json_path, "md_path": md_path}

    def to_markdown(self, result: dict) -> str:
        lines = [
            "# Retrieval Preflight",
            "",
            f"- Passed: {result.get('passed')}",
            f"- Checked questions: {result.get('checked_count')}",
            f"- Failures: {result.get('failure_count')}",
            f"- Critical failures: {result.get('critical_failure_count')}",
            "",
            "## Critical Failures",
        ]
        for item in result.get("critical_failures", []):
            lines.append(f"- {item.get('question_id')}: {', '.join(item.get('reasons', []))}")
        lines.extend(["", "## All Failures"])
        for item in result.get("failures", [])[:80]:
            lines.append(f"- {item.get('question_id')}: {', '.join(item.get('reasons', []))}")
        return "\n".join(lines) + "\n"

    def _check_rule(self, rule: dict, plan: dict, retrieval: dict, candidates: list[dict]) -> dict:
        qid = rule.get("id")
        reasons = []
        if not plan:
            reasons.append("missing_metadata")
        if not plan.get("search_queries"):
            reasons.append("missing_metadata_queries")
        if not candidates:
            reasons.append("no_candidates")

        qtype = rule.get("question_type")
        strategy = str(plan.get("strategy") or plan.get("metadata_strategy") or "")
        evidence_shape = str(plan.get("evidence_shape") or "")
        haystack = self._candidate_haystack(candidates[:8])

        if rule.get("is_multi_select") or strategy == "multi_option":
            positive_options = self._positive_options(rule)
            focus = plan.get("option_focus") if isinstance(plan.get("option_focus"), dict) else {}
            missing_focus = [letter for letter in positive_options if not focus.get(letter)]
            if missing_focus:
                reasons.append(f"missing_option_focus={','.join(missing_focus)}")
            if len(plan.get("search_queries", [])) < max(2, min(len(positive_options), 4)):
                reasons.append("too_few_multi_queries")

        numeric_like = (
            qtype in {"numeric_disclosure", "ratio_calculation"}
            or str(plan.get("evidence_profile") or "") in {"metric_disclosure", "ratio_with_revenue"}
        )
        if numeric_like:
            if not any(self._is_table_like(item) for item in candidates[:8]):
                reasons.append("no_table_like_candidate")
            anchors = " ".join(plan.get("must_have_terms", []) + plan.get("search_queries", [])).lower()
            if not any(token in anchors for token in ["số liệu", "bảng", "tỷ lệ", "kwh", "mj", "m3", "vnd", "doanh thu", "scope"]):
                reasons.append("missing_numeric_anchors")
            if str(plan.get("evidence_profile") or "") == "ratio_with_revenue" and "doanh thu" not in anchors and "revenue" not in anchors:
                reasons.append("missing_revenue_anchor")

        if str(qid).startswith("G."):
            preferred = set(plan.get("required_doc_types", []))
            if preferred and not any(item.get("document_type") in preferred for item in candidates[:8]):
                reasons.append("no_preferred_governance_doc_type")

        if qid in self.CRITICAL_IDS:
            critical_terms = self._critical_terms(qid)
            if critical_terms and not any(term in haystack for term in critical_terms):
                reasons.append("critical_anchor_not_seen")

        return {
            "question_id": qid,
            "strategy": strategy,
            "candidate_count": len(candidates),
            "actual_top_k": retrieval.get("actual_top_k"),
            "max_queries_used": retrieval.get("max_queries_used"),
            "top_sources": [
                {
                    "source_file": item.get("source_file"),
                    "document_type": item.get("document_type"),
                    "page_start": item.get("page_start"),
                    "page_end": item.get("page_end"),
                    "rerank_score": item.get("rerank_score"),
                }
                for item in candidates[:5]
            ],
            "passed": not reasons,
            "reasons": reasons,
        }

    def _positive_options(self, rule: dict) -> dict[str, str]:
        options = self.builder._parse_options(rule.get("options", ""))
        return {
            letter: body
            for letter, body in options.items()
            if not self.builder._is_negative_option(body)
        }

    def _is_table_like(self, item: dict) -> bool:
        return (
            item.get("chunk_type") in {"table_section", "metric_kv_section"}
            or bool(item.get("table_family"))
            or any(token in str(item.get("content", "")).lower() for token in ["|", "đơn vị", "kwh", "mj", "m3", "tỷ đồng", "scope"])
        )

    def _candidate_haystack(self, candidates: list[dict]) -> str:
        parts = []
        for item in candidates:
            parts.append(str(item.get("section_title", "")))
            parts.append(str(item.get("content", ""))[:2500])
        return self.builder._norm(" ".join(parts))

    def _critical_terms(self, qid: str) -> list[str]:
        raw = {
            "S.3.2.1": ["cong dong", "thien nguyen", "quy sua", "trach nhiem xa hoi"],
            "S.3.2.2": ["cong dong", "thien nguyen", "ty dong", "doanh thu"],
            "G.15": ["thu lao", "luong", "thuong", "hoi dong quan tri"],
            "G.19": ["hoc van", "kinh nghiem", "chuyen mon", "hoi dong quan tri"],
            "E.3.2.1": ["nang luong", "kwh", "mj", "doanh thu"],
            "E.3.3.1": ["nuoc", "m3", "doanh thu"],
            "E.3.5.2": ["scope", "phat thai", "co2", "khi nha kinh"],
        }.get(qid, [])
        return raw
