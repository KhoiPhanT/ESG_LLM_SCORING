"""
Post-run advisor for NULL, negative, weak, and recoverable VNSI answers.
"""
from __future__ import annotations

from collections import defaultdict
import json
import os

from core.scoring.scoring_contract import ScoringContract


class ReviewAdvisor:
    def __init__(self, rules: list[dict], factor_max_scores: dict | None = None):
        self.contract = ScoringContract(rules, factor_max_scores=factor_max_scores)
        self.rules_by_id = self.contract.rules_by_id

    def build(self, report: dict) -> dict:
        details = report.get("scoring_details", [])
        rows = self.contract.build_audit_rows(details)
        score_summary = self.contract.summarize(
            details,
            weights=report.get("scores", {}).get("weights", {}),
        )
        items = []
        totals = defaultdict(float)

        for row in rows:
            if not self._needs_review(row):
                continue
            rule = self.rules_by_id.get(row["question_id"], {})
            advice = self._advice_for(row, rule)
            item = {
                **row,
                "question": rule.get("question", ""),
                "issue_type": advice["issue_type"],
                "recommendation": advice["recommendation"],
                "next_retrieval_hint": advice["next_retrieval_hint"],
            }
            items.append(item)
            totals[advice["issue_type"]] += float(row.get("lost_points", 0.0) or 0.0)

        items.sort(key=lambda item: (float(item.get("lost_points", 0.0) or 0.0), item.get("question_id") or ""), reverse=True)
        current_score = float(score_summary.get("raw_total", 0.0) or 0.0)
        recoverable = round(sum(float(item.get("lost_points", 0.0) or 0.0) for item in items), 4)
        return {
            "company": report.get("company"),
            "year": report.get("year"),
            "current_score": current_score,
            "raw_max": score_summary.get("raw_max", 0.0),
            "percentage": score_summary.get("percentage", 0.0),
            "recoverable_points": recoverable,
            "best_case_score": round(current_score + recoverable, 4),
            "issue_totals": {k: round(v, 4) for k, v in sorted(totals.items())},
            "top_actions": self._top_actions(items),
            "items": items,
        }

    def write(self, advisor: dict, output_dir: str, company: str, year: int) -> dict:
        os.makedirs(output_dir, exist_ok=True)
        json_path = os.path.join(output_dir, f"{company}_{year}_review_advisor.json")
        md_path = os.path.join(output_dir, f"{company}_{year}_review_advisor.md")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(advisor, f, ensure_ascii=False, indent=2)
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(self.to_markdown(advisor))
        return {"json_path": json_path, "md_path": md_path}

    def to_markdown(self, advisor: dict) -> str:
        lines = [
            f"# Review Advisor: {advisor.get('company')} ({advisor.get('year')})",
            "",
            f"- Current raw score: {advisor.get('current_score')} / {advisor.get('raw_max')}",
            f"- Reference percentage: {advisor.get('percentage')}%",
            f"- Recoverable points: {advisor.get('recoverable_points')}",
            f"- Best-case score if all review items are recovered: {advisor.get('best_case_score')}",
            "",
            "## Issue Totals",
        ]
        for issue, value in advisor.get("issue_totals", {}).items():
            lines.append(f"- {issue}: {value}")
        lines.extend(["", "## Top Actions"])
        for action in advisor.get("top_actions", []):
            lines.append(f"- {action}")
        lines.extend(["", "## Highest Impact Review Items"])
        for item in advisor.get("items", [])[:40]:
            lines.extend([
                "",
                f"### {item.get('question_id')} ({item.get('factor')})",
                f"- Score: {item.get('score')} / {item.get('max_score')} | Lost: {item.get('lost_points')}",
                f"- Answer: {item.get('answer')} | Status: {item.get('resolution_status')}",
                f"- Issue: {item.get('issue_type')} | Reason: {item.get('loss_reason')}",
                f"- Sources checked: {item.get('evidence_source_ref') or item.get('top_source_refs') or 'no grounded source'}",
                f"- Recommendation: {item.get('recommendation')}",
                f"- Retrieval hint: {item.get('next_retrieval_hint')}",
            ])
        return "\n".join(lines) + "\n"

    def _needs_review(self, row: dict) -> bool:
        return (
            row.get("answer") in {"NULL", "", None}
            or float(row.get("score", 0.0) or 0.0) < 0
            or row.get("resolution_status") in {"insufficient", "weakly_supported", "contested"}
            or float(row.get("lost_points", 0.0) or 0.0) > 0.25
        )

    def _advice_for(self, row: dict, rule: dict) -> dict:
        qtype = rule.get("question_type", row.get("question_type", "default"))
        qid = row.get("question_id", "")
        loss_reason = row.get("loss_reason", "")
        if loss_reason == "llm_null_or_no_answer":
            return {
                "issue_type": "null_answer",
                "recommendation": "Review query plan, source filters và các disclosure còn thiếu; nếu tài liệu có bằng chứng, bổ sung alias/anchor đúng vào plan hoặc goldset.",
                "next_retrieval_hint": self._hint(qid, qtype),
            }
        if loss_reason == "numeric_or_table_evidence_missing" or qtype in {"numeric_disclosure", "ratio_calculation"}:
            return {
                "issue_type": "numeric_missing",
                "recommendation": "Kiểm tra bảng số liệu trong BCTN/BC PTBV/BCTC, đặc biệt cột năm, đơn vị và doanh thu nếu câu cần tỷ lệ.",
                "next_retrieval_hint": "review metric/table query plan + source filters + financial_metrics nếu cần doanh thu",
            }
        if loss_reason == "negative_disclosure_or_non_compliance":
            return {
                "issue_type": "negative_score",
                "recommendation": "Xác minh wording VNSI và evidence nguồn. Nếu công ty thật sự không công bố nội dung này thì giữ điểm âm; nếu có, chỉnh query plan/source filter để kéo đúng tài liệu/trang.",
                "next_retrieval_hint": self._hint(qid, qtype),
            }
        if loss_reason == "multi_select_incomplete" or qtype == "multi_select":
            return {
                "issue_type": "multi_select_incomplete",
                "recommendation": "Review từng option A/B/C... riêng, vì mỗi option có thể cộng điểm độc lập nếu có bằng chứng.",
                "next_retrieval_hint": "extract evidence_by_option from policy/table sections",
            }
        if row.get("resolution_status") == "weakly_supported":
            return {
                "issue_type": "weak_evidence",
                "recommendation": "Evidence đang yếu hoặc không grounded; cần quote ngắn sát nguồn hơn và đúng trang.",
                "next_retrieval_hint": self._hint(qid, qtype),
            }
        return {
            "issue_type": "recoverable_zero",
            "recommendation": "Có điểm mất nhưng chưa rõ nguyên nhân; ưu tiên review query_plan, source_sections và disclosure còn thiếu.",
            "next_retrieval_hint": self._hint(qid, qtype),
        }

    def _hint(self, qid: str, qtype: str) -> str:
        if qid.startswith("G."):
            return "governance profile: bao cao quan tri, DHDCD, bien ban, nghi quyet, co tuc, thu lao"
        if qtype in {"numeric_disclosure", "ratio_calculation"}:
            return "table profile: so lieu, ty le, tong luong, don vi, doanh thu"
        if qtype == "policy":
            return "policy profile: chinh sach, quy trinh, he thong, chung nhan, con hieu luc"
        return "review query plan aliases + exact phrase anchors"

    def _top_actions(self, items: list[dict]) -> list[str]:
        issue_counts = defaultdict(int)
        for item in items:
            issue_counts[item["issue_type"]] += 1
        actions = []
        if issue_counts.get("numeric_missing"):
            actions.append("Run table extraction review for E3/S3 numeric questions first.")
        if issue_counts.get("negative_score"):
            actions.append("Manually verify negative-score governance/social disclosure questions against source pages.")
        if issue_counts.get("null_answer"):
            actions.append("Review query plan aliases/source filters or add goldset evidence for NULL questions.")
        if issue_counts.get("multi_select_incomplete"):
            actions.append("Review multi-select questions option-by-option.")
        if issue_counts.get("weak_evidence"):
            actions.append("Replace weak evidence with exact grounded quotes from source sections.")
        return actions[:5]
