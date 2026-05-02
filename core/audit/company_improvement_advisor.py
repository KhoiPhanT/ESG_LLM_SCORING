"""
Business-facing ESG improvement advisor based on NULL, weak, and negative answers.
"""
from __future__ import annotations

from collections import defaultdict
import json
import os

from core.scoring.scoring_contract import ScoringContract


class CompanyImprovementAdvisor:
    def __init__(self, rules: list[dict], factor_max_scores: dict | None = None, llm_client=None):
        self.contract = ScoringContract(rules, factor_max_scores=factor_max_scores)
        self.rules_by_id = self.contract.rules_by_id
        self.llm_client = llm_client

    def build(self, report: dict) -> dict:
        details = report.get("scoring_details", [])
        rows = self.contract.build_audit_rows(details)
        gaps = []
        for row in rows:
            if not self._is_company_gap(row):
                continue
            rule = self.rules_by_id.get(row["question_id"], {})
            gap = self._gap_for(row, rule)
            gaps.append({
                **row,
                "question": rule.get("question", ""),
                "gap_category": gap["gap_category"],
                "business_recommendation": gap["business_recommendation"],
                "evidence_to_disclose_next_year": gap["evidence_to_disclose_next_year"],
                "implementation_priority": gap["implementation_priority"],
                "llm_note": "",
            })

        gaps.sort(key=lambda item: (
            self._priority_rank(item.get("implementation_priority", "")),
            float(item.get("lost_points", 0.0) or 0.0),
            item.get("question_id") or "",
        ), reverse=True)
        issue_totals = defaultdict(float)
        for gap in gaps:
            issue_totals[gap["gap_category"]] += float(gap.get("lost_points", 0.0) or 0.0)

        result = {
            "company": report.get("company"),
            "year": report.get("year"),
            "summary": self._summary(gaps),
            "gap_totals": {key: round(value, 4) for key, value in sorted(issue_totals.items())},
            "priority_actions": self._priority_actions(gaps),
            "items": gaps,
        }
        self._add_llm_notes(result)
        return result

    def write(self, advisor: dict, output_dir: str, company: str, year: int) -> dict:
        os.makedirs(output_dir, exist_ok=True)
        json_path = os.path.join(output_dir, f"{company}_{year}_company_improvement_advisor.json")
        md_path = os.path.join(output_dir, f"{company}_{year}_company_improvement_advisor.md")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(advisor, f, ensure_ascii=False, indent=2)
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(self.to_markdown(advisor))
        return {"json_path": json_path, "md_path": md_path}

    def to_markdown(self, advisor: dict) -> str:
        lines = [
            f"# ESG Improvement Advisor: {advisor.get('company')} ({advisor.get('year')})",
            "",
            "## Executive Summary",
        ]
        for line in advisor.get("summary", []):
            lines.append(f"- {line}")
        lines.extend(["", "## Priority Actions"])
        for action in advisor.get("priority_actions", []):
            lines.append(f"- {action}")
        lines.extend(["", "## Gap Totals"])
        for category, value in advisor.get("gap_totals", {}).items():
            lines.append(f"- {category}: {value}")
        lines.extend(["", "## Detailed Improvement Items"])
        for item in advisor.get("items", [])[:45]:
            lines.extend([
                "",
                f"### {item.get('question_id')} ({item.get('factor')})",
                f"- Lost points: {item.get('lost_points')} | Priority: {item.get('implementation_priority')}",
                f"- Gap: {item.get('gap_category')}",
                f"- Sources checked: {item.get('evidence_source_ref') or item.get('top_source_refs') or 'no grounded source'}",
                f"- Recommendation: {item.get('business_recommendation')}",
                f"- Evidence to disclose: {item.get('evidence_to_disclose_next_year')}",
            ])
            if item.get("llm_note"):
                lines.append(f"- LLM note: {item.get('llm_note')}")
        return "\n".join(lines) + "\n"

    def _is_company_gap(self, row: dict) -> bool:
        return (
            row.get("answer") in {"NULL", "", None}
            or float(row.get("score", 0.0) or 0.0) < 0
            or float(row.get("lost_points", 0.0) or 0.0) > 0.25
            or row.get("resolution_status") in {"insufficient", "weakly_supported", "contested"}
        )

    def _gap_for(self, row: dict, rule: dict) -> dict:
        qid = str(row.get("question_id", ""))
        question = str(rule.get("question", "") or "").lower()
        qtype = rule.get("question_type", row.get("question_type", "default"))
        factor = str(row.get("factor", ""))

        if qtype in {"numeric_disclosure", "ratio_calculation"} or any(token in question for token in ["tỷ lệ", "tổng số", "số liệu", "doanh thu"]):
            return {
                "gap_category": "numeric_disclosure_gap",
                "business_recommendation": "Thiết lập bảng KPI định lượng theo năm, đơn vị đo và phạm vi tính toán; công bố cùng kỳ trong báo cáo thường niên hoặc báo cáo PTBV.",
                "evidence_to_disclose_next_year": "Bảng số liệu 2024/2025, đơn vị, phương pháp tính, mẫu số nếu là tỷ lệ và giải thích biến động.",
                "implementation_priority": "high" if float(row.get("lost_points", 0.0) or 0.0) >= 1 else "medium",
            }
        if qid.startswith("G.") or factor.startswith("G"):
            return {
                "gap_category": "governance_disclosure_gap",
                "business_recommendation": "Bổ sung công bố quản trị bằng bảng/đoạn riêng cho cơ cấu HĐQT, thù lao, độc lập, xung đột lợi ích, đánh giá HĐQT và cơ chế ủy ban.",
                "evidence_to_disclose_next_year": "Trích dẫn quy chế, nghị quyết, biên bản ĐHĐCĐ, bảng thù lao từng thành viên và hồ sơ năng lực HĐQT.",
                "implementation_priority": "high",
            }
        if factor.startswith("S") and any(token in question for token in ["cộng đồng", "thiện nguyện", "trách nhiệm xã hội"]):
            return {
                "gap_category": "community_impact_gap",
                "business_recommendation": "Chuẩn hóa phần hoạt động cộng đồng thành bảng tổng hợp: chương trình, địa bàn, số người hưởng lợi, ngân sách, đối tác và kết quả đo lường.",
                "evidence_to_disclose_next_year": "Tổng ngân sách CSR trong năm, % doanh thu, danh sách chương trình chính và chỉ số tác động.",
                "implementation_priority": "medium",
            }
        if factor.startswith("S"):
            return {
                "gap_category": "social_workforce_gap",
                "business_recommendation": "Công bố đầy đủ dữ liệu nhân sự theo giới, độ tuổi, cấp bậc, tuyển dụng, nghỉ việc, đào tạo, an toàn lao động và công đoàn.",
                "evidence_to_disclose_next_year": "Bảng workforce KPI có năm so sánh, công thức tỷ lệ và phạm vi nhân sự được tính.",
                "implementation_priority": "medium",
            }
        return {
            "gap_category": "policy_or_evidence_gap",
            "business_recommendation": "Công bố rõ chính sách, phạm vi áp dụng, năm hiệu lực, đơn vị phụ trách và bằng chứng triển khai.",
            "evidence_to_disclose_next_year": "Chính sách/quy trình, ngày hiệu lực, minh chứng hoạt động và KPI theo dõi.",
            "implementation_priority": "medium",
        }

    def _priority_actions(self, gaps: list[dict]) -> list[str]:
        categories = {gap.get("gap_category") for gap in gaps}
        actions = []
        if "numeric_disclosure_gap" in categories:
            actions.append("Lập phụ lục KPI ESG định lượng, dùng cùng cấu trúc năm sau để tránh mất điểm do thiếu số liệu.")
        if "governance_disclosure_gap" in categories:
            actions.append("Tạo bảng quản trị công ty riêng cho HĐQT, thù lao, ủy ban, xung đột lợi ích và đánh giá độc lập.")
        if "community_impact_gap" in categories:
            actions.append("Biến phần CSR từ infographic rời rạc thành bảng tổng hợp có ngân sách năm và % doanh thu.")
        if "social_workforce_gap" in categories:
            actions.append("Chuẩn hóa workforce dashboard: tuyển dụng, nghỉ việc, công đoàn, đào tạo, tai nạn, thu nhập theo nhóm.")
        return actions[:6]

    def _summary(self, gaps: list[dict]) -> list[str]:
        total_lost = round(sum(float(gap.get("lost_points", 0.0) or 0.0) for gap in gaps), 4)
        high_count = sum(1 for gap in gaps if gap.get("implementation_priority") == "high")
        return [
            f"{len(gaps)} câu cần cải thiện hoặc cần công bố rõ hơn.",
            f"Tổng điểm có thể cần xem xét/cải thiện: {total_lost}.",
            f"{high_count} hạng mục ưu tiên cao, chủ yếu nên xử lý ở tầng dữ liệu định lượng và quản trị.",
        ]

    def _add_llm_notes(self, advisor: dict) -> None:
        if not self.llm_client or not advisor.get("items"):
            return
        for item in advisor["items"][:8]:
            prompt = f"""
Bạn là cố vấn ESG cho doanh nghiệp Việt Nam. Viết một khuyến nghị ngắn, thực tế, không phóng đại.
Câu hỏi VNSI: {item.get('question')}
Khoảng trống: {item.get('gap_category')}
Khuyến nghị hiện có: {item.get('business_recommendation')}
Trả lời 1 câu tiếng Việt, tối đa 35 từ.
"""
            note = self.llm_client._call([{"role": "user", "content": prompt}], temperature=0.2, max_tokens=256)
            if note:
                item["llm_note"] = " ".join(str(note).split())[:300]

    def _priority_rank(self, priority: str) -> int:
        return {"low": 1, "medium": 2, "high": 3}.get(priority, 0)
