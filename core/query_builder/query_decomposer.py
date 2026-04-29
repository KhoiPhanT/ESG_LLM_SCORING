"""
Query decomposer — split complex VNSI questions into multiple sub-queries
for broader retrieval coverage. Does NOT use LLM (too slow for 80+ questions),
instead uses rule-based decomposition from question structure and options.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class DecomposedQuery:
    original_question: str
    sub_queries: list[str] = field(default_factory=list)
    is_compound: bool = False


class QueryDecomposer:
    """Decompose complex VNSI questions into targeted sub-queries."""

    # Connectors that indicate compound questions
    COMPOUND_INDICATORS = [
        r"\bvà\b",
        r"\bhoặc\b",
        r"\bđồng thời\b",
        r"\bngoài ra\b",
        r"\bbao gồm\b",
        r"\bcũng như\b",
    ]

    def decompose(self, rule: dict) -> DecomposedQuery:
        """
        Analyze a rule and generate sub-queries if the question is compound.
        Always returns at least the original question as a sub-query.
        """
        question = rule.get("question", "")
        options = rule.get("options", "")
        factor = rule.get("factor", "")
        sub_category = rule.get("sub_category", "")

        sub_queries = [question]  # Always include original

        # Strategy 1: Extract sub-topics from options
        option_queries = self._queries_from_options(options, question, factor)
        sub_queries.extend(option_queries)

        # Strategy 2: Split compound questions
        if self._is_compound_question(question):
            compound_queries = self._split_compound(question)
            sub_queries.extend(compound_queries)

        # Strategy 3: Add targeted queries based on factor/sub_category
        targeted = self._targeted_queries(rule)
        sub_queries.extend(targeted)

        # Deduplicate while preserving order
        seen = set()
        unique = []
        for q in sub_queries:
            q_clean = q.strip()
            if q_clean and q_clean.lower() not in seen:
                seen.add(q_clean.lower())
                unique.append(q_clean)

        return DecomposedQuery(
            original_question=question,
            sub_queries=unique,
            is_compound=len(unique) > 1,
        )

    def _is_compound_question(self, question: str) -> bool:
        """Check if question contains multiple requirements."""
        lowered = question.lower()
        indicator_count = sum(
            1 for pattern in self.COMPOUND_INDICATORS if re.search(pattern, lowered)
        )
        # Check for multiple question marks or semicolons
        has_multiple_clauses = question.count("?") > 1 or question.count(";") > 0
        return indicator_count >= 1 or has_multiple_clauses

    def _split_compound(self, question: str) -> list[str]:
        """Split a compound question into its constituent parts."""
        parts = []

        # Split on "và", "đồng thời", semicolons
        split_pattern = r"(?:\bvà\b|\bđồng thời\b|;)"
        segments = re.split(split_pattern, question, flags=re.IGNORECASE)

        for segment in segments:
            segment = segment.strip().rstrip("?").strip()
            if len(segment) > 15:  # Meaningful segment
                parts.append(segment)

        return parts

    def _queries_from_options(self, options: str, question: str, factor: str) -> list[str]:
        """Generate sub-queries from the answer options themselves."""
        if not options:
            return []

        queries = []
        option_lines = [
            line.strip()
            for line in options.split("\n")
            if line.strip() and re.match(r"^[A-Z][.\)]", line.strip())
        ]

        for line in option_lines:
            # Remove the letter prefix
            content = re.sub(r"^[A-Z][.\)]\s*", "", line).strip()
            if not content or len(content) < 10:
                continue

            # Skip generic/negative options
            lowered = content.lower()
            if any(neg in lowered for neg in [
                "không có", "không công bố", "chưa", "không rõ",
                "không áp dụng", "null", "chọn nhiều",
            ]):
                continue

            # The option content itself is a valid search query
            # Because it describes what the company SHOULD have
            queries.append(content)

        return queries[:4]  # Max 4 option-derived queries

    def _targeted_queries(self, rule: dict) -> list[str]:
        """Generate targeted queries based on factor and sub_category."""
        factor = rule.get("factor", "")
        sub_category = rule.get("sub_category", "")
        question = rule.get("question", "").lower()
        queries = []

        # For performance questions, add explicit metric queries
        if sub_category == "Hiệu quả":
            if factor.startswith("E"):
                queries.extend([
                    "số liệu phát thải khí nhà kính CO2",
                    "tiêu thụ năng lượng nước chất thải theo năm",
                ])
            elif factor.startswith("S"):
                queries.extend([
                    "thống kê nhân sự lao động đào tạo",
                    "tỷ lệ tai nạn lao động phúc lợi nhân viên",
                ])

        # For governance questions about specific structures
        if factor.startswith("G"):
            if "hđqt" in question or "hội đồng quản trị" in question:
                queries.append("thành viên hội đồng quản trị danh sách tiểu sử")
            if "độc lập" in question:
                queries.append("thành viên độc lập hội đồng quản trị")
            if "kiểm toán" in question:
                queries.append("ủy ban kiểm toán báo cáo tài chính ý kiến")
            if "cổ đông" in question or "đhđcđ" in question:
                queries.append("đại hội đồng cổ đông nghị quyết biểu quyết")
            if "thù lao" in question or "lương" in question:
                queries.append("thù lao lương thưởng hội đồng quản trị ban điều hành")

        # For policy questions, search for both policy AND implementation
        if sub_category == "Chính sách":
            if "môi trường" in question:
                queries.append("chính sách quản lý môi trường phát triển bền vững ISO 14001")
            if "lao động" in question or "nhân sự" in question:
                queries.append("chính sách nhân sự lao động quyền lợi nhân viên")
            if "chống tham nhũng" in question or "đạo đức" in question:
                queries.append("quy tắc đạo đức bộ quy tắc ứng xử chống tham nhũng")

        return queries[:3]
