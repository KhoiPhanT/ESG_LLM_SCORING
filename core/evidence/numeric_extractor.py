"""
Numeric Extractor — Trích xuất giá trị số cụ thể từ context và tính ratio.

Chạy TRƯỚC LLM answering để enrichment context, và cung cấp structured data
cho scoring engine. Nếu tìm thấy số liệu mà LLM vẫn nói "không đề cập" →
override thành "có đề cập" (Option A strategy).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict


# ── Unit & metric mapping ──────────────────────────────────

UNIT_PATTERNS: list[tuple[str, str]] = [
    # Emissions
    (r"tấn\s*CO[2₂](?:e| tương đương)?", "tấn CO2e"),
    (r"tCO[2₂]e?", "tấn CO2e"),
    (r"kg\s*CO[2₂](?:e| tương đương)?", "kg CO2e"),
    # Energy
    (r"kWh", "kWh"),
    (r"MWh", "MWh"),
    (r"GJ", "GJ"),
    (r"MJ", "MJ"),
    (r"joules?", "J"),
    # Water & waste
    (r"m3|m³", "m3"),
    (r"lít", "lít"),
    (r"tấn", "tấn"),
    (r"kg", "kg"),
    # Financial
    (r"tỷ\s*(?:đồng|VND|vnđ)", "tỷ đồng"),
    (r"triệu\s*(?:đồng|VND|vnđ)", "triệu đồng"),
    (r"nghìn\s*(?:đồng|VND|vnđ)", "nghìn đồng"),
    (r"(?:đồng|VND|vnđ)", "đồng"),
    (r"USD", "USD"),
    # Workforce
    (r"người", "người"),
    (r"nhân\s*viên", "người"),
    (r"lao\s*động", "người"),
    (r"giờ(?:/người)?", "giờ"),
    # Percentage
    (r"%", "%"),
]

METRIC_KEYWORDS: dict[str, list[str]] = {
    "recycled_material_ratio": ["vật liệu tái chế", "vật liệu tái tạo", "nguyên vật liệu đầu vào"],
    "packaging_recovery_ratio": ["thu hồi", "tái sử dụng", "tái chế", "vật liệu đóng gói", "sản phẩm đã bán"],
    "ghg_scope1": ["scope 1", "phát thải trực tiếp", "phạm vi 1"],
    "ghg_scope2": ["scope 2", "phát thải gián tiếp", "phạm vi 2"],
    "ghg_scope3": ["scope 3", "phạm vi 3"],
    "ghg_total": ["tổng phát thải", "tổng lượng phát thải", "total ghg", "total emission"],
    "energy_saving": ["năng lượng tiết kiệm", "tiết kiệm năng lượng", "sáng kiến tiết kiệm"],
    "energy": ["năng lượng", "điện năng", "tiêu thụ năng lượng", "energy consumption"],
    "wastewater": ["nước thải", "xử lý nước thải", "wastewater"],
    "water_reuse": ["nước tuần hoàn", "tái sử dụng nước", "tuần hoàn và tái sử dụng"],
    "water": ["nước", "tiêu thụ nước", "water consumption", "water usage"],
    "waste": ["chất thải", "rác thải", "waste", "xử lý chất thải"],
    "revenue": ["doanh thu", "tổng doanh thu", "revenue", "net revenue"],
    "recruitment_rate": ["tỷ lệ tuyển dụng", "tuyển dụng nhân viên mới", "nhân viên mới", "tuyển mới"],
    "turnover_rate": ["tỷ lệ chấm dứt", "chấm dứt hợp đồng", "nghỉ việc", "thôi việc", "turnover"],
    "employees": ["nhân viên", "nhân sự", "lao động", "tổng số nhân viên", "employee"],
    "training": ["đào tạo", "training hours", "giờ đào tạo"],
    "social_investment": ["đầu tư cộng đồng", "csr", "an sinh xã hội", "thiện nguyện", "cộng đồng địa phương", "đóng góp cộng đồng"],
    "female_ratio": ["nữ", "tỷ lệ nữ", "female", "phụ nữ"],
    "rd_expense": ["r&d", "nghiên cứu và phát triển", "đổi mới công nghệ", "research and development"],
}

METRIC_UNIT_ALLOWLIST: dict[str, set[str]] = {
    "recycled_material_ratio": {"%", "tấn", "kg"},
    "packaging_recovery_ratio": {"%", "tấn", "kg"},
    "ghg_scope1": {"tấn CO2e", "kg CO2e"},
    "ghg_scope2": {"tấn CO2e", "kg CO2e"},
    "ghg_scope3": {"tấn CO2e", "kg CO2e"},
    "ghg_total": {"tấn CO2e", "kg CO2e"},
    "energy": {"kWh", "MWh", "GJ", "MJ", "J"},
    "energy_saving": {"kWh", "MWh", "GJ", "MJ", "J"},
    "water": {"m3", "lít"},
    "water_reuse": {"m3", "lít", "%"},
    "wastewater": {"m3", "lít"},
    "waste": {"tấn", "kg"},
    "revenue": {"tỷ đồng", "triệu đồng", "nghìn đồng", "đồng", "USD"},
    "recruitment_rate": {"%"},
    "turnover_rate": {"%"},
    "social_investment": {"tỷ đồng", "triệu đồng", "nghìn đồng", "đồng", "USD", "%"},
    "rd_expense": {"tỷ đồng", "triệu đồng", "nghìn đồng", "đồng", "USD"},
}


@dataclass
class ExtractedMetric:
    """A single numeric value extracted from text."""
    metric_name: str = ""
    value: float | None = None
    unit: str = ""
    year: int | None = None
    source_id: str = ""
    page: int | None = None
    context_snippet: str = ""
    extraction_method: str = "regex"
    confidence: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class NumericResult:
    """Full numeric extraction result for a question."""
    metric_name: str = ""
    value: float | None = None
    unit: str = ""
    year: int | None = None
    source_id: str = ""
    page: int | None = None
    year_validated: bool = False
    extraction_method: str = "regex"
    confidence: float = 0.0
    ratio: dict | None = None
    raw_candidates: list[dict] = field(default_factory=list)
    context_enrichment: str = ""

    def to_dict(self) -> dict:
        result = asdict(self)
        result["raw_candidates"] = [c for c in result.get("raw_candidates", [])[:5]]
        return result


class NumericExtractor:
    """Extract structured numeric values from retrieval context for VNSI scoring."""

    def __init__(self, llm_client=None, target_year: int | None = None):
        self.llm_client = llm_client
        self.target_year = target_year or 2024
        self._number_pattern = re.compile(
            r"(?<!\w)"                            # not preceded by word char
            r"([+-]?)(\d{1,3}(?:[.,]\d{3})*(?:[.,]\d+)?|\d+(?:[.,]\d+)?)"  # number
            r"\s*"                                 # optional space
            r"("                                   # unit group
            + "|".join(p for p, _ in UNIT_PATTERNS)
            + r")",
            re.IGNORECASE,
        )

    def extract(self, rule: dict, context_bundle: dict) -> dict | None:
        """
        Extract numeric data for a numeric_disclosure or ratio_calculation question.
        Returns a NumericResult dict if data found, else None.
        """
        question_type = rule.get("question_type", "default")
        if question_type not in {"numeric_disclosure", "ratio_calculation"}:
            return None

        context = context_bundle.get("context", "")
        sections = context_bundle.get("sections", [])
        if not context:
            return None

        # Layer 1: Regex pre-scan
        candidates = self._regex_prescan(context, rule, sections)

        # Layer 2: LLM structured extraction (only if regex found raw numbers but needs disambiguation)
        llm_extraction = None
        if self.llm_client and question_type == "ratio_calculation":
            llm_extraction = self._llm_extract(context, rule)
        elif self.llm_client and not candidates:
            llm_extraction = self._llm_extract(context, rule)

        # Build result
        result = self._build_result(rule, candidates, llm_extraction)
        if result is None:
            return None

        # Build context enrichment string for LLM prompt injection
        result.context_enrichment = self._build_enrichment(result, question_type)
        return result.to_dict()

    def _regex_prescan(self, context: str, rule: dict, sections: list[dict]) -> list[ExtractedMetric]:
        """Scan context for numeric values matching expected metrics."""
        question = str(rule.get("question", "")).lower()
        question_type = rule.get("question_type", "default")
        candidates: list[ExtractedMetric] = []

        # Determine what metric we're looking for based on question text
        target_metrics = self._detect_target_metrics(question)
        if question_type == "ratio_calculation" and "revenue" not in target_metrics:
            target_metrics = list(dict.fromkeys(target_metrics + ["revenue"]))

        candidates.extend(self._table_year_prescan(context, rule, sections, target_metrics))

        for match in self._number_pattern.finditer(context):
            sign_str, num_str, unit_raw = match.groups()
            value = self._parse_number(sign_str + num_str)
            if value is None or value == 0:
                continue

            unit = self._normalize_unit(unit_raw)
            if not unit:
                continue

            # Extract surrounding context for metric identification
            start = max(0, match.start() - 120)
            end = min(len(context), match.end() + 60)
            snippet = context[start:end].replace("\n", " ").strip()

            # Detect year near the value
            year = self._detect_year_near(context, match.start(), match.end())

            # Detect which source section this belongs to
            source_id = self._find_source_id(context, match.start(), sections)

            # Detect page
            page = self._find_page(context, match.start(), sections)

            # Match against target metrics
            metric_name = self._identify_metric(snippet, target_metrics)

            confidence = self._score_candidate_confidence(
                value, unit, year, metric_name, target_metrics
            )

            if confidence >= 0.3:
                candidates.append(ExtractedMetric(
                    metric_name=metric_name,
                    value=value,
                    unit=unit,
                    year=year,
                    source_id=source_id,
                    page=page,
                    context_snippet=snippet[:200],
                    extraction_method="regex",
                    confidence=confidence,
                ))

        # Sort by confidence, keep top candidates
        candidates.sort(key=lambda c: c.confidence, reverse=True)
        return candidates[:10]

    def _llm_extract(self, context: str, rule: dict) -> dict | None:
        """Use LLM to extract structured numeric values when regex is insufficient."""
        if not self.llm_client:
            return None

        question = rule.get("question", "")
        question_type = rule.get("question_type", "default")
        is_ratio = question_type == "ratio_calculation"

        ratio_fields = ""
        if is_ratio:
            ratio_fields = """
    "denominator_value": <number or null>,
    "denominator_unit": "đơn vị mẫu số hoặc null",
    "denominator_metric": "tên chỉ số mẫu số (ví dụ: doanh thu) hoặc null",
    "ratio_result": <computed ratio or null>,
    "ratio_unit": "đơn vị ratio (ví dụ: tấn CO2e/tỷ đồng) hoặc null","""

        prompt = f"""Extract the SPECIFIC numeric value asked by this VNSI question from the report context below.
Look carefully in tables, bullet points, and narrative text for the exact metric.

QUESTION: {question}

RULES:
- Return the value for year {self.target_year} if available, otherwise the most recent year.
- For ratio questions, extract BOTH numerator and denominator values, then compute the ratio.
- value must be a raw number (no commas, no unit text).
- source_id must match one of the SOURCE_ID labels in the context (e.g. "S1", "S2").
- If the metric is NOT found in the context, return {{"found": false}}.

CONTEXT:
{context[:12000]}

Return JSON only:
{{"found": true,
    "metric_name": "tên chỉ số",
    "value": <number>,
    "unit": "đơn vị",
    "year": <year as integer>,
    "source_id": "SOURCE_ID",{ratio_fields}
    "confidence": <0.0 to 1.0>
}}"""

        raw = self.llm_client._call(
            [{"role": "user", "content": prompt}],
            temperature=0.05,
            max_tokens=1024,
            retries=1,
        )
        parsed = self.llm_client._parse_json(raw)
        if not parsed or not parsed.get("found"):
            return None
        return parsed

    def _build_result(
        self,
        rule: dict,
        candidates: list[ExtractedMetric],
        llm_extraction: dict | None,
    ) -> NumericResult | None:
        """Merge regex candidates and LLM extraction into a final result."""
        question_type = rule.get("question_type", "default")
        is_ratio = question_type == "ratio_calculation"

        # Prefer LLM extraction for ratio questions (more reliable for multi-value extraction)
        if llm_extraction and llm_extraction.get("value") is not None:
            try:
                value = float(llm_extraction["value"])
            except (TypeError, ValueError):
                value = None
            if value is None:
                return None
            year = self._safe_int(llm_extraction.get("year"))
            year_validated = year == self.target_year if year else False

            ratio = None
            if is_ratio and llm_extraction.get("denominator_value") is not None:
                try:
                    denom = float(llm_extraction["denominator_value"])
                except (TypeError, ValueError):
                    denom = 0.0
                if denom > 0:
                    ratio_result = llm_extraction.get("ratio_result")
                    if ratio_result is None:
                        ratio_result = round(value / denom, 6)
                    ratio = {
                        "numerator": {"value": value, "unit": llm_extraction.get("unit", ""), "metric": llm_extraction.get("metric_name", "")},
                        "denominator": {"value": denom, "unit": llm_extraction.get("denominator_unit", ""), "metric": llm_extraction.get("denominator_metric", "")},
                        "result": ratio_result,
                        "result_unit": llm_extraction.get("ratio_unit", ""),
                    }

            return NumericResult(
                metric_name=str(llm_extraction.get("metric_name", "")),
                value=value,
                unit=str(llm_extraction.get("unit", "")),
                year=year,
                source_id=str(llm_extraction.get("source_id", "")),
                year_validated=year_validated,
                extraction_method="llm_structured",
                confidence=float(llm_extraction.get("confidence", 0.7)),
                ratio=ratio,
                raw_candidates=[c.to_dict() for c in candidates[:5]],
            )

        if is_ratio:
            ratio_result = self._build_ratio_result(rule, candidates)
            if ratio_result:
                return ratio_result

        # Fall back to best regex candidate
        if candidates:
            best = candidates[0]
            year_validated = best.year == self.target_year if best.year else False
            return NumericResult(
                metric_name=best.metric_name,
                value=best.value,
                unit=best.unit,
                year=best.year,
                source_id=best.source_id,
                page=best.page,
                year_validated=year_validated,
                extraction_method="regex",
                confidence=best.confidence,
                raw_candidates=[c.to_dict() for c in candidates[:5]],
            )

        return None

    def _table_year_prescan(
        self,
        context: str,
        rule: dict,
        sections: list[dict],
        target_metrics: list[str],
    ) -> list[ExtractedMetric]:
        candidates = []
        lines = context.splitlines()
        header_years: list[int] = []
        current_source = ""
        current_page = None
        source_pattern = re.compile(r"\[SOURCE_ID:\s*(S\d+).*?PAGES?:\s*(\d+)", re.IGNORECASE)
        for line in lines:
            source_match = source_pattern.search(line)
            if source_match:
                current_source = source_match.group(1)
                try:
                    current_page = int(source_match.group(2))
                except ValueError:
                    current_page = None
                header_years = []
                continue

            years = [int(value) for value in re.findall(r"\b(20[12]\d)\b", line)]
            if len(years) >= 2:
                header_years = years
                continue
            if not header_years:
                continue

            normalized_line = line.replace("|", " ")
            values = list(self._number_pattern.finditer(normalized_line))
            if not values:
                continue
            metric_name = self._identify_metric(normalized_line, target_metrics)
            if metric_name == "unknown":
                continue

            value_index = None
            if self.target_year in header_years:
                idx = header_years.index(self.target_year)
                if idx < len(values):
                    value_index = idx
            if value_index is None:
                value_index = min(len(values) - 1, len(header_years) - 1)
            match = values[value_index]
            sign_str, num_str, unit_raw = match.groups()
            value = self._parse_number(sign_str + num_str)
            unit = self._normalize_unit(unit_raw)
            if value is None or not unit:
                continue
            year = self.target_year if self.target_year in header_years else header_years[min(value_index, len(header_years) - 1)]
            confidence = self._score_candidate_confidence(value, unit, year, metric_name, target_metrics) + 0.08
            if confidence >= 0.3:
                candidates.append(ExtractedMetric(
                    metric_name=metric_name,
                    value=value,
                    unit=unit,
                    year=year,
                    source_id=current_source or self._source_for_page(current_page, sections),
                    page=current_page,
                    context_snippet=normalized_line[:200],
                    extraction_method="table_year_regex",
                    confidence=min(1.0, confidence),
                ))
        return candidates

    def _build_ratio_result(self, rule: dict, candidates: list[ExtractedMetric]) -> NumericResult | None:
        question = str(rule.get("question", "")).lower()
        target_metrics = [metric for metric in self._detect_target_metrics(question) if metric != "revenue"]
        numerator_candidates = [
            item for item in candidates
            if item.metric_name in target_metrics and item.value is not None and item.value > 0
        ]
        denominator_candidates = [
            item for item in candidates
            if item.metric_name == "revenue" and item.value is not None and item.value > 0
        ]
        if not numerator_candidates or not denominator_candidates:
            return None
        numerator_candidates.sort(key=lambda item: (item.year == self.target_year, item.confidence), reverse=True)
        denominator_candidates.sort(key=lambda item: (item.year == self.target_year, item.confidence), reverse=True)
        numerator = numerator_candidates[0]
        denominator = denominator_candidates[0]
        result = round(float(numerator.value) / float(denominator.value), 6)
        confidence = min(0.95, (numerator.confidence + denominator.confidence) / 2 + 0.08)
        return NumericResult(
            metric_name=numerator.metric_name,
            value=numerator.value,
            unit=numerator.unit,
            year=numerator.year,
            source_id=numerator.source_id,
            page=numerator.page,
            year_validated=numerator.year == self.target_year if numerator.year else False,
            extraction_method="deterministic_ratio",
            confidence=round(confidence, 3),
            ratio={
                "numerator": {"value": numerator.value, "unit": numerator.unit, "metric": numerator.metric_name, "source_id": numerator.source_id},
                "denominator": {"value": denominator.value, "unit": denominator.unit, "metric": denominator.metric_name, "source_id": denominator.source_id},
                "result": result,
                "result_unit": f"{numerator.unit}/{denominator.unit}".strip("/"),
            },
            raw_candidates=[c.to_dict() for c in candidates[:8]],
        )

    def _safe_int(self, value) -> int | None:
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return None

    def _build_enrichment(self, result: NumericResult, question_type: str) -> str:
        """Build a structured text block to inject into the LLM prompt context."""
        lines = ["\n[NUMERIC_EXTRACTION_DATA]"]
        lines.append(f"Metric: {result.metric_name}")
        lines.append(f"Value: {result.value} {result.unit}")
        if result.year:
            lines.append(f"Year: {result.year} ({'✓ matches target' if result.year_validated else '⚠ different from target'})")
        if result.source_id:
            lines.append(f"Source: {result.source_id}")
        if result.ratio:
            r = result.ratio
            lines.append(f"Ratio: {r['numerator']['value']} {r['numerator']['unit']} / {r['denominator']['value']} {r['denominator']['unit']} = {r['result']} {r.get('result_unit', '')}")
        lines.append(f"Extraction method: {result.extraction_method} (confidence: {result.confidence:.2f})")
        lines.append("[/NUMERIC_EXTRACTION_DATA]\n")
        return "\n".join(lines)

    # ── Helper methods ──────────────────────────────────────

    def _parse_number(self, text: str) -> float | None:
        """Parse a Vietnamese/European formatted number."""
        text = text.strip()
        if not text:
            return None
        # Handle Vietnamese/European number formats:
        # "1.234.567" → 1234567 (dot as thousands sep)
        # "1,234.56"  → 1234.56 (comma as thousands sep)
        # "1.234,56"  → 1234.56 (European)
        # "12.5"      → 12.5 (regular decimal)

        # Count dots and commas
        dots = text.count(".")
        commas = text.count(",")

        if dots > 1:
            # Multiple dots → thousand separators, comma might be decimal
            text = text.replace(".", "")
            text = text.replace(",", ".")
        elif commas > 1:
            # Multiple commas → thousand separators, dot might be decimal
            text = text.replace(",", "")
        elif dots == 1 and commas == 1:
            # One of each → last one is decimal
            dot_pos = text.index(".")
            comma_pos = text.index(",")
            if dot_pos < comma_pos:
                # "1.234,56" → European
                text = text.replace(".", "").replace(",", ".")
            else:
                # "1,234.56" → US format
                text = text.replace(",", "")
        elif commas == 1 and dots == 0:
            # Could be decimal comma: "12,5" or thousands: "1,234"
            parts = text.split(",")
            if len(parts[1]) <= 2:
                # Likely decimal: "12,5" or "0,25"
                text = text.replace(",", ".")
            else:
                # Likely thousands: "1,234"
                text = text.replace(",", "")

        try:
            return float(text)
        except ValueError:
            return None

    def _normalize_unit(self, raw_unit: str) -> str:
        """Map raw unit text to standardized unit string."""
        raw = raw_unit.strip()
        for pattern, normalized in UNIT_PATTERNS:
            if re.match(pattern, raw, re.IGNORECASE):
                return normalized
        return raw

    def _detect_target_metrics(self, question: str) -> list[str]:
        """Detect which metrics the question is asking about."""
        question_lower = question.lower()
        if "chấm dứt hợp đồng" in question_lower or "nghỉ việc" in question_lower:
            return ["turnover_rate"]
        if "tuyển dụng" in question_lower and "nhân viên mới" in question_lower:
            return ["recruitment_rate"]
        if "nước thải" in question_lower:
            return ["wastewater"]
        if "tuần hoàn" in question_lower or "tái sử dụng nước" in question_lower:
            return ["water_reuse"]
        if "tiết kiệm" in question_lower and "năng lượng" in question_lower:
            return ["energy_saving"]
        if "phát thải khí nhà kính" in question_lower or "phat thai khi nha kinh" in question_lower:
            return ["ghg_total"]
        if "phát thải" in question_lower or "phat thai" in question_lower:
            return ["ghg_total"]
        if "doanh thu" in question_lower or "revenue" in question_lower:
            return ["revenue"]
        if "thiện nguyện" in question_lower or "cộng đồng địa phương" in question_lower:
            return ["social_investment"]
        if "r&d" in question_lower or "đổi mới công nghệ" in question_lower or "nghiên cứu và phát triển" in question_lower:
            return ["rd_expense"]
        if "vật liệu tái chế" in question_lower or "vật liệu tái tạo" in question_lower:
            return ["recycled_material_ratio"]
        if "vật liệu đóng gói" in question_lower or "thu hồi" in question_lower:
            return ["packaging_recovery_ratio"]
        found = []
        for metric_key, keywords in METRIC_KEYWORDS.items():
            if any(kw in question_lower for kw in keywords):
                found.append(metric_key)
        return found or ["unknown"]

    def _detect_year_near(self, text: str, start: int, end: int, window: int = 200) -> int | None:
        """Find the closest year mention near a numeric value."""
        region_start = max(0, start - window)
        region_end = min(len(text), end + window)
        region = text[region_start:region_end]
        year_matches = re.findall(r"\b(20[12]\d)\b", region)
        if not year_matches:
            return None
        # Prefer target_year if found
        for y in year_matches:
            if int(y) == self.target_year:
                return self.target_year
        # Otherwise return closest to target_year
        return min((int(y) for y in year_matches), key=lambda y: abs(y - self.target_year))

    def _find_source_id(self, context: str, position: int, sections: list[dict]) -> str:
        """Find which SOURCE_ID block contains the position."""
        # Search backwards for [SOURCE_ID: Sx]
        region = context[max(0, position - 2000):position]
        matches = re.findall(r"\[SOURCE_ID:\s*(S\d+)", region)
        if matches:
            return matches[-1]
        if sections:
            return sections[0].get("source_id", "")
        return ""

    def _find_page(self, context: str, position: int, sections: list[dict]) -> int | None:
        """Find page number near position."""
        region = context[max(0, position - 1500):position]
        matches = re.findall(r"PAGES?:\s*(\d+)", region, re.IGNORECASE)
        if matches:
            try:
                return int(matches[-1])
            except ValueError:
                pass
        return None

    def _source_for_page(self, page: int | None, sections: list[dict]) -> str:
        if page is None:
            return sections[0].get("source_id", "") if sections else ""
        for section in sections:
            start = section.get("page_start")
            end = section.get("page_end")
            if start is not None and end is not None and int(start) <= page <= int(end):
                return section.get("source_id", "")
        return sections[0].get("source_id", "") if sections else ""

    def _identify_metric(self, snippet: str, target_metrics: list[str]) -> str:
        """Identify which metric a snippet corresponds to."""
        snippet_lower = snippet.lower()
        for metric_key in target_metrics:
            keywords = METRIC_KEYWORDS.get(metric_key, [])
            if any(kw in snippet_lower for kw in keywords):
                return metric_key
        # Generic identification
        for metric_key, keywords in METRIC_KEYWORDS.items():
            if any(kw in snippet_lower for kw in keywords):
                return metric_key
        return "unknown"

    def _score_candidate_confidence(
        self,
        value: float,
        unit: str,
        year: int | None,
        metric_name: str,
        target_metrics: list[str],
    ) -> float:
        """Score confidence of a regex extraction candidate."""
        if target_metrics and target_metrics != ["unknown"] and metric_name not in target_metrics:
            return 0.0
        if metric_name == "unknown":
            return 0.0

        score = 0.3  # base

        # Year match
        if year == self.target_year:
            score += 0.25
        elif year and abs(year - self.target_year) <= 1:
            score += 0.1

        # Metric match
        if metric_name in target_metrics:
            score += 0.25

        allowed_units = METRIC_UNIT_ALLOWLIST.get(metric_name)
        if allowed_units and unit not in allowed_units:
            return 0.0

        # Reasonable value range
        if value > 0:
            score += 0.1
        if unit in {"tấn CO2e", "m3", "kWh", "MWh", "GJ", "MJ", "tỷ đồng", "%"}:
            score += 0.1

        return min(1.0, score)
