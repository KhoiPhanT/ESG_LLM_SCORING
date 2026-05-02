"""
LLM Client — Giao tiếp với Qwen3:30b qua Ollama REST API.
Đây là module trung tâm của toàn bộ pipeline ESG.
Enhanced: expanded context window, stricter prompts for accuracy.
"""
import json
import os
import requests
import re
import time


class OllamaClient:
    def __init__(self, model="qwen3:30b", base_url="http://localhost:11434"):
        self.model = model
        self.base_url = base_url
        self.chat_url = f"{base_url}/api/chat"
        self.num_ctx = int(os.environ.get("ESG_OLLAMA_NUM_CTX", "24576"))
        self.timeout = int(os.environ.get("ESG_OLLAMA_TIMEOUT", "300"))
        self._last_parse_info = {}
        self._last_call_info = {}

    def _call(self, messages, temperature=0.3, max_tokens=3072, retries=1):
        """Gửi request tới Ollama và trả về response text."""
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                "num_ctx": self.num_ctx,
            },
        }
        for attempt in range(retries + 1):
            try:
                resp = requests.post(self.chat_url, json=payload, timeout=self.timeout)
                resp.raise_for_status()
                body = resp.json()
                message = body.get("message", {}) if isinstance(body, dict) else {}
                raw_content = str(message.get("content", "") or "")
                content = self._strip_think_tags(raw_content)
                self._last_call_info = {
                    "status": "ok",
                    "raw_response_before_strip": raw_content,
                    "response_after_strip": content,
                    "raw_response_length": len(raw_content),
                    "response_after_strip_length": len(content),
                    "done": body.get("done") if isinstance(body, dict) else None,
                    "done_reason": body.get("done_reason") if isinstance(body, dict) else None,
                    "total_duration": body.get("total_duration") if isinstance(body, dict) else None,
                    "load_duration": body.get("load_duration") if isinstance(body, dict) else None,
                    "prompt_eval_count": body.get("prompt_eval_count") if isinstance(body, dict) else None,
                    "eval_count": body.get("eval_count") if isinstance(body, dict) else None,
                }
                return content
            except Exception as e:
                self._last_call_info = {
                    "status": "error",
                    "error": str(e),
                }
                if attempt < retries:
                    print(f"  [LLM retry {attempt+1}] {e}")
                    time.sleep(2 + attempt * 2)
                else:
                    print(f"  [LLM ERROR] {e}")
                    return None

    @staticmethod
    def _strip_think_tags(text: str) -> str:
        """Robustly strip all <think>...</think> blocks from Qwen3 output.
        Handles: nested blocks, truncated blocks (no closing tag), multiple blocks."""
        if not text:
            return ""
        open_tag = "<think>"
        close_tag = "</think>"
        result = []
        idx = 0
        depth = 0
        while idx < len(text):
            if text.startswith(open_tag, idx):
                depth += 1
                idx += len(open_tag)
                continue
            if text.startswith(close_tag, idx):
                if depth > 0:
                    depth -= 1
                idx += len(close_tag)
                continue
            if depth == 0:
                result.append(text[idx])
            idx += 1
        return "".join(result).strip()

    def _parse_json(self, text):
        """Trích xuất JSON từ response LLM (có thể lẫn text thừa hoặc bị cắt dở)."""
        call_info = dict(self._last_call_info or {})
        call_status = call_info.get("status")
        raw_before_strip = call_info.get("raw_response_before_strip")
        after_strip = call_info.get("response_after_strip")
        self._last_parse_info = {
            "parse_status": "failed",
            "parse_error": None,
            "raw_response_preview": self._response_preview(text),
            "call_info": call_info,
        }
        if call_status == "error":
            self._last_parse_info["parse_status"] = "llm_transport_error"
            self._last_parse_info["parse_error"] = call_info.get("error")
            return None
        if not text:
            if raw_before_strip == "":
                self._last_parse_info["parse_status"] = "empty_raw_response"
            elif raw_before_strip is not None and after_strip == "":
                self._last_parse_info["parse_status"] = "empty_after_think_strip"
            else:
                self._last_parse_info["parse_status"] = "failed"
            return None
        text = str(text).strip()
        text = self._strip_think_tags(text)
        candidate_text = self._trim_to_json_start(text)

        # Strategy 0: markdown fence
        fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", candidate_text, re.DOTALL | re.IGNORECASE)
        if fence:
            candidate = fence.group(1).strip()
            for attempt_text in [candidate, self._fix_trailing_commas(candidate)]:
                try:
                    return self._with_parse_info(json.loads(attempt_text), "valid_json")
                except json.JSONDecodeError as exc:
                    self._last_parse_info["parse_error"] = str(exc)

        # Strategy 1: balanced brace extraction (most reliable)
        balanced = self._extract_balanced_json_object(candidate_text)
        if balanced:
            for attempt_text in [balanced, self._fix_trailing_commas(balanced)]:
                try:
                    return self._with_parse_info(json.loads(attempt_text), "valid_json")
                except json.JSONDecodeError as exc:
                    self._last_parse_info["parse_error"] = str(exc)

        # Strategy 2: greedy first-{ to last-} match
        match = re.search(r"(\{.*\})", candidate_text, re.DOTALL)
        if match:
            clean_json = match.group(1).strip()
            for attempt_text in [clean_json, self._fix_trailing_commas(clean_json)]:
                try:
                    return self._with_parse_info(json.loads(attempt_text), "valid_json")
                except json.JSONDecodeError as exc:
                    self._last_parse_info["parse_error"] = str(exc)

        # Strategy 3: truncated JSON repair (max_tokens reached)
        match2 = re.search(r"(\{.*)", candidate_text, re.DOTALL)
        if match2:
            truncated = match2.group(1).strip()
            repaired = self._repair_truncated_json(truncated)
            if repaired is not None:
                return self._with_parse_info(repaired, "repaired_json")
            # Last resort: extract answer via regex
            answer_match = re.search(r'"answer"\s*:\s*"([A-Z,]+|NULL)"', truncated)
            reason_match = re.search(r'"reason"\s*:\s*"([^"]+)"', truncated)
            evidence_match = re.search(r'"(?:evidence|evidence_quote)"\s*:\s*"([^"]{10,})"?', truncated)
            source_match = re.search(r'"evidence_source_id"\s*:\s*"([^"]+)"', truncated)
            confidence_match = re.search(r'"confidence"\s*:\s*([01](?:\.\d+)?)', truncated)
            if answer_match:
                answer = answer_match.group(1)
                return self._with_parse_info({
                    "answer": answer,
                    "selected_options": [] if answer == "NULL" else [x.strip() for x in answer.split(",")],
                    "reason": reason_match.group(1) if reason_match else "",
                    "evidence": evidence_match.group(1) if evidence_match else None,
                    "evidence_quote": evidence_match.group(1) if evidence_match else None,
                    "evidence_source_id": source_match.group(1) if source_match else None,
                    "confidence": float(confidence_match.group(1)) if confidence_match else 0.0,
                }, "answer_regex_only")

        # Fallback: thử parse toàn bộ
        try:
            return self._with_parse_info(json.loads(candidate_text), "valid_json")
        except json.JSONDecodeError as exc:
            self._last_parse_info["parse_error"] = str(exc)
            if "unterminated" in str(exc).lower() or "expecting" in str(exc).lower():
                self._last_parse_info["parse_status"] = "json_malformed"
            return None

    @staticmethod
    def _trim_to_json_start(text: str) -> str:
        """Drop leading narrative/prefix text before the first JSON object."""
        if not text:
            return ""
        first_brace = text.find("{")
        if first_brace < 0:
            return text.strip()
        prefix = text[:first_brace].strip()
        if not prefix:
            return text[first_brace:].strip()
        if "{" not in prefix and "}" not in prefix:
            return text[first_brace:].strip()
        return text.strip()

    @staticmethod
    def _fix_trailing_commas(text: str) -> str:
        """Remove trailing commas before } or ] which are common LLM JSON errors."""
        text = re.sub(r",\s*}", "}", text)
        text = re.sub(r",\s*]", "]", text)
        return text

    @staticmethod
    def _repair_truncated_json(text: str) -> dict | None:
        """Try to repair truncated JSON by closing open strings/braces/brackets."""
        if not text:
            return None
        candidate = text.strip()
        if not candidate.startswith("{"):
            return None
        stack = []
        in_string = False
        escape = False
        for char in candidate:
            if in_string:
                if escape:
                    escape = False
                elif char == "\\":
                    escape = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char in "{[":
                stack.append(char)
            elif char == "}" and stack and stack[-1] == "{":
                stack.pop()
            elif char == "]" and stack and stack[-1] == "[":
                stack.pop()
        if in_string:
            candidate += '"'
        closing = []
        while stack:
            closing.append("}" if stack.pop() == "{" else "]")
        candidate = OllamaClient._fix_trailing_commas(candidate) + "".join(closing)
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass
        return None

    def _with_parse_info(self, parsed, status: str):
        if isinstance(parsed, dict):
            self._last_parse_info = {
                "parse_status": status,
                "parse_error": None if status in {"valid_json", "repaired_json"} else self._last_parse_info.get("parse_error"),
                "raw_response_preview": self._last_parse_info.get("raw_response_preview"),
                "call_info": self._last_parse_info.get("call_info", {}),
            }
            parsed.setdefault("__parse_status", status)
            parsed.setdefault("__parse_error", self._last_parse_info.get("parse_error"))
            return parsed
        return parsed

    def _response_preview(self, text, limit: int = 2000) -> str | None:
        if text is None:
            return None
        value = str(text)
        if len(value) <= limit:
            return value
        half = limit // 2
        return value[:half] + "\n...[truncated]...\n" + value[-half:]

    def _extract_balanced_json_object(self, text: str) -> str | None:
        start = text.find("{")
        if start < 0:
            return None
        depth = 0
        in_string = False
        escape = False
        for idx in range(start, len(text)):
            char = text[idx]
            if in_string:
                if escape:
                    escape = False
                elif char == "\\":
                    escape = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return text[start:idx + 1]
        return None

    # ─── Query Planning ──────────────────────────────────────

    def plan_retrieval_query(self, rule: dict, target_year: int | None = None) -> dict | None:
        """
        Ask the LLM to plan retrieval once for a scoring rule.
        No document context is provided here; invalid JSON returns None so callers can
        fall back deterministically without another LLM call.
        """
        target_year = target_year or 2024
        payload = {
            "id": rule.get("id", ""),
            "question": rule.get("question", ""),
            "options": rule.get("options", ""),
            "logic": rule.get("logic", ""),
            "question_type": rule.get("question_type", "default"),
            "time_policy": rule.get("time_policy", "unspecified"),
            "factor": rule.get("factor", ""),
            "target_year": target_year,
        }
        prompt = f"""
You are planning retrieval for a Vietnamese ESG/VNSI scoring system.
Do NOT answer the question. Do NOT use document context. Your job is to inspect the question, options, and scoring logic, then produce tight search instructions.

Inputs:
{json.dumps(payload, ensure_ascii=False, indent=2)}

Planning rules:
- Analyze every option in the rubric, especially multi-select options and options with negative score.
- Generate queries that find evidence matching the scoring logic, not just repeated question words.
- Prefer Vietnamese aliases used in annual reports, governance reports, sustainability reports, financial statements, AGM minutes/resolutions.
- `required_doc_types` must use only these labels when relevant: policy_document, annual_report, sustainability_report, financial_report, resolution, other.
- `year_policy` must be one of: current_year_required, historical_allowed, latest_valid_allowed, future_target_allowed, unspecified.
- `evidence_shape` must be one of: policy_text, narrative_text, metric_table, financial_table, governance_profile, meeting_resolution, certificate, numeric_value, mixed.
- For numeric/table questions, include metric-family anchors in `must_have_terms` such as energy/water/emissions/waste/workforce/revenue/CSR spending.
- Put distracting topics in `avoid_terms`; include 2050/2060/future-target terms when the question is not about future commitments.

Return one JSON object only with exactly these fields:
{{
  "search_queries": ["2-5 short, specific Vietnamese queries"],
  "semantic_aliases": ["synonyms/near terms that bridge the wording gap"],
  "required_doc_types": ["document type labels"],
  "must_have_terms": ["anchors that should appear in a good source"],
  "avoid_terms": ["terms that indicate noisy/off-topic source"],
  "year_policy": "current_year_required | historical_allowed | latest_valid_allowed | future_target_allowed | unspecified",
  "evidence_shape": "policy_text | narrative_text | metric_table | financial_table | governance_profile | meeting_resolution | certificate | numeric_value | mixed",
  "option_focus": {{"A": ["terms to verify option A"], "B": ["terms to verify option B"]}}
}}"""

        raw = self._call(
            [{"role": "user", "content": prompt}],
            temperature=0.05,
            max_tokens=2048,
            retries=1,
        )
        parsed = self._parse_json(raw)
        return self._validate_query_plan(parsed)

    def _validate_query_plan(self, plan: dict | None) -> dict | None:
        if not isinstance(plan, dict):
            return None

        required = [
            "search_queries",
            "semantic_aliases",
            "required_doc_types",
            "must_have_terms",
            "avoid_terms",
            "year_policy",
            "evidence_shape",
            "option_focus",
        ]
        if any(field not in plan for field in required):
            return None

        cleaned = {}
        for field in ["search_queries", "semantic_aliases", "required_doc_types", "must_have_terms", "avoid_terms"]:
            values = self._coerce_string_list(plan.get(field))
            if field == "search_queries" and not values:
                return None
            if field == "required_doc_types":
                values = self._clean_doc_types(values)
            cleaned[field] = values[:12]

        allowed_year = {
            "current_year_required",
            "historical_allowed",
            "latest_valid_allowed",
            "future_target_allowed",
            "unspecified",
        }
        year_policy = str(plan.get("year_policy") or "unspecified").strip().lower()
        cleaned["year_policy"] = year_policy if year_policy in allowed_year else "unspecified"

        allowed_shapes = {
            "policy_text",
            "narrative_text",
            "metric_table",
            "financial_table",
            "governance_profile",
            "meeting_resolution",
            "certificate",
            "numeric_value",
            "mixed",
        }
        evidence_shape = str(plan.get("evidence_shape") or "mixed").strip().lower()
        cleaned["evidence_shape"] = evidence_shape if evidence_shape in allowed_shapes else "mixed"

        option_focus = plan.get("option_focus")
        if not isinstance(option_focus, dict):
            return None
        cleaned["option_focus"] = {
            str(key).strip().upper(): self._coerce_string_list(value)[:8]
            for key, value in option_focus.items()
            if str(key).strip()
        }
        return cleaned

    def _clean_doc_types(self, values: list[str]) -> list[str]:
        aliases = {
            "annual report": ["annual_report"],
            "annual_report": ["annual_report"],
            "policy": ["policy_document"],
            "policy document": ["policy_document"],
            "policy_document": ["policy_document"],
            "environmental policy": ["policy_document"],
            "sustainability policy": ["policy_document"],
            "sustainability report": ["sustainability_report"],
            "sustainability_report": ["sustainability_report"],
            "financial report": ["financial_report"],
            "financial_report": ["financial_report"],
            "financial_statement": ["financial_report"],
            "financial statements": ["financial_report"],
            "resolution": ["resolution"],
            "meeting minutes": ["resolution"],
            "meeting_minutes": ["resolution"],
            "agm minutes": ["resolution"],
            "governance_report": ["annual_report", "resolution", "other"],
            "governance report": ["annual_report", "resolution", "other"],
            "corporate governance report": ["annual_report", "resolution", "other"],
            "other": ["other"],
        }
        cleaned = []
        for value in values:
            key = str(value).strip().lower()
            for mapped in aliases.get(key, []):
                cleaned.append(mapped)
        return list(dict.fromkeys(cleaned))

    def _coerce_string_list(self, value) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            raw_items = [value]
        elif isinstance(value, (list, tuple, set)):
            raw_items = list(value)
        else:
            return []
        items = []
        for item in raw_items:
            text = str(item).strip()
            if text and text.lower() not in {"null", "none", "n/a"}:
                items.append(text)
        return list(dict.fromkeys(items))

    # ─── VNSI Question Answering ─────────────────────────────

    def ask_vnsi_question(
        self,
        context,
        question,
        options,
        q_id="",
        is_multi_select=False,
        question_type="default",
        time_policy="unspecified",
        query_plan=None,
        context_limit=22000,
        response_mode="full",
        attempt_index=0,
    ):
        """
        Trả lời 1 câu hỏi VNSI dựa trên context báo cáo.
        Returns: {"answer": "C", "selected_options": ["A"], "reason": "...", "evidence": "..."}
        """
        if is_multi_select:
            response_example = {
                "answer": "A,C",
                "selected_options": ["A", "C"],
                "confidence": 0.82,
                "reason": "Nguồn cung cấp bằng chứng riêng cho từng lựa chọn đã chọn.",
                "option_evidence": {
                    "A": {"source_id": "S1", "quote": "short exact quote without double quotes"},
                    "C": {"source_id": "S3", "quote": "short exact quote without double quotes"},
                },
            }
            multi_instruction = """- This question allows MULTIPLE selections. Evaluate EACH criterion (A, B, C, D...) independently.
- Select ALL options that the company meets based on the evidence.
- If no option is supported by the provided source context, return answer: "NULL" and selected_options: [].
- Evidence can come from different SOURCE_ID blocks for different options.
- If an OPTION_CANDIDATES block points to an option, inspect that option independently.
- If the source is an official policy/report and describes an activity, commitment, approval, measure, or disclosure matching an option, count that option as supported.
- Return option_evidence for every selected option only. Do not return evidence for unselected options.
- Keep each quote under 180 characters, single-line, and copied exactly. Do not put double quotes inside quote text.
- Example: if company meets both C and D, return selected_options: ["C", "D"] and answer: "C,D"."""
        else:
            response_example = {
                "answer": "A",
                "selected_options": ["A"],
                "confidence": 0.82,
                "reason": "Nguồn S1 đề cập đúng nội dung cần chấm.",
                "evidence_source_id": "S1",
                "evidence_quote": "short exact quote without double quotes",
            }
            multi_instruction = "- Select exactly ONE option only when the provided context supports it; otherwise return NULL."

        if response_mode == "minimal":
            response_example = {
                "answer": response_example["answer"],
                "selected_options": response_example["selected_options"],
                "confidence": response_example["confidence"],
            }
        elif response_mode == "answer_only":
            response_example = {
                "answer": response_example["answer"],
                "selected_options": response_example["selected_options"],
            }

        if time_policy == "current_year_required":
            temporal_instruction = "- This question is current-year specific. Prefer evidence from the assessment year/current fiscal year; do not award points from historical evidence alone."
        elif time_policy in {"historical_allowed", "latest_valid_allowed"}:
            temporal_instruction = "- Historical evidence is valid if it proves a policy/system/certification exists, was completed, or remains effective."
        else:
            temporal_instruction = "- Use the most relevant evidence; mention if evidence is historical or current-year."

        if question_type in {"numeric_disclosure", "ratio_calculation"}:
            numeric_instruction = """- This is a numeric disclosure question. Look for tables, values, units, years, revenue, ratios, Scope 1/2/3, water, waste, energy, workforce, or social spending data.
- For ratio questions, answer positively when the context contains the relevant numerator metric and revenue/denominator needed to compute the ratio, even if the report does not pre-calculate the ratio."""
        else:
            numeric_instruction = ""

        if is_multi_select:
            evidence_instruction = """- Each selected option must have an option_evidence entry with a valid source_id from the context.
- Each option quote must be a short exact quote proving that specific option.
- If answer is NULL, option_evidence must be an empty object."""
        else:
            evidence_instruction = """- The evidence_source_id MUST be one of the SOURCE_ID values in the context, or null.
- evidence_quote must be a short exact quote copied from that SOURCE_ID block. If answer is NULL, evidence_source_id and evidence_quote must be null."""

        plan_summary = self._format_query_plan_summary(query_plan) if query_plan else "N/A"
        response_schema = json.dumps(response_example, ensure_ascii=False, indent=2)

        prompt = f"""
You are an ESG analyst evaluating a Vietnamese company's annual report.
Answer the following VNSI assessment question based on the provided report context.
Use only the provided SOURCE_ID blocks. Do not use outside knowledge.

CRITICAL RULES:
- NULL is a valid answer. If the provided context is insufficient, return "NULL".
- Do not be stricter than the workbook. When the context mentions the requested policy, action, disclosure, metric, activity, committee, or practice with a relevant source, that is enough to select the positive workbook option.
- Many VNSI questions score disclosure/mention, not audit-grade proof. Do not require exhaustive details, implementation proof, or pre-calculated ratios unless the option explicitly asks for them.
- Do not infer a negative-score answer from vague silence. Select a negative option only when the context explicitly supports it, or when the workbook option is clearly an absence-of-disclosure finding and the provided context is sufficient for that finding.
- Select a positive option only when there is grounded evidence in one SOURCE_ID block.
- For historical_allowed/latest_valid_allowed questions, completed prior-year actions can count if the workbook permits them and the evidence proves they remain relevant.
{multi_instruction}

EVIDENCE QUALITY RULES:
- TEMPORAL RULE: {temporal_instruction}
- NUMERIC RULE: {numeric_instruction}
{evidence_instruction}
- Confidence is 0.0-1.0. Use <=0.4 if evidence is incomplete or indirect.

RETRIEVAL PLAN SUMMARY:
{plan_summary}

QUESTION [{q_id}]: {question}

OPTIONS:
{options}

REPORT CONTEXT:
{context[:context_limit]}

OUTPUT FORMAT:
- Respond with a single raw JSON object ONLY.
- Do NOT wrap the JSON in markdown code fences (``` or ```json).
- Do NOT include any text, explanation, or commentary before or after the JSON.
- The JSON must be valid and parseable.

{response_schema}

If answer is NULL, return this valid JSON shape:
{self._null_response_example(is_multi_select, response_mode)}"""

        messages = [{"role": "user", "content": prompt}]
        raw = self._call(
            messages,
            temperature=0.05,
            max_tokens=4096 if is_multi_select else 3072,
            retries=1,
        )
        result = self._parse_json(raw)
        parse_info = dict(self._last_parse_info or {})
        parse_status = parse_info.get("parse_status", "failed")
        if parse_status in {
            "answer_regex_only",
            "failed",
            "json_malformed",
            "empty_raw_response",
            "empty_after_think_strip",
            "llm_transport_error",
        }:
            self._write_llm_json_error(
                q_id=q_id,
                parse_info=parse_info,
                raw=raw,
                prompt=prompt,
                context_chars=len(context[:context_limit]),
                is_multi_select=is_multi_select,
                response_mode=response_mode,
                attempt_index=attempt_index,
            )
        if (
            is_multi_select
            and result
            and parse_status in {"answer_regex_only", "repaired_json"}
            and not result.get("option_evidence")
        ):
            result = None
        if result and "answer" in result:
            result["answer"] = str(result.get("answer", "NULL")).strip().upper() or "NULL"
            result["parse_status"] = result.pop("__parse_status", parse_status)
            if "__parse_error" in result:
                result["parse_error"] = result.pop("__parse_error")
            if "selected_options" not in result or not result["selected_options"]:
                answer = str(result.get("answer", "")).strip()
                if answer.upper() == "NULL":
                    result["selected_options"] = []
                else:
                    result["selected_options"] = [x.strip() for x in answer.split(",") if x.strip()]
            if "evidence" not in result:
                result["evidence"] = result.get("evidence_quote")
            try:
                result["confidence"] = max(0.0, min(1.0, float(result.get("confidence", 0.0) or 0.0)))
            except (TypeError, ValueError):
                result["confidence"] = 0.0
            return result
        # Fallback nếu parse thất bại hoặc LLM không trả về kết quả
        fallback_reason = "llm_error" if raw is None else "json_parse_failed_or_no_answer"
        return {
            "answer": "NULL",
            "selected_options": [],
            "reason": f"Không tìm thấy bằng chứng xác thực trong tài liệu (Fallback: {fallback_reason})",
            "evidence": None,
            "parse_status": parse_status,
            "parse_error": parse_info.get("parse_error"),
            "is_empty_response": (raw == ""),
            "call_info": parse_info.get("call_info", {}),
        }

    def _null_response_example(self, is_multi_select: bool, response_mode: str) -> str:
        payload = {"answer": "NULL", "selected_options": []}
        if response_mode != "answer_only":
            payload["confidence"] = 0.2
        if response_mode == "full":
            payload["reason"] = "Không đủ bằng chứng trong SOURCE_ID đã cung cấp."
            if is_multi_select:
                payload["option_evidence"] = {}
            else:
                payload["evidence_source_id"] = None
                payload["evidence_quote"] = None
        return json.dumps(payload, ensure_ascii=False)

    def _format_query_plan_summary(self, query_plan: dict) -> str:
        if not isinstance(query_plan, dict):
            return "N/A"
        lines = [
            f"- strategy: {query_plan.get('metadata_strategy') or query_plan.get('query_plan_source') or 'standard'}",
            f"- year_policy: {query_plan.get('year_policy', 'unspecified')}",
            f"- evidence_shape: {query_plan.get('evidence_shape', 'mixed')}",
            f"- required_doc_types: {', '.join(query_plan.get('required_doc_types') or []) or 'N/A'}",
        ]
        option_focus = query_plan.get("option_focus")
        if isinstance(option_focus, dict) and option_focus:
            lines.append("- option_focus:")
            for letter in sorted(option_focus):
                terms = self._coerce_string_list(option_focus.get(letter))[:5]
                lines.append(f"  {letter}: {', '.join(terms)}")
        else:
            aliases = self._coerce_string_list(query_plan.get("semantic_aliases"))[:8]
            if aliases:
                lines.append(f"- semantic_aliases: {', '.join(aliases)}")
        return "\n".join(lines)

    def _write_llm_json_error(
        self,
        q_id: str,
        parse_info: dict,
        raw,
        prompt: str,
        context_chars: int,
        is_multi_select: bool,
        response_mode: str,
        attempt_index: int,
    ) -> None:
        try:
            os.makedirs("outputs/debug/llm_json_errors", exist_ok=True)
            safe_qid = re.sub(r"[^A-Za-z0-9_.-]+", "_", q_id or "unknown")
            call_info = parse_info.get("call_info", {}) or {}
            payload = {
                "question_id": q_id,
                "parse_status": parse_info.get("parse_status"),
                "parse_error": parse_info.get("parse_error"),
                "is_multi_select": is_multi_select,
                "response_mode": response_mode,
                "attempt_index": attempt_index,
                "prompt_chars": len(prompt or ""),
                "context_chars": context_chars,
                "raw_response": raw,
                "raw_response_preview": parse_info.get("raw_response_preview"),
                "raw_response_before_strip": call_info.get("raw_response_before_strip"),
                "response_after_strip": call_info.get("response_after_strip"),
                "raw_response_length": call_info.get("raw_response_length"),
                "response_after_strip_length": call_info.get("response_after_strip_length"),
                "done_reason": call_info.get("done_reason"),
                "eval_count": call_info.get("eval_count"),
                "prompt_eval_count": call_info.get("prompt_eval_count"),
                "total_duration": call_info.get("total_duration"),
                "load_duration": call_info.get("load_duration"),
                "llm_call_status": call_info.get("status"),
            }
            path = os.path.join("outputs/debug/llm_json_errors", f"{safe_qid}.json")
            tmp_path = path + ".tmp"
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            os.replace(tmp_path, path)
        except Exception:
            pass

    # ─── Screening Questions ─────────────────────────────────

    def ask_screening_question(self, context, question, q_id=""):
        """
        Trả lời câu hỏi sàng lọc (SL1-SL5): Có/Không.
        """
        prompt = f"""
You are auditing a Vietnamese company's annual report for ESG violations.
Answer this screening question with ONLY "A" (Yes, violation found) or "B" (No violation found).
Be conservative: only answer "A" if there is CLEAR evidence of violation.
- Answer "B" for ordinary risk controls, policies, audit procedures, compliance statements, no-incident statements, or generic mentions of the violation topic.
- Answer "A" only if the context says the company was concluded, fined, investigated, or received a qualified audit opinion for the exact screening issue.
Think step-by-step concisely to prevent reaching token limits.

QUESTION [{q_id}]: {question}

REPORT CONTEXT:
{context[:30000]}

Respond with JSON only:
{{"answer": "A or B", "reason": "<1 sentence in Vietnamese>"}}"""

        messages = [{"role": "user", "content": prompt}]
        raw = self._call(messages, temperature=0.1, max_tokens=2048)
        result = self._parse_json(raw)
        if result and "answer" in result:
            return result
        return {"answer": "B", "reason": "Không phát hiện vi phạm rõ ràng"}

    # ─── Entity Extraction ───────────────────────────────────

    def extract_esg_entities(self, context):
        """Trích xuất các chỉ số định lượng ESG từ đoạn báo cáo."""
        prompt = f"""
Extract quantitative ESG metrics from this Vietnamese company report section.
Think step-by-step concisely before returning the JSON output.
Return JSON with these fields (use null if not found):

{{"co2_emission": "value with unit or null",
 "energy_consumption": "value with unit or null",
 "water_usage": "value with unit or null",
 "waste_treated": "value with unit or null",
 "female_leadership_ratio": "percentage or null",
 "training_hours_per_employee": "number or null",
 "social_investment": "value with unit or null",
 "employee_count": "number or null",
 "rd_expense": "value with unit or null"}}

TEXT:
{context[:8000]}

JSON:"""
        messages = [{"role": "user", "content": prompt}]
        raw = self._call(messages, temperature=0.2, max_tokens=2048)
        result = self._parse_json(raw)
        return result or {}


if __name__ == "__main__":
    client = OllamaClient()
    # Quick test
    result = client.ask_vnsi_question(
        context="ACB đã xây dựng chiến lược phát triển bền vững với cam kết bảo vệ môi trường.",
        question="Công ty có chính sách liên quan tới quản lý các tác động môi trường?",
        options="A. Không có\nB. Có nhưng không công khai\nC. Có và công khai",
        q_id="E.1.1.1",
    )
    print("Test result:", json.dumps(result, ensure_ascii=False, indent=2))
