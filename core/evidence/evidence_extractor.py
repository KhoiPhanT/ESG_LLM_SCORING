"""
Evidence-first extraction layer for VNSI scoring questions.
Enhanced with non-LLM evidence verification and numeric extraction.
"""
from __future__ import annotations

from core.evidence.evidence_verifier import EvidenceVerifier
from core.evidence.numeric_extractor import NumericExtractor


class EvidenceExtractor:
    def __init__(self, llm_client=None, target_year: int | None = None):
        self.llm_client = llm_client
        self.verifier = EvidenceVerifier()
        self.numeric_extractor = NumericExtractor(llm_client=llm_client, target_year=target_year)

    def extract(self, rule: dict, context_bundle: dict) -> dict:
        question_id = rule.get("id", "")
        source_sections = context_bundle.get("sections", [])
        context = context_bundle.get("context", "")
        question_type = rule.get("question_type", "default")

        # ── NumericExtractor: run BEFORE LLM to enrich context ───────
        numeric_extraction = None
        enriched_context = context
        if question_type in {"numeric_disclosure", "ratio_calculation"}:
            numeric_extraction = self.numeric_extractor.extract(rule, context_bundle)
            if numeric_extraction and numeric_extraction.get("context_enrichment"):
                enriched_context = numeric_extraction["context_enrichment"] + context

        # ── LLM answering with enriched context ──────────────────────
        if self.llm_client:
            llm_result = self.llm_client.ask_vnsi_question(
                context=enriched_context,
                question=rule.get("question", ""),
                options=rule.get("options", ""),
                q_id=question_id,
                is_multi_select=rule.get("is_multi_select", False),
                question_type=question_type,
                time_policy=rule.get("time_policy", "unspecified"),
                query_plan=context_bundle.get("retrieval_meta", {}).get("query_plan"),
                context_limit=context_bundle.get("context_char_limit", 22000),
            )
        else:
            llm_result = {
                "answer": "NULL",
                "selected_options": [],
                "reason": "LLM không khả dụng",
                "evidence": None,
            }

        answer = str(llm_result.get("answer", "NULL")).strip().upper() or "NULL"
        selected_options = self._normalize_selected_options(llm_result, answer)
        raw_evidence = llm_result.get("evidence_quote")
        if raw_evidence is None:
            raw_evidence = llm_result.get("evidence")
        evidence_source_id = llm_result.get("evidence_source_id")
        option_evidence = self._normalize_option_evidence(llm_result.get("option_evidence"))
        if (
            rule.get("is_multi_select")
            and (answer in {"", "NULL"} or selected_options)
            and not option_evidence
            and (self._is_llm_parse_fallback(llm_result) or selected_options)
        ):
            fallback_options = None if answer in {"", "NULL"} else set(selected_options)
            option_evidence = self._fallback_option_evidence_from_retrieval(
                rule=rule,
                source_sections=source_sections,
                query_plan=context_bundle.get("retrieval_meta", {}).get("query_plan") or {},
                allowed_options=fallback_options,
            )
            if option_evidence:
                selected_options = sorted(option_evidence.keys())
                answer = ",".join(selected_options)
                if self._is_llm_parse_fallback(llm_result):
                    llm_result["reason"] = (
                        "LLM trả JSON lỗi; dùng bằng chứng option-level từ retrieval đã được grounding kiểm tra."
                    )
        if (
            answer not in {"", "NULL", "SKIP"}
            and evidence_source_id
            and self._quote_too_weak(raw_evidence)
        ):
            replacement = self._supporting_quote_from_section(
                source_sections=source_sections,
                evidence_source_id=evidence_source_id,
                reason=llm_result.get("reason", ""),
                question=rule.get("question", ""),
            )
            if replacement:
                raw_evidence = replacement

        # ── NumericExtractor override: if numeric data found but LLM disagreed ──
        numeric_override = False
        if (
            numeric_extraction
            and numeric_extraction.get("value") is not None
            and numeric_extraction.get("confidence", 0) >= 0.5
            and answer in {"NULL", "B", ""}
            and question_type in {"numeric_disclosure", "ratio_calculation"}
        ):
            answer = "A"
            selected_options = ["A"]
            numeric_override = True
            # Use numeric data as evidence if LLM didn't provide any
            if not raw_evidence or str(raw_evidence).strip().lower() == "null":
                metric = numeric_extraction.get("metric_name", "")
                value = numeric_extraction.get("value", "")
                unit = numeric_extraction.get("unit", "")
                snippet = self._best_numeric_snippet(numeric_extraction)
                raw_evidence = snippet or f"{metric}: {value} {unit}"
                evidence_source_id = numeric_extraction.get("source_id")

        option_verification = {}
        if rule.get("is_multi_select") and option_evidence:
            option_result = self._build_option_level_evidence(
                rule=rule,
                selected_options=selected_options,
                option_evidence=option_evidence,
                source_sections=source_sections,
                llm_confidence=llm_result.get("confidence"),
            )
            selected_options = option_result["selected_options"]
            answer = ",".join(selected_options) if selected_options else "NULL"
            evidence_items = option_result["evidence_items"]
            option_verification = option_result["option_evidence_verification"]
            raw_evidence = option_result["raw_evidence"]
            evidence_source_id = option_result["evidence_source_id"]
        else:
            evidence_items = self._build_evidence_items(
                raw_evidence=raw_evidence,
                source_sections=source_sections,
                evidence_source_id=evidence_source_id,
                llm_confidence=llm_result.get("confidence"),
            )

        # Verify evidence grounding only. This never calls the LLM and never
        # changes the selected answer; scoring handles weak evidence later.
        verification = None
        if evidence_items and source_sections and not option_verification:
            verification_sections = self._sections_for_source_id(source_sections, evidence_source_id) or source_sections
            verification = self.verifier.verify(str(raw_evidence or ""), verification_sections)
            if (
                verification
                and not verification["grounded"]
                and self._allow_source_id_support(rule, answer, evidence_source_id, source_sections, raw_evidence)
            ):
                verification = dict(verification)
                verification["grounded"] = True
                verification["match_type"] = "source_id_context_support"
                verification["match_score"] = max(float(verification.get("match_score", 0.0) or 0.0), 0.55)
            if verification and not verification["grounded"]:
                # Hallucinated evidence — demote confidence
                # But if numeric_override, trust the structured extraction over verification
                if not numeric_override:
                    for item in evidence_items:
                        item["confidence"] = max(0.1, float(item.get("confidence", 0.5)) * 0.35)
                        item["verification_status"] = "ungrounded"
                else:
                    for item in evidence_items:
                        item["verification_status"] = "numeric_override"
            elif verification and verification["grounded"]:
                for item in evidence_items:
                    item["verification_status"] = "grounded"

        if answer in {"", "NULL", "SKIP"} or not evidence_items:
            extraction_status = "insufficient"
        elif verification and not verification["grounded"] and not numeric_override:
            extraction_status = "weakly_supported"
        else:
            extraction_status = "supported"

        result = {
            "question_id": question_id,
            "answer": answer,
            "selected_options": selected_options,
            "reason": llm_result.get("reason", ""),
            "raw_evidence": raw_evidence,
            "evidence_source_id": evidence_source_id,
            "llm_confidence": llm_result.get("confidence"),
            "evidence_items": evidence_items,
            "source_sections": source_sections,
            "status": extraction_status,
            "numeric_extraction": numeric_extraction,
            "numeric_override": numeric_override,
        }

        if verification:
            result["evidence_verification"] = verification
        if option_evidence:
            result["option_evidence"] = option_evidence
            result["option_evidence_verification"] = option_verification

        return result

    def _normalize_selected_options(self, llm_result: dict, answer: str) -> list[str]:
        selected_options = [
            str(opt).strip().upper()
            for opt in llm_result.get("selected_options", [])
            if str(opt).strip()
        ]
        if not selected_options and answer not in {"", "NULL"}:
            selected_options = [part.strip().upper() for part in answer.split(",") if part.strip()]
        return selected_options

    def _best_numeric_snippet(self, numeric_extraction: dict) -> str:
        for candidate in numeric_extraction.get("raw_candidates", []) or []:
            snippet = str(candidate.get("context_snippet", "") or "").strip()
            if snippet:
                return snippet[:500]
        metric = numeric_extraction.get("metric_name", "")
        value = numeric_extraction.get("value", "")
        unit = numeric_extraction.get("unit", "")
        if value not in {"", None}:
            return f"{metric}: {value} {unit}".strip()
        return ""

    def _quote_too_weak(self, quote) -> bool:
        text = str(quote or "").strip()
        if len(text) < 24:
            return True
        generic = {"trong năm 2024", "có", "có thực hiện", "có đề cập", "năm 2024"}
        return text.lower() in generic

    def _supporting_quote_from_section(
        self,
        source_sections: list[dict],
        evidence_source_id,
        reason: str = "",
        question: str = "",
    ) -> str:
        matched = self._sections_for_source_id(source_sections, evidence_source_id)
        if not matched:
            return ""
        content = str(matched[0].get("content", "") or "")
        if not content.strip():
            return ""

        signal_text = f"{reason} {question}".lower()
        raw_lines = []
        for line in content.splitlines():
            line = " ".join(line.split())
            if 35 <= len(line) <= 360:
                raw_lines.append(line)
        if not raw_lines:
            compact = " ".join(content.split())
            return compact[:320]

        priority_terms = [
            "cộng đồng",
            "thiện nguyện",
            "đào tạo",
            "chính sách",
            "doanh thu",
            "năng lượng",
            "nước",
            "phát thải",
            "chất thải",
            "hội đồng quản trị",
            "thù lao",
            "nhân viên",
            "người lao động",
        ]
        active_terms = [term for term in priority_terms if term in signal_text]
        if active_terms:
            for line in raw_lines:
                lowered = line.lower()
                if any(term in lowered for term in active_terms):
                    return line[:320]
        return raw_lines[0][:320]

    def _sections_for_source_id(self, source_sections: list[dict], evidence_source_id) -> list[dict]:
        if not evidence_source_id:
            return []
        source_id = str(evidence_source_id).strip()
        return [section for section in source_sections if str(section.get("source_id", "")).strip() == source_id]

    def _normalize_option_evidence(self, value) -> dict:
        if isinstance(value, list):
            converted = {}
            for item in value:
                if not isinstance(item, dict):
                    continue
                option = item.get("option") or item.get("letter")
                if option:
                    converted[str(option).strip().upper()] = item
            value = converted
        if not isinstance(value, dict):
            return {}
        normalized = {}
        for key, item in value.items():
            letter = str(key).strip().upper()
            if not letter:
                continue
            if isinstance(item, dict):
                source_id = item.get("source_id") or item.get("evidence_source_id")
                quote = item.get("quote") or item.get("evidence_quote") or item.get("evidence")
            else:
                source_id = None
                quote = item
            if source_id or quote:
                normalized[letter] = {
                    "source_id": str(source_id).strip() if source_id else None,
                    "quote": str(quote).strip() if quote else None,
                }
        return normalized

    def _is_llm_parse_fallback(self, llm_result: dict) -> bool:
        reason = str(llm_result.get("reason", "") or "").lower()
        return "fallback" in reason and (
            "json_parse" in reason
            or "llm_error" in reason
            or "no_answer" in reason
        )

    def _fallback_option_evidence_from_retrieval(
        self,
        rule: dict,
        source_sections: list[dict],
        query_plan: dict,
        allowed_options: set[str] | None = None,
    ) -> dict:
        option_focus = query_plan.get("option_focus") if isinstance(query_plan.get("option_focus"), dict) else {}
        if not option_focus:
            return {}

        fallback = {}
        for option in sorted(option_focus):
            if allowed_options is not None and option not in allowed_options:
                continue
            best_section = None
            best_quote = ""
            best_score = -1.0
            for section in source_sections or []:
                if option not in (section.get("matched_options") or []):
                    continue
                hit_terms = section.get("option_hit_terms") or []
                if not hit_terms:
                    continue
                quote = self._supporting_quote_for_option(section, option_focus.get(option, []), hit_terms)
                if not quote:
                    continue
                try:
                    score = float(section.get("rerank_score", section.get("score", 0.0)) or 0.0)
                except (TypeError, ValueError):
                    score = 0.0
                specificity_bonus = max(0.0, 8.0 - len(section.get("matched_options") or []))
                doc_bonus = {
                    "policy_document": 5.0,
                    "sustainability_report": 3.0,
                    "annual_report": 2.0,
                }.get(section.get("document_type"), 0.0)
                option_specific_bonus = 2.0 if len(hit_terms) == 1 else min(4.0, len(hit_terms))
                total = score + doc_bonus + option_specific_bonus + specificity_bonus
                if total > best_score:
                    best_section = section
                    best_quote = quote
                    best_score = total
            if best_section and best_quote:
                fallback[option] = {
                    "source_id": best_section.get("source_id"),
                    "quote": best_quote,
                }
        return fallback

    def _supporting_quote_for_option(self, section: dict, focus_terms, hit_terms) -> str:
        content = str(section.get("content", "") or "")
        if not content.strip():
            return ""
        terms = []
        for value in list(hit_terms or []) + self._coerce_option_terms(focus_terms):
            normalized = self._normalize_for_match(value)
            if normalized and normalized not in {"chinh sach", "moi truong", "cong ty"}:
                terms.append(normalized)
        terms = list(dict.fromkeys(terms))

        lines = []
        for line in content.splitlines():
            compact = " ".join(line.split())
            if 20 <= len(compact) <= 1200:
                lines.append(compact)
        if not lines:
            compact = " ".join(content.split())
            lines = [compact[:1200]] if compact else []

        best_line = ""
        best_hits = 0
        for line in lines:
            normalized_line = self._normalize_for_match(line)
            hits = sum(1 for term in terms if self._term_matches(normalized_line, term))
            if hits > best_hits:
                best_line = line
                best_hits = hits
        if best_line:
            return self._trim_quote_around_terms(best_line, terms)
        return ""

    def _trim_quote_around_terms(self, line: str, normalized_terms: list[str], limit: int = 360) -> str:
        if len(line) <= limit:
            return line
        normalized_line = self._normalize_for_match(line)
        best_pos = -1
        for term in normalized_terms:
            tokens = [
                token for token in term.split()
                if len(token) >= 4 and token not in {"chinh", "sach", "moi", "truong", "cong"}
            ]
            for token in tokens or [term]:
                pos = normalized_line.find(token)
                if pos >= 0:
                    best_pos = pos
                    break
            if best_pos >= 0:
                break
        if best_pos < 0:
            return line[:limit]
        start = max(0, best_pos - limit // 3)
        end = min(len(line), start + limit)
        return line[start:end].strip()

    def _coerce_option_terms(self, value) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        if isinstance(value, (list, tuple, set)):
            return [str(item) for item in value if str(item).strip()]
        return []

    def _normalize_for_match(self, text) -> str:
        import unicodedata
        normalized = unicodedata.normalize("NFD", str(text or "").lower())
        normalized = "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")
        normalized = normalized.replace("đ", "d")
        import re
        normalized = re.sub(r"[^a-z0-9\s]", " ", normalized)
        return re.sub(r"\s+", " ", normalized).strip()

    def _term_matches(self, normalized_text: str, normalized_term: str) -> bool:
        if not normalized_term:
            return False
        if normalized_term in normalized_text:
            return True
        tokens = [
            token for token in normalized_term.split()
            if len(token) >= 4 and token not in {"chinh", "sach", "moi", "truong", "cong"}
        ]
        if not tokens:
            return False
        matches = sum(1 for token in tokens if token in normalized_text)
        required = 1 if len(tokens) == 1 else min(2, len(tokens))
        return matches >= required

    def _build_option_level_evidence(
        self,
        rule: dict,
        selected_options: list[str],
        option_evidence: dict,
        source_sections: list[dict],
        llm_confidence=None,
    ) -> dict:
        kept_options = []
        evidence_items = []
        verification_by_option = {}
        raw_quotes = []
        first_source_id = None

        for option in selected_options:
            evidence = option_evidence.get(option) or {}
            source_id = evidence.get("source_id")
            quote = evidence.get("quote")
            if source_id and self._quote_too_weak(quote):
                replacement = self._supporting_quote_from_section(
                    source_sections=source_sections,
                    evidence_source_id=source_id,
                    reason=rule.get("options", ""),
                    question=rule.get("question", ""),
                )
                if replacement:
                    quote = replacement
            if not quote:
                verification_by_option[option] = {"grounded": False, "match_type": "missing_quote"}
                continue

            sections = self._sections_for_source_id(source_sections, source_id) or source_sections
            verification = self.verifier.verify(str(quote), sections)
            if (
                verification
                and not verification["grounded"]
                and self._allow_source_id_support(rule, option, source_id, source_sections, quote)
            ):
                verification = dict(verification)
                verification["grounded"] = True
                verification["match_type"] = "source_id_context_support"
                verification["match_score"] = max(float(verification.get("match_score", 0.0) or 0.0), 0.55)
            verification_by_option[option] = verification
            if not verification.get("grounded"):
                continue

            kept_options.append(option)
            raw_quotes.append(f"{option}: {quote}")
            if not first_source_id:
                first_source_id = source_id
            items = self._build_evidence_items(
                raw_evidence=quote,
                source_sections=source_sections,
                evidence_source_id=source_id,
                llm_confidence=llm_confidence,
            )
            for item in items:
                item["option"] = option
                item["verification_status"] = "grounded"
                evidence_items.append(item)

        return {
            "selected_options": kept_options,
            "evidence_items": evidence_items,
            "option_evidence_verification": verification_by_option,
            "raw_evidence": "; ".join(raw_quotes) if raw_quotes else None,
            "evidence_source_id": first_source_id,
        }

    def _allow_source_id_support(
        self,
        rule: dict,
        answer: str,
        evidence_source_id,
        source_sections: list[dict],
        raw_evidence,
    ) -> bool:
        """Accept mention-level evidence when the LLM points at a real source block.

        VNSI scoring often asks whether something is disclosed/mentioned. In that
        case a slightly paraphrased or short quote should not erase a positive
        answer if the SOURCE_ID is valid and the answer itself is positive.
        """
        if not raw_evidence or str(raw_evidence).strip().lower() == "null":
            return False
        if not self._sections_for_source_id(source_sections, evidence_source_id):
            return False
        return self._score_for_answer(rule, answer) > 0

    def _score_for_answer(self, rule: dict, answer: str) -> float:
        import re

        logic = str(rule.get("logic", "") or "")
        if not logic or not answer:
            return 0.0
        if rule.get("is_multi_select"):
            return 1.0
        first_letter = str(answer).split(",")[0].strip().upper()
        for line in logic.splitlines():
            match = re.match(
                rf"\s*{re.escape(first_letter)}[\.\)]\s*([+-]?\d+(?:[.,]\d+)?)",
                line.strip(),
            )
            if match:
                return float(match.group(1).replace(",", "."))
        text = logic.lower()
        if first_letter == "A" and (
            "đề cập số liệu" in text
            or "có đề cập" in text
            or "công bố" in text
        ):
            return 1.0
        return 0.0

    def _build_evidence_items(
        self,
        raw_evidence,
        source_sections: list[dict],
        evidence_source_id=None,
        llm_confidence=None,
    ) -> list[dict]:
        if not raw_evidence or str(raw_evidence).strip().lower() == "null":
            return []

        evidence_text = str(raw_evidence).strip()
        items = []
        matched_sections = self._sections_for_source_id(source_sections, evidence_source_id)
        evidence_sections = matched_sections or source_sections[:3]
        try:
            llm_confidence_float = float(llm_confidence)
        except (TypeError, ValueError):
            llm_confidence_float = None

        for section in evidence_sections:
            estimated = self._estimate_confidence(section)
            confidence = estimated
            if llm_confidence_float is not None:
                confidence = round(max(0.05, min(0.98, (estimated + llm_confidence_float) / 2)), 3)
            items.append(
                {
                    "quote": evidence_text,
                    "source_id": section.get("source_id"),
                    "source_file": section.get("source_file"),
                    "source_path": section.get("source_path"),
                    "document_type": section.get("document_type"),
                    "page_start": section.get("page_start"),
                    "page_end": section.get("page_end"),
                    "retrieval_score": section.get("score"),
                    "llm_confidence": llm_confidence_float,
                    "confidence": confidence,
                }
            )

        if items:
            return items

        return [
            {
                "quote": evidence_text,
                "source_id": evidence_source_id,
                "source_file": None,
                "source_path": None,
                "document_type": None,
                "page_start": None,
                "page_end": None,
                "retrieval_score": None,
                "llm_confidence": llm_confidence_float,
                "confidence": 0.3,
            }
        ]

    def _estimate_confidence(self, section: dict) -> float:
        retrieval_score = float(section.get("score", 0.0) or 0.0)
        quality_score = float(section.get("quality_score", 0.0) or 0.0)
        confidence = 0.35 + min(0.35, retrieval_score / 20) + min(0.2, quality_score / 4)
        return round(min(0.95, confidence), 3)
