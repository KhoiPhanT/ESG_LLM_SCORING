"""
VNSI Scorer — Chấm điểm ESG bám sát workbook VNSI, ưu tiên bằng chứng.
LLM query planning + one-pass retrieval/scoring, no answer/query retry loops.
"""
import json
import os
import re

from core.cache import CacheManager
from core.evidence import EvidenceExtractor
from core.query_builder.question_query_builder import QuestionQueryBuilder
from core.query_builder.question_retrieval_metadata import QuestionRetrievalMetadataBuilder
from core.retrieval.retrieval_engine import RetrievalEngine
from core.resolution import AnswerResolver
from core.scoring.scoring_contract import ScoringContract
from core.scoring.scoring_engine import ScoringEngine
from core.reporting import format_short_source, format_source_list


class VNSIScorer:
    def __init__(
        self,
        rules_path="outputs/vnsi_rules.json",
        structure_path="outputs/scoring_structure.json",
        llm_client=None,
        corpus=None,
        industry_sector: str = "",
        target_year: int | None = None,
        retrieval_engine=None,
        metadata_path: str = "outputs/question_retrieval_metadata.json",
    ):
        self.llm_client = llm_client
        self.rules_path = rules_path
        self.structure_path = structure_path
        self.corpus = corpus
        self.industry_sector = industry_sector
        self.target_year = target_year or getattr(corpus, "target_year", None) or 2024
        self.scoring_rules = self._load_json(rules_path).get("scoring", [])
        self.scoring_structure = self._load_json(structure_path)
        if isinstance(self.scoring_structure, list):
            self.scoring_structure = {"factor_max_scores": {}}
        self.factor_max_scores = self.scoring_structure.get("factor_max_scores", {})
        self.derived_factor_max_scores = self._derive_factor_max_scores()
        if retrieval_engine is not None:
            self.retrieval_engine = retrieval_engine
        elif corpus:
            self.retrieval_engine = RetrievalEngine(corpus, industry_sector=industry_sector, target_year=self.target_year)
        else:
            self.retrieval_engine = None
        self.query_builder = QuestionQueryBuilder()
        if industry_sector:
            self.query_builder.normalizer.set_industry(industry_sector)
        self.metadata_path = metadata_path
        self.retrieval_metadata = self._load_retrieval_metadata(metadata_path)
        if not self.retrieval_metadata and self.scoring_rules:
            self.retrieval_metadata = QuestionRetrievalMetadataBuilder(
                target_year=self.target_year,
            ).build_all(self.scoring_rules).get("metadata", {})
        self.evidence_extractor = EvidenceExtractor(llm_client=llm_client, target_year=self.target_year)
        self.answer_resolver = AnswerResolver()
        self.scoring_engine = ScoringEngine()
        self.scoring_contract = ScoringContract(self.scoring_rules, self.factor_max_scores)

    def _load_json(self, path):
        if not os.path.exists(path):
            return {}
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _load_retrieval_metadata(self, path: str) -> dict:
        if not path or not os.path.exists(path):
            return {}
        try:
            data = self._load_json(path)
            return data.get("metadata", data if isinstance(data, dict) else {})
        except Exception:
            return {}

    def _get_relevant_keywords(self, rule):
        question = rule.get("question", "")
        qid = rule.get("id", "")
        factor = rule.get("factor", "")
        keywords = []
        lowered = question.lower()
        for token in [
            "môi trường",
            "phát thải",
            "năng lượng",
            "nước",
            "chất thải",
            "khí nhà kính",
            "đa dạng sinh học",
            "lao động",
            "an toàn",
            "sức khỏe",
            "đào tạo",
            "nhân sự",
            "cộng đồng",
            "khách hàng",
            "nhà cung cấp",
            "bình đẳng",
            "hđqt",
            "cổ đông",
            "kiểm toán",
            "quản trị",
            "minh bạch",
            "đhđcđ",
            "nghị quyết",
            "rủi ro",
            "tuân thủ",
            "assurance",
        ]:
            if token in lowered:
                keywords.append(token)

        if qid.startswith("G.") and "đhđcđ" not in keywords:
            keywords.extend(["đhđcđ", "nghị quyết", "hội đồng quản trị", "cổ đông"])
        if factor == "E3":
            keywords.extend(["đơn vị", "doanh thu", "số liệu"])
        return list(dict.fromkeys(keywords)) or ["phát triển bền vững"]

    def _build_context(self, rule, report_text):
        """Plan retrieval once, then pack a clean bounded context for scoring."""
        query_plan, query_plan_fallback = self._build_query_plan(rule)
        evidence_profile = self._evidence_profile(rule)
        query_plan["evidence_profile"] = evidence_profile
        context_limit = 22000

        if not self.retrieval_engine:
            context = report_text[:context_limit]
            return {
                "context": context,
                "sections": [],
                "context_char_limit": context_limit,
                "retrieval_meta": {
                    "query_plan": query_plan,
                    "query_plan_fallback": True,
                    "evidence_profile": evidence_profile,
                    "context_char_count": len(context),
                    "top_source_years": [],
                    "top_source_doc_types": [],
                    "total_candidates": 0,
                },
            }

        retrieval = self.retrieval_engine.retrieve_for_plan(rule, query_plan, top_k=10)
        sections = self._with_source_ids(retrieval.get("candidates", []))
        context = self._pack_context(
            sections,
            char_limit=context_limit,
            group_by_option=bool(rule.get("is_multi_select")),
            query_plan=query_plan,
        ) if sections else report_text[:context_limit]

        return {
            "context": context,
            "sections": sections,
            "context_char_limit": context_limit,
            "retrieval_meta": {
                "query_plan": query_plan,
                "query_plan_fallback": query_plan_fallback,
                "evidence_profile": evidence_profile,
                "context_char_count": len(context),
                "top_source_years": self._top_values(sections, "year_guess"),
                "top_source_doc_types": self._top_values(sections, "document_type"),
                "sub_query_count": retrieval.get("sub_query_count", 0),
                "total_candidates": retrieval.get("total_unique_candidates", len(sections)),
                "actual_top_k": retrieval.get("actual_top_k"),
                "max_queries_used": retrieval.get("max_queries_used"),
                "option_retrieval_coverage": retrieval.get("option_retrieval_coverage", {}),
            },
        }

    def _build_context_with_plan(self, rule, report_text, cached_plan: dict):
        """Build context using a cached query plan (skips LLM planning call)."""
        context_limit = 22000
        cached_plan = dict(cached_plan or {})
        evidence_profile = cached_plan.get("evidence_profile") or self._evidence_profile(rule)
        cached_plan["evidence_profile"] = evidence_profile

        if not self.retrieval_engine:
            context = report_text[:context_limit]
            return {
                "context": context,
                "sections": [],
                "context_char_limit": context_limit,
                "retrieval_meta": {
                    "query_plan": cached_plan,
                    "query_plan_fallback": False,
                    "evidence_profile": evidence_profile,
                    "context_char_count": len(context),
                    "top_source_years": [],
                    "top_source_doc_types": [],
                    "total_candidates": 0,
                },
            }

        retrieval = self.retrieval_engine.retrieve_for_plan(rule, cached_plan, top_k=10)
        sections = self._with_source_ids(retrieval.get("candidates", []))
        context = self._pack_context(
            sections,
            char_limit=context_limit,
            group_by_option=bool(rule.get("is_multi_select")),
            query_plan=cached_plan,
        ) if sections else report_text[:context_limit]

        return {
            "context": context,
            "sections": sections,
            "context_char_limit": context_limit,
            "retrieval_meta": {
                "query_plan": cached_plan,
                "query_plan_fallback": False,
                "evidence_profile": evidence_profile,
                "context_char_count": len(context),
                "top_source_years": self._top_values(sections, "year_guess"),
                "top_source_doc_types": self._top_values(sections, "document_type"),
                "sub_query_count": retrieval.get("sub_query_count", 0),
                "total_candidates": retrieval.get("total_unique_candidates", len(sections)),
                "actual_top_k": retrieval.get("actual_top_k"),
                "max_queries_used": retrieval.get("max_queries_used"),
                "option_retrieval_coverage": retrieval.get("option_retrieval_coverage", {}),
            },
        }

    def _build_query_plan(self, rule: dict) -> tuple[dict, bool]:
        manual_plan = self._manual_query_plan(rule)
        if self.llm_client:
            try:
                plan = self.llm_client.plan_retrieval_query(rule, target_year=self.target_year)
            except Exception as exc:
                print(f" [plan-fallback:{exc}]", end=" ", flush=True)
                plan = None
            if self._valid_query_plan(plan):
                merged = self._merge_query_plans(manual_plan, plan)
                merged["query_plan_source"] = "metadata_plus_llm" if manual_plan else "llm_only"
                return merged, False
        if manual_plan:
            fallback = dict(manual_plan)
            fallback["query_plan_source"] = "metadata_only"
            fallback["llm_query_plan_failed"] = bool(self.llm_client)
            # Metadata plans are curated deterministic query plans, not an
            # emergency fallback. Mark fallback only when no plan exists.
            return fallback, False
        fallback = self._fallback_query_plan(rule)
        fallback["query_plan_source"] = "fallback"
        fallback["llm_query_plan_failed"] = bool(self.llm_client)
        return fallback, True

    def _manual_query_plan(self, rule: dict) -> dict:
        qid = str(rule.get("id", "") or "")
        record = self.retrieval_metadata.get(qid)
        if not isinstance(record, dict):
            return {}
        plan = {
            "search_queries": self._coerce_list(record.get("search_queries")),
            "semantic_aliases": self._coerce_list(record.get("semantic_aliases")),
            "required_doc_types": self._coerce_list(record.get("required_doc_types")),
            "must_have_terms": self._coerce_list(record.get("must_have_terms")),
            "avoid_terms": self._coerce_list(record.get("avoid_terms")),
            "year_policy": record.get("year_policy") or self._planner_year_policy(rule.get("time_policy", "unspecified")),
            "evidence_shape": record.get("evidence_shape") or self._fallback_evidence_shape(rule),
            "evidence_profile": record.get("evidence_profile") or self._evidence_profile(rule),
            "option_focus": record.get("option_focus") if isinstance(record.get("option_focus"), dict) else {},
            "isolated_option_queries": record.get("isolated_option_queries") if isinstance(record.get("isolated_option_queries"), dict) else {},
            "option_evidence_requirements": record.get("option_evidence_requirements") if isinstance(record.get("option_evidence_requirements"), dict) else {},
            "option_polarity": record.get("option_polarity") if isinstance(record.get("option_polarity"), dict) else {},
            "metadata_strategy": record.get("strategy"),
            "metadata_applied": True,
            "manual_profile_id": qid,
            "negative_options": self._coerce_list(record.get("negative_options")),
        }
        if not plan["search_queries"]:
            return {}
        return plan

    def _merge_query_plans(self, manual: dict, llm_plan: dict) -> dict:
        if not manual:
            merged = dict(llm_plan or {})
            merged.setdefault("metadata_applied", False)
            return merged
        if not llm_plan:
            return dict(manual)
        merged = dict(llm_plan)
        for field, limit in [
            ("search_queries", 10),
            ("semantic_aliases", 18),
            ("must_have_terms", 18),
            ("avoid_terms", 12),
            ("required_doc_types", 6),
        ]:
            merged[field] = list(dict.fromkeys(
                self._coerce_list(manual.get(field)) + self._coerce_list(llm_plan.get(field))
            ))[:limit]
        for field in ["year_policy", "evidence_shape", "evidence_profile", "metadata_strategy", "manual_profile_id"]:
            if manual.get(field):
                merged[field] = manual[field]
        option_focus = {}
        for source in [llm_plan.get("option_focus"), manual.get("option_focus")]:
            if not isinstance(source, dict):
                continue
            for key, value in source.items():
                letter = str(key).strip().upper()
                option_focus[letter] = list(dict.fromkeys(
                    option_focus.get(letter, []) + self._coerce_list(value)
                ))[:12]
        merged["option_focus"] = option_focus
        for field in ["isolated_option_queries", "option_evidence_requirements", "option_polarity"]:
            combined = {}
            for source in [llm_plan.get(field), manual.get(field)]:
                if isinstance(source, dict):
                    for key, value in source.items():
                        combined[str(key).strip().upper()] = value
            merged[field] = combined
        merged["metadata_applied"] = True
        merged["negative_options"] = self._coerce_list(manual.get("negative_options"))
        return merged

    def _coerce_list(self, value) -> list:
        if value is None:
            return []
        if isinstance(value, str):
            values = [value]
        elif isinstance(value, (list, tuple, set)):
            values = list(value)
        else:
            return []
        return [str(item).strip() for item in values if str(item).strip()]

    def _valid_query_plan(self, plan) -> bool:
        required = {
            "search_queries",
            "semantic_aliases",
            "required_doc_types",
            "must_have_terms",
            "avoid_terms",
            "year_policy",
            "evidence_shape",
            "option_focus",
        }
        return isinstance(plan, dict) and required.issubset(plan.keys()) and bool(plan.get("search_queries"))

    def _fallback_query_plan(self, rule: dict) -> dict:
        query = self.query_builder.build(rule, corpus=self.corpus)
        question = str(rule.get("question", "") or "").strip()
        qid = str(rule.get("id", "") or "")
        qtype = str(rule.get("question_type", "default") or "default")
        time_policy = str(rule.get("time_policy", "unspecified") or "unspecified")
        evidence_profile = self._evidence_profile(rule)
        exact = list(query.exact_phrases)
        primary = list(query.primary_terms)
        secondary = list(query.secondary_terms)
        option_focus = self._option_focus_terms(rule)

        focused = " ".join((exact + primary)[:8])
        search_queries = [question]
        if focused:
            search_queries.append(focused)
        if qid.startswith("G."):
            search_queries.append(" ".join((exact + secondary + ["hội đồng quản trị", "báo cáo quản trị"])[:10]))
        if qtype in {"numeric_disclosure", "ratio_calculation"}:
            search_queries.append(" ".join((exact + primary + ["số liệu", "bảng", str(self.target_year)])[:10]))
        if evidence_profile == "ratio_with_revenue":
            search_queries.append(" ".join((exact + primary + ["doanh thu", "tổng doanh thu", str(self.target_year)])[:12]))
        if evidence_profile == "policy_public":
            search_queries.append("chính sách môi trường công khai phát triển bền vững")

        avoid_terms = []
        if time_policy != "future_target_allowed" and evidence_profile not in {"policy_public", "metric_disclosure", "ratio_with_revenue"}:
            avoid_terms.extend(["2050", "2060", "trung hòa carbon", "net zero"])

        required_doc_types = list(query.preferred_document_types)[:5]
        if evidence_profile == "policy_public":
            required_doc_types = ["policy_document", "sustainability_report", "annual_report", "financial_report"]
        elif evidence_profile == "ratio_with_revenue":
            required_doc_types = ["sustainability_report", "annual_report", "financial_report"]

        return {
            "search_queries": [item for item in list(dict.fromkeys(search_queries)) if item][:5],
            "semantic_aliases": list(dict.fromkeys(secondary + query.intent_terms))[:12],
            "required_doc_types": required_doc_types[:5],
            "must_have_terms": list(dict.fromkeys(exact + primary[:8]))[:14],
            "avoid_terms": avoid_terms,
            "year_policy": self._planner_year_policy(time_policy),
            "evidence_shape": self._fallback_evidence_shape(rule),
            "evidence_profile": evidence_profile,
            "option_focus": option_focus,
            "isolated_option_queries": self._fallback_isolated_option_queries(rule, option_focus),
            "option_evidence_requirements": {},
            "option_polarity": {letter: "positive" for letter in option_focus},
        }

    def _fallback_isolated_option_queries(self, rule: dict, option_focus: dict) -> dict:
        if not rule.get("is_multi_select") or not isinstance(option_focus, dict):
            return {}
        question = str(rule.get("question", "") or "").lower()
        core = "chính sách" if "chính sách" in question else "bằng chứng"
        isolated = {}
        for letter, terms in option_focus.items():
            values = self._coerce_list(terms)
            target = " ".join(values[:5]).strip()
            if target:
                isolated[str(letter).strip().upper()] = [f"{core} {target}", target]
        return isolated

    def _evidence_profile(self, rule: dict) -> str:
        question = self.query_builder.normalizer.normalize_for_search(str(rule.get("question", "") or ""))
        options = self.query_builder.normalizer.normalize_for_search(str(rule.get("options", "") or ""))
        qtype = str(rule.get("question_type", "default") or "default")
        text = f"{question} {options}"
        if qtype == "ratio_calculation" or "doanh thu" in text or "revenue" in text:
            if any(token in text for token in ["phat thai", "nuoc", "chat thai", "nang luong", "nguyen vat lieu", "cong dong"]):
                return "ratio_with_revenue"
            return "financial_metric"
        if qtype == "numeric_disclosure" or str(rule.get("sub_category", "")) == "Hiệu quả":
            return "metric_disclosure"
        if qtype == "policy" or "chinh sach" in text or "cong khai" in text or "cam ket" in text:
            return "policy_public"
        if str(rule.get("id", "")).startswith("G."):
            return "governance_narrative"
        return "narrative"

    def _planner_year_policy(self, time_policy: str) -> str:
        if time_policy in {"current_year_required", "historical_allowed", "latest_valid_allowed"}:
            return time_policy
        return "unspecified"

    def _fallback_evidence_shape(self, rule: dict) -> str:
        question = str(rule.get("question", "") or "").lower()
        qtype = rule.get("question_type", "default")
        qid = str(rule.get("id", "") or "")
        if qtype in {"numeric_disclosure", "ratio_calculation"}:
            return "metric_table"
        if "kiểm toán" in question or "báo cáo tài chính" in question:
            return "financial_table"
        if qid.startswith("G.") and any(token in question for token in ["đa dạng", "kinh nghiệm", "kiến thức", "học vấn"]):
            return "governance_profile"
        if qid.startswith("G.") and any(token in question for token in ["đhđcđ", "cổ đông", "nghị quyết", "biên bản"]):
            return "meeting_resolution"
        if qtype == "policy":
            return "policy_text"
        if any(token in question for token in ["chứng nhận", "iso", "sa8000"]):
            return "certificate"
        return "mixed"

    def _option_focus_terms(self, rule: dict) -> dict:
        options = str(rule.get("options", "") or "")
        focus = {}
        stopwords = getattr(self.query_builder, "OPTION_STOPWORDS", set())
        for _, letter, text in re.findall(r"(^|\n)\s*([A-Z])[\.\)]\s*([^\n]+)", options):
            words = re.findall(r"[A-Za-zÀ-ỹà-ỹĐđ0-9]{3,}", text)
            cleaned = []
            normalized_text = self.query_builder.normalizer.normalize_for_search(text)
            if "cong khai" in normalized_text:
                cleaned.extend(["công khai", "minh bạch", "công bố thông tin"])
            if "doanh thu" in normalized_text:
                cleaned.extend(["doanh thu", "tổng doanh thu", "doanh thu hợp nhất"])
            for word in words:
                normalized = self.query_builder.normalizer.normalize_for_search(word)
                if normalized in stopwords or len(normalized) < 4:
                    continue
                cleaned.append(word)
            focus[letter] = list(dict.fromkeys(cleaned[:10]))
        return {key: value for key, value in focus.items() if key}

    def _with_source_ids(self, sections: list[dict]) -> list[dict]:
        enriched = []
        for index, section in enumerate(sections, start=1):
            item = dict(section)
            item["source_id"] = f"S{index}"
            enriched.append(item)
        return enriched

    def _pack_context(
        self,
        sections: list[dict],
        char_limit: int = 22000,
        group_by_option: bool = False,
        query_plan: dict | None = None,
    ) -> str:
        if group_by_option:
            option_focus = query_plan.get("option_focus") if isinstance(query_plan, dict) else {}
            if isinstance(option_focus, dict) and option_focus:
                return self._pack_option_context(sections, option_focus, char_limit)

        blocks = []
        total = 0
        current_group = None
        for section in sections:
            if group_by_option:
                options = section.get("matched_options") or []
                group = ",".join(options) if options else "GENERAL"
                if group != current_group:
                    header = f"[OPTION_CANDIDATES: {group}]"
                    remaining = char_limit - total
                    if remaining <= len(header) + 5:
                        break
                    blocks.append(header)
                    total += len(header) + 5
                    current_group = group
            block = self._format_section(section)
            remaining = char_limit - total
            if remaining <= 0:
                break
            if len(block) > remaining:
                if not blocks:
                    blocks.append(block[:remaining])
                break
            blocks.append(block)
            total += len(block) + 5
        return "\n---\n".join(blocks)

    def _pack_option_context(self, sections: list[dict], option_focus: dict, char_limit: int) -> str:
        blocks = []
        total = 0
        used_general = set()

        def add_block(block: str) -> bool:
            nonlocal total
            remaining = char_limit - total
            if remaining <= 0:
                return False
            if len(block) > remaining:
                return False
            blocks.append(block)
            total += len(block) + 5
            return True

        for option in sorted(str(key).strip().upper() for key in option_focus if str(key).strip()):
            option_sections = [
                section for section in sections
                if option in (section.get("matched_options") or [])
            ][:2]
            if not option_sections:
                continue
            if not add_block(f"[OPTION_CANDIDATES: {option}]"):
                break
            for section in option_sections:
                item = dict(section)
                item["matched_options"] = [option]
                item["option_hit_terms"] = self._context_option_terms(item, option_focus.get(option, []))
                source_key = item.get("chunk_id") or (
                    item.get("source_file"),
                    item.get("page_start"),
                    item.get("page_end"),
                )
                used_general.add(source_key)
                if not add_block(self._format_section(item)):
                    return "\n---\n".join(blocks)

        general_count = 0
        for section in sections:
            source_key = section.get("chunk_id") or (
                section.get("source_file"),
                section.get("page_start"),
                section.get("page_end"),
            )
            if source_key in used_general:
                continue
            if general_count == 0:
                if not add_block("[OPTION_CANDIDATES: GENERAL]"):
                    break
            if not add_block(self._format_section(section)):
                break
            general_count += 1
            if general_count >= 5:
                break

        return "\n---\n".join(blocks)

    def _context_option_terms(self, section: dict, focus_terms) -> list[str]:
        content = self.query_builder.normalizer.normalize_for_search(section.get("content", ""))
        terms = []
        raw_terms = focus_terms if isinstance(focus_terms, (list, tuple, set)) else [focus_terms]
        for term in raw_terms:
            text = str(term or "").strip()
            normalized = self.query_builder.normalizer.normalize_for_search(text)
            if normalized in {"chinh sach", "moi truong", "cong ty", "bang chung"}:
                continue
            if normalized and normalized in content:
                terms.append(text)
        return list(dict.fromkeys(terms))

    def _top_values(self, sections: list[dict], field: str) -> list:
        values = []
        for section in sections[:10]:
            value = section.get(field)
            if value is not None and value not in values:
                values.append(value)
        return values

    def _prerequisite_satisfied(self, rule, answer_registry):
        prerequisite = rule.get("prerequisite")
        if not prerequisite:
            return True, ""

        previous = answer_registry.get(prerequisite["question_id"]) or answer_registry.get(
            self._alternate_question_id(prerequisite["question_id"])
        )
        if not previous:
            return False, f"Thiếu kết quả câu điều kiện {prerequisite['question_id']}"

        if previous.get("answer") in prerequisite.get("allowed_answers", []):
            return True, ""
        return False, (
            f"Bỏ qua vì câu điều kiện {prerequisite['question_id']} trả lời "
            f"{previous.get('answer')} thay vì {','.join(prerequisite['allowed_answers'])}"
        )

    def _canonical_question_id(self, question_id):
        qid = str(question_id).upper().strip().replace("..", ".")
        if re.match(r"^[A-Z]\d", qid):
            qid = f"{qid[0]}.{qid[1:]}"
        return qid

    def _alternate_question_id(self, question_id):
        qid = self._canonical_question_id(question_id)
        return qid.replace(".", "", 1) if "." in qid else qid

    def _register_answer(self, registry, question_id, answer, selected_options):
        canonical = self._canonical_question_id(question_id)
        payload = {"answer": answer, "selected_options": selected_options}
        registry[canonical] = payload
        registry[self._alternate_question_id(canonical)] = payload

    def _derive_factor_max_scores(self):
        factor_max = {}
        for rule in self.scoring_rules:
            factor = rule.get("factor", "")
            if not factor:
                continue
            factor_max[factor] = round(factor_max.get(factor, 0.0) + float(rule.get("max_score", 0.0) or 0.0), 4)
        return factor_max

    def score_all_questions(self, report_text, industry_sector="Financials", company_name="Company"):
        if not self.scoring_rules:
            print("  [WARN] Không có scoring rules")
            return self._empty_result(industry_sector)

        print(
            f"\n  [SCORING] Ngành: {industry_sector} | "
            f"Tổng số câu hỏi: {len(self.scoring_rules)} | "
            f"Điểm tổng = raw_percentage (không nhân trọng số ngành)"
        )

        factor_scores = {}
        all_details = []
        answer_registry = {}
        query_plan_cache = {}

        # Configurable thermal throttle
        thermal_cooldown = int(os.environ.get("ESG_THERMAL_COOLDOWN", "0"))
        thermal_interval = int(os.environ.get("ESG_THERMAL_INTERVAL", "8"))

        progress_file = f"outputs/cache/{company_name}_{self.target_year}_scoring_progress.json"
        progress_schema_version = 18
        progress_fingerprint = self._scoring_progress_fingerprint(company_name, industry_sector)
        progress_forced = CacheManager.is_forced("scoring") or os.environ.get("ESG_NO_RESUME_SCORING", "0") == "1"
        cache_manager = CacheManager(run_key=f"{company_name}:{self.target_year}:scoring")
        if progress_forced:
            cache_manager.record(
                "scoring_progress",
                "rebuilt",
                progress_schema_version,
                progress_fingerprint,
                path=progress_file,
                reason="forced_or_no_resume",
            )
            print("  [RESUME] Bỏ qua progress cache theo cấu hình.")
        if os.path.exists(progress_file) and not progress_forced:
            try:
                with open(progress_file, 'r', encoding='utf-8') as f:
                    progress = json.load(f)
                if (
                    progress.get("schema_version") == progress_schema_version
                    and progress.get("input_fingerprint") == progress_fingerprint
                ):
                    factor_scores = progress.get("factor_scores", {})
                    all_details = progress.get("all_details", [])
                    answer_registry = progress.get("answer_registry", {})
                    query_plan_cache = progress.get("query_plan_cache", {})
                    print(f"  [RESUME] Đã nạp lại kết quả của {len(all_details)} câu hỏi từ lần chạy trước.")
                    cache_manager.record(
                        "scoring_progress",
                        "hit",
                        progress_schema_version,
                        progress_fingerprint,
                        path=progress_file,
                    )
                else:
                    print("  [RESUME] Bỏ qua cache tiến độ cũ vì schema/input fingerprint đã đổi.")
                    cache_manager.record(
                        "scoring_progress",
                        "rebuilt",
                        progress_schema_version,
                        progress_fingerprint,
                        path=progress_file,
                        reason="schema_or_fingerprint_changed",
                    )
            except Exception as e:
                print(f"  [WARN] Không thể đọc file tiến độ: {e}")
                cache_manager.record(
                    "scoring_progress",
                    "failed",
                    progress_schema_version,
                    progress_fingerprint,
                    path=progress_file,
                    reason="cache_parse_error",
                    error=str(e),
                )

        questions_processed_in_this_run = 0

        for i, rule in enumerate(self.scoring_rules):
            q_id = rule["id"]
            factor = rule.get("factor", "")
            
            canonical_qid = self._canonical_question_id(q_id)
            if canonical_qid in answer_registry:
                # Tìm answer đã lưu để in ra cho đẹp
                saved_ans = answer_registry[canonical_qid].get("answer", "SKIP")
                print(f"    [{i+1}/{len(self.scoring_rules)}] {q_id}... → {saved_ans} (Đã chấm)")
                continue

            print(f"    [{i+1}/{len(self.scoring_rules)}] {q_id}...", end=" ", flush=True)

            satisfied, prerequisite_note = self._prerequisite_satisfied(rule, answer_registry)
            if not satisfied:
                print("→ SKIP (điều kiện không thỏa)")
                detail = {
                    "id": q_id,
                    "factor": factor,
                    "pillar": rule.get("pillar", ""),
                    "sub_category": rule.get("sub_category", ""),
                    "question": rule["question"][:140],
                    "answer": "SKIP",
                    "selected_options": [],
                    "base_score": 0.0,
                    "score": 0.0,
                    "reason": prerequisite_note,
                    "evidence": None,
                    "evidence_present": False,
                }
                self._register_answer(answer_registry, q_id, "SKIP", [])
                all_details.append(detail)
                factor_scores.setdefault(factor, 0.0)
                continue

            # ═══ Query plan -> retrieval -> one scoring pass ═══
            cached_plan = query_plan_cache.get(q_id)
            if cached_plan:
                context_bundle = self._build_context_with_plan(rule, report_text, cached_plan)
            else:
                context_bundle = self._build_context(rule, report_text)
                built_plan = context_bundle.get("retrieval_meta", {}).get("query_plan")
                if built_plan:
                    query_plan_cache[q_id] = built_plan
            extraction = self.evidence_extractor.extract(rule, context_bundle)
            resolution = self.answer_resolver.resolve(rule, extraction)
            score_result = self.scoring_engine.score_rule(rule, resolution)

            answer_letter = score_result["answer"]
            selected_options = score_result["selected_options"]
            base_score = score_result["base_score"]
            final_score = score_result["score"]
            evidence_present = score_result["evidence_present"]

            query_plan_fallback = context_bundle.get("retrieval_meta", {}).get("query_plan_fallback", False)
            status_tag = " [plan-fallback]" if query_plan_fallback else ""
            display_answer = ",".join(selected_options) if selected_options else answer_letter
            print(f"→ {display_answer} ({final_score:+.2f}){status_tag}")
            
            # Print reason and evidence for better live feedback
            reason = resolution.get("reason", "")
            if reason:
                print(f"      Lý do: {reason}")
            
            evidence = extraction.get("raw_evidence")
            if evidence and str(evidence).upper() != "NULL":
                clean_ev = str(evidence).replace('\n', ' ').strip()
                print(f"      Bằng chứng: {clean_ev}")

            factor_scores[factor] = factor_scores.get(factor, 0.0) + final_score
            self._register_answer(answer_registry, q_id, answer_letter, selected_options)
            retrieval_meta = context_bundle.get("retrieval_meta", {})
            evidence_items = resolution["evidence_items"]
            source_sections = context_bundle["sections"]
            evidence_source_ref = self._evidence_source_ref(
                extraction.get("evidence_source_id"),
                evidence_items,
                source_sections,
            )
            option_source_refs = self._option_source_refs(evidence_items)
            top_source_refs = format_source_list(source_sections, limit=3)
            all_details.append({
                "id": q_id,
                "factor": factor,
                "pillar": rule.get("pillar", ""),
                "sub_category": rule.get("sub_category", ""),
                "question_bucket": self._question_bucket(rule, retrieval_meta.get("query_plan") or {}),
                "question_type": rule.get("question_type", "default"),
                "time_policy": rule.get("time_policy", "unspecified"),
                "question": rule["question"][:140],
                "answer": answer_letter,
                "selected_options": selected_options,
                "base_score": base_score,
                "score": final_score,
                "max_score": rule.get("max_score", 0.0),
                "resolution_status": score_result["resolution_status"],
                "confidence": resolution["confidence"],
                "conflict_detected": resolution["conflict_detected"],
                "reason": resolution["reason"],
                "answer_origin": resolution.get("answer_origin"),
                "parse_status": resolution.get("parse_status"),
                "parse_error": resolution.get("parse_error"),
                "evidence": extraction.get("raw_evidence"),
                "evidence_source_id": extraction.get("evidence_source_id"),
                "llm_confidence": extraction.get("llm_confidence"),
                "evidence_items": evidence_items,
                "evidence_source_ref": evidence_source_ref,
                "option_source_refs": option_source_refs,
                "option_evidence": extraction.get("option_evidence", {}),
                "option_evidence_verification": extraction.get("option_evidence_verification", {}),
                "option_retrieval_coverage": retrieval_meta.get("option_retrieval_coverage", {}),
                "top_source_refs": top_source_refs,
                "evidence_present": evidence_present,
                "evidence_verification": extraction.get("evidence_verification"),
                "retry_used": resolution.get("retry_used", False),
                "retry_attempts": resolution.get("retry_attempts", 0),
                "retry_profiles": resolution.get("retry_profiles", []),
                "llm_call_info": extraction.get("llm_call_info", {}),
                "query_plan": retrieval_meta.get("query_plan"),
                "query_plan_fallback": retrieval_meta.get("query_plan_fallback", False),
                "query_plan_source": (retrieval_meta.get("query_plan") or {}).get("query_plan_source"),
                "metadata_applied": bool((retrieval_meta.get("query_plan") or {}).get("metadata_applied")),
                "metadata_strategy": (retrieval_meta.get("query_plan") or {}).get("metadata_strategy"),
                "metadata_search_queries": (retrieval_meta.get("query_plan") or {}).get("search_queries", []),
                "actual_top_k": retrieval_meta.get("actual_top_k"),
                "max_queries_used": retrieval_meta.get("max_queries_used"),
                "context_char_count": retrieval_meta.get("context_char_count", 0),
                "top_source_years": retrieval_meta.get("top_source_years", []),
                "top_source_doc_types": retrieval_meta.get("top_source_doc_types", []),
                "retrieval_meta": retrieval_meta,
                "numeric_extraction": extraction.get("numeric_extraction"),
                "source_sections": [
                    {
                        "source_id": section.get("source_id"),
                        "source_file": section.get("source_file"),
                        "document_type": section.get("document_type"),
                        "year_guess": section.get("year_guess"),
                        "coverage_source": section.get("coverage_source"),
                        "page_start": section.get("page_start"),
                        "page_end": section.get("page_end"),
                        "rerank_score": section.get("rerank_score"),
                        "rerank_reasons": section.get("rerank_reasons", []),
                        "matched_keywords": section.get("matched_keywords", []),
                        "matched_terms": section.get("matched_terms", []),
                        "matched_options": section.get("matched_options", []),
                        "option_hit_terms": section.get("option_hit_terms", []),
                        "source_ref": format_short_source(section),
                    }
                    for section in source_sections
                ],
            })

            # Save progress after each question (atomic write to prevent corruption)
            try:
                tmp_progress_file = progress_file + ".tmp"
                with open(tmp_progress_file, 'w', encoding='utf-8') as f:
                    json.dump({
                        "schema_version": progress_schema_version,
                        "input_fingerprint": progress_fingerprint,
                        "factor_scores": factor_scores,
                        "all_details": all_details,
                        "answer_registry": answer_registry,
                        "query_plan_cache": query_plan_cache,
                    }, f, ensure_ascii=False, indent=2)
                os.replace(tmp_progress_file, progress_file)
            except Exception:
                pass
            
            questions_processed_in_this_run += 1
            if thermal_cooldown > 0 and thermal_interval > 0 and questions_processed_in_this_run % thermal_interval == 0:
                print(f"\n  [TẢN NHIỆT] Đã chạy xong {questions_processed_in_this_run} câu. Máy sẽ nghỉ {thermal_cooldown // 60} phút để hạ nhiệt...")
                print(f"  (Bạn có thể nhấn Ctrl+C để dừng hẳn, tiến trình đã được tự động lưu)")
                import time
                time.sleep(thermal_cooldown)
                print("  [TẢN NHIỆT] Đã nghỉ xong, tiếp tục phân tích!\n")

        summary = self.scoring_contract.summarize(all_details)
        e_score = summary["pillar_scores"]["E"]["percentage"]
        s_score = summary["pillar_scores"]["S"]["percentage"]
        g_score = summary["pillar_scores"]["G"]["percentage"]
        total = summary["raw_total"]

        return {
            "E_score": e_score,
            "S_score": s_score,
            "G_score": g_score,
            "total_score": total,
            "raw_total": total,
            "raw_max": summary["raw_max"],
            "raw_percentage": summary["raw_percentage"],
            "percentage": summary["percentage"],
            "score_100": min(100.0, max(0.0, summary["score_100"])),
            "pillar_scores": summary["pillar_scores"],
            "factor_scores": summary["factor_scores"],
            "factor_max_mismatches": summary["factor_max_mismatches"],
            "diagnostics": self._build_diagnostics(all_details),
            "details": all_details,
        }

    def _scoring_progress_fingerprint(self, company_name: str, industry_sector: str) -> str:
        metadata_payload = CacheManager.load_json(self.metadata_path) or {}
        source_docs = []
        if self.corpus is not None:
            for doc in getattr(self.corpus, "documents", []):
                source_docs.append({
                    "path": os.path.abspath(doc.path),
                    "file_hash": getattr(doc.metadata, "file_hash", ""),
                    "doc_type": doc.doc_type,
                    "year_guess": doc.metadata.year_guess,
                })
        retrieval_fp = ""
        if self.retrieval_engine is not None and hasattr(self.retrieval_engine, "corpus_window_fingerprint"):
            retrieval_fp = self.retrieval_engine.corpus_window_fingerprint()
        return CacheManager.hash_json({
            "schema_version": "scoring_progress_v18",
            "score_formula": "raw_percentage_only",
            "llm_resilience": "retry_ladder_with_retrieval_fallback",
            "company": company_name,
            "industry_sector": industry_sector,
            "target_year": self.target_year,
            "rules_path": os.path.abspath(self.rules_path),
            "rules_hash": CacheManager.hash_file(self.rules_path) if os.path.exists(self.rules_path) else "",
            "structure_hash": CacheManager.hash_file(self.structure_path) if os.path.exists(self.structure_path) else "",
            "metadata_hash": CacheManager.hash_json(metadata_payload),
            "source_documents": source_docs,
            "retrieval_window_fingerprint": retrieval_fp,
        })

    def _evidence_source_ref(
        self,
        evidence_source_id,
        evidence_items: list[dict],
        source_sections: list[dict],
    ) -> str:
        if evidence_source_id:
            source_id = str(evidence_source_id).strip()
            for section in source_sections or []:
                if str(section.get("source_id", "")).strip() == source_id:
                    return format_short_source(section)
        refs = format_source_list(evidence_items, limit=1)
        if refs:
            return refs[0]
        refs = format_source_list(source_sections, limit=1)
        return refs[0] if refs else ""

    def _option_source_refs(self, evidence_items: list[dict]) -> dict:
        refs = {}
        for item in evidence_items or []:
            option = str(item.get("option", "") or "").strip().upper()
            if not option:
                continue
            ref = format_short_source(item)
            if not ref:
                continue
            refs.setdefault(option, [])
            if ref not in refs[option]:
                refs[option].append(ref)
        return refs

    def apply_screening_penalties(self, scores, screening_results):
        if screening_results.get("G_killed"):
            print("  ⚠ TOTAL KILL: Điểm G bị đưa về 0!")
            scores["G_score"] = 0
            if "pillar_scores" in scores:
                self._zero_pillar_score(scores["pillar_scores"], "G")
        if screening_results.get("ES_killed"):
            print("  ⚠ TOTAL KILL: Điểm E/S bị đưa về 0!")
            scores["E_score"] = 0
            scores["S_score"] = 0
            if "pillar_scores" in scores:
                self._zero_pillar_score(scores["pillar_scores"], "E")
                self._zero_pillar_score(scores["pillar_scores"], "S")

        deductions = screening_results.get("direct_deductions", 0)
        pillar_scores = scores.get("pillar_scores", {})
        raw_total = sum(float(pillar_scores.get(p, {}).get("raw_score", 0.0) or 0.0) for p in ["E", "S", "G"])
        scores["total_score"] = max(0.0, round(raw_total - deductions, 4))
        scores["raw_total"] = scores["total_score"]
        raw_max = float(scores.get("raw_max", 0.0) or 0.0)
        scores["raw_percentage"] = round(scores["total_score"] / raw_max * 100, 2) if raw_max > 0 else 0.0
        # score_100 = raw_percentage sau khi total_score đã áp dụng screening penalty.
        scores["percentage"] = round(min(100.0, max(0.0, scores["raw_percentage"])), 2)
        scores["score_100"] = scores["percentage"]
        return scores

    def _zero_pillar_score(self, pillar_scores: dict, pillar: str):
        data = pillar_scores.setdefault(pillar, {})
        data["raw_score"] = 0.0
        data["raw_percentage"] = 0.0
        data["percentage"] = 0.0

    def _empty_result(self, industry_sector):
        return {
            "E_score": 0,
            "S_score": 0,
            "G_score": 0,
            "total_score": 0,
            "raw_total": 0,
            "raw_max": 0,
            "raw_percentage": 0,
            "percentage": 0,
            "score_100": 0,
            "details": [],
            "diagnostics": self._build_diagnostics([]),
            "factor_scores": {},
        }

    def _question_bucket(self, rule: dict, query_plan: dict) -> str:
        if rule.get("question_type") == "ratio_calculation":
            return "ratio_calculation"
        if rule.get("question_type") == "numeric_disclosure":
            return "numeric_disclosure"
        if rule.get("is_multi_select"):
            return "multi_select"
        negative_options = set((query_plan or {}).get("negative_options") or [])
        if negative_options:
            return "single_select_negative"
        if str(rule.get("time_policy", "")) == "current_year_required":
            return "current_year_required"
        if str(rule.get("time_policy", "")) in {"historical_allowed", "latest_valid_allowed"}:
            return "historical_allowed"
        return "single_select_positive"

    def _build_diagnostics(self, details: list[dict]) -> dict:
        from collections import Counter

        counters = {
            "question_bucket": Counter(),
            "answer_origin": Counter(),
            "parse_status": Counter(),
            "resolution_status": Counter(),
            "failure_reason": Counter(),
        }
        zero_score_positive = 0
        retrieval_weak = 0
        risky_questions = []
        for detail in details or []:
            counters["question_bucket"][detail.get("question_bucket") or "unknown"] += 1
            counters["answer_origin"][detail.get("answer_origin") or "unknown"] += 1
            counters["parse_status"][detail.get("parse_status") or "unknown"] += 1
            counters["resolution_status"][detail.get("resolution_status") or "unknown"] += 1
            failure_reason = self._diagnostic_failure_reason(detail)
            counters["failure_reason"][failure_reason] += 1
            if (
                detail.get("answer") not in {"NULL", "SKIP", "", None}
                and float(detail.get("score", 0.0) or 0.0) <= 0.0
            ):
                zero_score_positive += 1
            if self._is_retrieval_weak(detail):
                retrieval_weak += 1
            if failure_reason not in {"ok"}:
                risky_questions.append({
                    "id": detail.get("id"),
                    "bucket": detail.get("question_bucket"),
                    "answer_origin": detail.get("answer_origin"),
                    "parse_status": detail.get("parse_status"),
                    "failure_reason": failure_reason,
                    "score": detail.get("score"),
                    "reason": detail.get("reason", ""),
                })
        return {
            "counts": {name: dict(counter) for name, counter in counters.items()},
            "zero_score_positive_answers": zero_score_positive,
            "retrieval_weak_count": retrieval_weak,
            "risky_questions": risky_questions[:25],
        }

    def _diagnostic_failure_reason(self, detail: dict) -> str:
        if detail.get("parse_status") in {"empty_raw_response", "empty_after_think_strip", "llm_transport_error"}:
            return "llm_completion_failure"
        if detail.get("parse_status") in {"answer_regex_only", "repaired_json", "json_malformed"}:
            return "llm_json_failure"
        if detail.get("answer_origin") in {"retrieval_fallback_multi", "retrieval_fallback_single"}:
            return "retrieval_fallback_used"
        if detail.get("resolution_status") == "weakly_supported":
            return "weak_evidence"
        if detail.get("answer") == "NULL":
            return "null_or_insufficient"
        if (
            detail.get("answer") not in {"NULL", "SKIP", "", None}
            and float(detail.get("score", 0.0) or 0.0) <= 0.0
        ):
            return "positive_answer_without_grounded_evidence"
        return "ok"

    def _is_retrieval_weak(self, detail: dict) -> bool:
        sections = detail.get("source_sections") or []
        if not sections:
            return True
        try:
            best_score = max(float(section.get("rerank_score", section.get("score", 0.0)) or 0.0) for section in sections)
        except (TypeError, ValueError):
            best_score = 0.0
        return best_score < 20.0 and not detail.get("evidence_present")

    def _format_section(self, section):
        try:
            score = float(section.get("rerank_score", section.get("score", 0)) or 0.0)
        except (TypeError, ValueError):
            score = 0.0
        return (
            f"[SOURCE_ID: {section.get('source_id', 'S?')} | DOC: {section.get('source_file')} | "
            f"TYPE: {section.get('document_type')} | YEAR: {section.get('year_guess')} | "
            f"PAGES: {section.get('page_start')}-{section.get('page_end')} | "
            f"SCORE: {score:.2f} | OPTIONS: {','.join(section.get('matched_options', []) or [])} | "
            f"OPTION_TERMS: {', '.join(section.get('option_hit_terms', []) or [])}]\n"
            f"{section.get('content', '')}"
        )
