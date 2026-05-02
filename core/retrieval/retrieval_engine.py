"""
Hybrid retrieval engine with lexical + semantic search, document-aware ranking,
and provenance tracking. Uses real embeddings when available.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import re

from core.cache import CacheManager
from core.normalization.text_normalizer import TextNormalizer
from core.query_builder.question_query_builder import QuestionQueryBuilder, RetrievalQuery
from core.retrieval.reranker import RetrievalReranker
from core.retrieval.semantic_index import SemanticIndex


@dataclass
class RetrievalCandidate:
    chunk_id: str
    source_file: str
    source_path: str
    document_type: str
    section_title: str
    chunk_type: str
    table_family: str | None
    year_guess: int | None
    coverage_source: str | None
    page_start: int
    page_end: int
    content: str
    normalized_content: str
    score: float
    quality_score: float
    labels: list[str] = field(default_factory=list)
    exact_phrase_hits: list[str] = field(default_factory=list)
    matched_terms: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    semantic_score: float = 0.0
    rerank_score: float = 0.0
    rerank_reasons: list[str] = field(default_factory=list)
    low_value: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class IndexedWindow:
    chunk_id: str
    source_file: str
    source_path: str
    document_type: str
    section_title: str
    chunk_type: str
    table_family: str | None
    year_guess: int | None
    coverage_source: str | None
    page_start: int
    page_end: int
    content: str
    quality_score: float
    low_value: bool
    no_evidence_hint: bool
    evidence_signal: float
    labels: list[str]
    normalized_content: str
    normalized_title: str


class RetrievalEngine:
    GENERIC_PERFORMANCE_TERMS = {
        "chi tieu",
        "so lieu",
        "thong ke",
        "dinh luong",
        "ket qua",
        "hieu qua",
        "environmental",
        "ems",
        "moi truong",
        "xa hoi",
        "nguoi lao dong",
        "thu hoi",
        "tai su dung",
        "thu nhap",
        "nhan vien",
    }
    PLAN_STOPWORDS = {
        "cong",
        "ty",
        "khong",
        "co",
        "duoc",
        "lien",
        "quan",
        "cac",
        "tac",
        "dong",
        "toi",
        "den",
        "nhung",
        "bat",
        "ky",
        "hieu",
        "luc",
        "rong",
        "rai",
    }

    def __init__(self, corpus, industry_sector: str = "", target_year: int | None = None):
        self.corpus = corpus
        self.target_year = target_year or getattr(corpus, "target_year", None) or 2024
        self.normalizer = TextNormalizer(industry_sector=industry_sector)
        self.query_builder = QuestionQueryBuilder()
        # Pass industry to query builder's normalizer too
        if industry_sector:
            self.query_builder.normalizer.set_industry(industry_sector)
        self.reranker = RetrievalReranker(target_year=self.target_year)
        self._normalized_term_cache: dict[str, str] = {}
        self._windows: list[IndexedWindow] = []
        self._token_index: dict[str, set[int]] = {}
        self._doc_type_index: dict[str, set[int]] = {}
        self._chunk_type_index: dict[str, set[int]] = {}
        self._table_family_index: dict[str, set[int]] = {}
        self._build_index()

        # Build semantic index with embedding cache key based on corpus
        cache_key = self._corpus_cache_key()
        semantic_documents = [
            f"{window.normalized_title}\n{window.normalized_content}" for window in self._windows
        ]
        self.semantic_index = SemanticIndex(semantic_documents, cache_key=cache_key)

    def retrieve_for_rule(self, rule: dict, top_k: int = 10) -> dict:
        query = self.query_builder.build(rule, corpus=self.corpus)
        candidates = self._score_candidates(query, rule=rule)
        selected = self._select_candidates(candidates, rule=rule, top_k=top_k)
        return {
            "query": query.to_dict(),
            "candidates": [candidate.to_dict() for candidate in selected],
        }

    def retrieve_for_plan(self, rule: dict, query_plan: dict, top_k: int = 10) -> dict:
        """
        Retrieve once from an LLM-generated query plan.
        Each planned search query is executed once, then merged by rerank score.
        """
        plan = query_plan or {}
        rule_with_plan = dict(rule)
        rule_with_plan["retrieval_plan"] = plan
        plan_queries = self._plan_search_queries(rule, plan)
        max_queries = self._max_plan_queries(rule, plan)
        selected_plan_queries = plan_queries[:max_queries]
        actual_top_k = self._actual_plan_top_k(rule, plan, top_k, len(selected_plan_queries))
        all_candidates: dict[str, dict] = {}
        hit_counts: dict[str, int] = {}
        isolated_queries = self._isolated_option_queries(plan)
        option_coverage: dict[str, bool] = {}

        # One broad plan query gives aliases/options a chance even when each
        # individual search query is narrow.
        base_query = self._build_query_from_plan(rule, plan)
        base_candidates = self._select_candidates(
            self._score_candidates(base_query, rule=rule_with_plan),
            rule=rule_with_plan,
            top_k=max(min(actual_top_k, 10), top_k),
        )
        for candidate in base_candidates:
            self._merge_candidate(all_candidates, hit_counts, candidate.to_dict())

        if isolated_queries:
            option_focus = plan.get("option_focus") if isinstance(plan.get("option_focus"), dict) else {}
            per_option_top_k = max(2, min(4, actual_top_k // max(1, len(isolated_queries)) + 1))
            for letter, option_queries in isolated_queries.items():
                option_terms = self._coerce_plan_list(option_focus.get(letter, []))
                option_coverage[letter] = False
                for query_text in option_queries[:3]:
                    planned_query = self._build_query_from_plan(
                        rule,
                        plan,
                        query_text=query_text,
                        option_terms=option_terms,
                    )
                    candidates = self._select_candidates(
                        self._score_candidates(planned_query, rule=rule_with_plan),
                        rule=rule_with_plan,
                        top_k=per_option_top_k,
                    )
                    for candidate in candidates:
                        candidate_dict = candidate.to_dict()
                        candidate_dict["option_query"] = query_text
                        candidate_dict["option_hit_terms"] = self._option_hit_terms(candidate_dict, option_terms)
                        candidate_dict["matched_options"] = [letter] if candidate_dict["option_hit_terms"] else []
                        if candidate_dict["option_hit_terms"]:
                            option_coverage[letter] = True
                        self._merge_candidate(all_candidates, hit_counts, candidate_dict)
        else:
            for query_text in selected_plan_queries:
                planned_query = self._build_query_from_plan(rule, plan, query_text=query_text)
                candidates = self._select_candidates(
                    self._score_candidates(planned_query, rule=rule_with_plan),
                    rule=rule_with_plan,
                    top_k=max(4, min(8, actual_top_k)),
                )
                for candidate in candidates:
                    self._merge_candidate(all_candidates, hit_counts, candidate.to_dict())

        for cid, count in hit_counts.items():
            if count <= 1 or cid not in all_candidates:
                continue
            bonus = min(2.4, 0.55 * (count - 1))
            item = all_candidates[cid]
            item["score"] = round(float(item.get("score", 0.0) or 0.0) + bonus, 4)
            item["rerank_score"] = round(float(item.get("rerank_score", item.get("score", 0.0)) or 0.0) + bonus, 4)
            reasons = list(item.get("reasons", []))
            reasons.append(f"query_plan_hit_boost={count}hits")
            item["reasons"] = reasons

        merged = self._apply_plan_soft_filters(
            list(all_candidates.values()),
            plan=plan,
            top_k=actual_top_k,
        )
        ranked = sorted(
            merged,
            key=lambda item: (
                float(item.get("rerank_score", item.get("score", 0.0)) or 0.0),
                float(item.get("score", 0.0) or 0.0),
                float(item.get("quality_score", 0.0) or 0.0),
            ),
            reverse=True,
        )
        selected = self._select_by_profile(ranked, plan=plan, top_k=actual_top_k)

        isolated_count = sum(len(value[:3]) for value in isolated_queries.values()) if isolated_queries else 0
        return {
            "query": base_query.to_dict(),
            "query_plan": plan,
            "sub_query_count": isolated_count + 1 if isolated_queries else len(selected_plan_queries) + 1,
            "total_unique_candidates": len(all_candidates),
            "actual_top_k": actual_top_k,
            "max_queries_used": isolated_count if isolated_queries else len(selected_plan_queries),
            "option_retrieval_coverage": option_coverage,
            "candidates": selected,
        }

    def _max_plan_queries(self, rule: dict, plan: dict) -> int:
        profile = str(plan.get("evidence_profile") or "")
        strategy = str(plan.get("metadata_strategy") or plan.get("strategy") or "")
        if rule.get("is_multi_select") or strategy == "multi_option":
            return 8
        if rule.get("question_type") in {"numeric_disclosure", "ratio_calculation"} or profile in {"metric_disclosure", "ratio_with_revenue"}:
            return 5
        return 4

    def _actual_plan_top_k(self, rule: dict, plan: dict, requested_top_k: int, query_count: int) -> int:
        strategy = str(plan.get("metadata_strategy") or plan.get("strategy") or "")
        is_multi = bool(rule.get("is_multi_select") or strategy == "multi_option")
        cap = 16 if is_multi else 14
        desired = max(requested_top_k, max(1, query_count) * 2)
        return min(cap, desired)

    def _select_by_profile(self, candidates: list[dict], plan: dict, top_k: int) -> list[dict]:
        profile = str(plan.get("evidence_profile") or "")
        strategy = str(plan.get("metadata_strategy") or plan.get("strategy") or "")
        if not candidates:
            return []
        if strategy == "multi_option":
            return self._select_multi_option_candidates(candidates, plan=plan, top_k=top_k)
        if profile == "policy_public":
            return self._quota_select_dicts(
                candidates,
                top_k=top_k,
                priority=lambda item: item.get("document_type") == "policy_document"
                or self._candidate_contains_any(item, ["chinh sach", "cong khai", "cam ket"]),
                min_priority=min(3, top_k),
            )
        if profile == "metric_disclosure":
            return self._quota_select_dicts(
                candidates,
                top_k=top_k,
                priority=lambda item: item.get("chunk_type") in {"table_section", "metric_kv_section"} or bool(item.get("table_family")),
                min_priority=min(4, top_k),
            )
        if profile == "ratio_with_revenue":
            selected = []
            selected_keys = set()
            groups = [
                lambda item: self._is_ratio_metric_candidate(item, plan),
                self._is_revenue_candidate,
                lambda item: item.get("chunk_type") in {"table_section", "metric_kv_section"},
            ]
            for predicate in groups:
                for item in candidates:
                    key = (item.get("source_path"), item.get("page_start"), item.get("page_end"))
                    if key in selected_keys or not predicate(item):
                        continue
                    selected.append(item)
                    selected_keys.add(key)
                    break
            for item in candidates:
                if len(selected) >= top_k:
                    break
                key = (item.get("source_path"), item.get("page_start"), item.get("page_end"))
                if key not in selected_keys:
                    selected.append(item)
                    selected_keys.add(key)
            return selected[:top_k]
        return candidates[:top_k]

    def _select_multi_option_candidates(self, candidates: list[dict], plan: dict, top_k: int) -> list[dict]:
        selected = []
        selected_keys = set()
        option_focus = plan.get("option_focus") if isinstance(plan.get("option_focus"), dict) else {}
        for letter, terms in option_focus.items():
            best = None
            best_score = -1.0
            normalized_terms = [
                self.normalizer.normalize_for_search(str(term))
                for term in self._as_list(terms)
            ]
            normalized_terms = [
                term for term in normalized_terms
                if len(term) >= 4 and term not in {"chinh sach", "cong ty", "trong", "duoc", "moi truong"}
            ]
            if not normalized_terms:
                continue
            for item in candidates:
                key = (item.get("source_path"), item.get("page_start"), item.get("page_end"))
                if key in selected_keys:
                    continue
                text = self._candidate_text(item)
                hits = sum(1 for term in normalized_terms if self._term_matches(text, term))
                if hits <= 0:
                    continue
                isolated_bonus = 8.0 if str(letter).strip().upper() in (item.get("matched_options") or []) else 0.0
                doc_bonus = {
                    "policy_document": 4.0,
                    "sustainability_report": 2.0,
                    "annual_report": 1.5,
                    "financial_report": 0.3,
                }.get(item.get("document_type"), 0.0)
                policy_bonus = 1.0 if any(term in text for term in ["chinh sach", "quan ly moi truong", "iso 14001", "tuan thu phap luat"]) else 0.0
                score = hits * 10 + isolated_bonus + doc_bonus + policy_bonus + float(item.get("rerank_score", item.get("score", 0.0)) or 0.0) / 100.0
                if score > best_score:
                    best = item
                    best_score = score
            if best:
                key = (best.get("source_path"), best.get("page_start"), best.get("page_end"))
                selected.append(best)
                selected_keys.add(key)

        for item in candidates:
            if len(selected) >= top_k:
                break
            key = (item.get("source_path"), item.get("page_start"), item.get("page_end"))
            if key not in selected_keys:
                selected.append(item)
                selected_keys.add(key)
        return selected[:top_k]

    def _as_list(self, value) -> list:
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        if isinstance(value, (list, tuple, set)):
            return list(value)
        return []

    def _candidate_text(self, item: dict) -> str:
        return " ".join([
            str(item.get("normalized_content", "") or ""),
            self.normalizer.normalize_for_search(str(item.get("content", "") or "")[:3500]),
            self.normalizer.normalize_for_search(str(item.get("section_title", "") or "")),
        ])

    def _plan_text(self, plan: dict) -> str:
        parts = []
        for key in ("question_id", "strategy", "evidence_shape", "evidence_profile"):
            parts.append(str(plan.get(key, "") or ""))
        for key in ("search_queries", "semantic_aliases", "must_have_terms"):
            value = plan.get(key)
            if isinstance(value, list):
                parts.extend(str(x) for x in value)
        return self.normalizer.normalize_for_search(" ".join(parts))

    def _is_ratio_metric_candidate(self, item: dict, plan: dict) -> bool:
        text = self._candidate_text(item)
        plan_text = self._plan_text(plan)
        is_metric_like = (
            item.get("chunk_type") in {"table_section", "metric_kv_section"}
            or bool(item.get("table_family"))
        )

        if any(term in plan_text for term in ["nang luong", "kwh", "mj", "dien evn"]):
            return any(term in text for term in ["tong nang luong", "nang luong", "kwh", "mj", "dien evn"])
        if any(term in plan_text for term in ["nuoc", "m3"]):
            return "nuoc" in text and any(term in text for term in ["m3", "tong luong nuoc", "tai nguyen nuoc", "nuoc cap"])
        if any(term in plan_text for term in ["phat thai", "co2", "scope", "khi nha kinh"]):
            return any(term in text for term in ["phat thai", "co2", "scope 1", "scope 2", "scope 3", "khi nha kinh"])
        if any(term in plan_text for term in ["chat thai", "rac", "phe lieu"]):
            return any(term in text for term in ["chat thai", "rac thai", "phe lieu", "rac cong nghiep", "rac sinh hoat"])

        return item.get("table_family") in {"environmental_metrics", "workforce_metrics", "financial_metrics"}

    def _is_revenue_candidate(self, item: dict) -> bool:
        if item.get("table_family") == "financial_metrics":
            return True
        if item.get("document_type") not in {"annual_report", "financial_report"}:
            return False
        normalized = self._candidate_text(item)
        strong_revenue = any(term in normalized for term in [
            "doanh thu thuan",
            "doanh thu ban hang",
            "tong doanh thu hop nhat",
            "tong doanh thu nam",
            "vinamilk ghi nhan tong doanh thu",
            "revenue",
        ])
        has_money = any(term in normalized for term in ["ty dong", "trieu dong", "vnd", "dong"])
        if not (strong_revenue and has_money):
            return False
        if item.get("document_type") == "financial_report":
            return True
        if item.get("chunk_type") in {"table_section", "metric_kv_section"}:
            return True
        if item.get("document_type") == "annual_report" and int(item.get("page_start") or 9999) <= 10:
            return True
        return False

    def _quota_select_dicts(self, candidates: list[dict], top_k: int, priority, min_priority: int) -> list[dict]:
        selected = []
        selected_keys = set()
        priority_items = [item for item in candidates if priority(item)]
        regular_items = [item for item in candidates if not priority(item)]
        for item in priority_items[:min_priority]:
            key = (item.get("source_path"), item.get("page_start"), item.get("page_end"))
            if key in selected_keys:
                continue
            selected.append(item)
            selected_keys.add(key)
        for item in priority_items[min_priority:] + regular_items:
            if len(selected) >= top_k:
                break
            key = (item.get("source_path"), item.get("page_start"), item.get("page_end"))
            if key in selected_keys:
                continue
            selected.append(item)
            selected_keys.add(key)
        return selected[:top_k]

    def retrieve_multi_query(self, rule: dict, sub_queries: list[str], top_k: int = 10) -> dict:
        """
        Run retrieval for multiple sub-queries and merge results.
        Each sub-query retrieves independently, then results are merged
        with deduplication and score boosting for chunks hit by multiple queries.
        """
        all_candidates: dict[str, dict] = {}  # chunk_id -> candidate dict
        hit_counts: dict[str, int] = {}  # chunk_id -> number of queries that found it

        # Always run the primary rule-based retrieval
        primary = self.retrieve_for_rule(rule, top_k=top_k)
        for cand in primary["candidates"]:
            cid = cand["chunk_id"]
            all_candidates[cid] = cand
            hit_counts[cid] = hit_counts.get(cid, 0) + 1

        # Run additional retrievals for each sub-query
        for sub_query in sub_queries:
            if sub_query == rule.get("question", ""):
                continue  # Skip if same as original question

            sub_candidates = self._retrieve_by_text(sub_query, rule, top_k=max(5, top_k // 2))
            for cand in sub_candidates:
                cid = cand["chunk_id"]
                if cid in all_candidates:
                    # Boost score for chunks found by multiple queries
                    hit_counts[cid] = hit_counts.get(cid, 0) + 1
                    existing = all_candidates[cid]
                    existing["score"] = max(float(existing.get("score", 0)), float(cand.get("score", 0)))
                    existing["rerank_score"] = max(
                        float(existing.get("rerank_score", existing.get("score", 0)) or 0),
                        float(cand.get("rerank_score", cand.get("score", 0)) or 0),
                    )
                else:
                    all_candidates[cid] = cand
                    hit_counts[cid] = 1

        # Apply multi-query boost: chunks found by multiple queries get bonus
        for cid, count in hit_counts.items():
            if count > 1 and cid in all_candidates:
                bonus = min(2.0, 0.5 * (count - 1))
                all_candidates[cid]["score"] = round(
                    float(all_candidates[cid].get("score", 0)) + bonus, 4
                )
                all_candidates[cid]["rerank_score"] = round(
                    float(all_candidates[cid].get("rerank_score", all_candidates[cid].get("score", 0)) or 0) + bonus,
                    4,
                )
                reasons = all_candidates[cid].get("reasons", [])
                reasons.append(f"multi_query_boost={count}hits")
                all_candidates[cid]["reasons"] = reasons

        # Sort and select top_k. Use rerank_score so recency/doc-type/layout
        # corrections are preserved after multi-query merging.
        merged = sorted(
            all_candidates.values(),
            key=lambda x: (
                float(x.get("rerank_score", x.get("score", 0)) or 0),
                float(x.get("score", 0) or 0),
                float(x.get("quality_score", 0) or 0),
            ),
            reverse=True,
        )
        selected = merged[:top_k]

        return {
            "query": primary["query"],
            "sub_query_count": len(sub_queries) + 1,
            "total_unique_candidates": len(all_candidates),
            "candidates": selected,
        }

    def _retrieve_by_text(self, query_text: str, rule: dict, top_k: int = 5) -> list[dict]:
        """
        Retrieve candidates using free-form text as query.
        Kept for audits and legacy probes; the main scorer uses retrieve_for_plan().
        """
        # Build a minimal query from the text
        normalized = self.normalizer.normalize_for_search(query_text)
        tokens = [t for t in normalized.split() if len(t) >= 3]
        phrases = []
        primary = tokens[:8]
        secondary = tokens[8:15]

        # Also try to match phrase_library items
        for phrase in self.query_builder.phrase_library:
            phrase_norm = self.normalizer.normalize_for_search(phrase)
            if phrase_norm in normalized:
                phrases.append(phrase_norm)

        preferred_doc_types = []
        if self.corpus:
            preferred_doc_types = self.corpus.choose_preferred_doc_types(
                q_id=rule.get("id", ""),
                question=rule.get("question", query_text),
            ) or []

        query = RetrievalQuery(
            question_id=rule.get("id", ""),
            question_text=query_text,
            exact_phrases=phrases,
            primary_terms=primary,
            secondary_terms=secondary,
            intent_terms=[],
            preferred_document_types=preferred_doc_types or rule.get("preferred_document_types", []),
        )

        candidates = self._score_candidates(query, rule=rule)
        selected = self._select_candidates(candidates, rule=rule, top_k=top_k)
        return [c.to_dict() for c in selected]

    def _build_query_from_plan(
        self,
        rule: dict,
        plan: dict,
        query_text: str | None = None,
        option_terms: list[str] | None = None,
    ) -> RetrievalQuery:
        fallback = self.query_builder.build(rule, corpus=self.corpus)
        plan_queries = self._plan_search_queries(rule, plan)
        aliases = self._clean_plan_terms(self._list_from_plan(plan, "semantic_aliases"), keep_phrases=True)
        must_have = self._clean_plan_terms(self._list_from_plan(plan, "must_have_terms"), keep_phrases=True)
        raw_option_terms = []
        if option_terms is not None:
            raw_option_terms = self._coerce_plan_list(option_terms)
        else:
            strategy = str(plan.get("metadata_strategy") or plan.get("strategy") or "")
            option_focus = plan.get("option_focus") if isinstance(plan, dict) else {}
            # For multi-option subqueries, keep options isolated. The broad
            # base query may still use all option terms as recall hints.
            if isinstance(option_focus, dict) and not (query_text and strategy == "multi_option"):
                for terms in option_focus.values():
                    raw_option_terms.extend(self._coerce_plan_list(terms))
        option_terms_clean = self._clean_plan_terms(raw_option_terms, keep_phrases=True)

        text = query_text or " ".join(plan_queries[:3]) or rule.get("question", "")
        normalized = self.normalizer.normalize_for_search(text)
        tokens = [
            token for token in normalized.split()
            if len(token) >= 3 and token not in self.PLAN_STOPWORDS
        ]

        preferred_doc_types = self._list_from_plan(plan, "required_doc_types")
        if not preferred_doc_types:
            preferred_doc_types = fallback.preferred_document_types

        query_must_have = must_have
        if query_text:
            normalized_query_text = self.normalizer.normalize_for_search(query_text)
            if "doanh thu" in normalized_query_text or "revenue" in normalized_query_text:
                query_must_have = [
                    term for term in must_have
                    if any(token in self.normalizer.normalize_for_search(term) for token in ["doanh thu", "revenue"])
                ] or ["doanh thu", "doanh thu thuần", "tổng doanh thu"]
            elif any(token in normalized_query_text for token in ["nang luong", "nuoc", "phat thai", "chat thai", "kwh", "m3", "co2"]):
                query_must_have = [
                    term for term in must_have
                    if not any(token in self.normalizer.normalize_for_search(term) for token in ["doanh thu", "revenue"])
                ]

        exact_phrases = list(dict.fromkeys(query_must_have + aliases[:8] + fallback.exact_phrases[:6]))
        primary_terms = list(dict.fromkeys(tokens[:12] + query_must_have + fallback.primary_terms[:8]))
        secondary_terms = list(dict.fromkeys(aliases + option_terms_clean + fallback.secondary_terms[:12]))
        intent_terms = list(dict.fromkeys(option_terms_clean + aliases[:6] + fallback.intent_terms[:8]))

        return RetrievalQuery(
            question_id=rule.get("id", ""),
            question_text=text,
            exact_phrases=exact_phrases[:24],
            primary_terms=primary_terms[:28],
            secondary_terms=secondary_terms[:32],
            intent_terms=intent_terms[:24],
            preferred_document_types=preferred_doc_types[:6],
        )

    def _isolated_option_queries(self, plan: dict) -> dict[str, list[str]]:
        isolated = plan.get("isolated_option_queries") if isinstance(plan, dict) else {}
        if not isinstance(isolated, dict):
            return {}
        result = {}
        for key, value in isolated.items():
            letter = str(key).strip().upper()
            queries = self._coerce_plan_list(value)
            if letter and queries:
                result[letter] = queries[:4]
        return result

    def _option_hit_terms(self, candidate: dict, terms: list[str]) -> list[str]:
        text = self._candidate_text(candidate)
        hits = []
        for term in self._clean_plan_terms(self._coerce_plan_list(terms), keep_phrases=True):
            normalized = self.normalizer.normalize_for_search(str(term))
            if normalized in {"chinh sach", "cong ty", "moi truong", "bang chung"}:
                continue
            if normalized and self._term_matches(text, normalized):
                hits.append(term)
        return list(dict.fromkeys(hits))[:10]

    def _term_matches(self, normalized_text: str, normalized_term: str) -> bool:
        if not normalized_term:
            return False
        if normalized_term in normalized_text:
            return True
        tokens = [
            token for token in normalized_term.split()
            if len(token) >= 4
            and token not in self.PLAN_STOPWORDS
            and token not in {"chinh", "sach", "cong", "moi", "truong"}
        ]
        if not tokens:
            return False
        matches = sum(1 for token in tokens if token in normalized_text)
        required = 1 if len(tokens) == 1 else min(2, len(tokens))
        return matches >= required

    def _plan_search_queries(self, rule: dict, plan: dict) -> list[str]:
        queries = self._list_from_plan(plan, "search_queries")
        if queries:
            return queries
        return [str(rule.get("question", "") or "").strip()]

    def _list_from_plan(self, plan: dict, field: str) -> list[str]:
        if not isinstance(plan, dict):
            return []
        return self._coerce_plan_list(plan.get(field))

    def _coerce_plan_list(self, value) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            values = [value]
        elif isinstance(value, (list, tuple, set)):
            values = list(value)
        else:
            return []
        cleaned = []
        for item in values:
            text = str(item).strip()
            if text and text.lower() not in {"null", "none", "n/a"}:
                cleaned.append(text)
        return list(dict.fromkeys(cleaned))

    def _clean_plan_terms(self, terms: list[str], keep_phrases: bool = False) -> list[str]:
        cleaned = []
        for term in terms:
            normalized = self.normalizer.normalize_for_search(str(term))
            if not normalized:
                continue
            tokens = normalized.split()
            if len(tokens) == 1 and tokens[0] in self.PLAN_STOPWORDS:
                continue
            if len(tokens) == 2 and all(token in self.PLAN_STOPWORDS for token in tokens):
                continue
            if keep_phrases:
                cleaned.append(term)
            else:
                cleaned.extend(token for token in tokens if token not in self.PLAN_STOPWORDS)
        return list(dict.fromkeys(cleaned))

    def _merge_candidate(self, all_candidates: dict[str, dict], hit_counts: dict[str, int], candidate: dict) -> None:
        cid = candidate.get("chunk_id")
        if not cid:
            return
        if cid in all_candidates:
            hit_counts[cid] = hit_counts.get(cid, 0) + 1
            existing = all_candidates[cid]
            existing["score"] = max(float(existing.get("score", 0) or 0), float(candidate.get("score", 0) or 0))
            existing["rerank_score"] = max(
                float(existing.get("rerank_score", existing.get("score", 0)) or 0),
                float(candidate.get("rerank_score", candidate.get("score", 0)) or 0),
            )
            existing["matched_terms"] = list(dict.fromkeys(existing.get("matched_terms", []) + candidate.get("matched_terms", [])))
            existing["matched_options"] = list(dict.fromkeys(existing.get("matched_options", []) + candidate.get("matched_options", [])))
            existing["option_hit_terms"] = list(dict.fromkeys(existing.get("option_hit_terms", []) + candidate.get("option_hit_terms", [])))
            if not existing.get("option_query") and candidate.get("option_query"):
                existing["option_query"] = candidate.get("option_query")
            existing["exact_phrase_hits"] = list(dict.fromkeys(existing.get("exact_phrase_hits", []) + candidate.get("exact_phrase_hits", [])))
            existing["rerank_reasons"] = list(dict.fromkeys(existing.get("rerank_reasons", []) + candidate.get("rerank_reasons", [])))
            existing["reasons"] = list(dict.fromkeys(existing.get("reasons", []) + candidate.get("reasons", [])))
            return
        all_candidates[cid] = dict(candidate)
        hit_counts[cid] = 1

    def _apply_plan_soft_filters(self, candidates: list[dict], plan: dict, top_k: int) -> list[dict]:
        if not candidates:
            return []

        year_policy = str(plan.get("year_policy") or "")
        if year_policy == "current_year_required":
            current = [
                item for item in candidates
                if self._candidate_effective_year(item) == self.target_year
            ]
            non_old = [
                item for item in candidates
                if (self._candidate_effective_year(item) is None or self._candidate_effective_year(item) >= self.target_year)
            ]
            if current and len(non_old) >= min(2, top_k):
                candidates = non_old

        required_types = set(self._list_from_plan(plan, "required_doc_types"))
        if required_types:
            preferred = [item for item in candidates if item.get("document_type") in required_types]
            if len(preferred) >= min(3, top_k):
                candidates = preferred

        avoid_terms = [self.normalizer.normalize_for_search(term) for term in self._list_from_plan(plan, "avoid_terms")]
        avoid_terms = [term for term in avoid_terms if term]
        if avoid_terms:
            clean = [item for item in candidates if not self._candidate_contains_any(item, avoid_terms)]
            if len(clean) >= min(4, top_k):
                candidates = clean

        must_have = [self.normalizer.normalize_for_search(term) for term in self._list_from_plan(plan, "must_have_terms")]
        must_have = [term for term in must_have if term]
        if must_have:
            anchored = [item for item in candidates if self._candidate_contains_any(item, must_have)]
            if len(anchored) >= min(3, top_k):
                candidates = anchored + [item for item in candidates if item not in anchored]

        evidence_shape = str(plan.get("evidence_shape", "") or "")
        if evidence_shape in {"metric_table", "financial_table", "numeric_value"}:
            table_like = [
                item for item in candidates
                if item.get("chunk_type") in {"table_section", "metric_kv_section"} or item.get("table_family")
            ]
            if len(table_like) >= min(2, top_k):
                candidates = table_like + [item for item in candidates if item not in table_like]

        return candidates

    def _candidate_effective_year(self, candidate: dict) -> int | None:
        year_guess = candidate.get("year_guess")
        if year_guess:
            try:
                return int(year_guess)
            except (TypeError, ValueError):
                pass
        text = "\n".join([
            str(candidate.get("source_file", "") or ""),
            str(candidate.get("section_title", "") or ""),
            str(candidate.get("content", "") or "")[:1200],
        ])
        years = []
        for match in re.finditer(r"\b(20[0-4]\d)\b", text):
            try:
                year = int(match.group(1))
            except ValueError:
                continue
            if 2010 <= year <= 2049:
                years.append(year)
        if not years:
            return None
        if self.target_year in years:
            return self.target_year
        past_or_current = [year for year in years if year <= self.target_year]
        if past_or_current:
            return max(past_or_current)
        return min(years)

    def _candidate_contains_any(self, candidate: dict, normalized_terms: list[str]) -> bool:
        haystack = " ".join([
            str(candidate.get("section_title", "") or ""),
            str(candidate.get("normalized_content", "") or ""),
            self.normalizer.normalize_for_search(str(candidate.get("content", "") or "")[:2000]),
        ])
        terms = [self.normalizer.normalize_for_search(term) for term in normalized_terms]
        return any(term and term in haystack for term in terms)

    def _score_candidates(self, query, rule: dict) -> list[RetrievalCandidate]:
        candidates = []
        semantic_scores = self._semantic_scores(query, rule)
        for window_index in self._candidate_window_indexes(query, rule, semantic_scores=semantic_scores):
            candidate = self._score_window(
                self._windows[window_index],
                query,
                rule=rule,
                semantic_score=semantic_scores.get(window_index, 0.0),
            )
            if candidate:
                candidates.append(candidate)

        candidates.sort(key=lambda item: item.score, reverse=True)
        deduped = []
        seen = set()
        for item in candidates:
            key = (item.source_path, item.page_start, item.page_end)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(item)
        reranked = self.reranker.rerank(
            [item.to_dict() for item in deduped],
            query=query,
            rule=rule,
        )
        return [RetrievalCandidate(**item) for item in reranked]

    def _select_candidates(self, candidates: list[RetrievalCandidate], rule: dict, top_k: int) -> list[RetrievalCandidate]:
        if len(candidates) <= top_k:
            return candidates

        quota = self._selection_quota(rule, top_k=top_k)
        if quota <= 0:
            return candidates[:top_k]

        priority = [candidate for candidate in candidates if self._matches_priority_candidate(candidate, rule)]
        regular = [candidate for candidate in candidates if not self._matches_priority_candidate(candidate, rule)]

        selected = []
        selected_keys = set()
        priority_target = min(quota, len(priority))
        priority_taken = 0
        p_idx = 0
        r_idx = 0

        while len(selected) < top_k and (p_idx < len(priority) or r_idx < len(regular)):
            should_take_priority = (
                priority_taken < priority_target
                and p_idx < len(priority)
                and (
                    r_idx >= len(regular)
                    or len(selected) % 2 == 0
                    or len(selected) >= top_k - (priority_target - priority_taken)
                )
            )

            pool = priority if should_take_priority else regular
            index = p_idx if should_take_priority else r_idx

            if index >= len(pool):
                if should_take_priority:
                    p_idx += 1
                else:
                    r_idx += 1
                continue

            candidate = pool[index]
            key = (candidate.source_path, candidate.page_start, candidate.page_end)
            if should_take_priority:
                p_idx += 1
            else:
                r_idx += 1

            if key in selected_keys:
                continue

            selected.append(candidate)
            selected_keys.add(key)
            if should_take_priority:
                priority_taken += 1

        return selected[:top_k] if selected else candidates[:top_k]

    def _build_windows(self, doc, pages: list[dict], window_size: int = 2) -> list[dict]:
        structured_sections = self.corpus.get_document_sections(doc.path)
        if structured_sections:
            return [
                {
                    "chunk_id": section.get("chunk_id"),
                    "source_file": section["source_file"],
                    "source_path": section["source_path"],
                    "document_type": section["document_type"],
                    "section_title": section.get("section_title", ""),
                    "chunk_type": section.get("chunk_type", "section"),
                    "table_family": section.get("table_family"),
                    "year_guess": section.get("year_guess"),
                    "coverage_source": section.get("coverage_source"),
                    "page_start": section["page_start"],
                    "page_end": section["page_end"],
                    "content": section["content"],
                    "quality_score": section.get("quality_score", 0.0),
                    "low_value": section.get("is_low_value", self.normalizer.is_low_value_text(section["content"])),
                    "no_evidence_hint": section.get("is_no_evidence_hint", False),
                    "evidence_signal": section.get("evidence_signal", 0.0),
                    "labels": section.get("labels", []),
                }
                for section in structured_sections
                if section.get("content")
            ]

        windows = []
        useful_pages = [page for page in pages if page.get("text")]
        for idx in range(len(useful_pages)):
            start = idx
            end = min(len(useful_pages), idx + window_size)
            chunk_pages = useful_pages[start:end]
            if not chunk_pages:
                continue
            content = "\n\n".join(page.get("text", "") for page in chunk_pages if page.get("text")).strip()
            if not content:
                continue
            windows.append(
                {
                    "source_file": doc.label,
                    "chunk_id": f"{doc.label}:{chunk_pages[0]['page']}-{chunk_pages[-1]['page']}",
                    "source_path": doc.path,
                    "document_type": doc.doc_type,
                    "section_title": f"pages {chunk_pages[0]['page']}-{chunk_pages[-1]['page']}",
                    "chunk_type": "page_window",
                    "table_family": None,
                    "year_guess": doc.metadata.year_guess,
                    "coverage_source": "retrieval_page_window",
                    "page_start": chunk_pages[0]["page"],
                    "page_end": chunk_pages[-1]["page"],
                    "content": content[:7000],
                    "quality_score": self.corpus.section_quality(content),
                    "low_value": all(self.corpus.is_low_value_page(page) for page in chunk_pages),
                    "no_evidence_hint": False,
                    "evidence_signal": 0.0,
                    "labels": [],
                }
            )
        return windows

    def _build_index(self) -> None:
        for doc in self.corpus.documents:
            pages = self.corpus.get_document_pages(doc.path)
            for raw_window in self._build_windows(doc, pages):
                indexed = self._index_window(raw_window)
                if indexed is None:
                    continue
                window_index = len(self._windows)
                self._windows.append(indexed)
                self._register_window_tokens(window_index, indexed)

    def _index_window(self, window: dict) -> IndexedWindow | None:
        normalized_content = self.normalizer.normalize_for_search(window["content"])
        if not normalized_content:
            return None

        normalized_title = self.normalizer.normalize_for_search(window.get("section_title", ""))
        low_value = window["low_value"] or self.normalizer.is_low_value_text(window["content"])
        return IndexedWindow(
            source_file=window["source_file"],
            chunk_id=window["chunk_id"],
            source_path=window["source_path"],
            document_type=window["document_type"],
            section_title=window.get("section_title", ""),
            chunk_type=window.get("chunk_type", "section"),
            table_family=window.get("table_family"),
            year_guess=window.get("year_guess"),
            coverage_source=window.get("coverage_source"),
            page_start=window["page_start"],
            page_end=window["page_end"],
            content=window["content"],
            quality_score=float(window["quality_score"]),
            low_value=low_value,
            no_evidence_hint=bool(window.get("no_evidence_hint", False)),
            evidence_signal=float(window.get("evidence_signal", 0.0) or 0.0),
            labels=list(window.get("labels", [])),
            normalized_content=normalized_content,
            normalized_title=normalized_title,
        )

    def _register_window_tokens(self, window_index: int, window: IndexedWindow) -> None:
        self._doc_type_index.setdefault(window.document_type, set()).add(window_index)
        self._chunk_type_index.setdefault(window.chunk_type, set()).add(window_index)
        if window.table_family:
            self._table_family_index.setdefault(window.table_family, set()).add(window_index)

        token_source = f"{window.normalized_title} {window.normalized_content}"
        for token in self._extract_index_tokens(token_source):
            self._token_index.setdefault(token, set()).add(window_index)

    def _candidate_window_indexes(self, query, rule: dict, semantic_scores: dict[int, float] | None = None) -> list[int]:
        candidate_indexes = self._lexical_candidate_indexes(query, rule)
        if semantic_scores:
            candidate_indexes.update(semantic_scores.keys())

        if not candidate_indexes:
            candidate_indexes = self._fallback_candidate_indexes(query)

        return sorted(candidate_indexes)

    def _lexical_candidate_indexes(self, query, rule: dict) -> set[int]:
        candidate_indexes: set[int] = set()
        for term in (
            list(query.exact_phrases)
            + list(query.primary_terms)
            + list(query.secondary_terms)
            + list(getattr(query, "intent_terms", []))
        ):
            normalized_term = self._normalize_term(term)
            if not normalized_term:
                continue
            if normalized_term in self._token_index:
                candidate_indexes.update(self._token_index[normalized_term])
            for token in self._extract_index_tokens(normalized_term):
                candidate_indexes.update(self._token_index.get(token, set()))

            # Also search industry-specific expansions
            for expansion in self.normalizer.get_industry_expansions(term):
                exp_normalized = self._normalize_term(expansion)
                if exp_normalized and exp_normalized in self._token_index:
                    candidate_indexes.update(self._token_index[exp_normalized])

        if str(rule.get("sub_category", "") or "") == "Hiệu quả" or rule.get("question_type") in {"numeric_disclosure", "ratio_calculation"}:
            candidate_indexes.update(self._chunk_type_index.get("table_section", set()))
            factor = str(rule.get("factor", "") or "")
            preferred_families = {"financial_metrics", "general_table"}
            if factor.startswith("E"):
                preferred_families.add("environmental_metrics")
            if factor.startswith("S"):
                preferred_families.update({"workforce_metrics", "csr_impact_metrics"})
            for family in preferred_families:
                candidate_indexes.update(self._table_family_index.get(family, set()))

        if query.preferred_document_types and candidate_indexes:
            preferred_indexes = set()
            for doc_type in query.preferred_document_types:
                preferred_indexes.update(self._doc_type_index.get(doc_type, set()))
            narrowed = candidate_indexes & preferred_indexes
            if narrowed:
                candidate_indexes = narrowed
        return candidate_indexes

    def _fallback_candidate_indexes(self, query) -> set[int]:
        fallback_indexes: set[int] = set()
        if query.preferred_document_types:
            for doc_type in query.preferred_document_types:
                fallback_indexes.update(self._doc_type_index.get(doc_type, set()))
        if not fallback_indexes:
            fallback_indexes = set(range(len(self._windows)))
        return fallback_indexes

    def _semantic_scores(self, query, rule: dict) -> dict[int, float]:
        if not self._windows:
            return {}

        preferred_pool = self._fallback_candidate_indexes(query)
        semantic_query = self._build_semantic_query(query, rule)
        top_k = 30 if str(rule.get("sub_category", "") or "") == "Hiệu quả" else 20
        min_score = 0.08 if str(rule.get("sub_category", "") or "") == "Hiệu quả" else 0.06

        # Use lower thresholds with real embeddings since scores are more meaningful
        if self.semantic_index.has_embeddings():
            min_score = max(0.2, min_score)
            top_k = min(top_k + 10, 40)

        matches = self.semantic_index.search(
            semantic_query,
            allowed_indexes=preferred_pool,
            top_k=top_k,
            min_score=min_score,
        )
        return {match.window_index: match.score for match in matches}

    def _build_semantic_query(self, query, rule: dict) -> str:
        parts = [query.question_text]
        parts.extend(query.exact_phrases[:8])
        parts.extend(query.primary_terms[:12])
        parts.extend(query.secondary_terms[:10])
        parts.extend(getattr(query, "intent_terms", [])[:10])
        options = str(rule.get("options", "") or "").strip()
        if options:
            parts.append(options[:500])
        return "\n".join(part for part in parts if part)

    def _extract_index_tokens(self, text: str) -> set[str]:
        return {
            token
            for token in re.findall(r"[0-9a-zA-Zà-ỹđ]{3,}", text)
            if len(token) >= 3
        }

    def _normalize_term(self, term: str) -> str:
        if term not in self._normalized_term_cache:
            self._normalized_term_cache[term] = self.normalizer.normalize_for_search(term)
        return self._normalized_term_cache[term]

    def _score_window(self, window: IndexedWindow, query, rule: dict, semantic_score: float = 0.0) -> RetrievalCandidate | None:
        normalized_content = window.normalized_content
        combined_haystack = f"{window.normalized_title} {normalized_content}".strip()
        exact_phrase_hits = self._matched_terms(query.exact_phrases, normalized_content)
        primary_hits = self._matched_terms(query.primary_terms, normalized_content)
        secondary_hits = self._matched_terms(query.secondary_terms, normalized_content)
        intent_hits = self._matched_terms(getattr(query, "intent_terms", []), normalized_content)
        chunk_type = window.chunk_type
        table_family = window.table_family
        family_match_bonus = self._table_family_bonus(rule=rule, table_family=table_family)
        performance_signal_hits = self._performance_signal_hits(query, combined_haystack)
        strict_anchor_required = self._has_strict_performance_anchor(query)

        if str(rule.get("sub_category", "") or "") == "Hiệu quả" or rule.get("question_type") in {"numeric_disclosure", "ratio_calculation"}:
            if strict_anchor_required and not exact_phrase_hits and semantic_score < 0.24:
                return None
            if query.exact_phrases and semantic_score < 0.16 and not exact_phrase_hits and not performance_signal_hits:
                return None
            if semantic_score < 0.16 and not performance_signal_hits and table_family in {"financial_metrics", "general_table"}:
                return None
            if semantic_score < 0.14 and not performance_signal_hits and not exact_phrase_hits and not primary_hits and not secondary_hits:
                return None

        if (
            not exact_phrase_hits
            and not primary_hits
            and not secondary_hits
            and not intent_hits
            and family_match_bonus <= 0
            and semantic_score < 0.1
        ):
            return None

        score = 0.0
        reasons = []
        if exact_phrase_hits:
            score += 3.0 * len(exact_phrase_hits)
            reasons.append(f"exact_phrase_hits={len(exact_phrase_hits)}")
        if primary_hits:
            score += 1.2 * len(primary_hits)
            reasons.append(f"primary_hits={len(primary_hits)}")
        if secondary_hits:
            score += 0.7 * min(4, len(secondary_hits))
            reasons.append(f"secondary_hits={len(secondary_hits)}")
        if intent_hits:
            score += 0.45 * min(4, len(intent_hits))
            reasons.append(f"intent_hits={len(intent_hits)}")

        # Higher weight for semantic score when using real embeddings
        if semantic_score > 0:
            if self.semantic_index.has_embeddings():
                semantic_contribution = min(3.0, semantic_score * 5.0)
            else:
                semantic_contribution = min(1.6, semantic_score * 2.4)
            score += semantic_contribution
            reasons.append(f"semantic_score={semantic_score:.2f}")

        if family_match_bonus > 0:
            score += family_match_bonus
            reasons.append(f"table_family_bonus={table_family}")

        if query.preferred_document_types and window.document_type in query.preferred_document_types:
            preference_rank = query.preferred_document_types.index(window.document_type) + 1
            score += max(0.2, 0.9 - 0.2 * (preference_rank - 1))
            reasons.append("preferred_document_type")

        if rule.get("question_type") == "governance" and window.document_type in {"resolution", "annual_report", "financial_report"}:
            score += 0.7
            reasons.append("governance_doc_bonus")
        if rule.get("question_type") in {"numeric_disclosure", "ratio_calculation"} and chunk_type in {"table_section", "metric_kv_section"}:
            score += 1.0
            reasons.append("numeric_table_bonus")

        if chunk_type == "section":
            score += 0.35
            reasons.append("structured_section_bonus")
        elif chunk_type == "table_section":
            score += 0.45
            reasons.append("table_section_bonus")
        elif chunk_type == "metric_kv_section":
            score += 0.65
            reasons.append("metric_kv_bonus")

        section_title = window.section_title
        normalized_title = window.normalized_title
        title_hits = self._matched_terms(query.primary_terms + query.exact_phrases, normalized_title)
        if title_hits:
            score += 0.8
            reasons.append("section_title_match")

        quality_score = window.quality_score
        score += quality_score
        score += window.evidence_signal
        if window.evidence_signal:
            reasons.append(f"evidence_signal={window.evidence_signal:.2f}")
        if window.low_value:
            score -= 1.2
            reasons.append("low_value_penalty")
        if window.no_evidence_hint:
            score -= 0.45
            reasons.append("no_evidence_hint_penalty")
        if window.page_start <= 2:
            score -= 0.25
            reasons.append("early_page_penalty")
        if len(normalized_content.split()) > 80:
            score += 0.15
            reasons.append("sufficient_context")
        if not exact_phrase_hits and len(primary_hits) < 2 and len(intent_hits) < 2:
            score -= 0.75
            reasons.append("weak_lexical_match_penalty")

        if score <= 0:
            return None

        matched_terms = list(dict.fromkeys(exact_phrase_hits + primary_hits + secondary_hits + intent_hits))
        return RetrievalCandidate(
            source_file=window.source_file,
            chunk_id=window.chunk_id,
            source_path=window.source_path,
            document_type=window.document_type,
            section_title=window.section_title,
            chunk_type=window.chunk_type,
            table_family=window.table_family,
            year_guess=window.year_guess,
            coverage_source=window.coverage_source,
            page_start=window.page_start,
            page_end=window.page_end,
            content=window.content,
            normalized_content=normalized_content,
            score=round(score, 4),
            quality_score=quality_score,
            labels=window.labels,
            exact_phrase_hits=exact_phrase_hits,
            matched_terms=matched_terms,
            reasons=reasons + [f"labels={','.join(window.labels)}"] if window.labels else reasons,
            semantic_score=round(semantic_score, 4),
            rerank_score=round(score, 4),
            rerank_reasons=[],
            low_value=window.low_value,
        )

    def _matched_terms(self, terms: list[str], normalized_content: str) -> list[str]:
        hits = []
        for term in terms:
            normalized_term = self._normalize_term(term)
            if normalized_term and normalized_term in normalized_content:
                hits.append(normalized_term)
        return list(dict.fromkeys(hits))

    def _performance_signal_hits(self, query, normalized_content: str) -> list[str]:
        terms = []
        for term in list(query.exact_phrases) + list(query.primary_terms) + list(query.secondary_terms):
            normalized_term = self._normalize_term(term)
            if not normalized_term or normalized_term in self.GENERIC_PERFORMANCE_TERMS:
                continue
            if normalized_term in normalized_content:
                terms.append(normalized_term)
        return list(dict.fromkeys(terms))

    def _has_strict_performance_anchor(self, query) -> bool:
        for term in query.exact_phrases:
            normalized_term = self._normalize_term(term)
            if normalized_term and normalized_term not in self.GENERIC_PERFORMANCE_TERMS and len(normalized_term) >= 8:
                return True
        return False

    def _table_family_bonus(self, rule: dict, table_family: str | None) -> float:
        if not table_family:
            return 0.0

        sub_category = str(rule.get("sub_category", "") or "")
        factor = str(rule.get("factor", "") or "")
        question = str(rule.get("question", "") or "").lower()

        if sub_category == "Hiệu quả" or rule.get("question_type") in {"numeric_disclosure", "ratio_calculation"}:
            if factor.startswith("E") and table_family in {"environmental_metrics", "financial_metrics"}:
                return 1.4
            if factor.startswith("S") and table_family in {"workforce_metrics", "financial_metrics", "csr_impact_metrics"}:
                return 1.4
            if table_family in {"general_table", "financial_metrics", "workforce_metrics", "environmental_metrics", "csr_impact_metrics"}:
                return 0.7

        if sub_category == "Quyền cổ đông" and table_family == "voting_results":
            return 1.3

        if "kiểm toán" in question and table_family == "financial_metrics":
            return 0.8

        return 0.0

    def _selection_quota(self, rule: dict, top_k: int) -> int:
        sub_category = str(rule.get("sub_category", "") or "")
        plan = rule.get("retrieval_plan") if isinstance(rule.get("retrieval_plan"), dict) else {}
        profile = str(plan.get("evidence_profile") or "")
        if profile == "policy_public":
            return min(max(3, top_k // 3), top_k)
        if profile == "ratio_with_revenue":
            return min(max(4, top_k // 2), top_k)
        if sub_category == "Hiệu quả" or rule.get("question_type") in {"numeric_disclosure", "ratio_calculation"}:
            return min(max(2, top_k // 3), top_k)
        if sub_category == "Quyền cổ đông":
            return 1
        return 0

    def _matches_priority_candidate(self, candidate: RetrievalCandidate, rule: dict) -> bool:
        sub_category = str(rule.get("sub_category", "") or "")
        factor = str(rule.get("factor", "") or "")
        plan = rule.get("retrieval_plan") if isinstance(rule.get("retrieval_plan"), dict) else {}
        profile = str(plan.get("evidence_profile") or "")

        if profile == "policy_public":
            return (
                candidate.document_type == "policy_document"
                or "chinh sach" in candidate.normalized_content
                or "cong khai" in candidate.normalized_content
            )

        if profile == "ratio_with_revenue":
            return (
                candidate.chunk_type in {"table_section", "metric_kv_section"}
                or candidate.table_family in {"environmental_metrics", "financial_metrics"}
            )

        if sub_category == "Hiệu quả" or rule.get("question_type") in {"numeric_disclosure", "ratio_calculation"}:
            if candidate.chunk_type not in {"table_section", "metric_kv_section"}:
                return False
            if factor.startswith("E"):
                return candidate.table_family in {"environmental_metrics", "financial_metrics", "general_table"}
            if factor.startswith("S"):
                return candidate.table_family in {"workforce_metrics", "financial_metrics", "general_table", "csr_impact_metrics"}
            return candidate.table_family in {"financial_metrics", "workforce_metrics", "environmental_metrics", "general_table", "csr_impact_metrics"}

        if sub_category == "Quyền cổ đông":
            return candidate.table_family == "voting_results"

        return False

    def _corpus_cache_key(self) -> str:
        """Generate a cache key from actual indexed windows for embedding caching."""
        return CacheManager.hash_json({
            "schema_version": "retrieval_schema_v7",
            "target_year": self.target_year,
            "window_count": len(self._windows),
            "window_fingerprint": self.corpus_window_fingerprint(),
        })[:24]

    def corpus_window_fingerprint(self) -> str:
        """Fingerprint the real retrieval windows, not just source PDFs."""
        return CacheManager.hash_json([
            {
                "chunk_id": window.chunk_id,
                "source_path": window.source_path,
                "source_file": window.source_file,
                "document_type": window.document_type,
                "year_guess": window.year_guess,
                "page_start": window.page_start,
                "page_end": window.page_end,
                "chunk_type": window.chunk_type,
                "table_family": window.table_family,
                "content_hash": CacheManager.hash_text(window.content),
            }
            for window in self._windows
        ])
