"""
Lexical retrieval engine with document-aware ranking and provenance.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import re

from core.normalization.text_normalizer import TextNormalizer
from core.query_builder.question_query_builder import QuestionQueryBuilder
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

    def __init__(self, corpus):
        self.corpus = corpus
        self.normalizer = TextNormalizer()
        self.query_builder = QuestionQueryBuilder()
        self.reranker = RetrievalReranker()
        self._normalized_term_cache: dict[str, str] = {}
        self._windows: list[IndexedWindow] = []
        self._token_index: dict[str, set[int]] = {}
        self._doc_type_index: dict[str, set[int]] = {}
        self._chunk_type_index: dict[str, set[int]] = {}
        self._table_family_index: dict[str, set[int]] = {}
        self._build_index()
        semantic_documents = [
            f"{window.normalized_title}\n{window.normalized_content}" for window in self._windows
        ]
        self.semantic_index = SemanticIndex(semantic_documents)

    def retrieve_for_rule(self, rule: dict, top_k: int = 6) -> dict:
        query = self.query_builder.build(rule, corpus=self.corpus)
        candidates = self._score_candidates(query, rule=rule)
        selected = self._select_candidates(candidates, rule=rule, top_k=top_k)
        return {
            "query": query.to_dict(),
            "candidates": [candidate.to_dict() for candidate in selected],
        }

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

        if str(rule.get("sub_category", "") or "") == "Hiệu quả":
            candidate_indexes.update(self._chunk_type_index.get("table_section", set()))
            factor = str(rule.get("factor", "") or "")
            preferred_families = {"financial_metrics", "general_table"}
            if factor.startswith("E"):
                preferred_families.add("environmental_metrics")
            if factor.startswith("S"):
                preferred_families.add("workforce_metrics")
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
        top_k = 24 if str(rule.get("sub_category", "") or "") == "Hiệu quả" else 16
        matches = self.semantic_index.search(
            semantic_query,
            allowed_indexes=preferred_pool,
            top_k=top_k,
            min_score=0.1 if str(rule.get("sub_category", "") or "") == "Hiệu quả" else 0.08,
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

        if str(rule.get("sub_category", "") or "") == "Hiệu quả":
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
        if semantic_score > 0:
            score += min(1.6, semantic_score * 2.4)
            reasons.append(f"semantic_score={semantic_score:.2f}")
        if family_match_bonus > 0:
            score += family_match_bonus
            reasons.append(f"table_family_bonus={table_family}")

        if query.preferred_document_types and window.document_type in query.preferred_document_types:
            preference_rank = query.preferred_document_types.index(window.document_type) + 1
            score += max(0.2, 0.9 - 0.2 * (preference_rank - 1))
            reasons.append("preferred_document_type")

        if chunk_type == "section":
            score += 0.35
            reasons.append("structured_section_bonus")
        elif chunk_type == "table_section":
            score += 0.45
            reasons.append("table_section_bonus")

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

        if sub_category == "Hiệu quả":
            if factor.startswith("E") and table_family in {"environmental_metrics", "financial_metrics"}:
                return 1.4
            if factor.startswith("S") and table_family in {"workforce_metrics", "financial_metrics"}:
                return 1.4
            if table_family in {"general_table", "financial_metrics", "workforce_metrics", "environmental_metrics"}:
                return 0.7

        if sub_category == "Quyền cổ đông" and table_family == "voting_results":
            return 1.3

        if "kiểm toán" in question and table_family == "financial_metrics":
            return 0.8

        return 0.0

    def _selection_quota(self, rule: dict, top_k: int) -> int:
        sub_category = str(rule.get("sub_category", "") or "")
        if sub_category == "Hiệu quả":
            return min(max(2, top_k // 3), top_k)
        if sub_category == "Quyền cổ đông":
            return 1
        return 0

    def _matches_priority_candidate(self, candidate: RetrievalCandidate, rule: dict) -> bool:
        sub_category = str(rule.get("sub_category", "") or "")
        factor = str(rule.get("factor", "") or "")

        if sub_category == "Hiệu quả":
            if candidate.chunk_type != "table_section":
                return False
            if factor.startswith("E"):
                return candidate.table_family in {"environmental_metrics", "financial_metrics", "general_table"}
            if factor.startswith("S"):
                return candidate.table_family in {"workforce_metrics", "financial_metrics", "general_table"}
            return candidate.table_family in {"financial_metrics", "workforce_metrics", "environmental_metrics", "general_table"}

        if sub_category == "Quyền cổ đông":
            return candidate.table_family == "voting_results"

        return False
