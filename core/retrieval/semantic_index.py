"""
Hybrid semantic index — combines TF-IDF (fast, always available) with dense
embeddings (high quality, requires sentence-transformers + FAISS).
Falls back gracefully if embedding libraries are not installed.
"""
from __future__ import annotations

from dataclasses import dataclass
import os

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer


@dataclass
class SemanticMatch:
    window_index: int
    score: float


class SemanticIndex:
    """Hybrid index: dense embeddings (primary) + TF-IDF (fallback)."""

    def __init__(self, documents: list[str], cache_key: str = ""):
        self.documents = [doc or "" for doc in documents]
        self.enabled = bool(self.documents)
        self._embedding_index = None
        self._tfidf_enabled = False
        self.vectorizer: TfidfVectorizer | None = None
        self.svd: TruncatedSVD | None = None
        self.document_vectors = None

        if not self.enabled:
            return

        # Try to build embedding index (primary)
        self._try_build_embedding_index(cache_key)

        # Always build TF-IDF as supplementary / fallback
        self._fit_tfidf()

    def search(
        self,
        query_text: str,
        allowed_indexes: set[int] | None = None,
        top_k: int = 20,
        min_score: float = 0.08,
    ) -> list[SemanticMatch]:
        if not self.enabled or not query_text.strip():
            return []

        # Get results from both systems and fuse
        embedding_results = self._embedding_search(query_text, allowed_indexes, top_k * 2)
        tfidf_results = self._tfidf_search(query_text, allowed_indexes, top_k * 2)

        if embedding_results and tfidf_results:
            # Reciprocal Rank Fusion
            return self._reciprocal_rank_fusion(
                embedding_results, tfidf_results, top_k=top_k, min_score=min_score
            )
        elif embedding_results:
            return [m for m in embedding_results[:top_k] if m.score >= min_score]
        elif tfidf_results:
            return [m for m in tfidf_results[:top_k] if m.score >= min_score]
        return []

    def has_embeddings(self) -> bool:
        """Check if dense embedding index is available."""
        return self._embedding_index is not None

    def _reciprocal_rank_fusion(
        self,
        list_a: list[SemanticMatch],
        list_b: list[SemanticMatch],
        k: int = 60,
        top_k: int = 20,
        min_score: float = 0.08,
    ) -> list[SemanticMatch]:
        """Combine two ranked lists using Reciprocal Rank Fusion."""
        fused_scores: dict[int, float] = {}
        raw_scores: dict[int, float] = {}

        for rank, match in enumerate(list_a, start=1):
            fused_scores[match.window_index] = fused_scores.get(match.window_index, 0) + 1 / (k + rank)
            raw_scores[match.window_index] = max(
                raw_scores.get(match.window_index, 0), match.score
            )

        for rank, match in enumerate(list_b, start=1):
            fused_scores[match.window_index] = fused_scores.get(match.window_index, 0) + 1 / (k + rank)
            raw_scores[match.window_index] = max(
                raw_scores.get(match.window_index, 0), match.score
            )

        ranked = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
        results = []
        for window_index, fused_score in ranked:
            original_score = raw_scores.get(window_index, 0)
            if original_score < min_score:
                continue
            results.append(SemanticMatch(window_index=window_index, score=round(original_score, 4)))
            if len(results) >= top_k:
                break
        return results

    def _try_build_embedding_index(self, cache_key: str) -> None:
        """Try to initialize the dense embedding index."""
        if os.environ.get("ESG_DISABLE_DENSE_EMBEDDINGS") == "1":
            print("  [SEMANTIC] Dense embeddings disabled by ESG_DISABLE_DENSE_EMBEDDINGS=1")
            self._embedding_index = None
            return

        try:
            from core.retrieval.embedding_index import EmbeddingIndex

            self._embedding_index = EmbeddingIndex(
                self.documents,
                cache_key=cache_key,
            )
            if not self._embedding_index.enabled:
                self._embedding_index = None
        except ImportError as e:
            print(f"  [SEMANTIC] Dense embeddings not available ({e}), using TF-IDF only")
            self._embedding_index = None
        except Exception as e:
            print(f"  [SEMANTIC] Embedding index failed ({e}), falling back to TF-IDF")
            self._embedding_index = None

    def _embedding_search(
        self, query_text: str, allowed_indexes: set[int] | None, top_k: int
    ) -> list[SemanticMatch]:
        if self._embedding_index is None:
            return []
        try:
            return self._embedding_index.search(
                query_text,
                allowed_indexes=allowed_indexes,
                top_k=top_k,
                min_score=0.2,
            )
        except Exception:
            return []

    def _tfidf_search(
        self, query_text: str, allowed_indexes: set[int] | None, top_k: int
    ) -> list[SemanticMatch]:
        if not self._tfidf_enabled or self.vectorizer is None:
            return []

        query_vector = self.vectorizer.transform([query_text])
        if query_vector.nnz == 0:
            return []

        if self.svd is not None and isinstance(self.document_vectors, np.ndarray):
            projected_query = self.svd.transform(query_vector)
            query_norm = np.linalg.norm(projected_query, axis=1, keepdims=True)
            query_norm[query_norm == 0.0] = 1.0
            normalized_query = projected_query / query_norm
            scores = (self.document_vectors @ normalized_query.T).ravel()
        else:
            scores = (self.document_vectors @ query_vector.T).toarray().ravel()

        if allowed_indexes is not None:
            score_rows = [(index, float(scores[index])) for index in allowed_indexes if index < len(scores)]
        else:
            score_rows = [(index, float(score)) for index, score in enumerate(scores)]

        ranked = sorted(score_rows, key=lambda item: item[1], reverse=True)
        matches = []
        for index, score in ranked:
            if score < 0.01:
                continue
            matches.append(SemanticMatch(window_index=index, score=round(score, 4)))
            if len(matches) >= top_k:
                break
        return matches

    def _fit_tfidf(self) -> None:
        if not self.documents:
            self._tfidf_enabled = False
            return

        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)
        tfidf_matrix = self.vectorizer.fit_transform(self.documents)
        self.document_vectors = tfidf_matrix

        feature_count = tfidf_matrix.shape[1]
        sample_count = tfidf_matrix.shape[0]
        if feature_count < 4 or sample_count < 4:
            self._tfidf_enabled = True
            return

        component_count = min(64, sample_count - 1, feature_count - 1)
        if component_count < 2:
            self._tfidf_enabled = True
            return

        self.svd = TruncatedSVD(n_components=component_count, random_state=42)
        projected = self.svd.fit_transform(tfidf_matrix)
        norms = np.linalg.norm(projected, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        self.document_vectors = projected / norms
        self._tfidf_enabled = True
