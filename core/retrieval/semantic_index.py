"""
Lightweight semantic index scaffold for hybrid retrieval.
Uses TF-IDF + latent semantic projection so the retrieval pipeline can
switch to real embeddings later without changing the surrounding API.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer


@dataclass
class SemanticMatch:
    window_index: int
    score: float


class SemanticIndex:
    def __init__(self, documents: list[str]):
        self.documents = [doc or "" for doc in documents]
        self.enabled = bool(self.documents)
        self.vectorizer: TfidfVectorizer | None = None
        self.svd: TruncatedSVD | None = None
        self.document_vectors = None
        self._fit()

    def search(
        self,
        query_text: str,
        allowed_indexes: set[int] | None = None,
        top_k: int = 20,
        min_score: float = 0.08,
    ) -> list[SemanticMatch]:
        if not self.enabled or not query_text.strip() or self.vectorizer is None:
            return []

        query_vector = self.vectorizer.transform([query_text])
        if query_vector.nnz == 0:
            return []

        if self.svd is not None and self.document_vectors is not None:
            projected_query = self.svd.transform(query_vector)
            query_norm = np.linalg.norm(projected_query, axis=1, keepdims=True)
            query_norm[query_norm == 0.0] = 1.0
            normalized_query = projected_query / query_norm
            scores = (self.document_vectors @ normalized_query.T).ravel()
        else:
            scores = (self.document_vectors @ query_vector.T).toarray().ravel()

        if allowed_indexes is not None:
            score_rows = [(index, float(scores[index])) for index in allowed_indexes]
        else:
            score_rows = [(index, float(score)) for index, score in enumerate(scores)]

        ranked = sorted(score_rows, key=lambda item: item[1], reverse=True)
        matches = []
        for index, score in ranked:
            if score < min_score:
                continue
            matches.append(SemanticMatch(window_index=index, score=round(score, 4)))
            if len(matches) >= top_k:
                break
        return matches

    def _fit(self) -> None:
        if not self.documents:
            self.enabled = False
            return

        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)
        tfidf_matrix = self.vectorizer.fit_transform(self.documents)
        self.document_vectors = tfidf_matrix

        feature_count = tfidf_matrix.shape[1]
        sample_count = tfidf_matrix.shape[0]
        if feature_count < 4 or sample_count < 4:
            return

        component_count = min(64, sample_count - 1, feature_count - 1)
        if component_count < 2:
            return

        self.svd = TruncatedSVD(n_components=component_count, random_state=42)
        projected = self.svd.fit_transform(tfidf_matrix)
        norms = np.linalg.norm(projected, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        self.document_vectors = projected / norms
