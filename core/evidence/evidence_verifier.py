"""
Evidence verification — ground-truth checking for LLM-generated evidence quotes.
Detects hallucinated evidence that doesn't actually appear in source documents.
"""
from __future__ import annotations

import re
from difflib import SequenceMatcher


class EvidenceVerifier:
    """Verify that LLM-claimed evidence actually exists in the source documents."""

    # Minimum similarity score to consider a quote as grounded
    GROUNDING_THRESHOLD = 0.35
    # Minimum length for a meaningful evidence quote
    MIN_EVIDENCE_LENGTH = 8

    def verify(self, claimed_evidence: str, source_sections: list[dict]) -> dict:
        """
        Check if the claimed evidence quote exists in any source section.
        Returns verification result with match details.
        """
        if not claimed_evidence or len(claimed_evidence.strip()) < self.MIN_EVIDENCE_LENGTH:
            return {
                "grounded": False,
                "match_score": 0.0,
                "match_type": "too_short",
                "matched_section": None,
                "verified_quote": None,
            }

        evidence_clean = self._normalize(claimed_evidence)
        best_match = None
        best_score = 0.0
        best_type = "no_match"
        best_quote = None

        for section in source_sections:
            content = section.get("content", "")
            if not content:
                continue

            content_clean = self._normalize(content)

            # Level 1: Exact substring match (strongest signal)
            if evidence_clean in content_clean:
                return {
                    "grounded": True,
                    "match_score": 1.0,
                    "match_type": "exact",
                    "matched_section": self._section_summary(section),
                    "verified_quote": claimed_evidence.strip(),
                }

            # Level 2: Key phrase matching
            phrase_score = self._phrase_overlap_score(evidence_clean, content_clean)
            if phrase_score > best_score:
                best_score = phrase_score
                best_match = section
                best_type = "phrase_overlap"
                best_quote = self._extract_best_matching_span(evidence_clean, content_clean)

            # Level 3: Fuzzy sequence matching
            fuzzy_score = self._fuzzy_match(evidence_clean, content_clean)
            if fuzzy_score > best_score:
                best_score = fuzzy_score
                best_match = section
                best_type = "fuzzy"
                best_quote = self._extract_best_matching_span(evidence_clean, content_clean)

        grounded = best_score >= self.GROUNDING_THRESHOLD
        return {
            "grounded": grounded,
            "match_score": round(best_score, 4),
            "match_type": best_type if grounded else "ungrounded",
            "matched_section": self._section_summary(best_match) if best_match else None,
            "verified_quote": best_quote,
        }

    def verify_batch(self, evidence_items: list[dict], source_sections: list[dict]) -> list[dict]:
        """Verify multiple evidence items against source sections."""
        results = []
        for item in evidence_items:
            quote = item.get("quote", "")
            verification = self.verify(quote, source_sections)
            enriched = dict(item)
            enriched["verification"] = verification
            if not verification["grounded"]:
                enriched["confidence"] = max(0.1, float(item.get("confidence", 0.5)) * 0.4)
            results.append(enriched)
        return results

    def _normalize(self, text: str) -> str:
        """Normalize text for comparison."""
        normalized = text.lower().strip()
        normalized = re.sub(r"\s+", " ", normalized)
        normalized = re.sub(r"[^\w\s]", "", normalized)
        return normalized

    def _phrase_overlap_score(self, evidence: str, content: str) -> float:
        """Score based on how many key phrases from evidence appear in content."""
        # Extract meaningful phrases (3+ word sequences)
        evidence_words = evidence.split()
        if len(evidence_words) < 3:
            return 0.0

        total_phrases = 0
        matched_phrases = 0
        for i in range(len(evidence_words) - 2):
            phrase = " ".join(evidence_words[i : i + 3])
            if len(phrase) >= 8:  # Skip very short phrases
                total_phrases += 1
                if phrase in content:
                    matched_phrases += 1

        if total_phrases == 0:
            return 0.0
        return matched_phrases / total_phrases

    def _fuzzy_match(self, evidence: str, content: str) -> float:
        """Fuzzy substring matching using SequenceMatcher."""
        # For very long content, sample windows around potential matches
        if len(content) > 5000:
            windows = self._extract_candidate_windows(evidence, content, window_size=500)
        else:
            windows = [content]

        best_ratio = 0.0
        for window in windows:
            matcher = SequenceMatcher(None, evidence[:200], window[:500])
            ratio = matcher.ratio()
            best_ratio = max(best_ratio, ratio)

        return best_ratio

    def _extract_candidate_windows(
        self, evidence: str, content: str, window_size: int = 500
    ) -> list[str]:
        """Extract windows from content that might contain the evidence."""
        key_words = [w for w in evidence.split()[:5] if len(w) >= 4]
        windows = []
        for word in key_words:
            pos = content.find(word)
            while pos >= 0 and len(windows) < 6:
                start = max(0, pos - window_size // 2)
                end = min(len(content), pos + window_size // 2)
                windows.append(content[start:end])
                pos = content.find(word, pos + 1)
        return windows or [content[:window_size]]

    def _extract_best_matching_span(self, evidence: str, content: str) -> str | None:
        """Extract the span from content that best matches the evidence."""
        key_words = [w for w in evidence.split() if len(w) >= 4][:3]
        for word in key_words:
            pos = content.find(word)
            if pos >= 0:
                start = max(0, pos - 100)
                end = min(len(content), pos + 200)
                return content[start:end].strip()
        return None

    def _section_summary(self, section: dict) -> dict:
        return {
            "source_file": section.get("source_file"),
            "document_type": section.get("document_type"),
            "page_start": section.get("page_start"),
            "page_end": section.get("page_end"),
        }
