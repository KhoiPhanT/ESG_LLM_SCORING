"""
Semantic chunker — split oversized sections into topic-coherent chunks.
Uses embedding similarity to detect topic boundaries within long sections.
Falls back to sliding-window overlap if embeddings are not available.
"""
from __future__ import annotations

import re


class SemanticChunker:
    """Split long document sections into semantically coherent chunks."""

    # Target chunk size in characters (roughly 400-600 tokens for Vietnamese)
    TARGET_CHUNK_SIZE = 2000
    MAX_CHUNK_SIZE = 5000
    MIN_CHUNK_SIZE = 300
    OVERLAP_SIZE = 300

    def chunk_sections(self, sections: list[dict], embedding_model=None) -> list[dict]:
        """
        Process a list of sections, splitting oversized ones into smaller chunks.
        Keeps small sections unchanged.
        """
        result = []
        for section in sections:
            content = section.get("content", "")
            if not content or len(content) <= self.MAX_CHUNK_SIZE:
                result.append(section)
                continue

            # Split oversized section
            chunks = self._split_section(section, embedding_model)
            result.extend(chunks)

        return result

    def _split_section(self, section: dict, embedding_model=None) -> list[dict]:
        """Split a single oversized section into smaller coherent chunks."""
        content = section.get("content", "")
        paragraphs = self._extract_paragraphs(content)

        if len(paragraphs) <= 1:
            # Can't split by paragraphs, use sliding window
            return self._sliding_window_split(section)

        # Group paragraphs into coherent chunks
        if embedding_model is not None:
            groups = self._embedding_based_grouping(paragraphs, embedding_model)
        else:
            groups = self._size_based_grouping(paragraphs)

        return self._build_chunks_from_groups(section, groups)

    def _sliding_window_split(self, section: dict) -> list[dict]:
        """Split a single section using a sliding window when paragraph splitting fails."""
        content = section.get("content", "")
        chunks = []
        start = 0
        chunk_idx = 0
        while start < len(content):
            end = min(start + self.MAX_CHUNK_SIZE, len(content))
            # Try to break at a newline boundary
            if end < len(content):
                newline_pos = content.rfind("\n", start + self.MIN_CHUNK_SIZE, end)
                if newline_pos > start:
                    end = newline_pos + 1
            chunk_content = content[start:end].strip()
            if len(chunk_content) >= self.MIN_CHUNK_SIZE:
                chunk = dict(section)
                chunk["content"] = chunk_content
                chunk["chunk_id"] = f"{section.get('chunk_id', '')}:win{chunk_idx}"
                chunk["section_title"] = section.get("section_title", "") + (
                    f" (phần {chunk_idx + 1})" if True else ""
                )
                chunks.append(chunk)
                chunk_idx += 1
            # Move start with overlap
            start = max(start + 1, end - self.OVERLAP_SIZE)
        return chunks if chunks else [section]

    def _extract_paragraphs(self, text: str) -> list[str]:
        """Split text into paragraphs based on double newlines or heading patterns."""
        # Split on double newlines
        raw_paragraphs = re.split(r"\n\s*\n", text)

        # Further split very long paragraphs on single newlines with heading patterns
        paragraphs = []
        for para in raw_paragraphs:
            para = para.strip()
            if not para:
                continue
            if len(para) > self.TARGET_CHUNK_SIZE:
                sub_paragraphs = self._split_on_headings(para)
                paragraphs.extend(sub_paragraphs)
            else:
                paragraphs.append(para)

        return [p for p in paragraphs if p.strip()]

    def _split_on_headings(self, text: str) -> list[str]:
        """Split text on heading-like patterns."""
        heading_pattern = re.compile(
            r"(?=^(?:\d+(?:\.\d+)*[\.\-:\s]|[A-ZĐ][A-ZĐ\s]{5,}|(?:Phần|Chương|Mục)\s+))",
            re.MULTILINE,
        )
        parts = heading_pattern.split(text)
        if len(parts) <= 1:
            # Fallback: split by single newlines for very dense text
            lines = text.split("\n")
            parts = []
            current = []
            for line in lines:
                current.append(line)
                if len("\n".join(current)) >= self.TARGET_CHUNK_SIZE:
                    parts.append("\n".join(current))
                    current = []
            if current:
                parts.append("\n".join(current))
        return [p.strip() for p in parts if p.strip()]

    def _size_based_grouping(self, paragraphs: list[str]) -> list[list[str]]:
        """Group paragraphs into chunks based on size constraints."""
        groups = []
        current_group = []
        current_size = 0

        for para in paragraphs:
            para_size = len(para)

            if current_size + para_size > self.MAX_CHUNK_SIZE and current_group:
                groups.append(current_group)
                # Add overlap from last paragraph
                overlap = current_group[-1][-self.OVERLAP_SIZE:] if current_group else ""
                current_group = [overlap] if overlap else []
                current_size = len(overlap)

            current_group.append(para)
            current_size += para_size

        if current_group:
            groups.append(current_group)

        return groups

    def _embedding_based_grouping(self, paragraphs: list[str], embedding_model) -> list[list[str]]:
        """Group paragraphs using embedding similarity to detect topic shifts."""
        if len(paragraphs) <= 2:
            return [paragraphs]

        try:
            embeddings = embedding_model.encode(
                [f"passage: {p[:500]}" for p in paragraphs],
                normalize_embeddings=True,
            )

            # Compute cosine similarity between consecutive paragraphs
            similarities = []
            for i in range(len(embeddings) - 1):
                sim = float(embeddings[i] @ embeddings[i + 1])
                similarities.append(sim)

            # Find natural break points (low similarity = topic shift)
            threshold = max(0.3, sum(similarities) / len(similarities) - 0.15)

            groups = []
            current_group = [paragraphs[0]]
            current_size = len(paragraphs[0])

            for i, para in enumerate(paragraphs[1:], start=1):
                is_topic_shift = similarities[i - 1] < threshold
                is_too_large = current_size + len(para) > self.MAX_CHUNK_SIZE
                is_target_reached = current_size >= self.TARGET_CHUNK_SIZE

                if (is_topic_shift and is_target_reached) or is_too_large:
                    groups.append(current_group)
                    # Add overlap
                    overlap = current_group[-1][-self.OVERLAP_SIZE:] if current_group else ""
                    current_group = [overlap, para] if overlap else [para]
                    current_size = len(overlap) + len(para)
                else:
                    current_group.append(para)
                    current_size += len(para)

            if current_group:
                groups.append(current_group)

            return groups

        except Exception:
            # Fallback if embedding fails
            return self._size_based_grouping(paragraphs)

    def _build_chunks_from_groups(self, section: dict, groups: list[list[str]]) -> list[dict]:
        """Build chunk dicts from paragraph groups."""
        chunks = []
        page_start = section.get("page_start", 0)
        page_end = section.get("page_end", 0)
        total_pages = max(1, page_end - page_start + 1)

        for group_idx, group in enumerate(groups):
            content = "\n\n".join(group).strip()
            if len(content) < self.MIN_CHUNK_SIZE:
                continue

            # Estimate page range for this chunk
            progress = group_idx / max(1, len(groups))
            estimated_page_start = page_start + int(progress * total_pages)
            estimated_page_end = min(
                page_end,
                page_start + int((group_idx + 1) / max(1, len(groups)) * total_pages),
            )

            chunk = dict(section)
            chunk["content"] = content[:self.MAX_CHUNK_SIZE]
            chunk["chunk_id"] = f"{section.get('chunk_id', '')}:sub{group_idx}"
            chunk["page_start"] = estimated_page_start
            chunk["page_end"] = max(estimated_page_start, estimated_page_end)
            chunk["section_title"] = section.get("section_title", "") + (
                f" (phần {group_idx + 1})" if len(groups) > 1 else ""
            )
            chunks.append(chunk)

        return chunks if chunks else [section]
