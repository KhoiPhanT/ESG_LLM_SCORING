"""
Audit how much extracted document content is represented in the retrieval index.
"""
from __future__ import annotations

import json
import os


class IndexCoverageAudit:
    def audit(self, corpus, verbose: bool = False) -> dict:
        documents = []
        total_docs = len(corpus.documents)
        for index, doc in enumerate(corpus.documents, start=1):
            if verbose:
                print(
                    f"  [INDEX COVERAGE] {index}/{total_docs} {doc.label}",
                    flush=True,
                )
            pages = corpus.get_document_pages(doc.path)
            sections = corpus.get_document_sections(doc.path)
            total_chars = sum(len(page.get("text", "") or "") for page in pages)
            indexed_chars = sum(len(section.get("content", "") or "") for section in sections)
            indexed_pages = set()
            sources = {}
            for section in sections:
                source = section.get("coverage_source") or section.get("chunk_type") or "unknown"
                sources[source] = sources.get(source, 0) + 1
                start = int(section.get("page_start", 0) or 0)
                end = int(section.get("page_end", start) or start)
                for page_no in range(start, end + 1):
                    indexed_pages.add(page_no)

            text_pages = {
                int(page.get("page", 0) or 0)
                for page in pages
                if (page.get("text") or "").strip()
            }
            page_coverage = len(indexed_pages & text_pages) / max(1, len(text_pages))
            char_ratio = indexed_chars / max(1, total_chars)
            documents.append({
                "source_file": doc.label,
                "document_type": doc.doc_type,
                "year_guess": doc.metadata.year_guess,
                "page_count": len(pages),
                "text_pages": len(text_pages),
                "indexed_pages": len(indexed_pages & text_pages),
                "page_coverage": round(page_coverage, 4),
                "total_chars": total_chars,
                "indexed_chars_with_overlap": indexed_chars,
                "indexed_char_ratio_with_overlap": round(char_ratio, 4),
                "section_count": len(sections),
                "coverage_sources": sources,
                "low_coverage": page_coverage < 0.85,
            })

        low = [item for item in documents if item["low_coverage"]]
        return {
            "document_count": len(documents),
            "low_coverage_count": len(low),
            "average_page_coverage": round(
                sum(item["page_coverage"] for item in documents) / max(1, len(documents)),
                4,
            ),
            "documents": documents,
        }

    def write(self, result: dict, output_dir: str, company: str, year: int) -> dict:
        os.makedirs(output_dir, exist_ok=True)
        json_path = os.path.join(output_dir, f"{company}_{year}_index_coverage.json")
        md_path = os.path.join(output_dir, f"{company}_{year}_index_coverage.md")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(self.to_markdown(result))
        return {"json_path": json_path, "md_path": md_path}

    def to_markdown(self, result: dict) -> str:
        lines = [
            "# Index Coverage Audit",
            "",
            f"- Documents: {result.get('document_count', 0)}",
            f"- Low coverage documents: {result.get('low_coverage_count', 0)}",
            f"- Average page coverage: {result.get('average_page_coverage', 0) * 100:.1f}%",
            "",
            "| Document | Type | Year | Pages indexed | Coverage | Sections |",
            "|---|---|---:|---:|---:|---:|",
        ]
        for item in result.get("documents", []):
            lines.append(
                "| {source_file} | {document_type} | {year} | {indexed_pages}/{text_pages} | {coverage:.1f}% | {section_count} |".format(
                    source_file=item.get("source_file", ""),
                    document_type=item.get("document_type", ""),
                    year=item.get("year_guess") or "",
                    indexed_pages=item.get("indexed_pages", 0),
                    text_pages=item.get("text_pages", 0),
                    coverage=float(item.get("page_coverage", 0.0) or 0.0) * 100,
                    section_count=item.get("section_count", 0),
                )
            )
        return "\n".join(lines) + "\n"
