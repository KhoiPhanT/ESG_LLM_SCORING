"""
Concise source reference formatting for reports and advisors.
"""
from __future__ import annotations

import os
from typing import Iterable


def format_short_source(source: dict | None) -> str:
    if not isinstance(source, dict):
        return ""

    raw_file = (
        source.get("source_file")
        or source.get("file_name")
        or source.get("document")
        or source.get("source_path")
        or ""
    )
    file_name = os.path.basename(str(raw_file)) if raw_file else ""
    if not file_name:
        return ""

    start = _as_int(source.get("page_start", source.get("page")))
    end = _as_int(source.get("page_end", source.get("page")))
    if start is None and end is None:
        return file_name
    if start is None:
        start = end
    if end is None:
        end = start
    if start == end:
        return f"{file_name} p.{start}"
    return f"{file_name} pp.{start}-{end}"


def format_source_list(sources: Iterable[dict] | None, limit: int = 3) -> list[str]:
    refs = []
    seen = set()
    for source in sources or []:
        ref = format_short_source(source)
        if not ref or ref in seen:
            continue
        seen.add(ref)
        refs.append(ref)
        if len(refs) >= limit:
            break
    return refs


def _as_int(value) -> int | None:
    if value in {"", None, "?"}:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
