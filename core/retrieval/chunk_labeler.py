"""
Annotate retrieval chunks with evidence and noise labels.
Supports heuristic labels with optional manual overrides.
"""
from __future__ import annotations

import json
import os
import re


class ChunkLabeler:
    def __init__(self, override_path: str = "refactor/chunk_label_overrides.json"):
        self.override_path = override_path
        self.overrides = self._load_overrides()

    def annotate(self, chunk: dict, doc) -> dict:
        item = dict(chunk)
        labels = set(item.get("labels", []))
        title = str(item.get("section_title", "") or "").lower()
        content = str(item.get("content", "") or "").lower()
        chunk_type = str(item.get("chunk_type", "") or "")
        table_family = str(item.get("table_family", "") or "")
        page_start = int(item.get("page_start", 0) or 0)
        page_end = int(item.get("page_end", 0) or 0)
        page_span = max(1, page_end - page_start + 1)
        doc_type = str(item.get("document_type", getattr(doc, "doc_type", "")) or "")

        if page_start <= 2:
            labels.add("front_matter")
        if any(token in title + "\n" + content for token in ["mục lục", "table of contents", "nội dung"]):
            labels.update({"toc_like", "low_value"})
        if chunk_type != "table_section" and self._looks_like_heading_inventory(content):
            labels.update({"toc_like", "low_value", "no_evidence_hint"})
        if any(token in title + "\n" + content for token in ["kpmg", "nguyen hue street", "sun wah tower", "điện thoại", "dien thoai", "fax"]):
            labels.update({"contact_boilerplate", "low_value"})
        if doc_type == "financial_report" and chunk_type == "section" and page_span >= 20:
            labels.add("broad_financial_disclosure")
            labels.add("no_evidence_hint")
        if chunk_type == "table_section" and table_family == "general_table" and not self._has_metric_anchor(content):
            labels.add("generic_table")
        if self._has_performance_anchor(title + "\n" + content):
            labels.add("performance_evidence")
        if self._has_social_anchor(title + "\n" + content):
            labels.add("workforce_evidence")
        if self._has_environment_anchor(title + "\n" + content):
            labels.add("environment_evidence")
        if self._looks_cross_topic_mixed(content):
            labels.add("mixed_metric_chunk")

        override = self.overrides.get(item.get("chunk_id", ""))
        if override:
            labels.update(override.get("add_labels", []))
            for label in override.get("remove_labels", []):
                labels.discard(label)

        item["labels"] = sorted(labels)
        item["is_low_value"] = bool({"low_value", "front_matter", "toc_like", "contact_boilerplate"} & labels)
        item["is_no_evidence_hint"] = bool({"no_evidence_hint", "generic_table"} & labels)
        item["evidence_signal"] = self._evidence_signal(labels)
        return item

    def _load_overrides(self) -> dict:
        if not os.path.exists(self.override_path):
            return {}
        with open(self.override_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        return {item["chunk_id"]: item for item in payload.get("overrides", []) if item.get("chunk_id")}

    def _has_metric_anchor(self, text: str) -> bool:
        return any(token in text for token in [
            "kwh", "joules", "nước", "nuoc", "nước thải", "nuoc thai", "phát thải", "phat thai",
            "co2", "scope 1", "scope 2", "scope 3", "giờ đào tạo", "dao tao", "thu nhập bình quân",
            "thu nhap binh quan", "tuyển dụng", "tuyen dung", "nghỉ việc", "nghi viec",
        ])

    def _has_environment_anchor(self, text: str) -> bool:
        return any(token in text for token in [
            "năng lượng", "nang luong", "điện năng", "dien nang", "nước", "nuoc", "phát thải", "phat thai",
            "khí nhà kính", "khi nha kinh", "chất thải", "chat thai", "đa dạng sinh học", "da dang sinh hoc",
            "nhựa", "nhua", "giấy", "giay", "mực in", "muc in",
        ])

    def _has_social_anchor(self, text: str) -> bool:
        return any(token in text for token in [
            "nhân viên", "nhan vien", "quản lý", "quan ly", "thu nhập bình quân", "thu nhap binh quan",
            "giới tính", "gioi tinh", "độ tuổi", "do tuoi", "đào tạo", "dao tao", "tai nạn lao động",
            "tai nan lao dong", "tuyển dụng", "tuyen dung", "nghỉ việc", "nghi viec",
        ])

    def _has_performance_anchor(self, text: str) -> bool:
        return self._has_environment_anchor(text) or self._has_social_anchor(text)

    def _looks_cross_topic_mixed(self, text: str) -> bool:
        groups = 0
        if self._has_environment_anchor(text):
            groups += 1
        if self._has_social_anchor(text):
            groups += 1
        return groups >= 2

    def _looks_like_heading_inventory(self, text: str) -> bool:
        heading_markers = len(re.findall(r"\bchuong\b|\bchương\b|\b\d+\.\d+\b", text))
        return heading_markers >= 8

    def _evidence_signal(self, labels: set[str]) -> float:
        score = 0.0
        if "performance_evidence" in labels:
            score += 0.5
        if "environment_evidence" in labels or "workforce_evidence" in labels:
            score += 0.25
        if "mixed_metric_chunk" in labels:
            score += 0.1
        if "generic_table" in labels:
            score -= 0.15
        if "broad_financial_disclosure" in labels:
            score -= 0.25
        if "front_matter" in labels or "low_value" in labels:
            score -= 0.3
        return round(max(-1.0, min(1.0, score)), 3)
