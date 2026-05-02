"""
Offline retrieval metadata for VNSI questions.

This module turns workbook rules into deterministic retrieval guardrails.  The
runtime planner may add aliases/queries, but these records keep every question
anchored to the evidence shape required by the rubric.
"""
from __future__ import annotations

import json
import os
import re

from core.cache import CacheManager
from core.normalization.text_normalizer import TextNormalizer


class QuestionRetrievalMetadataBuilder:
    CACHE_SCHEMA = 4
    VALID_DOC_TYPES = {
        "policy_document",
        "annual_report",
        "sustainability_report",
        "financial_report",
        "resolution",
        "other",
    }
    VALID_YEAR_POLICIES = {
        "current_year_required",
        "historical_allowed",
        "latest_valid_allowed",
        "future_target_allowed",
        "cross_year_reference",
        "unspecified",
    }
    VALID_EVIDENCE_SHAPES = {
        "policy_text",
        "narrative_text",
        "metric_table",
        "financial_table",
        "governance_profile",
        "meeting_resolution",
        "certificate",
        "numeric_value",
        "mixed",
    }

    NEGATIVE_HINTS = [
        "không có",
        "không công bố",
        "không thực hiện",
        "không áp dụng",
        "chưa",
    ]
    POSITIVE_NEGATION_PHRASES = [
        "không ngừng",
        "không phân biệt",
        "không làm tổn hại",
        "không gây",
    ]

    def __init__(self, target_year: int = 2024):
        self.target_year = target_year
        self.normalizer = TextNormalizer()

    def build_all(self, rules: list[dict]) -> dict:
        records = {}
        for rule in rules or []:
            qid = str(rule.get("id", "") or "").strip()
            if not qid:
                continue
            records[qid] = self.build(rule)
        return {
            "schema_version": self.CACHE_SCHEMA,
            "target_year": self.target_year,
            "question_count": len(records),
            "metadata": records,
        }

    def write(self, rules: list[dict], output_path: str) -> dict:
        fingerprint = CacheManager.hash_json({
            "schema_version": self.CACHE_SCHEMA,
            "target_year": self.target_year,
            "rules": rules or [],
        })
        cache_manager = CacheManager(run_key="question_metadata")
        if not CacheManager.is_forced("metadata"):
            cached = CacheManager.load_json(output_path)
            if (
                isinstance(cached, dict)
                and cached.get("schema_version") == self.CACHE_SCHEMA
                and cached.get("input_fingerprint") == fingerprint
                and isinstance(cached.get("metadata"), dict)
            ):
                cache_manager.record(
                    "question_metadata",
                    "hit",
                    self.CACHE_SCHEMA,
                    fingerprint,
                    path=output_path,
                )
                return cached
        payload = self.build_all(rules)
        payload["input_fingerprint"] = fingerprint
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        CacheManager.atomic_write_json(output_path, payload, indent=2)
        cache_manager.record(
            "question_metadata",
            "rebuilt",
            self.CACHE_SCHEMA,
            fingerprint,
            path=output_path,
            reason="forced_rebuild" if CacheManager.is_forced("metadata") else "missing_or_stale_cache",
        )
        return payload

    def build(self, rule: dict) -> dict:
        qid = str(rule.get("id", "") or "")
        qtype = str(rule.get("question_type", "default") or "default")
        time_policy = str(rule.get("time_policy", "unspecified") or "unspecified")
        text = self._norm(" ".join([
            str(rule.get("question", "") or ""),
            str(rule.get("options", "") or ""),
            str(rule.get("logic", "") or ""),
        ]))
        options = self._parse_options(rule.get("options", ""))
        positive_options = {
            letter: body
            for letter, body in options.items()
            if not self._is_negative_option(body)
        }

        strategy = self._strategy(rule, text)
        evidence_shape = self._evidence_shape(rule, strategy, text)
        evidence_profile = self._evidence_profile(rule, strategy, text)
        required_doc_types = self._doc_types(rule, strategy, text)
        aliases = self._aliases(rule, strategy, text)
        must_have = self._must_have(rule, strategy, text, positive_options)
        avoid_terms = self._avoid_terms(rule, strategy, text)
        option_focus = self._option_focus(rule, positive_options, strategy)
        search_queries = self._search_queries(rule, strategy, text, option_focus, must_have)
        isolated_option_queries = self._isolated_option_queries(rule, strategy, option_focus)
        option_evidence_requirements = self._option_evidence_requirements(rule, positive_options, strategy)
        option_polarity = {
            letter: "positive" for letter in positive_options
        }
        negative_options = [
            letter for letter, body in options.items() if self._is_negative_option(body)
        ]
        year_policy = time_policy if time_policy in self.VALID_YEAR_POLICIES else "unspecified"
        if strategy in {"metric_table", "ratio_with_revenue"} and year_policy not in {
            "historical_allowed",
            "latest_valid_allowed",
            "future_target_allowed",
        }:
            year_policy = "current_year_required"
        elif year_policy == "unspecified" and strategy in {
            "csr_narrative",
            "governance_agm_resolution",
            "governance_compensation",
            "governance_board_profile",
            "governance_committee",
            "governance_related_party",
            "governance_audit_control",
            "governance_esg",
            "governance_narrative",
        }:
            year_policy = "current_year_required"

        return {
            "question_id": qid,
            "strategy": strategy,
            "search_queries": self._dedupe(search_queries)[:10],
            "semantic_aliases": self._dedupe(aliases)[:18],
            "required_doc_types": [doc for doc in self._dedupe(required_doc_types) if doc in self.VALID_DOC_TYPES][:6],
            "must_have_terms": self._dedupe(must_have)[:18],
            "avoid_terms": self._dedupe(avoid_terms)[:12],
            "year_policy": year_policy,
            "evidence_shape": evidence_shape,
            "evidence_profile": evidence_profile,
            "option_focus": option_focus,
            "isolated_option_queries": isolated_option_queries,
            "option_evidence_requirements": option_evidence_requirements,
            "option_polarity": option_polarity,
            "negative_options": negative_options,
            "coverage_notes": self._coverage_notes(rule, strategy),
        }

    def validate(self, payload: dict, rules: list[dict]) -> dict:
        metadata = payload.get("metadata", {}) if isinstance(payload, dict) else {}
        rule_ids = [str(rule.get("id", "") or "") for rule in rules or [] if rule.get("id")]
        missing = [qid for qid in rule_ids if qid not in metadata]
        invalid = []
        multi_missing = []
        numeric_missing = []
        governance_missing = []

        for rule in rules or []:
            qid = str(rule.get("id", "") or "")
            record = metadata.get(qid, {})
            if not record:
                continue
            for field in ["strategy", "search_queries", "required_doc_types", "evidence_shape"]:
                if not record.get(field):
                    invalid.append({"question_id": qid, "field": field})
            if record.get("evidence_shape") not in self.VALID_EVIDENCE_SHAPES:
                invalid.append({"question_id": qid, "field": "evidence_shape"})
            if rule.get("is_multi_select"):
                positives = [
                    letter
                    for letter, body in self._parse_options(rule.get("options", "")).items()
                    if not self._is_negative_option(body)
                ]
                focus = record.get("option_focus") or {}
                isolated = record.get("isolated_option_queries") or {}
                missing_options = [
                    letter for letter in positives
                    if not focus.get(letter) or not isolated.get(letter)
                ]
                if missing_options:
                    multi_missing.append({"question_id": qid, "missing_options": missing_options})
            if rule.get("question_type") in {"numeric_disclosure", "ratio_calculation"}:
                anchors = " ".join(record.get("must_have_terms", []) + record.get("search_queries", [])).lower()
                if not any(token in anchors for token in ["số liệu", "bảng", "tỷ lệ", "kwh", "mj", "m3", "vnd", "doanh thu", "scope"]):
                    numeric_missing.append(qid)
            if str(rule.get("id", "")).startswith("G.") and not str(record.get("strategy", "")).startswith("governance"):
                governance_missing.append(qid)

        passed = not (missing or invalid or multi_missing or numeric_missing or governance_missing)
        return {
            "passed": passed,
            "question_count": len(rule_ids),
            "metadata_count": len(metadata),
            "missing": missing,
            "invalid": invalid,
            "multi_missing": multi_missing,
            "numeric_missing": numeric_missing,
            "governance_missing": governance_missing,
        }

    def _strategy(self, rule: dict, text: str) -> str:
        qid = str(rule.get("id", "") or "")
        qtype = str(rule.get("question_type", "default") or "default")
        if rule.get("is_multi_select"):
            return "multi_option"
        if qtype == "ratio_calculation":
            return "ratio_with_revenue" if "doanh thu" in text or "revenue" in text else "metric_table"
        if qtype == "numeric_disclosure":
            if "doanh thu" in text or "revenue" in text:
                return "ratio_with_revenue"
            return "metric_table"
        if qid.startswith("G."):
            return self._governance_strategy(qid, text)
        if qtype == "policy":
            if any(token in text for token in ["iso", "sa8000", "chung nhan", "chung nhan"]):
                return "certificate"
            return "policy_document"
        if any(token in text for token in ["cong dong", "thien nguyen", "csr", "trach nhiem xa hoi"]):
            return "csr_narrative"
        return "narrative_disclosure"

    def _governance_strategy(self, qid: str, text: str) -> str:
        if any(token in text for token in ["dhdcd", "dai hoi dong co dong", "co dong", "bien ban", "thu moi", "kiem phieu", "co tuc"]):
            return "governance_agm_resolution"
        if any(token in text for token in ["thu lao", "luong", "thuong", "loi ich", "chi phi"]):
            return "governance_compensation"
        if any(token in text for token in ["kinh nghiem", "hoc van", "chuyen mon", "da dang", "can bang gioi", "doc lap"]):
            return "governance_board_profile"
        if any(token in text for token in ["uy ban", "kiem toan noi bo", "quan ly rui ro"]):
            return "governance_committee"
        if any(token in text for token in ["giao dich ben lien quan", "bo phieu"]):
            return "governance_related_party"
        if any(token in text for token in ["kiem toan", "quy tac dao duc", "ung xu", "tuan thu"]):
            return "governance_audit_control"
        if any(token in text for token in ["esg", "phat trien ben vung", "moi truong va xa hoi"]):
            return "governance_esg"
        return "governance_narrative"

    def _evidence_shape(self, rule: dict, strategy: str, text: str) -> str:
        if strategy == "ratio_with_revenue":
            return "metric_table"
        if strategy == "metric_table":
            return "metric_table"
        if strategy == "certificate":
            return "certificate"
        if strategy == "policy_document":
            return "policy_text"
        if strategy == "governance_board_profile":
            return "governance_profile"
        if strategy in {"governance_agm_resolution", "governance_committee"}:
            return "meeting_resolution"
        if strategy == "governance_compensation":
            return "financial_table"
        if strategy.startswith("governance"):
            return "narrative_text"
        if strategy == "csr_narrative":
            return "narrative_text"
        return "mixed"

    def _evidence_profile(self, rule: dict, strategy: str, text: str) -> str:
        if strategy == "ratio_with_revenue":
            return "ratio_with_revenue"
        if strategy == "metric_table":
            return "metric_disclosure"
        if strategy in {"policy_document", "certificate"}:
            return "policy_public"
        if strategy.startswith("governance"):
            return "governance_narrative"
        return "narrative"

    def _doc_types(self, rule: dict, strategy: str, text: str) -> list[str]:
        if strategy == "ratio_with_revenue":
            return ["sustainability_report", "annual_report", "financial_report"]
        if strategy == "metric_table":
            return ["sustainability_report", "annual_report", "financial_report"]
        if strategy == "certificate":
            return ["policy_document", "sustainability_report", "annual_report", "other"]
        if strategy == "policy_document":
            return ["policy_document", "sustainability_report", "annual_report", "financial_report"]
        if strategy == "governance_agm_resolution":
            return ["resolution", "annual_report", "other"]
        if strategy.startswith("governance"):
            return ["annual_report", "resolution", "financial_report", "other"]
        if strategy == "csr_narrative":
            return ["sustainability_report", "annual_report", "other"]
        if str(rule.get("pillar", "")) == "E":
            return ["sustainability_report", "annual_report", "policy_document", "financial_report"]
        if str(rule.get("pillar", "")) == "S":
            return ["annual_report", "sustainability_report", "policy_document", "other"]
        return ["annual_report", "sustainability_report", "financial_report", "other"]

    def _aliases(self, rule: dict, strategy: str, text: str) -> list[str]:
        aliases = []
        topic_aliases = {
            "moi truong": ["môi trường", "quản lý môi trường", "phát triển bền vững", "ISO 14001"],
            "nang luong": ["năng lượng", "điện năng", "kWh", "MJ", "tiết kiệm năng lượng"],
            "nuoc": ["nước", "nước cấp", "nước thải", "m3", "tuần hoàn nước"],
            "phat thai": ["phát thải", "khí nhà kính", "CO2", "scope 1", "scope 2", "scope 3"],
            "chat thai": ["chất thải", "chất thải nguy hại", "không nguy hại", "tái chế"],
            "cong dong": ["cộng đồng", "thiện nguyện", "đóng góp cộng đồng", "quỹ sữa", "bão Yagi"],
            "nguoi lao dong": ["người lao động", "nhân viên", "phúc lợi", "đào tạo", "an toàn lao động"],
            "khach hang": ["khách hàng", "người tiêu dùng", "bảo mật thông tin", "an toàn sức khỏe"],
            "nha cung cap": ["nhà cung cấp", "chuỗi cung ứng", "đánh giá nhà cung cấp"],
        }
        for key, values in topic_aliases.items():
            if key in text:
                aliases.extend(values)
        if strategy.startswith("governance"):
            aliases.extend(["HĐQT", "hội đồng quản trị", "ĐHĐCĐ", "quản trị công ty", "báo cáo quản trị"])
        if strategy == "governance_compensation":
            aliases.extend(["thù lao từng thành viên", "lương thưởng", "ban điều hành", "tổng giám đốc"])
        if strategy == "governance_board_profile":
            aliases.extend(["học vấn", "kinh nghiệm", "trình độ chuyên môn", "thành viên độc lập", "cân bằng giới"])
        if strategy == "csr_narrative":
            aliases.extend(["hoạt động cộng đồng", "trách nhiệm xã hội", "lan tỏa giá trị", "gắn kết yêu thương"])
        return aliases

    def _must_have(self, rule: dict, strategy: str, text: str, positive_options: dict[str, str]) -> list[str]:
        terms = []
        question = str(rule.get("question", "") or "")
        terms.extend(self._significant_terms(question)[:8])
        if strategy == "ratio_with_revenue":
            terms.extend(["số liệu", "bảng", "doanh thu", "tổng doanh thu", "doanh thu thuần"])
            terms.extend(self._metric_family_terms(text))
        elif strategy == "metric_table":
            terms.extend(["số liệu", "bảng", "tỷ lệ", "đơn vị"])
            terms.extend(self._metric_family_terms(text))
        elif strategy == "certificate":
            terms.extend(["chứng nhận", "hệ thống quản lý"])
            if str(rule.get("pillar", "")) == "E" or "moi truong" in text:
                terms.extend(["ISO 14001", "EMS", "môi trường"])
            if str(rule.get("pillar", "")) == "S" or "xa hoi" in text:
                terms.extend(["ISO 26000", "SA8000", "trách nhiệm xã hội"])
        elif strategy == "policy_document":
            terms.extend(["chính sách", "cam kết", "quy trình", "công bố"])
        elif strategy == "governance_compensation":
            terms.extend(["thù lao từng thành viên", "lương thưởng", "HĐQT", "Ban điều hành"])
        elif strategy == "governance_board_profile":
            terms.extend(["học vấn", "kinh nghiệm", "trình độ chuyên môn", "tiểu sử HĐQT"])
        elif strategy.startswith("governance"):
            terms.extend(["hội đồng quản trị", "báo cáo thường niên", "quản trị công ty"])
        for body in positive_options.values():
            terms.extend(self._significant_terms(body)[:4])
        return terms

    def _avoid_terms(self, rule: dict, strategy: str, text: str) -> list[str]:
        terms = []
        if str(rule.get("time_policy", "")) != "future_target_allowed":
            terms.extend(["2050", "2060", "net zero", "trung hòa carbon"])
        if strategy in {"metric_table", "ratio_with_revenue"}:
            terms.extend(["mục tiêu", "cam kết tương lai"])
        return terms

    def _option_focus(self, rule: dict, positive_options: dict[str, str], strategy: str) -> dict:
        focus = {}
        core = self._core_action(rule, strategy)
        for letter, body in positive_options.items():
            target = self._option_query_target([body])
            terms = [
                term for term in self._significant_terms(body)
                if " " in str(term).strip()
                and len(self._norm(term)) >= 5
                and self._norm(term) not in {"trong", "duoc", "chinh", "sach", "cong", "ty"}
            ]
            anchors = [core]
            if target:
                anchors.append(target)
            elif body:
                anchors.append(body)
            anchors.extend(terms[:5])
            focus[letter] = self._dedupe(anchors)[:8]
        return focus

    def _search_queries(self, rule: dict, strategy: str, text: str, option_focus: dict, must_have: list[str]) -> list[str]:
        question = str(rule.get("question", "") or "").strip()
        core = self._core_action(rule, strategy)
        queries = []
        if strategy == "multi_option":
            for terms in option_focus.values():
                target = self._option_query_target(terms)
                if not target:
                    continue
                queries.append(f"{core} {target}")
            queries.append(question)
            return queries
        if strategy == "ratio_with_revenue":
            metric_queries = self._metric_family_queries(text)
            queries.extend(metric_queries)
            queries.append(" ".join(self._dedupe(must_have + ["doanh thu", str(self.target_year)])[:10]))
            queries.append("doanh thu thuần tổng doanh thu báo cáo tài chính")
        elif strategy == "metric_table":
            metric_queries = self._metric_family_queries(text)
            queries.extend(metric_queries)
            queries.append(" ".join(self._dedupe(must_have + ["số liệu", "bảng", str(self.target_year)])[:10]))
        elif strategy.startswith("governance"):
            queries.append(" ".join(self._dedupe(must_have + ["HĐQT", "báo cáo quản trị"])[:10]))
        elif strategy == "csr_narrative":
            queries.append("hoạt động cộng đồng thiện nguyện đóng góp cộng đồng quỹ sữa")
            queries.append("trách nhiệm xã hội cộng đồng địa phương trong năm")
        elif strategy in {"policy_document", "certificate"}:
            queries.append(" ".join(self._dedupe(must_have + ["chính sách", "còn hiệu lực"])[:10]))
        queries.append(question)
        return queries

    def _isolated_option_queries(self, rule: dict, strategy: str, option_focus: dict) -> dict:
        if strategy != "multi_option" or not isinstance(option_focus, dict):
            return {}
        core = self._core_action(rule, strategy)
        isolated = {}
        for letter, terms in option_focus.items():
            target = self._option_query_target(terms)
            if not target:
                continue
            queries = [
                f"{core} {target}",
                target,
            ]
            if core == "chính sách":
                queries.append(f"chính sách môi trường {target}")
            isolated[str(letter).strip().upper()] = self._dedupe([q for q in queries if q])[:3]
        return isolated

    def _option_evidence_requirements(self, rule: dict, positive_options: dict[str, str], strategy: str) -> dict:
        if strategy != "multi_option":
            return {}
        requirements = {}
        for letter, body in positive_options.items():
            text = self._norm(body)
            if "hoi dong quan tri" in text or "hdqt" in text or "phe duyet" in text:
                req = "approval"
            elif "tuan thu" in text or "phap luat" in text:
                req = "legal_commitment"
            elif "cach thuc" in text or "bien phap" in text or "quan ly" in text:
                req = "management_measure"
            elif any(token in text for token in ["so lieu", "ty le", "tong", "luong", "phat thai", "nuoc", "nang luong", "chat thai"]):
                req = "metric_scope"
            else:
                req = "policy_topic"
            requirements[str(letter).strip().upper()] = req
        return requirements

    def _option_query_target(self, terms: list[str]) -> str:
        cleaned = []
        generic = {
            "các hình thức khác",
            "hình thức khác",
            "phúc lợi khác",
            "khác",
        }
        candidates = terms[1:] if len(terms) > 1 else terms
        if candidates:
            first = str(candidates[0]).strip(" ;,.")
            normalized_first = self._norm(first)
            phrase_overrides = [
                (["nha cung cap"], "quản lý môi trường nhà cung cấp"),
                (["tuan thu", "phap luat"], "tuân thủ pháp luật môi trường"),
                (["hoi dong quan tri", "phe duyet"], "Hội đồng quản trị phê duyệt"),
                (["cai thien", "khong ngung"], "cải thiện không ngừng hiệu suất môi trường"),
                (["cach thuc", "quan ly"], "cách thức quản lý biện pháp bảo vệ môi trường"),
                (["bien phap", "tai nguyen"], "biện pháp sử dụng tài nguyên bảo vệ môi trường"),
            ]
            for required, replacement in phrase_overrides:
                if all(token in normalized_first for token in required):
                    return replacement
            if first and self._norm(first) not in {self._norm(item) for item in generic} and len(first) <= 70:
                return first
        for term in candidates:
            text = str(term).strip(" ;,.")
            if not text:
                continue
            if self._norm(text) in {self._norm(item) for item in generic}:
                continue
            if len(text) > 55:
                sig = self._significant_terms(text)
                text = " ".join(sig[:5])
            cleaned.append(text)
            if len(" ".join(cleaned)) >= 45:
                break
        result = " ".join(cleaned[:4]).strip()
        if result:
            return result
        if candidates:
            return str(candidates[0]).strip(" ;,.")
        return ""

    def _core_action(self, rule: dict, strategy: str) -> str:
        question = self._norm(str(rule.get("question", "") or ""))
        if "danh gia rui ro" in question:
            return "đánh giá rủi ro"
        if "chinh sach" in question or "cam ket" in question:
            return "chính sách"
        if "quy trinh" in question:
            return "quy trình"
        if "phuc loi" in question:
            return "phúc lợi người lao động"
        if "hai long" in question:
            return "khảo sát hài lòng người lao động"
        if "trach nhiem xa hoi" in question or strategy == "csr_narrative":
            return "hoạt động trách nhiệm xã hội cộng đồng"
        if strategy.startswith("governance"):
            return "quản trị công ty"
        if strategy in {"metric_table", "ratio_with_revenue"}:
            return "số liệu chỉ tiêu"
        return "bằng chứng công bố"

    def _metric_family_terms(self, text: str) -> list[str]:
        terms = []
        if "nang luong" in text or "kwh" in text or "joules" in text or "mj" in text:
            terms.extend(["tổng năng lượng tiêu thụ", "năng lượng", "MJ", "kWh", "điện EVN"])
        if "nuoc" in text or "m3" in text:
            terms.extend(["tài nguyên nước", "tổng lượng nước", "nước", "m3"])
        if "phat thai" in text or "co2" in text or "scope" in text or "khi nha kinh" in text:
            terms.extend(["phát thải CO2", "khí nhà kính", "Scope 1", "Scope 2", "tCO2"])
        if "chat thai" in text or "rac" in text or "tai che" in text:
            terms.extend(["tổng lượng chất thải", "rác công nghiệp", "chất thải nguy hại", "tái sử dụng tái chế"])
        if "nguyen vat lieu" in text or "bao bi" in text:
            terms.extend(["nguyên vật liệu", "bao bì", "vật liệu tái chế"])
        if "thien nguyen" in text or "cong dong" in text:
            terms.extend(["đóng góp cộng đồng", "thiện nguyện", "quỹ sữa", "tỷ đồng"])
        if "nhan vien" in text or "nguoi lao dong" in text:
            terms.extend(["nhân viên", "người lao động", "tổng số lao động"])
        return terms

    def _metric_family_queries(self, text: str) -> list[str]:
        queries = []
        if "nang luong" in text or "kwh" in text or "joules" in text or "mj" in text:
            queries.append(f"tổng năng lượng tiêu thụ MJ kWh {self.target_year}")
            queries.append(f"năng lượng Đơn vị 2023 {self.target_year} điện EVN điện mặt trời")
        if "nuoc" in text or "m3" in text:
            queries.append(f"tài nguyên nước tổng lượng nước m3 {self.target_year}")
            queries.append(f"nước Đơn vị 2023 {self.target_year} tổng lượng nước")
        if "phat thai" in text or "co2" in text or "scope" in text or "khi nha kinh" in text:
            queries.append(f"phát thải CO2 scope 1 scope 2 tCO2 {self.target_year}")
        if "chat thai" in text or "rac" in text or "tai che" in text:
            queries.append(f"tổng lượng chất thải rác công nghiệp chất thải nguy hại {self.target_year}")
        if "thien nguyen" in text or "cong dong" in text:
            queries.append(f"đóng góp cộng đồng thiện nguyện quỹ sữa tỷ đồng {self.target_year}")
        return queries

    def _coverage_notes(self, rule: dict, strategy: str) -> str:
        if strategy == "multi_option":
            return "Each positive option has independent retrieval anchors."
        if strategy in {"metric_table", "ratio_with_revenue"}:
            return "Requires quantitative disclosure; ratio questions include revenue anchors."
        if strategy.startswith("governance"):
            return "Governance evidence should come from annual report, governance report, AGM/resolution, or financial report depending on subtype."
        return "General metadata generated from question/rubric terms."

    def _parse_options(self, options: str) -> dict[str, str]:
        parsed = {}
        for _, letter, body in re.findall(r"(^|\n)\s*([A-Z])[\.\)]\s*([^\n]+)", str(options or "")):
            parsed[letter.strip().upper()] = body.strip()
        return parsed

    def _is_negative_option(self, body: str) -> bool:
        normalized = self._norm(body)
        if any(self._norm(phrase) in normalized for phrase in self.POSITIVE_NEGATION_PHRASES):
            return False
        negative_patterns = [
            r"^(khong|chua)\b",
            r"\bkhong\s+(co|cong bo|thuc hien|ap dung|duoc cong bo)\b",
            r"\bchua\s+(co|cong bo|thuc hien|ap dung)\b",
        ]
        if any(re.search(pattern, normalized) for pattern in negative_patterns):
            return True
        return any(self._norm(hint) in normalized for hint in self.NEGATIVE_HINTS)

    def _significant_terms(self, text: str) -> list[str]:
        stop = {
            "cong", "ty", "khong", "co", "cac", "cua", "cho", "voi", "trong", "nam",
            "duoc", "thuc", "hien", "lien", "quan", "cung", "cap", "cau", "hoi",
            "vui", "long", "biet", "neu", "tra", "loi", "phan", "noi", "dung",
            "bao", "gom", "mot", "nhieu", "ve", "va", "hoac", "theo", "tren",
        }
        terms = []
        for word in re.findall(r"[A-Za-zÀ-ỹà-ỹĐđ0-9%]{2,}", str(text or "")):
            norm = self._norm(word)
            if len(norm) < 2 or norm in stop:
                continue
            terms.append(word.strip(" ;,."))
        return self._dedupe(terms)

    def _norm(self, text: str) -> str:
        return self.normalizer.normalize_for_search(str(text or ""))

    def _dedupe(self, values: list[str]) -> list[str]:
        out = []
        seen = set()
        for value in values or []:
            text = str(value).strip()
            key = text.lower()
            if text and key not in seen:
                seen.add(key)
                out.append(text)
        return out
