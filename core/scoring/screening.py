"""
Screening Module — Đánh giá câu hỏi sàng lọc SL1-SL5 (Total Kill Switch).
Sử dụng LLM thật để phân tích báo cáo.
"""
import json
import os
import re

from core.retrieval.retrieval_engine import RetrievalEngine


class ScreeningModule:
    def __init__(self, rules_path="outputs/vnsi_rules.json", llm_client=None, corpus=None, industry_sector: str = "", target_year: int | None = None, retrieval_engine=None):
        self.llm_client = llm_client
        self.corpus = corpus
        self.screening_rules = self._load_rules(rules_path)
        if retrieval_engine is not None:
            self.retrieval_engine = retrieval_engine
        elif corpus:
            self.retrieval_engine = RetrievalEngine(corpus, industry_sector=industry_sector, target_year=target_year)
        else:
            self.retrieval_engine = None

    def _load_rules(self, path):
        if not os.path.exists(path):
            print(f"  [WARN] Không tìm thấy rules: {path}")
            return []
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f).get("screening", [])

    def evaluate(self, report_text):
        """
        Đánh giá 5 câu hỏi sàng lọc SL1-SL5.
        Returns: dict with penalties and details.
        """
        results = {
            "passed": True,
            "G_killed": False,
            "ES_killed": False,
            "direct_deductions": 0,
            "details": [],
        }

        if not self.screening_rules:
            return results

        print("\n  [SCREENING] Đang đánh giá 5 câu hỏi sàng lọc...")

        for rule in self.screening_rules:
            q_id = rule["id"]
            question = rule["question"]
            logic = rule.get("logic", "")
            context = report_text[:10000]
            source_sections = []
            if self.retrieval_engine:
                query_plan = self._screening_query_plan(rule)
                retrieval = self.retrieval_engine.retrieve_for_plan(rule, query_plan, top_k=8)
                source_sections = retrieval["candidates"]
                if source_sections:
                    context = "\n---\n".join(
                        self._format_section(section) for section in source_sections
                    )[:16000]

            if not self._has_screening_violation_signal(q_id, context):
                answer = "B"
                reason = "Không thấy bằng chứng rõ ràng về vi phạm trong nguồn được truy xuất."
            # Gọi LLM thật
            elif self.llm_client:
                answer_data = self.llm_client.ask_screening_question(
                    context=context,
                    question=question,
                    q_id=q_id,
                )
                answer = answer_data.get("answer", "B").strip().upper()
                reason = answer_data.get("reason", "")
            else:
                answer = "B"
                reason = "LLM không khả dụng, mặc định: Không vi phạm"

            # Xử lý logic
            violation = answer.startswith("A")
            penalty_text = ""

            if violation:
                results["passed"] = False
                if "Governace = 0" in logic or "Governance = 0" in logic:
                    results["G_killed"] = True
                    penalty_text = "⚠ ĐIỂM G = 0"
                elif "E/S = 0" in logic:
                    results["ES_killed"] = True
                    penalty_text = "⚠ ĐIỂM E/S = 0"
                elif "-1" in logic:
                    results["direct_deductions"] += 1
                    penalty_text = "⚠ Trừ điểm trực tiếp"

            status = "❌ CÓ VI PHẠM" if violation else "✅ Không vi phạm"
            detail = {
                "id": q_id,
                "question": question[:80] + "...",
                "answer": answer,
                "status": status,
                "reason": reason,
                "penalty": penalty_text,
                "source_sections": [
                    {
                        "source_file": section["source_file"],
                        "document_type": section["document_type"],
                        "page_start": section["page_start"],
                        "page_end": section["page_end"],
                        "score": section.get("score"),
                    }
                    for section in source_sections
                ],
            }
            results["details"].append(detail)
            print(f"    [{q_id}] {status} {penalty_text}")

        return results

    def _screening_query_plan(self, rule: dict) -> dict:
        q_id = str(rule.get("id", "") or "")
        question = str(rule.get("question", "") or "")
        base = {
            "search_queries": [question],
            "semantic_aliases": [],
            "required_doc_types": ["annual_report", "financial_report", "sustainability_report", "resolution", "other"],
            "must_have_terms": [],
            "avoid_terms": [],
            "year_policy": "current_year_required",
            "evidence_shape": "narrative_text",
            "evidence_profile": "screening_violation",
            "option_focus": {},
            "metadata_strategy": "screening",
        }
        profiles = {
            "SL1": {
                "search_queries": [
                    "giao dịch nội gián thành viên HĐQT ban điều hành nhân viên vi phạm",
                    "xử phạt giao dịch nội gián chứng khoán Vinamilk",
                ],
                "semantic_aliases": ["giao dịch nội gián", "insider trading", "xử phạt", "vi phạm chứng khoán"],
                "must_have_terms": ["giao dịch nội gián", "vi phạm", "xử phạt"],
            },
            "SL2": {
                "search_queries": [
                    "vi phạm môi trường lao động an toàn lao động quyền con người bị xử phạt",
                    "xử phạt cơ quan quản lý môi trường lao động Vinamilk",
                ],
                "semantic_aliases": ["vi phạm môi trường", "vi phạm lao động", "xử phạt", "an toàn lao động"],
                "must_have_terms": ["vi phạm", "xử phạt", "môi trường", "lao động"],
            },
            "SL3": {
                "search_queries": [
                    "tham nhũng hối lộ gian lận trốn thuế điều tra trong năm",
                    "vi phạm tham nhũng hối lộ gian lận trốn thuế Vinamilk",
                ],
                "semantic_aliases": ["tham nhũng", "hối lộ", "gian lận", "trốn thuế", "điều tra"],
                "must_have_terms": ["tham nhũng", "hối lộ", "gian lận", "trốn thuế"],
            },
            "SL4": {
                "search_queries": [
                    "không tuân thủ giao dịch bên liên quan trọng yếu vi phạm",
                    "giao dịch bên liên quan xử phạt không tuân thủ quy định",
                ],
                "semantic_aliases": ["giao dịch bên liên quan", "không tuân thủ", "trọng yếu", "xử phạt"],
                "must_have_terms": ["giao dịch bên liên quan", "không tuân thủ", "vi phạm"],
            },
            "SL5": {
                "search_queries": [
                    "ý kiến kiểm toán ngoại trừ báo cáo tài chính năm",
                    "qualified opinion ngoại trừ kiểm toán báo cáo tài chính",
                ],
                "semantic_aliases": ["ý kiến kiểm toán", "ngoại trừ", "qualified opinion", "báo cáo kiểm toán"],
                "required_doc_types": ["financial_report", "annual_report"],
                "must_have_terms": ["ý kiến kiểm toán", "ngoại trừ", "báo cáo tài chính"],
                "evidence_shape": "financial_table",
            },
        }
        base.update(profiles.get(q_id, {}))
        return base

    def _has_screening_violation_signal(self, q_id: str, context: str) -> bool:
        text = str(context or "").lower()
        # High precision gate: screening is a kill switch, so do not send
        # ordinary policy/controls/no-incident disclosures to the LLM as
        # potential violations.
        adverse_patterns = [
            r"bị\s+(?:xử phạt|phạt|kết luận|điều tra)",
            r"xử phạt\s+(?:.*\s)?(?:vi phạm|hành chính)",
            r"kết luận\s+(?:.*\s)?vi phạm",
            r"điều tra\s+(?:đang diễn ra|liên quan)",
            r"không tuân thủ\s+(?:quy định|pháp luật)",
            r"ý kiến kiểm toán ngoại trừ",
            r"qualified opinion",
        ]
        has_adverse = any(re.search(pattern, text) for pattern in adverse_patterns)
        if q_id == "SL1":
            return has_adverse and any(term in text for term in ["giao dịch nội gián", "insider trading"])
        if q_id == "SL2":
            return has_adverse and any(term in text for term in ["môi trường", "lao động", "an toàn lao động", "quyền con người", "lao động trẻ em"])
        if q_id == "SL3":
            return has_adverse and any(term in text for term in ["tham nhũng", "hối lộ", "gian lận", "trốn thuế"])
        if q_id == "SL4":
            return has_adverse and "giao dịch bên liên quan" in text
        if q_id == "SL5":
            return any(term in text for term in ["ý kiến kiểm toán ngoại trừ", "qualified opinion", "ngoại trừ"])
        return has_adverse

    def _keywords_from_question(self, question):
        keywords = []
        lowered = question.lower()
        for token in [
            "giao dịch nội gián",
            "tham nhũng",
            "hối lộ",
            "gian lận",
            "trốn thuế",
            "môi trường",
            "lao động",
            "giao dịch bên liên quan",
            "kiểm toán ngoại trừ",
            "báo cáo tài chính",
            "xử phạt",
        ]:
            if token in lowered:
                keywords.append(token)
        return keywords or ["kiểm toán", "vi phạm"]

    def _format_section(self, section):
        return (
            f"[DOC: {section['source_file']} | TYPE: {section['document_type']} | "
            f"PAGES: {section['page_start']}-{section['page_end']} | SCORE: {section.get('score', 0):.2f}]\n"
            f"{section['content']}"
        )
