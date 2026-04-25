"""
Screening Module — Đánh giá câu hỏi sàng lọc SL1-SL5 (Total Kill Switch).
Sử dụng LLM thật để phân tích báo cáo.
"""
import json
import os

from core.retrieval.retrieval_engine import RetrievalEngine


class ScreeningModule:
    def __init__(self, rules_path="outputs/vnsi_rules.json", llm_client=None, corpus=None):
        self.llm_client = llm_client
        self.corpus = corpus
        self.screening_rules = self._load_rules(rules_path)
        self.retrieval_engine = RetrievalEngine(corpus) if corpus else None

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
                retrieval = self.retrieval_engine.retrieve_for_rule(rule, top_k=5)
                source_sections = retrieval["candidates"]
                if source_sections:
                    context = "\n---\n".join(
                        self._format_section(section) for section in source_sections
                    )[:16000]

            # Gọi LLM thật
            if self.llm_client:
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
