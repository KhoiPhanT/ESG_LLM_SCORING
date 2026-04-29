"""
VNSI Scorer — Chấm điểm ESG bám sát workbook VNSI, ưu tiên bằng chứng.
Adaptive RAG: multi-query retrieval + iterative retry for low-confidence answers.
"""
import json
import os
import re

from core.evidence import EvidenceExtractor
from core.query_builder.query_decomposer import QueryDecomposer
from core.retrieval.retrieval_engine import RetrievalEngine
from core.resolution import AnswerResolver
from core.scoring.scoring_engine import ScoringEngine


class VNSIScorer:
    def __init__(
        self,
        rules_path="outputs/vnsi_rules.json",
        weights_path="outputs/industry_weights.json",
        structure_path="outputs/scoring_structure.json",
        llm_client=None,
        corpus=None,
        industry_sector: str = "",
    ):
        self.llm_client = llm_client
        self.corpus = corpus
        self.industry_sector = industry_sector
        self.scoring_rules = self._load_json(rules_path).get("scoring", [])
        self.industry_weights = self._load_json(weights_path)
        self.scoring_structure = self._load_json(structure_path)
        if isinstance(self.scoring_structure, list):
            self.scoring_structure = {"factor_max_scores": {}}
        self.factor_max_scores = self.scoring_structure.get("factor_max_scores", {})
        self.derived_factor_max_scores = self._derive_factor_max_scores()
        self.retrieval_engine = RetrievalEngine(corpus, industry_sector=industry_sector) if corpus else None
        self.query_decomposer = QueryDecomposer()
        self.evidence_extractor = EvidenceExtractor(llm_client=llm_client, enable_self_consistency=True)
        self.answer_resolver = AnswerResolver()
        self.scoring_engine = ScoringEngine()

    def _load_json(self, path):
        if not os.path.exists(path):
            return {}
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _fallback_weights(self, industry_sector):
        defaults = {
            "Financials": {"G": 0.6, "S": 0.3, "E": 0.1},
            "Information Technology": {"G": 0.45, "S": 0.45, "E": 0.1},
            "Energy": {"G": 0.3, "S": 0.2, "E": 0.5},
            "Materials": {"G": 0.3, "S": 0.2, "E": 0.5},
        }
        return defaults.get(industry_sector, {"G": 0.35, "S": 0.35, "E": 0.30})

    def _get_relevant_keywords(self, rule):
        question = rule.get("question", "")
        qid = rule.get("id", "")
        factor = rule.get("factor", "")
        keywords = []
        lowered = question.lower()
        for token in [
            "môi trường",
            "phát thải",
            "năng lượng",
            "nước",
            "chất thải",
            "khí nhà kính",
            "đa dạng sinh học",
            "lao động",
            "an toàn",
            "sức khỏe",
            "đào tạo",
            "nhân sự",
            "cộng đồng",
            "khách hàng",
            "nhà cung cấp",
            "bình đẳng",
            "hđqt",
            "cổ đông",
            "kiểm toán",
            "quản trị",
            "minh bạch",
            "đhđcđ",
            "nghị quyết",
            "rủi ro",
            "tuân thủ",
            "assurance",
        ]:
            if token in lowered:
                keywords.append(token)

        if qid.startswith("G.") and "đhđcđ" not in keywords:
            keywords.extend(["đhđcđ", "nghị quyết", "hội đồng quản trị", "cổ đông"])
        if factor == "E3":
            keywords.extend(["đơn vị", "doanh thu", "số liệu"])
        return list(dict.fromkeys(keywords)) or ["phát triển bền vững"]

    def _build_context(self, rule, report_text):
        """Phase 1 of Adaptive RAG: multi-query retrieval for broad coverage."""
        if not self.retrieval_engine:
            return {"context": report_text[:25000], "sections": [], "retrieval_meta": {}}

        # Decompose question into sub-queries
        decomposition = self.query_decomposer.decompose(rule)

        if decomposition.is_compound and len(decomposition.sub_queries) > 1:
            # Multi-query retrieval for compound questions
            retrieval = self.retrieval_engine.retrieve_multi_query(
                rule, decomposition.sub_queries, top_k=10
            )
        else:
            # Standard single-query retrieval
            retrieval = self.retrieval_engine.retrieve_for_rule(rule, top_k=10)

        sections = retrieval["candidates"]
        if sections:
            context = "\n---\n".join(self._format_section(section) for section in sections)[:30000]
        else:
            context = report_text[:25000]

        return {
            "context": context,
            "sections": sections,
            "retrieval_meta": {
                "sub_queries": decomposition.sub_queries,
                "is_compound": decomposition.is_compound,
                "total_candidates": retrieval.get("total_unique_candidates", len(sections)),
            },
        }

    def _adaptive_retry(self, rule, report_text, first_extraction, first_context_bundle):
        """
        Phase 2 of Adaptive RAG: when first pass has low confidence or
        ungrounded evidence, ask LLM what additional info is needed,
        then retrieve again with expanded context.
        """
        if not self.retrieval_engine or not self.llm_client:
            return None

        # Ask LLM: what info are you missing?
        question = rule.get("question", "")
        first_answer = first_extraction.get("answer", "NULL")
        first_evidence = first_extraction.get("raw_evidence", "")

        hint_prompt = f"""
You previously answered the following ESG question but your evidence was weak.
Question: {question}
Your answer: {first_answer}
Your evidence: {first_evidence or "None"}

What SPECIFIC information would help you answer more confidently?
Think step-by-step concisely, then respond with 1-2 short search queries in Vietnamese (one per line). Be specific.
Example: "chính sách quản lý nước thải" or "tỷ lệ phát thải CO2 năm 2024"

Search queries:"""

        messages = [{"role": "user", "content": hint_prompt}]
        raw_hint = self.llm_client._call(messages, temperature=0.1, max_tokens=1536)
        if not raw_hint:
            return None

        # Parse the hint into search queries
        hint_queries = [
            line.strip().strip('"').strip("'").strip("-").strip()
            for line in raw_hint.strip().split("\n")
            if line.strip() and len(line.strip()) > 5
        ][:3]  # Max 3 retry queries

        if not hint_queries:
            return None

        # Retrieve additional context using LLM's hints
        additional_sections = []
        for hint_query in hint_queries:
            new_candidates = self.retrieval_engine._retrieve_by_text(
                hint_query, rule, top_k=5
            )
            additional_sections.extend(new_candidates)

        if not additional_sections:
            return None

        # Deduplicate against existing sections
        existing_chunk_ids = {
            s.get("chunk_id") for s in first_context_bundle.get("sections", [])
        }
        new_sections = [
            s for s in additional_sections
            if s.get("chunk_id") not in existing_chunk_ids
        ]

        if not new_sections:
            return None

        # Build expanded context: original + new sections
        all_sections = first_context_bundle.get("sections", []) + new_sections
        expanded_context = "\n---\n".join(
            self._format_section(s) for s in all_sections
        )[:30000]

        expanded_bundle = {
            "context": expanded_context,
            "sections": all_sections,
            "retrieval_meta": {
                "is_retry": True,
                "retry_queries": hint_queries,
                "new_sections_added": len(new_sections),
            },
        }

        # Re-extract with expanded context (full pipeline including self-consistency)
        retry_extraction = self.evidence_extractor.extract(rule, expanded_bundle)
        return {
            "extraction": retry_extraction,
            "context_bundle": expanded_bundle,
            "hint_queries": hint_queries,
        }

    def _prerequisite_satisfied(self, rule, answer_registry):
        prerequisite = rule.get("prerequisite")
        if not prerequisite:
            return True, ""

        previous = answer_registry.get(prerequisite["question_id"]) or answer_registry.get(
            self._alternate_question_id(prerequisite["question_id"])
        )
        if not previous:
            return False, f"Thiếu kết quả câu điều kiện {prerequisite['question_id']}"

        if previous.get("answer") in prerequisite.get("allowed_answers", []):
            return True, ""
        return False, (
            f"Bỏ qua vì câu điều kiện {prerequisite['question_id']} trả lời "
            f"{previous.get('answer')} thay vì {','.join(prerequisite['allowed_answers'])}"
        )

    def _canonical_question_id(self, question_id):
        qid = str(question_id).upper().strip().replace("..", ".")
        if re.match(r"^[A-Z]\d", qid):
            qid = f"{qid[0]}.{qid[1:]}"
        return qid

    def _alternate_question_id(self, question_id):
        qid = self._canonical_question_id(question_id)
        return qid.replace(".", "", 1) if "." in qid else qid

    def _register_answer(self, registry, question_id, answer, selected_options):
        canonical = self._canonical_question_id(question_id)
        payload = {"answer": answer, "selected_options": selected_options}
        registry[canonical] = payload
        registry[self._alternate_question_id(canonical)] = payload

    def _derive_factor_max_scores(self):
        factor_max = {}
        for rule in self.scoring_rules:
            factor = rule.get("factor", "")
            if not factor:
                continue
            factor_max[factor] = round(factor_max.get(factor, 0.0) + float(rule.get("max_score", 0.0) or 0.0), 4)
        return factor_max

    def score_all_questions(self, report_text, industry_sector="Financials", company_name="Company"):
        if not self.scoring_rules:
            print("  [WARN] Không có scoring rules")
            return self._empty_result(industry_sector)

        weights = self.industry_weights.get(industry_sector, self._fallback_weights(industry_sector))
        print(
            f"\n  [SCORING] Ngành: {industry_sector} → "
            f"Trọng số: E={weights['E']:.0%}, S={weights['S']:.0%}, G={weights['G']:.0%}"
        )
        print(f"  [SCORING] Tổng số câu hỏi: {len(self.scoring_rules)}")

        factor_scores = {}
        all_details = []
        answer_registry = {}

        progress_file = f"outputs/cache/{company_name}_scoring_progress.json"
        if os.path.exists(progress_file):
            try:
                with open(progress_file, 'r', encoding='utf-8') as f:
                    progress = json.load(f)
                    factor_scores = progress.get("factor_scores", {})
                    all_details = progress.get("all_details", [])
                    answer_registry = progress.get("answer_registry", {})
                print(f"  [RESUME] Đã nạp lại kết quả của {len(all_details)} câu hỏi từ lần chạy trước.")
            except Exception as e:
                print(f"  [WARN] Không thể đọc file tiến độ: {e}")

        questions_processed_in_this_run = 0

        for i, rule in enumerate(self.scoring_rules):
            q_id = rule["id"]
            factor = rule.get("factor", "")
            
            canonical_qid = self._canonical_question_id(q_id)
            if canonical_qid in answer_registry:
                # Tìm answer đã lưu để in ra cho đẹp
                saved_ans = answer_registry[canonical_qid].get("answer", "SKIP")
                print(f"    [{i+1}/{len(self.scoring_rules)}] {q_id}... → {saved_ans} (Đã chấm)")
                continue

            print(f"    [{i+1}/{len(self.scoring_rules)}] {q_id}...", end=" ", flush=True)

            satisfied, prerequisite_note = self._prerequisite_satisfied(rule, answer_registry)
            if not satisfied:
                print("→ SKIP (điều kiện không thỏa)")
                detail = {
                    "id": q_id,
                    "factor": factor,
                    "pillar": rule.get("pillar", ""),
                    "sub_category": rule.get("sub_category", ""),
                    "question": rule["question"][:140],
                    "answer": "SKIP",
                    "selected_options": [],
                    "base_score": 0.0,
                    "score": 0.0,
                    "reason": prerequisite_note,
                    "evidence": None,
                    "evidence_present": False,
                }
                self._register_answer(answer_registry, q_id, "SKIP", [])
                all_details.append(detail)
                factor_scores.setdefault(factor, 0.0)
                continue

            # ═══ Phase 1: Multi-query retrieval + first LLM pass ═══
            context_bundle = self._build_context(rule, report_text)
            extraction = self.evidence_extractor.extract(rule, context_bundle)
            resolution = self.answer_resolver.resolve(rule, extraction)

            # ═══ Phase 2: Adaptive retry if needed ═══
            retry_used = False
            needs_retry = self._should_retry(extraction, resolution)
            if needs_retry:
                print("↻", end=" ", flush=True)
                retry_result = self._adaptive_retry(rule, report_text, extraction, context_bundle)
                if retry_result:
                    retry_used = True
                    extraction = retry_result["extraction"]
                    context_bundle = retry_result["context_bundle"]
                    resolution = self.answer_resolver.resolve(rule, extraction)

            score_result = self.scoring_engine.score_rule(rule, resolution)

            answer_letter = score_result["answer"]
            selected_options = score_result["selected_options"]
            base_score = score_result["base_score"]
            final_score = score_result["score"]
            evidence_present = score_result["evidence_present"]

            retry_tag = " [retry]" if retry_used else ""
            display_answer = ",".join(selected_options) if selected_options else answer_letter
            print(f"→ {display_answer} ({final_score:+.2f}){retry_tag}")
            
            # Print reason and evidence for better live feedback
            reason = resolution.get("reason", "")
            if reason:
                print(f"      Lý do: {reason}")
            
            evidence = extraction.get("raw_evidence")
            if evidence and evidence.upper() != "NULL":
                clean_ev = str(evidence).replace('\n', ' ').strip()
                print(f"      Bằng chứng: {clean_ev}")

            factor_scores[factor] = factor_scores.get(factor, 0.0) + final_score
            self._register_answer(answer_registry, q_id, answer_letter, selected_options)
            all_details.append({
                "id": q_id,
                "factor": factor,
                "pillar": rule.get("pillar", ""),
                "sub_category": rule.get("sub_category", ""),
                "question": rule["question"][:140],
                "answer": answer_letter,
                "selected_options": selected_options,
                "base_score": base_score,
                "score": final_score,
                "resolution_status": score_result["resolution_status"],
                "confidence": resolution["confidence"],
                "conflict_detected": resolution["conflict_detected"],
                "reason": resolution["reason"],
                "evidence": extraction.get("raw_evidence"),
                "evidence_items": resolution["evidence_items"],
                "evidence_present": evidence_present,
                "evidence_verification": extraction.get("evidence_verification"),
                "consistency_check": extraction.get("consistency_check"),
                "retry_used": retry_used,
                "retrieval_meta": context_bundle.get("retrieval_meta", {}),
                "source_sections": [
                    {
                        "source_file": section["source_file"],
                        "document_type": section["document_type"],
                        "page_start": section["page_start"],
                        "page_end": section["page_end"],
                        "matched_keywords": section.get("matched_keywords", []),
                    }
                    for section in context_bundle["sections"]
                ],
            })

            # Save progress after each question
            try:
                with open(progress_file, 'w', encoding='utf-8') as f:
                    json.dump({
                        "factor_scores": factor_scores,
                        "all_details": all_details,
                        "answer_registry": answer_registry
                    }, f, ensure_ascii=False, indent=2)
            except Exception:
                pass
            
            questions_processed_in_this_run += 1
            if questions_processed_in_this_run % 8 == 0:
                print(f"\n  [TẢN NHIỆT] Đã chạy xong {questions_processed_in_this_run} câu. Máy sẽ nghỉ 8 phút để hạ nhiệt...")
                print(f"  (Bạn có thể nhấn Ctrl+C để dừng hẳn, tiến trình đã được tự động lưu)")
                import time
                time.sleep(8 * 60)
                print("  [TẢN NHIỆT] Đã nghỉ xong, tiếp tục phân tích!\n")

        factor_percentages = {}
        pillar_groups = {"E": [], "S": [], "G": []}
        for factor, raw_score in factor_scores.items():
            max_info = self.factor_max_scores.get(factor, {})
            summary_max = float(max_info.get("max_points", 0.0) or 0.0)
            derived_max = float(self.derived_factor_max_scores.get(factor, 0.0) or 0.0)
            max_points = max(summary_max, derived_max)
            pillar = factor[:1]
            percentage = (raw_score / max_points * 100) if max_points > 0 else 0.0
            percentage = min(100.0, percentage)
            factor_percentages[factor] = {
                "raw_score": round(raw_score, 4),
                "max_points": round(max_points, 4),
                "percentage": round(percentage, 2),
                "pillar": pillar,
                "label": max_info.get("content", factor),
            }
            if pillar in pillar_groups:
                pillar_groups[pillar].append(percentage)

        e_score = round(sum(pillar_groups["E"]) / max(len(pillar_groups["E"]), 1), 2) if pillar_groups["E"] else 0.0
        s_score = round(sum(pillar_groups["S"]) / max(len(pillar_groups["S"]), 1), 2) if pillar_groups["S"] else 0.0
        g_score = round(sum(pillar_groups["G"]) / max(len(pillar_groups["G"]), 1), 2) if pillar_groups["G"] else 0.0
        total = round(e_score * weights["E"] + s_score * weights["S"] + g_score * weights["G"], 2)

        return {
            "E_score": e_score,
            "S_score": s_score,
            "G_score": g_score,
            "weights": weights,
            "total_score": total,
            "factor_scores": factor_percentages,
            "details": all_details,
        }

    def apply_screening_penalties(self, scores, screening_results):
        if screening_results.get("G_killed"):
            print("  ⚠ TOTAL KILL: Điểm G bị đưa về 0!")
            scores["G_score"] = 0
        if screening_results.get("ES_killed"):
            print("  ⚠ TOTAL KILL: Điểm E/S bị đưa về 0!")
            scores["E_score"] = 0
            scores["S_score"] = 0

        deductions = screening_results.get("direct_deductions", 0)
        w = scores["weights"]
        scores["total_score"] = round(
            scores["E_score"] * w["E"] + scores["S_score"] * w["S"] + scores["G_score"] * w["G"]
            - deductions * 2,
            2,
        )
        scores["total_score"] = max(0, scores["total_score"])
        return scores

    def _empty_result(self, industry_sector):
        return {
            "E_score": 0,
            "S_score": 0,
            "G_score": 0,
            "total_score": 0,
            "details": [],
            "weights": self._fallback_weights(industry_sector),
            "factor_scores": {},
        }

    def _format_section(self, section):
        return (
            f"[DOC: {section['source_file']} | TYPE: {section['document_type']} | "
            f"PAGES: {section['page_start']}-{section['page_end']} | SCORE: {section.get('score', 0):.2f}]\n"
            f"{section['content']}"
        )

    def _should_retry(self, extraction: dict, resolution: dict) -> bool:
        """
        Determine if adaptive retry is needed.
        Only retry when there's signal that the answer might improve with more context.
        """
        answer = extraction.get("answer", "NULL")
        status = extraction.get("status", "")
        confidence = resolution.get("confidence", 0.0)

        # Don't retry NULL/SKIP — these likely mean the question is genuinely unanswerable
        if answer in {"", "NULL", "SKIP"}:
            return False

        # Retry if evidence was hallucinated (ungrounded)
        verification = extraction.get("evidence_verification")
        if verification and not verification.get("grounded", True):
            return True

        # Retry if self-consistency check found conflict
        consistency = extraction.get("consistency_check")
        if consistency and consistency.get("conflict", False):
            return True

        # Retry if confidence is very low despite having an answer
        if confidence < 0.35 and answer not in {"", "NULL", "SKIP"}:
            return True

        # Retry if weakly_supported
        if status == "weakly_supported":
            return True

        return False
