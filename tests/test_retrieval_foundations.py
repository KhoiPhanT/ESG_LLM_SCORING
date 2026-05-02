import os
import tempfile
import unittest
from types import SimpleNamespace

from core.evidence.numeric_extractor import NumericExtractor
from core.evidence.evidence_extractor import EvidenceExtractor
from core.ingestion.document_classifier import DocumentClassifier
from core.ingestion.document_corpus import DocumentCorpus, DocumentRecord
from core.llm_client import OllamaClient
from core.query_builder.question_query_builder import QuestionQueryBuilder
from core.query_builder.question_retrieval_metadata import QuestionRetrievalMetadataBuilder
from core.scoring.vnsi_scorer import VNSIScorer


class RetrievalFoundationTests(unittest.TestCase):
    def test_strip_think_tags_handles_nested_and_unclosed_blocks(self):
        text = '<think>inner <think>nested</think> still thinking</think>{"answer":"A"}'
        self.assertEqual(OllamaClient._strip_think_tags(text), '{"answer":"A"}')
        self.assertEqual(OllamaClient._strip_think_tags('preface <think>cut off'), "preface")

    def test_parse_json_trims_prefix_and_repairs_trailing_comma(self):
        client = OllamaClient()
        parsed = client._parse_json('Narrative first {"answer":"A","selected_options":["A"],}')
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed["answer"], "A")
        self.assertEqual(parsed["selected_options"], ["A"])

    def test_parse_json_repairs_truncated_nested_object(self):
        client = OllamaClient()
        parsed = client._parse_json(
            '{"answer":"A","selected_options":["A"],"option_evidence":{"A":{"source_id":"S1","quote":"abc"}}'
        )
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed["answer"], "A")
        self.assertEqual(parsed["option_evidence"]["A"]["source_id"], "S1")

    def test_parse_json_classifies_empty_after_think_strip(self):
        client = OllamaClient()
        client._last_call_info = {
            "status": "ok",
            "raw_response_before_strip": "<think>internal</think>",
            "response_after_strip": "",
        }
        parsed = client._parse_json("")
        self.assertIsNone(parsed)
        self.assertEqual(client._last_parse_info["parse_status"], "empty_after_think_strip")

    def test_llm_regex_only_parse_is_not_enough_for_multi_select(self):
        class BrokenJsonClient(OllamaClient):
            def _call(self, *args, **kwargs):
                return '{"answer":"A,B","selected_options":["A","B"],"confidence":0.8,"reason":"unterminated'

        result = BrokenJsonClient().ask_vnsi_question(
            context="[SOURCE_ID: S1] Chính sách năng lượng.",
            question="Chọn các chính sách môi trường phù hợp",
            options="A. Năng lượng\nB. Nước",
            q_id="E.test",
            is_multi_select=True,
        )

        self.assertEqual(result["answer"], "NULL")
        self.assertEqual(result["selected_options"], [])
        self.assertIn(result["parse_status"], {"answer_regex_only", "repaired_json"})

    def test_retry_ladder_recovers_after_empty_response(self):
        class RetryClient:
            def __init__(self):
                self.calls = []

            def ask_vnsi_question(self, **kwargs):
                self.calls.append((kwargs["response_mode"], kwargs["context_limit"]))
                if len(self.calls) == 1:
                    return {
                        "answer": "NULL",
                        "selected_options": [],
                        "reason": "fallback",
                        "parse_status": "empty_after_think_strip",
                        "call_info": {},
                    }
                return {
                    "answer": "A",
                    "selected_options": ["A"],
                    "reason": "ok",
                    "confidence": 0.8,
                    "evidence_source_id": "S1",
                    "evidence_quote": "Cam kết bảo vệ môi trường",
                    "parse_status": "valid_json",
                    "call_info": {},
                }

        extractor = EvidenceExtractor(llm_client=RetryClient(), target_year=2024)
        rule = {
            "id": "E.1.1.1",
            "question_type": "policy",
            "question": "Có chính sách môi trường còn hiệu lực hay không?",
            "options": "A. Không có\nB. Có nhưng không công khai\nC. Có và công khai",
            "logic": "A. 0\nB. 0.5\nC. 1",
        }
        context_bundle = {
            "context": "[SOURCE_ID: S1] Cam kết bảo vệ môi trường và công khai.",
            "sections": [{
                "source_id": "S1",
                "source_file": "policy.pdf",
                "source_path": "/tmp/policy.pdf",
                "document_type": "policy_document",
                "page_start": 1,
                "page_end": 1,
                "content": "Cam kết bảo vệ môi trường và công khai.",
                "score": 30,
                "quality_score": 1.0,
            }],
            "context_char_limit": 22000,
            "retrieval_meta": {"query_plan": {}},
        }
        result = extractor.extract(rule, context_bundle)
        self.assertEqual(result["answer"], "A")
        self.assertTrue(result["retry_used"])
        self.assertEqual(result["retry_attempts"], 2)

    def test_single_select_fallback_uses_dominant_retrieval_option(self):
        class EmptyClient:
            def ask_vnsi_question(self, **kwargs):
                return {
                    "answer": "NULL",
                    "selected_options": [],
                    "reason": "fallback",
                    "parse_status": "empty_raw_response",
                    "call_info": {},
                }

        extractor = EvidenceExtractor(llm_client=EmptyClient(), target_year=2024)
        rule = {
            "id": "E.1.1.1",
            "question_type": "policy",
            "question": "Công ty có chính sách liên quan tới tác động môi trường còn hiệu lực hay không?",
            "options": (
                "A. Không có chính sách\n"
                "B. Có chính sách nhưng không công khai rộng rãi\n"
                "C. Có chính sách liên quan tới tác động môi trường và được công khai rộng rãi"
            ),
            "logic": "A. 0\nB. 0.5\nC. 1",
            "time_policy": "current_year_required",
        }
        query_plan = {
            "option_focus": {
                "B": ["chính sách môi trường", "không công khai"],
                "C": ["chính sách môi trường", "công khai rộng rãi"],
            },
            "negative_options": ["A"],
            "option_evidence_requirements": {"B": "policy_topic", "C": "policy_topic"},
        }
        context_bundle = {
            "context": "",
            "sections": [{
                "source_id": "S1",
                "source_file": "annual.pdf",
                "source_path": "/tmp/annual.pdf",
                "document_type": "annual_report",
                "year_guess": 2024,
                "page_start": 10,
                "page_end": 10,
                "matched_options": ["C"],
                "content": "Công ty có chính sách môi trường và được công khai rộng rãi trong báo cáo thường niên.",
                "score": 35,
                "quality_score": 1.0,
            }],
            "context_char_limit": 22000,
            "retrieval_meta": {"query_plan": query_plan},
        }
        result = extractor.extract(rule, context_bundle)
        self.assertEqual(result["answer"], "C")
        self.assertEqual(result["answer_origin"], "retrieval_fallback_single")
        self.assertTrue(result["evidence_items"])

    def test_single_select_fallback_stays_null_when_ambiguous(self):
        class EmptyClient:
            def ask_vnsi_question(self, **kwargs):
                return {
                    "answer": "NULL",
                    "selected_options": [],
                    "reason": "fallback",
                    "parse_status": "empty_raw_response",
                    "call_info": {},
                }

        extractor = EvidenceExtractor(llm_client=EmptyClient(), target_year=2024)
        rule = {
            "id": "E.1.1.1",
            "question_type": "policy",
            "question": "Công ty có chính sách liên quan tới tác động môi trường còn hiệu lực hay không?",
            "options": "A. Không có\nB. Có nhưng không công khai\nC. Có và công khai",
            "logic": "A. 0\nB. 0.5\nC. 1",
        }
        query_plan = {
            "option_focus": {
                "B": ["chính sách môi trường", "không công khai"],
                "C": ["chính sách môi trường", "công khai rộng rãi"],
            },
            "negative_options": ["A"],
            "option_evidence_requirements": {"B": "policy_topic", "C": "policy_topic"},
        }
        sections = [
            {
                "source_id": "S1",
                "source_file": "policy1.pdf",
                "source_path": "/tmp/policy1.pdf",
                "document_type": "policy_document",
                "year_guess": 2024,
                "page_start": 1,
                "page_end": 1,
                "matched_options": ["B"],
                "content": "Có chính sách môi trường nhưng không công khai rộng rãi.",
                "score": 30,
                "quality_score": 1.0,
            },
            {
                "source_id": "S2",
                "source_file": "policy2.pdf",
                "source_path": "/tmp/policy2.pdf",
                "document_type": "policy_document",
                "year_guess": 2024,
                "page_start": 2,
                "page_end": 2,
                "matched_options": ["C"],
                "content": "Có chính sách môi trường và công khai rộng rãi trên website.",
                "score": 29,
                "quality_score": 1.0,
            },
        ]
        result = extractor.extract(rule, {
            "context": "",
            "sections": sections,
            "context_char_limit": 22000,
            "retrieval_meta": {"query_plan": query_plan},
        })
        self.assertEqual(result["answer"], "NULL")

    def test_option_evidence_requires_option_specific_quote_relevance(self):
        extractor = EvidenceExtractor()
        rule = {
            "id": "E.1.1.3",
            "is_multi_select": True,
            "question_type": "multi_select",
            "options": (
                "A. Được Hội đồng quản trị phê duyệt;\n"
                "B. Cam kết tuân thủ pháp luật về môi trường;"
            ),
        }
        source_sections = [{
            "source_id": "S1",
            "source_file": "governance.pdf",
            "document_type": "policy_document",
            "page_start": 41,
            "page_end": 42,
            "content": "đã có đầy đủ quyền hạn, nguồn lực, và tư cách độc lập để hỗ trợ Hội đồng quản trị thực hiện chức năng giám sát.",
            "score": 10,
            "quality_score": 0.9,
        }]
        query_plan = {
            "option_focus": {
                "A": ["chính sách", "Hội đồng quản trị phê duyệt"],
                "B": ["chính sách", "tuân thủ pháp luật môi trường"],
            },
            "option_evidence_requirements": {
                "A": "approval",
                "B": "legal_commitment",
            },
        }

        result = extractor._build_option_level_evidence(
            rule=rule,
            selected_options=["A"],
            option_evidence={
                "A": {
                    "source_id": "S1",
                    "quote": "đã có đầy đủ quyền hạn, nguồn lực, và tư cách độc lập để hỗ trợ Hội đồng quản trị thực hiện chức năng giám sát.",
                }
            },
            source_sections=source_sections,
            query_plan=query_plan,
        )

        self.assertEqual(result["selected_options"], [])
        self.assertEqual(
            result["option_evidence_verification"]["A"]["option_relevance_status"],
            "unsupported",
        )

    def test_multi_select_fallback_uses_distinct_option_anchors(self):
        plan = QuestionRetrievalMetadataBuilder(target_year=2024).build({
            "id": "E.1.1.2",
            "question_type": "multi_select",
            "is_multi_select": True,
            "question": "Chính sách môi trường đề cập khía cạnh chủ đề trọng yếu nào?",
            "options": (
                "A. Nguyên vật liệu\n"
                "B. Năng lượng\n"
                "C. Nước\n"
                "D. Đa dạng sinh học\n"
                "E. Phát thải\n"
                "F. Nước thải và Chất thải\n"
                "G. Quản lý môi trường nhà cung cấp\n"
                "H. Tuân thủ pháp luật môi trường"
            ),
            "logic": "+0,125 trên 1 yêu cầu đáp ứng",
        })
        sections = [{
            "source_id": "S1",
            "source_file": "ptbv.pdf",
            "document_type": "sustainability_report",
            "page_start": 19,
            "page_end": 19,
            "matched_options": list("ABCDEFGH"),
            "content": "\n".join([
                "Nguồn nguyên liệu bền vững",
                "Sử dụng năng lượng hiệu quả và sử dụng năng lượng xanh",
                "Nguồn nước và chất lượng nước",
                "Bảo vệ đa dạng sinh học",
                "Giảm lượng phát thải khí nhà kính",
                "Kiểm soát nước thải và chất thải",
                "Mở rộng hoạt động đến các nhà cung cấp trong chuỗi cung ứng",
                "Tuân thủ luật định về môi trường",
            ]),
            "score": 10,
            "quality_score": 1.0,
        }]

        fallback = EvidenceExtractor()._fallback_option_evidence_from_retrieval(
            rule={"is_multi_select": True},
            source_sections=sections,
            query_plan=plan,
        )

        self.assertEqual(set(fallback), set("ABCDEFGH"))
        self.assertEqual(len({item["quote"] for item in fallback.values()}), 8)

    def test_policy_question_does_not_emit_connector_bigrams(self):
        rule = {
            "id": "E.1.1.1",
            "factor": "E1",
            "pillar": "E",
            "sub_category": "Chính sách",
            "question_type": "policy",
            "question": "E.1.1.1 Công ty có chính sách liên quan tới quản lý các tác động môi trường còn hiệu lực hay không?",
            "options": (
                "A. Công ty không có bất kỳ chính sách liên quan đến môi trường còn hiệu lực.\n"
                "B. Công ty có chính sách môi trường nhưng không công khai rộng rãi\n"
                "C. Công ty có chính sách liên quan tới tác động môi trường và được công khai rộng rãi"
            ),
        }

        query = QuestionQueryBuilder().build(rule)
        all_terms = set(query.exact_phrases + query.primary_terms + query.secondary_terms + query.intent_terms)

        self.assertIn("chinh sach moi truong", all_terms)
        self.assertIn("quan ly moi truong", all_terms)
        self.assertNotIn("sach quan", all_terms)
        self.assertNotIn("quan cac", all_terms)

    def test_policy_pdf_is_classified_as_policy_document(self):
        classifier = DocumentClassifier()
        pages = [{
            "page": 1,
            "text": "Chính sách thực hành sản xuất nông nghiệp bền vững. Cam kết bảo vệ môi trường và giảm tác động môi trường.",
            "extraction_method": "native",
            "ocr_quality_score": 1.0,
        }]
        with tempfile.NamedTemporaryFile(prefix="Chinh_sach_moi_truong_", suffix=".pdf", delete=False) as handle:
            handle.write(b"%PDF-1.4 test")
            path = handle.name
        try:
            metadata = classifier.classify(path, pages=pages)
        finally:
            os.unlink(path)

        self.assertEqual(metadata.document_type, "policy_document")
        self.assertGreaterEqual(metadata.classification_confidence, 0.75)

    def test_numeric_extractor_calculates_ratio_with_revenue(self):
        rule = {
            "id": "E.3.test",
            "question_type": "ratio_calculation",
            "question": "Công ty có công bố tỷ lệ phát thải khí nhà kính trên doanh thu năm 2024 không?",
        }
        context = """
[SOURCE_ID: S1 | DOC: sustainability.pdf | TYPE: sustainability_report | YEAR: 2024 | PAGES: 10-10 | SCORE: 9.0]
Năm 2023 2024 2025
Tổng phát thải khí nhà kính 100 kg CO2e 80 kg CO2e 90 kg CO2e
---
[SOURCE_ID: S2 | DOC: annual.pdf | TYPE: annual_report | YEAR: 2024 | PAGES: 20-20 | SCORE: 9.0]
Tổng doanh thu hợp nhất năm 2024 đạt 40 tỷ đồng.
"""
        result = NumericExtractor(target_year=2024).extract(rule, {"context": context, "sections": []})

        self.assertIsNotNone(result)
        self.assertEqual(result["extraction_method"], "deterministic_ratio")
        self.assertEqual(result["ratio"]["numerator"]["value"], 80.0)
        self.assertEqual(result["ratio"]["denominator"]["value"], 40.0)
        self.assertEqual(result["ratio"]["result"], 2.0)

    def test_multi_select_metadata_isolates_options_and_keeps_positive_negation(self):
        rule = {
            "id": "E.1.1.3",
            "question_type": "multi_select",
            "is_multi_select": True,
            "question": "Nếu công ty có chính sách môi trường, chính sách đề cập mức độ cụ thể nào?",
            "options": (
                "A. Được Hội đồng quản trị phê duyệt;\n"
                "B. Cam kết tuân thủ pháp luật về môi trường;\n"
                "C. Cách thức quản lý / biện pháp sử dụng tài nguyên, bảo vệ môi trường;\n"
                "D. Cam kết cải thiện không ngừng hiệu suất môi trường;"
            ),
            "logic": "+0,25 trên 1 yêu cầu đáp ứng",
        }

        metadata = QuestionRetrievalMetadataBuilder(target_year=2024).build(rule)

        self.assertEqual(metadata["strategy"], "multi_option")
        self.assertEqual(set(metadata["option_focus"]), {"A", "B", "C", "D"})
        self.assertEqual(set(metadata["isolated_option_queries"]), {"A", "B", "C", "D"})
        self.assertNotIn("D", metadata["negative_options"])
        self.assertEqual(metadata["option_polarity"]["D"], "positive")

    def test_document_corpus_section_cache_round_trip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            old_cache_dir = DocumentCorpus.CACHE_DIR
            DocumentCorpus.CACHE_DIR = tmpdir
            try:
                with tempfile.NamedTemporaryFile(prefix="policy_doc_", suffix=".pdf", delete=False) as handle:
                    handle.write(b"%PDF-1.4 test")
                    pdf_path = handle.name

                metadata = DocumentClassifier().classify(
                    pdf_path,
                    pages=[{
                        "page": 1,
                        "text": "Chính sách môi trường và cam kết công khai rộng rãi.",
                        "extraction_method": "native",
                        "ocr_quality_score": 1.0,
                    }],
                )
                doc = DocumentRecord(
                    path=pdf_path,
                    label=os.path.basename(pdf_path),
                    doc_type=metadata.document_type,
                    parser=SimpleNamespace(extract_text=lambda: [{"page": 1, "text": "Nội dung chính sách", "extraction_method": "native", "ocr_quality_score": 1.0, "char_count": 18, "word_count": 3}]),
                    metadata=metadata,
                )

                first_corpus = DocumentCorpus.__new__(DocumentCorpus)
                first_corpus.target_year = 2024
                first_corpus._section_cache = {}
                first_corpus._table_cache = {}
                first_corpus._full_text_cache = {}
                first_corpus._pages_cache = {}
                first_corpus.section_builder = SimpleNamespace(build=lambda **kwargs: [{
                    "chunk_id": "chunk-1",
                    "source_file": doc.label,
                    "source_path": doc.path,
                    "document_type": doc.doc_type,
                    "year_guess": doc.metadata.year_guess,
                    "section_title": "policy",
                    "page_start": 1,
                    "page_end": 1,
                    "chunk_type": "section",
                    "table_family": None,
                    "content": "Chính sách môi trường",
                    "coverage_source": "section_builder",
                    "quality_score": 0.9,
                }])
                first_corpus.semantic_chunker = SimpleNamespace(chunk_sections=lambda sections: sections)
                first_corpus.table_extractor = SimpleNamespace(extract=lambda **kwargs: [{
                    "chunk_id": "table-1",
                    "source_file": doc.label,
                    "source_path": doc.path,
                    "document_type": doc.doc_type,
                    "year_guess": doc.metadata.year_guess,
                    "section_title": "table",
                    "page_start": 1,
                    "page_end": 1,
                    "chunk_type": "table_section",
                    "table_family": "environmental_metrics",
                    "content": "Bảng số liệu",
                    "coverage_source": "table_extractor",
                    "quality_score": 0.8,
                }])
                first_corpus.chunk_labeler = SimpleNamespace(annotate=lambda section, doc: dict(section, labeled=True))
                first_corpus._get_pages = lambda _doc: [{"page": 1, "text": "Nội dung chính sách", "extraction_method": "native", "ocr_quality_score": 1.0, "char_count": 18, "word_count": 3}]
                first_corpus._build_page_fallback_sections = lambda _doc, pages: []

                sections_first = first_corpus._get_sections(doc)
                self.assertTrue(sections_first)
                cache_files = []
                for root, _, files in os.walk(tmpdir):
                    for file_name in files:
                        cache_files.append(os.path.join(root, file_name))
                self.assertTrue(any(path.endswith(".json") for path in cache_files))

                second_corpus = DocumentCorpus.__new__(DocumentCorpus)
                second_corpus.target_year = 2024
                second_corpus._section_cache = {}
                second_corpus._table_cache = {}
                second_corpus._full_text_cache = {}
                second_corpus._pages_cache = {}
                second_corpus.section_builder = SimpleNamespace(build=lambda **kwargs: self.fail("section builder should not run on cache hit"))
                second_corpus.semantic_chunker = SimpleNamespace(chunk_sections=lambda sections: self.fail("semantic chunker should not run on cache hit"))
                second_corpus.table_extractor = SimpleNamespace(extract=lambda **kwargs: self.fail("table extractor should not run on cache hit"))
                second_corpus.chunk_labeler = SimpleNamespace(annotate=lambda section, doc: self.fail("labeler should not run on cache hit"))
                second_corpus._get_pages = lambda _doc: self.fail("page extraction should not run on cache hit")
                second_corpus._build_page_fallback_sections = lambda _doc, pages: self.fail("fallback builder should not run on cache hit")

                sections_second = second_corpus._get_sections(doc)
                self.assertEqual(sections_first, sections_second)
                self.assertEqual(sections_second[0]["content"], "Chính sách môi trường")
            finally:
                DocumentCorpus.CACHE_DIR = old_cache_dir
                try:
                    os.unlink(pdf_path)
                except Exception:
                    pass

    def test_screening_penalties_apply_deductions_once_and_keep_raw_formula(self):
        scorer = VNSIScorer(rules_path="missing_rules.json", structure_path="missing_structure.json")
        scores = {
            "raw_max": 20.0,
            "pillar_scores": {
                "E": {"raw_score": 4.0, "raw_percentage": 40.0, "percentage": 40.0},
                "S": {"raw_score": 3.0, "raw_percentage": 30.0, "percentage": 30.0},
                "G": {"raw_score": 5.0, "raw_percentage": 50.0, "percentage": 50.0},
            },
        }
        result = scorer.apply_screening_penalties(scores, {"direct_deductions": 2.0})
        self.assertEqual(result["total_score"], 10.0)
        self.assertEqual(result["raw_total"], 10.0)
        self.assertEqual(result["raw_percentage"], 50.0)
        self.assertEqual(result["score_100"], 50.0)

    def test_diagnostics_classify_llm_and_fallback_failures(self):
        scorer = VNSIScorer(rules_path="missing_rules.json", structure_path="missing_structure.json")
        diagnostics = scorer._build_diagnostics([
            {
                "id": "E.1.1.1",
                "question_bucket": "single_select_positive",
                "answer_origin": "llm_empty_failure",
                "parse_status": "empty_raw_response",
                "resolution_status": "insufficient",
                "answer": "NULL",
                "score": 0.0,
                "evidence_present": False,
                "source_sections": [],
                "reason": "",
            },
            {
                "id": "E.1.1.2",
                "question_bucket": "multi_select",
                "answer_origin": "retrieval_fallback_multi",
                "parse_status": "answer_regex_only",
                "resolution_status": "supported",
                "answer": "A,B",
                "score": 0.25,
                "evidence_present": True,
                "source_sections": [{"rerank_score": 50}],
                "reason": "",
            },
        ])
        self.assertEqual(diagnostics["counts"]["failure_reason"]["llm_completion_failure"], 1)
        self.assertEqual(diagnostics["counts"]["failure_reason"]["llm_json_failure"], 1)


if __name__ == "__main__":
    unittest.main()
