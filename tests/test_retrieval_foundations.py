import os
import tempfile
import unittest
from types import SimpleNamespace

from core.evidence.numeric_extractor import NumericExtractor
from core.ingestion.document_classifier import DocumentClassifier
from core.ingestion.document_corpus import DocumentCorpus, DocumentRecord
from core.query_builder.question_query_builder import QuestionQueryBuilder
from core.query_builder.question_retrieval_metadata import QuestionRetrievalMetadataBuilder


class RetrievalFoundationTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
