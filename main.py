"""
ESG Scoring Pipeline — Main Entry Point
Hệ thống Đánh giá và Phân tích Chỉ số ESG tự động (VNSI 2025)
"""
import json
import os
import sys
import time
from datetime import datetime

from core.cache import CacheManager
from core.ingestion.document_corpus import DocumentCorpus, discover_related_pdf_paths
from core.ingestion.excel_parser import VNSIExcelParser
from core.ingestion.text_cleaner import TextCleaner
from core.extraction.keyword_matcher import KeywordMatcher
from core.llm_client import OllamaClient
from core.scoring.screening import ScreeningModule
from core.scoring.vnsi_scorer import VNSIScorer
from core.retrieval.retrieval_engine import RetrievalEngine
from core.analytics.risk_index import ESGUIRiskIndex
from core.analytics.causal_inference import PLS_SEM_Model
from core.audit.retrieval_audit import RetrievalAudit
from core.audit.review_console import ReviewConsole
from core.audit.review_advisor import ReviewAdvisor
from core.audit.company_improvement_advisor import CompanyImprovementAdvisor
from core.audit.index_coverage_audit import IndexCoverageAudit
from core.audit.goldset_benchmark import GoldsetBenchmark
from core.audit.retrieval_preflight import RetrievalPreflight
from core.query_builder.question_retrieval_metadata import QuestionRetrievalMetadataBuilder


def print_header():
    print("=" * 65)
    print("  HỆ THỐNG ĐÁNH GIÁ VÀ PHÂN TÍCH ESG (VNSI 2025)")
    print("  Powered by Qwen3:30b Local LLM")
    print("=" * 65)


def run_pipeline(
    pdf_path: str,
    company_name: str = "ACB",
    industry_sector: str = "Financials",
    year: int = 2024,
):
    start_time = time.time()
    print_header()

    os.makedirs("outputs", exist_ok=True)
    os.makedirs("outputs/cache", exist_ok=True)
    os.makedirs("outputs/reports", exist_ok=True)

    run_cache = CacheManager(run_key=f"{company_name}:{year}:pending")

    # ─── Phase 0: Parse VNSI Rules ────────────────────────
    print("\n[0/6] Parsing bộ quy tắc VNSI 2025...")
    excel_path = "inputs/20250506 - VNSI - Bo cau hoi PTBV 2025.xlsx"
    parser = VNSIExcelParser(excel_path)
    parsed_rules, _, _, _ = parser.parse_all("outputs")

    print("  Sinh metadata retrieval offline cho 113 câu hỏi...")
    metadata_builder = QuestionRetrievalMetadataBuilder(target_year=year)
    metadata_path = "outputs/question_retrieval_metadata.json"
    retrieval_metadata_payload = metadata_builder.write(
        parsed_rules.get("scoring", []),
        metadata_path,
    )
    metadata_validation = metadata_builder.validate(
        retrieval_metadata_payload,
        parsed_rules.get("scoring", []),
    )
    print(
        "  Metadata retrieval:"
        f" coverage={metadata_validation['metadata_count']}/{metadata_validation['question_count']},"
        f" passed={metadata_validation['passed']}"
    )
    if not metadata_validation["passed"]:
        print("  ❌ Metadata retrieval chưa đạt validation. Dừng trước khi scoring.")
        print(json.dumps(metadata_validation, ensure_ascii=False, indent=2))
        return

    document_paths = discover_related_pdf_paths(pdf_path, company_name=company_name, year=year)
    input_fingerprint = CacheManager.folder_fingerprint(
        document_paths,
        extra={
            "company": company_name,
            "year": year,
            "industry_sector": industry_sector,
            "entry_path": os.path.abspath(pdf_path),
        },
    )
    run_cache = CacheManager(run_key=f"{company_name}:{year}:{input_fingerprint[:16]}")
    if os.environ.get("ESG_CACHE_STATUS", "0") == "1":
        print(f"  [CACHE] Run fingerprint: {input_fingerprint[:16]}")

    # ─── Phase 1: Ingestion ───────────────────────────────
    corpus = DocumentCorpus(document_paths, target_year=year)
    print(f"\n[1/6] Đọc bộ tài liệu ({len(document_paths)} file)")
    for path in document_paths:
        print(f"  - {os.path.basename(path)}")

    extracted_documents = corpus.extract_all()
    if not extracted_documents:
        print("  ❌ Không đọc được PDF. Kết thúc.")
        return

    print("  Metadata registry:")
    for metadata in corpus.build_registry():
        year_display = metadata["year_guess"] if metadata["year_guess"] else "?"
        review_flag = "review" if metadata["needs_review"] else "ok"
        print(
            "  - "
            f"{metadata['file_name']} | {metadata['document_type']} | year={year_display} | "
            f"extract={metadata['text_extraction_method']} | ocr={metadata['average_ocr_quality']:.2f} | {review_flag}"
        )

    cleaner = TextCleaner()
    combined_text_parts = []
    total_pages = 0
    for extracted in extracted_documents:
        doc = extracted["document"]
        pages = extracted["pages"]
        total_pages += len(pages)
        full_doc_text = "\n\n".join(p["text"] for p in pages if p["text"])
        if full_doc_text.strip():
            combined_text_parts.append(
                f"[DOC: {doc.label} | TYPE: {doc.doc_type} | PAGES: 1-{doc.metadata.page_count}]\n{full_doc_text}"
            )

    full_text = "\n\n".join(combined_text_parts)
    full_text = cleaner.clean(full_text)
    word_count = cleaner.word_count(full_text)
    print(f"  ✅ {total_pages} trang, {word_count:,} từ")

    # ─── Phase 2: Keyword Analysis ────────────────────────
    print("\n[2/6] Phân tích từ khóa...")
    keyword_matcher = KeywordMatcher("core/extraction/dictionary.json")
    risk_count, risk_details = keyword_matcher.count_risk_keywords(full_text)
    detected_isos = keyword_matcher.detect_iso_standards(full_text)
    print(f"  Từ khóa rủi ro: {risk_count} ({risk_details})")
    print(f"  Chứng chỉ ISO : {detected_isos if detected_isos else 'Không phát hiện'}")

    print("  Đang kiểm tra coverage index/sections...", flush=True)
    index_coverage_audit = IndexCoverageAudit()
    index_coverage_result = index_coverage_audit.audit(corpus, verbose=True)
    low_coverage_count = index_coverage_result.get("low_coverage_count", 0)
    print(
        "  Index coverage:"
        f" avg_page_coverage={index_coverage_result.get('average_page_coverage', 0.0):.2f},"
        f" low_coverage_docs={low_coverage_count}"
    )

    print("  Đang dựng retrieval/semantic index cho audit và scoring...", flush=True)
    retrieval_engine = RetrievalEngine(corpus, industry_sector=industry_sector, target_year=year)

    retrieval_audit = RetrievalAudit(
        corpus,
        industry_sector=industry_sector,
        target_year=year,
        retrieval_engine=retrieval_engine,
    )
    print("  Chạy retrieval preflight toàn bộ câu hỏi scoring...", flush=True)
    preflight = RetrievalPreflight(
        parsed_rules.get("scoring", []),
        retrieval_metadata_payload,
        target_year=year,
    )
    preflight_full = os.environ.get("ESG_PREFLIGHT_CRITICAL_ONLY") != "1"
    preflight_cache_fingerprint = CacheManager.hash_json({
        "schema_version": "retrieval_preflight_v2",
        "input_fingerprint": input_fingerprint,
        "metadata_fingerprint": retrieval_metadata_payload.get("input_fingerprint"),
        "metadata_hash": CacheManager.hash_json(retrieval_metadata_payload),
        "retrieval_window_fingerprint": retrieval_engine.corpus_window_fingerprint(),
        "target_year": year,
        "full": preflight_full,
    })
    preflight_json_path = os.path.join("outputs/audit", f"{company_name}_{year}_retrieval_preflight.json")
    preflight_md_path = os.path.join("outputs/audit", f"{company_name}_{year}_retrieval_preflight.md")
    cached_preflight = None
    if not CacheManager.is_forced("preflight") and os.environ.get("ESG_DISABLE_PREFLIGHT_CACHE", "0") != "1":
        payload = CacheManager.load_json(preflight_json_path)
        if (
            isinstance(payload, dict)
            and payload.get("schema_version") == "retrieval_preflight_v2"
            and payload.get("input_fingerprint") == preflight_cache_fingerprint
        ):
            cached_preflight = payload

    if cached_preflight:
        preflight_result = cached_preflight
        preflight_paths = {"json_path": preflight_json_path, "md_path": preflight_md_path}
        if not os.path.exists(preflight_md_path):
            preflight_paths = preflight.write(
                preflight_result,
                output_dir="outputs/audit",
                company=company_name,
                year=year,
            )
        print("  Retrieval preflight: cache hit, bỏ qua quét 113 câu.", flush=True)
        run_cache.record(
            "retrieval_preflight",
            "hit",
            "retrieval_preflight_v2",
            preflight_cache_fingerprint,
            path=preflight_json_path,
            reason="cache_hit",
        )
    else:
        preflight_result = preflight.run(
            retrieval_engine,
            full=preflight_full,
            verbose=True,
        )
        preflight_result["schema_version"] = "retrieval_preflight_v2"
        preflight_result["input_fingerprint"] = preflight_cache_fingerprint
        preflight_paths = preflight.write(
            preflight_result,
            output_dir="outputs/audit",
            company=company_name,
            year=year,
        )
        run_cache.record(
            "retrieval_preflight",
            "rebuilt",
            "retrieval_preflight_v2",
            preflight_cache_fingerprint,
            path=preflight_paths.get("json_path") or preflight_paths.get("md_path"),
            reason="preflight_run",
        )
    print(
        "  Retrieval preflight:"
        f" passed={preflight_result['passed']},"
        f" checked={preflight_result['checked_count']},"
        f" failures={preflight_result['failure_count']},"
        f" critical_failures={preflight_result['critical_failure_count']}"
    )
    print(f"  📄 Retrieval preflight report: {preflight_paths['md_path']}")
    if not preflight_result["passed"]:
        print("  ❌ Retrieval preflight chưa đạt critical gate. Dừng trước khi gọi LLM scoring.")
        return

    audit_sample_rules = parsed_rules.get("screening", []) + parsed_rules.get("scoring", [])[:15]
    benchmark_path = "refactor/goldsets/performance_goldset_acb_2024.json"
    retrieval_audit_result = retrieval_audit.audit_rules(
        audit_sample_rules,
        top_k=3,
        benchmark_path=benchmark_path if os.path.exists(benchmark_path) else None,
    )
    retrieval_review_list = ReviewConsole().build_retrieval_review_list(retrieval_audit_result)
    print(
        "  Retrieval audit:"
        f" coverage={retrieval_audit_result['questions_with_results']}/{retrieval_audit_result['question_count']},"
        f" low_value_ratio={retrieval_audit_result['average_low_value_ratio']:.2f},"
        f" preferred_hit_ratio={retrieval_audit_result['average_preferred_hit_ratio']:.2f},"
        f" semantic_support={retrieval_audit_result.get('average_semantic_support_ratio', 0.0):.2f},"
        f" rerank_gain={retrieval_audit_result['average_reranked_improvement_ratio']:.2f},"
        f" review_items={len(retrieval_review_list)}"
    )
    if retrieval_audit_result.get("benchmark"):
        benchmark = retrieval_audit_result["benchmark"]
        print(
            "  Retrieval benchmark:"
            f" recall@3={benchmark['recall_at_k']:.2f},"
            f" mrr={benchmark['mrr']:.2f},"
            f" no_evidence_guardrail={benchmark['no_evidence_guardrail']:.2f}"
        )

    # ─── Phase 3: LLM Scoring ─────────────────────────────
    print("\n[3/6] Khởi tạo Qwen3:30b LLM...")
    llm = OllamaClient(model="qwen3:30b")

    # 3a. Screening
    screener = ScreeningModule("outputs/vnsi_rules.json", llm_client=llm, corpus=corpus, industry_sector=industry_sector, target_year=year, retrieval_engine=retrieval_engine)
    screening_results = screener.evaluate(full_text)

    # 3b. VNSI Scoring (82 câu hỏi)
    print(f"\n[4/6] Chấm điểm VNSI ({industry_sector})...")
    scorer = VNSIScorer(
        rules_path="outputs/vnsi_rules.json",
        structure_path="outputs/scoring_structure.json",
        llm_client=llm,
        corpus=corpus,
        industry_sector=industry_sector,
        target_year=year,
        retrieval_engine=retrieval_engine,
        metadata_path=metadata_path,
    )
    scores = scorer.score_all_questions(full_text, industry_sector=industry_sector, company_name=company_name)
    scoring_review_list = ReviewConsole().build_scoring_review_list(scores.get("details", []))

    # Áp dụng penalties
    scores = scorer.apply_screening_penalties(scores, screening_results)

    # ─── Phase 4: ESGUI ───────────────────────────────────
    print(f"\n[5/6] Tính toán ESGUI (năm {year})...")
    esgui_calc = ESGUIRiskIndex()
    score_100 = min(100.0, max(0.0, float(scores.get("score_100", scores.get("percentage", scores["total_score"])) or 0.0)))
    esgui_result = esgui_calc.calculate_esgui(score_100, year=year)
    print(f"  ESGUI = (1 - {score_100}/100) × (1 + {esgui_result['wui']}) = {esgui_result['esgui']}")

    # ─── Phase 5: PLS-SEM ─────────────────────────────────
    print("\n[6/6] Phân tích nhân quả PLS-SEM...")
    sem_model = PLS_SEM_Model()
    sem_model.load_data(companies=["ACB", "VCB", "MBB", "FPT", "VNM"], min_year=2016)
    sem_results = sem_model.run_sem()

    # ─── Output Report ────────────────────────────────────
    elapsed = time.time() - start_time

    print("\n" + "=" * 65)
    print("  BÁO CÁO KẾT QUẢ ĐÁNH GIÁ ESG")
    print("=" * 65)
    print(f"  Doanh nghiệp    : {company_name}")
    print(f"  Ngành            : {industry_sector}")
    print(f"  Năm đánh giá     : {year}")
    print(f"  Báo cáo          : {os.path.basename(pdf_path)}")
    print(f"  Thời gian xử lý  : {elapsed:.0f} giây")
    print("-" * 65)

    print(f"\n  ĐIỂM VNSI RAW THEO WORKBOOK: {scores['total_score']:.2f} / {scores.get('raw_max', 0):.2f}")
    print(f"  ĐIỂM QUY ĐỔI THANG 100   : {scores.get('score_100', scores.get('percentage', 0)):.2f}")
    print(f"  ┌───────────────────────────────────────────┐")
    print(f"  │  E (Môi trường) : {scores['E_score']:6.2f}%                  │")
    print(f"  │  S (Xã hội)     : {scores['S_score']:6.2f}%                  │")
    print(f"  │  G (Quản trị)   : {scores['G_score']:6.2f}%                  │")
    print(f"  └───────────────────────────────────────────┘")

    print(f"\n  Screening        : {'✅ PASS' if screening_results['passed'] else '❌ FAIL'}")
    if not screening_results["passed"]:
        for d in screening_results["details"]:
            if "CÓ VI PHẠM" in d["status"]:
                print(f"    {d['id']}: {d['penalty']}")

    print(f"\n  ESGUI Index      : {esgui_result['esgui']:.4f}")
    print(f"  WUI ({year})       : {esgui_result['wui']:.4f}")
    print(f"  Đánh giá         : {esgui_result['interpretation']}")

    if sem_results:
        print(f"\n  PLS-SEM Analysis :")
        print(f"  {sem_model.interpret()}")

    # Keyword summary
    print(f"\n  Từ khóa rủi ro   : {risk_count} lần xuất hiện")
    print(f"  Chứng chỉ ISO    : {', '.join(detected_isos) if detected_isos else 'Không có'}")
    print("=" * 65)

    # ─── Save detailed report ─────────────────────────────
    report = {
        "company": company_name,
        "industry": industry_sector,
        "year": year,
        "pdf": os.path.basename(pdf_path),
        "source_documents": corpus.build_registry(),
        "timestamp": datetime.now().isoformat(),
        "processing_time_seconds": round(elapsed, 1),
        "scores": {
            "E": scores["E_score"],
            "S": scores["S_score"],
            "G": scores["G_score"],
            "total": scores["total_score"],
            "raw_total": scores.get("raw_total", scores["total_score"]),
            "raw_max": scores.get("raw_max", 0.0),
            "raw_percentage": scores.get("raw_percentage", 0.0),
            "percentage": scores.get("percentage", 0.0),
            "score_100": scores.get("score_100", scores.get("percentage", 0.0)),
            "pillar_scores": scores.get("pillar_scores", {}),
            "factor_scores": scores.get("factor_scores", {}),
            "factor_max_mismatches": scores.get("factor_max_mismatches", []),
            "diagnostics": scores.get("diagnostics", {}),
        },
        "screening": screening_results,
        "esgui": esgui_result,
        "keyword_analysis": {
            "risk_keywords": risk_count,
            "risk_details": risk_details,
            "iso_standards": detected_isos,
        },
        "retrieval_audit": retrieval_audit_result,
        "retrieval_preflight": preflight_result,
        "question_retrieval_metadata": {
            "path": metadata_path,
            "validation": metadata_validation,
        },
        "index_coverage_audit": index_coverage_result,
        "retrieval_review_list": retrieval_review_list,
        "scoring_review_list": scoring_review_list,
        "sem_analysis": sem_results,
        "scoring_details": scores.get("details", []),
    }

    report_path = f"outputs/reports/{company_name}_{year}_esg_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n  📄 Báo cáo chi tiết: {report_path}")

    try:
        coverage_paths = index_coverage_audit.write(
            index_coverage_result,
            output_dir="outputs/audit",
            company=company_name,
            year=year,
        )
        print(f"  📄 Index coverage audit: {coverage_paths['md_path']}")
    except Exception as e:
        print(f"  ❌ Lỗi khi xuất index coverage audit: {e}")

    try:
        audit_info = scorer.scoring_contract.write_audit(
            scores.get("details", []),
            output_dir="outputs/audit",
            company=company_name,
            year=year,
        )
        print(f"  📄 Scoring audit: {audit_info['csv_path']}")
    except Exception as e:
        print(f"  ❌ Lỗi khi xuất scoring audit: {e}")

    try:
        advisor_builder = ReviewAdvisor(
            scorer.scoring_rules,
            factor_max_scores=scorer.factor_max_scores,
        )
        advisor = advisor_builder.build(report)
        advisor_paths = advisor_builder.write(
            advisor,
            output_dir="outputs/reports",
            company=company_name,
            year=year,
        )
        report["review_advisor"] = {
            "summary": {
                "recoverable_points": advisor.get("recoverable_points"),
                "best_case_score": advisor.get("best_case_score"),
                "issue_totals": advisor.get("issue_totals"),
                "top_actions": advisor.get("top_actions"),
            },
            "paths": advisor_paths,
        }
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2, default=str)
        print(f"  📄 Review advisor: {advisor_paths['md_path']}")
        run_cache.record(
            "advisor",
            "rebuilt",
            "review_advisor_v1",
            CacheManager.hash_json({"report_path": report_path, "advisor": advisor.get("issue_totals")}),
            path=advisor_paths.get("json_path"),
            reason="advisor_generated",
        )
    except Exception as e:
        print(f"  ❌ Lỗi khi xuất review advisor: {e}")

    try:
        company_advisor_builder = CompanyImprovementAdvisor(
            scorer.scoring_rules,
            factor_max_scores=scorer.factor_max_scores,
            llm_client=llm,
        )
        company_advisor = company_advisor_builder.build(report)
        company_advisor_paths = company_advisor_builder.write(
            company_advisor,
            output_dir="outputs/reports",
            company=company_name,
            year=year,
        )
        report["company_improvement_advisor"] = {
            "summary": company_advisor.get("summary"),
            "gap_totals": company_advisor.get("gap_totals"),
            "priority_actions": company_advisor.get("priority_actions"),
            "paths": company_advisor_paths,
        }
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2, default=str)
        print(f"  📄 Company improvement advisor: {company_advisor_paths['md_path']}")
        run_cache.record(
            "advisor",
            "rebuilt",
            "company_improvement_advisor_v1",
            CacheManager.hash_json({"report_path": report_path, "advisor": company_advisor.get("gap_totals")}),
            path=company_advisor_paths.get("json_path"),
            reason="advisor_generated",
        )
    except Exception as e:
        print(f"  ❌ Lỗi khi xuất company improvement advisor: {e}")

    goldset_path = f"goldsets/{company_name.lower()}_{year}_manual.csv"
    if os.path.exists(goldset_path):
        try:
            benchmark = GoldsetBenchmark().compare(scores.get("details", []), goldset_path)
            report["goldset_benchmark"] = benchmark
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(report, f, ensure_ascii=False, indent=2, default=str)
            print(
                "  📄 Goldset benchmark:"
                f" count={benchmark['count']},"
                f" answer_accuracy={benchmark['answer_accuracy']},"
                f" evidence_hit_rate={benchmark['evidence_hit_rate']}"
            )
        except Exception as e:
            print(f"  ❌ Lỗi khi chạy goldset benchmark: {e}")

    try:
        from core.analytics.clean_reporter import generate_clean_markdown
        md_path = f"outputs/reports/{company_name}_{year}_esg_report_clean.md"
        generate_clean_markdown(report_path, md_path)
        print(f"  📄 Báo cáo Clean (Dễ đọc): {md_path}")
    except Exception as e:
        print(f"  ❌ Lỗi khi xuất báo cáo Clean: {e}")

    if os.environ.get("ESG_CACHE_STATUS", "0") == "1":
        for cache_key in [
            "document_cache",
            "question_metadata",
            "embedding_index",
            f"{company_name}:{year}:scoring",
            run_cache.run_key,
        ]:
            CacheManager(run_key=cache_key).print_summary()

    return report


if __name__ == "__main__":
    # Mặc định: dùng bộ tài liệu ACB 2024 có sẵn trong repo để test
    # Có thể truyền argument: python main.py <pdf_path> <company> <sector> <year>
    # Lưu ý: nếu <pdf_path> nằm trong một folder mang tên công ty (vd:
    # "VNM - 2024 test"), pipeline sẽ nạp toàn bộ PDF/ảnh trong folder đó như
    # một dossier bằng chứng của công ty. Điều này là chủ đích vì nhiều câu hỏi
    # cho phép historical evidence; các câu current-year-only sẽ được lọc sau
    # theo `time_policy`.
    if len(sys.argv) > 1:
        pdf = sys.argv[1]
        company = sys.argv[2] if len(sys.argv) > 2 else "ACB"
        sector = sys.argv[3] if len(sys.argv) > 3 else "Financials"
        yr = int(sys.argv[4]) if len(sys.argv) > 4 else 2024
    else:
        pdf = "inputs/ACB/ACB_Baocaothuongnien_2024.pdf"
        company = "ACB"
        sector = "Financials"
        yr = 2024

    run_pipeline(pdf, company, sector, yr)
