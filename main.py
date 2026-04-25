"""
ESG Scoring Pipeline — Main Entry Point
Hệ thống Đánh giá và Phân tích Chỉ số ESG tự động (VNSI 2025)
"""
import json
import os
import sys
import time
from datetime import datetime

from core.ingestion.document_corpus import DocumentCorpus, discover_related_pdf_paths
from core.ingestion.excel_parser import VNSIExcelParser
from core.ingestion.text_cleaner import TextCleaner
from core.extraction.keyword_matcher import KeywordMatcher
from core.llm_client import OllamaClient
from core.scoring.screening import ScreeningModule
from core.scoring.vnsi_scorer import VNSIScorer
from core.analytics.risk_index import ESGUIRiskIndex
from core.analytics.causal_inference import PLS_SEM_Model
from core.audit.retrieval_audit import RetrievalAudit
from core.audit.review_console import ReviewConsole


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

    # ─── Phase 0: Parse VNSI Rules ────────────────────────
    print("\n[0/6] Parsing bộ quy tắc VNSI 2025...")
    excel_path = "inputs/20250506 - VNSI - Bo cau hoi PTBV 2025.xlsx"
    parser = VNSIExcelParser(excel_path)
    parser.parse_all("outputs")

    # ─── Phase 1: Ingestion ───────────────────────────────
    document_paths = discover_related_pdf_paths(pdf_path, company_name=company_name, year=year)
    corpus = DocumentCorpus(document_paths)
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

    retrieval_audit = RetrievalAudit(corpus)
    audit_sample_rules = parser.parse_vnsi_rules().get("screening", []) + parser.parse_vnsi_rules().get("scoring", [])[:15]
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
    screener = ScreeningModule("outputs/vnsi_rules.json", llm_client=llm, corpus=corpus)
    screening_results = screener.evaluate(full_text)

    # 3b. VNSI Scoring (82 câu hỏi)
    print(f"\n[4/6] Chấm điểm VNSI ({industry_sector})...")
    scorer = VNSIScorer(
        rules_path="outputs/vnsi_rules.json",
        weights_path="outputs/industry_weights.json",
        structure_path="outputs/scoring_structure.json",
        llm_client=llm,
        corpus=corpus,
    )
    scores = scorer.score_all_questions(full_text, industry_sector=industry_sector)
    scoring_review_list = ReviewConsole().build_scoring_review_list(scores.get("details", []))

    # Áp dụng penalties
    scores = scorer.apply_screening_penalties(scores, screening_results)

    # ─── Phase 4: ESGUI ───────────────────────────────────
    print(f"\n[5/6] Tính toán ESGUI (năm {year})...")
    esgui_calc = ESGUIRiskIndex()
    esgui_result = esgui_calc.calculate_esgui(scores["total_score"], year=year)
    print(f"  ESGUI = (1 - {scores['total_score']}/100) × (1 + {esgui_result['wui']}) = {esgui_result['esgui']}")

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

    w = scores["weights"]
    print(f"\n  ĐIỂM ESG TỔNG HỢP: {scores['total_score']:.2f} / 100")
    print(f"  ┌──────────────────────────────────────────────┐")
    print(f"  │  E (Môi trường) : {scores['E_score']:6.2f}/100  (trọng số {w['E']:.0%}) │")
    print(f"  │  S (Xã hội)     : {scores['S_score']:6.2f}/100  (trọng số {w['S']:.0%}) │")
    print(f"  │  G (Quản trị)   : {scores['G_score']:6.2f}/100  (trọng số {w['G']:.0%}) │")
    print(f"  └──────────────────────────────────────────────┘")

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
            "weights": scores["weights"],
            "factor_scores": scores.get("factor_scores", {}),
        },
        "screening": screening_results,
        "esgui": esgui_result,
        "keyword_analysis": {
            "risk_keywords": risk_count,
            "risk_details": risk_details,
            "iso_standards": detected_isos,
        },
        "retrieval_audit": retrieval_audit_result,
        "retrieval_review_list": retrieval_review_list,
        "scoring_review_list": scoring_review_list,
        "sem_analysis": sem_results,
        "scoring_details": scores.get("details", []),
    }

    report_path = f"outputs/reports/{company_name}_{year}_esg_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n  📄 Báo cáo chi tiết: {report_path}")

    return report


if __name__ == "__main__":
    # Mặc định: dùng bộ tài liệu ACB 2024 có sẵn trong repo để test
    # Có thể truyền argument: python main.py <pdf_path> <company> <sector> <year>
    if len(sys.argv) > 1:
        pdf = sys.argv[1]
        company = sys.argv[2] if len(sys.argv) > 2 else "ACB"
        sector = sys.argv[3] if len(sys.argv) > 3 else "Financials"
        yr = int(sys.argv[4]) if len(sys.argv) > 4 else 2024
    else:
        pdf = "inputs/ACB/reports/ACB_Baocaothuongnien_2024.pdf"
        company = "ACB"
        sector = "Financials"
        yr = 2024

    run_pipeline(pdf, company, sector, yr)
