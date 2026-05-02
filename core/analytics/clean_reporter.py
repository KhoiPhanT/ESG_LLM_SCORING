import json
import os

from core.reporting import format_short_source

def generate_clean_markdown(json_report_path: str, output_md_path: str, rules_path: str = "outputs/vnsi_rules.json"):
    """
    Reads the JSON report, recalculates the score using derived_max, and outputs a clean markdown report.
    """
    if not os.path.exists(json_report_path):
        print(f"File not found: {json_report_path}")
        return

    with open(json_report_path, "r", encoding="utf-8") as f:
        report = json.load(f)

    # 1. Recalculate the scores properly using derived_max
    rules = []
    if os.path.exists(rules_path):
        with open(rules_path, "r", encoding="utf-8") as f:
            rules_data = json.load(f)
            rules = rules_data.get("scoring", [])
            
    derived_max = {}
    for r in rules:
        factor = r.get("factor")
        if factor:
            derived_max[factor] = derived_max.get(factor, 0.0) + float(r.get("max_score", 0.0))

    scoring_details = report.get("scoring_details", [])
    factor_raw_scores = {}
    for q in scoring_details:
        f = q.get("factor")
        score = q.get("score", 0.0)
        if f:
            factor_raw_scores[f] = factor_raw_scores.get(f, 0.0) + score

    pillar_percentages = {"E": [], "S": [], "G": []}
    factor_percentages = {}
    for factor, raw_score in factor_raw_scores.items():
        max_score = derived_max.get(factor, 0.0)
        # If derived_max is 0 but raw_score exists, we have a problem. Fallback to 1.0 to avoid zero division.
        if max_score <= 0:
            max_score = max(1.0, raw_score)
            
        percentage = min(100.0, max(0.0, (raw_score / max_score) * 100))
        pillar = factor[0]
        pillar_percentages[pillar].append(percentage)
        factor_percentages[factor] = {
            "raw": raw_score,
            "max": max_score,
            "percentage": percentage
        }

    scores = report.get("scores", {})
    weights = scores.get("weights", {"E": 0.3, "S": 0.35, "G": 0.35})
    total_raw = float(scores.get("total", sum(factor_raw_scores.values())) or 0.0)
    raw_max = float(scores.get("raw_max", sum(derived_max.values())) or 0.0)
    total_score = float(scores.get("score_100", scores.get("percentage", (total_raw / raw_max * 100 if raw_max else 0.0))) or 0.0)
    pillar_scores = scores.get("pillar_scores", {})
    e_score = float(scores.get("E", pillar_scores.get("E", {}).get("percentage", 0.0)) or 0.0)
    s_score = float(scores.get("S", pillar_scores.get("S", {}).get("percentage", 0.0)) or 0.0)
    g_score = float(scores.get("G", pillar_scores.get("G", {}).get("percentage", 0.0)) or 0.0)

    # 2. Write the Markdown Report
    company = report.get("company", "Unknown")
    year = report.get("year", "Unknown")
    industry = report.get("industry", "Unknown")
    
    md_lines = []
    md_lines.append(f"# Báo cáo đánh giá ESG: {company} ({year})")
    md_lines.append(f"**Ngành:** {industry}")
    md_lines.append(f"**Tổng điểm VNSI (Raw theo workbook):** {total_raw:.2f} / {raw_max:.2f}")
    md_lines.append(f"**Điểm quy đổi thang 100:** {total_score:.2f}")
    md_lines.append("")
    md_lines.append("## Điểm thành phần")
    md_lines.append(f"- **Môi trường (E)**: {e_score:.2f}/100 (Trọng số ngành tham chiếu: {weights.get('E', 0)*100:.0f}%)")
    md_lines.append(f"- **Xã hội (S)**: {s_score:.2f}/100 (Trọng số ngành tham chiếu: {weights.get('S', 0)*100:.0f}%)")
    md_lines.append(f"- **Quản trị (G)**: {g_score:.2f}/100 (Trọng số ngành tham chiếu: {weights.get('G', 0)*100:.0f}%)")
    md_lines.append("")
    md_lines.append("### Chi tiết theo Factor")
    for factor, data in sorted(factor_percentages.items()):
        md_lines.append(f"- **{factor}**: {data['raw']:.2f} / {data['max']:.2f} điểm ({data['percentage']:.2f}%)")
    md_lines.append("\n---\n")
    md_lines.append("## Chi tiết câu trả lời & Bằng chứng")
    
    # Group questions by factor
    questions_by_factor = {}
    for q in scoring_details:
        f = q.get("factor", "Unknown")
        questions_by_factor.setdefault(f, []).append(q)

    for factor in sorted(questions_by_factor.keys()):
        md_lines.append(f"\n### Mảng {factor}")
        for q in questions_by_factor[factor]:
            q_id = q.get("id", "N/A")
            score = q.get("score", 0.0)
            answer = q.get("answer", "N/A")
            question_text = q.get("question", "")
            reason = q.get("reason", "")
            
            md_lines.append(f"#### Câu {q_id}: {question_text}")
            md_lines.append(f"**Trả lời:** {answer} | **Điểm đạt được:** {score}")
            if reason:
                md_lines.append(f"**Lập luận (Reason):** {reason}")
            
            evidence_items = q.get("evidence_items", [])
            if not evidence_items:
                md_lines.append("> *Không tìm thấy bằng chứng trong tài liệu.*")
            else:
                for idx, ev in enumerate(evidence_items, 1):
                    quote = str(ev.get("quote", "")).replace("\n", " ")
                    source_ref = format_short_source(ev) or q.get("evidence_source_ref") or "Unknown Document"
                    option = str(ev.get("option", "") or "").strip()
                    label = f"Bằng chứng {idx}" if not option else f"Bằng chứng {idx} (option {option})"
                    md_lines.append(f"> **{label}:** \"{quote}\"")
                    md_lines.append(f"> *Nguồn: {source_ref}*")
                    md_lines.append(">")
            md_lines.append("")

    with open(output_md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
    
    print(f"Generated clean report at: {output_md_path}")

if __name__ == "__main__":
    # Test execution
    generate_clean_markdown(
        "outputs/reports/VNM_2024_esg_report.json",
        "outputs/reports/VNM_2024_esg_report_clean.md"
    )
