"""
Excel Parser — Parse bộ câu hỏi VNSI, trọng số theo ngành, cấu trúc điểm tối đa.
"""
import json
import os
import re

import pandas as pd


class VNSIExcelParser:
    def __init__(self, file_path):
        self.file_path = file_path

    def parse_all(self, output_dir="outputs"):
        """Parse tất cả sheets và xuất ra JSON."""
        os.makedirs(output_dir, exist_ok=True)

        rules = self.parse_vnsi_rules()
        with open(os.path.join(output_dir, "vnsi_rules.json"), "w", encoding="utf-8") as f:
            json.dump(rules, f, ensure_ascii=False, indent=2)

        weights = self.parse_industry_weights()
        with open(os.path.join(output_dir, "industry_weights.json"), "w", encoding="utf-8") as f:
            json.dump(weights, f, ensure_ascii=False, indent=2)

        rd = self.parse_rd_benchmarks()
        with open(os.path.join(output_dir, "rd_benchmarks.json"), "w", encoding="utf-8") as f:
            json.dump(rd, f, ensure_ascii=False, indent=2)

        scoring_structure = self.parse_scoring_structure()
        with open(os.path.join(output_dir, "scoring_structure.json"), "w", encoding="utf-8") as f:
            json.dump(scoring_structure, f, ensure_ascii=False, indent=2)

        print(f"[Excel Parser] Đã xuất tất cả rules vào {output_dir}/")
        return rules, weights, rd, scoring_structure

    def parse_vnsi_rules(self):
        """Parse sheet VNSI → screening + scoring rules."""
        df = pd.read_excel(self.file_path, sheet_name="VNSI", header=3)
        factor_max_scores = self.parse_scoring_structure().get("factor_max_scores", {})
        rules = {"screening": [], "scoring": []}

        for _, row in df.iterrows():
            q_id = str(row.get("#", "")).strip()
            if pd.isna(q_id) or q_id == "nan" or not q_id:
                continue

            factor = str(row.get("Mục", "")).strip()
            category = str(row.get("Mảng đánh giá", "")).strip()
            question = str(row.get("Câu hỏi", "")).strip()
            options = str(row.get("Unnamed: 4", "")).strip()
            logic = str(row.get("Unnamed: 5", "")).strip()

            entry = {
                "id": q_id,
                "factor": factor if factor != "nan" else "",
                "category": category if category != "nan" else "",
                "question": question,
                "options": options,
                "logic": logic if logic != "nan" else "",
            }

            if q_id.startswith("SL"):
                rules["screening"].append(entry)
            else:
                # Xác định pillar (E/S/G)
                if q_id.startswith("E"):
                    entry["pillar"] = "E"
                elif q_id.startswith("S"):
                    entry["pillar"] = "S"
                elif q_id.startswith("G"):
                    entry["pillar"] = "G"
                else:
                    entry["pillar"] = "Unknown"

                entry["sub_category"] = self._infer_sub_category(entry["pillar"], factor)
                entry["is_multi_select"] = (
                    "có thể chọn nhiều đáp án" in options.lower()
                    or self._is_implicit_cumulative(logic)
                )
                entry["requires_evidence"] = "dẫn chứng" in options.lower()
                entry["prerequisite"] = self._extract_prerequisite(question)
                entry["base_max_score"] = self._infer_base_max_score(logic, options)
                entry["max_score"] = self._compute_max_score(logic, options)
                if factor in factor_max_scores:
                    entry["factor_max_score"] = factor_max_scores[factor]["max_points"]

                rules["scoring"].append(entry)

        return rules

    def parse_industry_weights(self):
        """Parse bảng trọng số E/S/G theo ngành GICS."""
        df = pd.read_excel(self.file_path, sheet_name="Nguyên tắc đánh giá", header=None)

        weights = {}
        for i in range(len(df)):
            val = str(df.iloc[i, 2]).strip()
            if "GICS Classification Sector" in val:
                for j in range(i + 1, len(df)):
                    sector = str(df.iloc[j, 2]).strip()
                    if sector == "nan" or not sector:
                        break
                    g = float(df.iloc[j, 3]) if pd.notna(df.iloc[j, 3]) else 0
                    s = float(df.iloc[j, 4]) if pd.notna(df.iloc[j, 4]) else 0
                    e = float(df.iloc[j, 5]) if pd.notna(df.iloc[j, 5]) else 0
                    weights[sector] = {"G": g, "S": s, "E": e}
                break

        return weights

    def parse_rd_benchmarks(self):
        """Parse bảng benchmark R&D theo ngành."""
        try:
            df = pd.read_excel(self.file_path, sheet_name=2, header=None)
            benchmarks = {}
            for i in range(1, len(df)):
                sector = str(df.iloc[i, 0]).strip()
                threshold = str(df.iloc[i, 1]).strip()
                if sector != "nan" and threshold != "nan":
                    benchmarks[sector] = threshold
            return benchmarks
        except Exception:
            return {}

    def parse_scoring_structure(self):
        """Parse cấu trúc điểm tối đa theo factor từ sheet nguyên tắc."""
        df = pd.read_excel(self.file_path, sheet_name=0, header=None)
        factor_rows = []
        current_category = ""
        counters = {"E": 0, "S": 0, "G": 0}

        for i in range(len(df)):
            category = str(df.iloc[i, 1]).strip() if pd.notna(df.iloc[i, 1]) else ""
            content = str(df.iloc[i, 2]).strip() if pd.notna(df.iloc[i, 2]) else ""
            num_q = df.iloc[i, 3] if pd.notna(df.iloc[i, 3]) else None
            max_pts = df.iloc[i, 4] if pd.notna(df.iloc[i, 4]) else None

            if category:
                current_category = category

            if not content or num_q is None or max_pts is None:
                continue

            if not re.match(r"^\d+\.", content):
                continue

            pillar = self._pillar_from_category(current_category)
            if not pillar:
                continue

            counters[pillar] += 1
            factor = f"{pillar}{counters[pillar]}"
            factor_rows.append({
                "factor": factor,
                "pillar": pillar,
                "category": current_category,
                "content": content,
                "num_questions": int(float(num_q)),
                "max_points": float(max_pts),
            })

        return {
            "factor_max_scores": {row["factor"]: row for row in factor_rows},
            "factor_rows": factor_rows,
        }

    def _pillar_from_category(self, category):
        normalized = category.lower().replace("\n", " ")
        if "môi trường" in normalized:
            return "E"
        if "xã hội" in normalized:
            return "S"
        if "quản trị" in normalized:
            return "G"
        return ""

    def _infer_sub_category(self, pillar, factor):
        if pillar == "E":
            return {"E1": "Chính sách", "E2": "Quản lý", "E3": "Hiệu quả"}.get(factor, "Khác")
        if pillar == "S":
            return {"S1": "Chính sách", "S2": "Quản lý", "S3": "Hiệu quả"}.get(factor, "Khác")
        if pillar == "G":
            return {
                "G1": "Quyền cổ đông",
                "G2": "Bên liên quan",
                "G3": "Công bố thông tin",
                "G4": "Trách nhiệm HĐQT",
                "G5": "Kiểm soát",
            }.get(factor, "Khác")
        return "Khác"

    def _extract_prerequisite(self, question):
        text = str(question)
        match = re.search(
            r"trả lời\s+([A-Z](?:\s*hoặc\s*[A-Z])*)\s+cho\s+Câu\s+([A-Z]\.?\d(?:[\.\d]+)*)",
            text,
            re.IGNORECASE,
        )
        if not match:
            return None

        allowed_expr = match.group(1).upper().replace("HOẶC", ",")
        allowed = [token.strip() for token in re.split(r"[^A-Z]+", allowed_expr) if len(token.strip()) == 1]
        ref_question = self._canonical_question_id(match.group(2))
        return {"question_id": ref_question, "allowed_answers": allowed}

    def _canonical_question_id(self, question_id):
        qid = str(question_id).upper().strip().replace("..", ".")
        if re.match(r"^[A-Z]\d", qid):
            qid = f"{qid[0]}.{qid[1:]}"
        return qid

    def _infer_base_max_score(self, logic, options):
        logic_text = str(logic or "")
        options_text = str(options or "")
        if not logic_text or logic_text == "nan":
            return 0.0

        frac_match = re.search(r"([+-]?\d+(?:[.,]\d+)?)\s*trên\s*1\s*yêu cầu", logic_text, re.IGNORECASE)
        if frac_match:
            per_item = float(frac_match.group(1).replace(",", "."))
            option_count = len(re.findall(r"(^|\n)([A-Z])[\.\)]", options_text))
            return round(per_item * option_count, 4)

        letter_scores = []
        for line in logic_text.splitlines():
            match = re.match(r"\s*[A-Z][\.\)]\s*([+-]?\d+(?:[.,]\d+)?)", line.strip())
            if match:
                score = float(match.group(1).replace(",", "."))
                if score > 0:
                    letter_scores.append(score)
        return max(letter_scores, default=0.0)

    def _is_implicit_cumulative(self, logic):
        """Phát hiện câu có 2+ options dương → cộng dồn ẩn."""
        positive_count = 0
        for line in str(logic or "").splitlines():
            m = re.match(r"\s*[A-Z][.\)]\s*([+-]?\d+(?:[.,]\d+)?)", line.strip())
            if m and float(m.group(1).replace(",", ".")) > 0:
                positive_count += 1
        return positive_count >= 2

    def _compute_max_score(self, logic, options):
        """Tính max_score: single-select = max dương, cumulative = tổng dương."""
        logic_text = str(logic or "")
        if not logic_text or logic_text == "nan":
            return 0.0

        # Multi-select kiểu "+X trên 1 yêu cầu"
        frac_match = re.search(
            r"([+-]?\d+(?:[.,]\d+?))\s*trên\s*1", logic_text, re.IGNORECASE
        )
        if frac_match:
            per_item = float(frac_match.group(1).replace(",", "."))
            option_count = len(re.findall(r"(^|\n)([A-Z])[.\)]", str(options or "")))
            return round(per_item * max(option_count, 1), 4)

        # Parse all positive scores
        positive_scores = []
        for line in logic_text.splitlines():
            m = re.match(r"\s*[A-Z][.\)]\s*([+-]?\d+(?:[.,]\d+)?)", line.strip())
            if m:
                val = float(m.group(1).replace(",", "."))
                if val > 0:
                    positive_scores.append(val)

        if len(positive_scores) >= 2:
            return round(sum(positive_scores), 4)  # Cộng dồn
        return max(positive_scores, default=0.0)    # Single-select


if __name__ == "__main__":
    parser = VNSIExcelParser("inputs/20250506 - VNSI - Bo cau hoi PTBV 2025.xlsx")
    rules, weights, rd, structure = parser.parse_all()
    print(f"\nScreening rules: {len(rules['screening'])}")
    print(f"Scoring rules: {len(rules['scoring'])}")
    print(f"Industry weights: {json.dumps(weights, indent=2)}")
    print(f"R&D benchmarks: {json.dumps(rd, ensure_ascii=False, indent=2)}")
