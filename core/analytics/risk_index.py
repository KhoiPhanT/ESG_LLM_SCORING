"""
ESGUI Risk Index — Công thức đúng theo tài liệu:
    ESGUI = (1 - ESG/100) * (1 + WUI)
ESG càng cao → ESGUI càng nhỏ (tốt)
WUI càng cao → ESGUI càng lớn (xấu)
"""
from core.analytics.wui_loader import WUILoader


class ESGUIRiskIndex:
    def __init__(self, wui_file="inputs/WUI.xlsx"):
        self.wui_loader = WUILoader(wui_file)

    def calculate_esgui(self, esg_score, year=2024):
        """
        ESGUI = (1 - ESG/100) * (1 + WUI)
        - esg_score: Điểm ESG tuân thủ VNSI (0-100)
        - year: Năm để lấy WUI tương ứng
        """
        wui = self.wui_loader.get_wui_by_year(year)
        esgui = (1 - esg_score / 100) * (1 + wui)
        return {
            "esgui": round(esgui, 4),
            "esg_score": esg_score,
            "wui": round(wui, 4),
            "year": year,
            "interpretation": self._interpret(esgui),
        }

    def _interpret(self, esgui):
        """Giải thích mức độ rủi ro."""
        if esgui < 0.1:
            return "Rủi ro RẤT THẤP — ESG tuân thủ tốt, môi trường ổn định"
        elif esgui < 0.3:
            return "Rủi ro THẤP — ESG khá tốt"
        elif esgui < 0.5:
            return "Rủi ro TRUNG BÌNH — Cần cải thiện ESG hoặc lưu ý biến động"
        elif esgui < 0.8:
            return "Rủi ro CAO — ESG yếu hoặc môi trường kinh doanh bất ổn"
        else:
            return "Rủi ro RẤT CAO — Nguy cơ vỡ nợ, cần hành động ngay"

    def compare_years(self, esg_scores_by_year):
        """So sánh ESGUI qua nhiều năm. Input: {2020: 65, 2021: 71, ...}"""
        results = []
        for year, esg in sorted(esg_scores_by_year.items()):
            r = self.calculate_esgui(esg, year)
            results.append(r)
        return results


if __name__ == "__main__":
    calculator = ESGUIRiskIndex()

    # Test với dữ liệu ACB từ Data(4).xlsx
    acb_esg = {2020: 65, 2021: 71, 2022: 78, 2023: 89.2, 2024: 91}
    print("ESGUI cho ACB qua các năm:")
    print(f"{'Năm':>6} {'ESG':>6} {'WUI':>8} {'ESGUI':>8}  Interpretation")
    print("-" * 70)
    for r in calculator.compare_years(acb_esg):
        print(f"{r['year']:>6} {r['esg_score']:>6.1f} {r['wui']:>8.4f} {r['esgui']:>8.4f}  {r['interpretation']}")
