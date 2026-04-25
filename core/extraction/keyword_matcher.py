import json
import os

class KeywordMatcher:
    def __init__(self, dictionary_path="core/extraction/dictionary.json"):
        self.dictionary_path = dictionary_path
        self.dict_data = self._load_dictionary()

    def _load_dictionary(self):
        if not os.path.exists(self.dictionary_path):
            raise FileNotFoundError(f"Không tìm thấy file từ điển tại: {self.dictionary_path}")
        with open(self.dictionary_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def count_risk_keywords(self, text):
        """
        Đếm tần suất xuất hiện của các từ khóa rủi ro trong văn bản.
        Sử dụng cho module tính toán chỉ số ESGUI.
        """
        text_lower = text.lower()
        risk_counts = {}
        total_risk_words = 0
        
        for keyword in self.dict_data.get("risk_keywords", []):
            count = text_lower.count(keyword.lower())
            if count > 0:
                risk_counts[keyword] = count
                total_risk_words += count
                
        return total_risk_words, risk_counts

    def detect_iso_standards(self, text):
        """
        Phát hiện các chứng chỉ ISO được nhắc đến trong văn bản.
        Sử dụng cho chấm điểm Quản lý (Management Score).
        """
        text_upper = text.upper()
        detected_iso = []
        
        for iso in self.dict_data.get("iso_standards", []):
            if iso.upper() in text_upper:
                detected_iso.append(iso)
                
        return detected_iso

if __name__ == "__main__":
    # Test matcher
    sample_text = "Năm nay công ty đối mặt với nhiều rủi ro và biến động do thị trường. Tuy nhiên chúng tôi đã đạt được chứng chỉ ISO 14001."
    matcher = KeywordMatcher()
    total, details = matcher.count_risk_keywords(sample_text)
    isos = matcher.detect_iso_standards(sample_text)
    
    print(f"Tổng từ khóa rủi ro: {total} - Chi tiết: {details}")
    print(f"ISO phát hiện: {isos}")
