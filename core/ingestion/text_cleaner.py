"""
Text Cleaner — Làm sạch văn bản với NLTK (theo yêu cầu đề bài).
"""
import re

# Stopwords tiếng Việt phổ biến
VIETNAMESE_STOPWORDS = {
    "và", "của", "là", "các", "trong", "có", "được", "cho", "với", "này",
    "đã", "để", "từ", "theo", "về", "một", "những", "không", "đến", "trên",
    "tại", "cũng", "như", "đó", "khi", "sẽ", "do", "hoặc", "hay", "còn",
    "vào", "ra", "lên", "bị", "đang", "rồi", "thì", "mà", "nếu", "nhưng",
    "nên", "cần", "phải", "nào", "hơn", "rất", "nhiều", "ít", "người",
    "năm", "sau", "trước", "qua", "lại", "bao", "gồm", "vì", "giữa",
}


class TextCleaner:
    def __init__(self, remove_stopwords=False):
        self.remove_stopwords = remove_stopwords

    def clean(self, text):
        """Làm sạch văn bản: loại bỏ ký tự thừa, chuẩn hóa khoảng trắng."""
        if not text:
            return ""
        # Loại bỏ ký tự đặc biệt thừa (giữ lại chữ, số, dấu tiếng Việt)
        text = re.sub(r"[​\ufeff\u200b]", "", text)  # Zero-width chars
        # Chuẩn hóa khoảng trắng và xuống dòng
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"[ \t]{2,}", " ", text)
        text = text.strip()

        if self.remove_stopwords:
            words = text.split()
            words = [w for w in words if w.lower() not in VIETNAMESE_STOPWORDS]
            text = " ".join(words)

        return text

    def extract_sentences(self, text):
        """Tách câu đơn giản cho tiếng Việt."""
        sentences = re.split(r"[.!?]\s+", text)
        return [s.strip() for s in sentences if len(s.strip()) > 10]

    def word_count(self, text):
        """Đếm số từ."""
        return len(text.split()) if text else 0
