"""
Build retrieval queries from VNSI rules.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

from core.normalization.text_normalizer import TextNormalizer


@dataclass
class RetrievalQuery:
    question_id: str
    question_text: str
    exact_phrases: list[str] = field(default_factory=list)
    primary_terms: list[str] = field(default_factory=list)
    secondary_terms: list[str] = field(default_factory=list)
    intent_terms: list[str] = field(default_factory=list)
    preferred_document_types: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


class QuestionQueryBuilder:
    BIGRAM_STOPWORDS = {
        "viec",
        "hoat",
        "dong",
        "trong",
        "ngoai",
        "lien",
        "phong",
        "xuat",
        "nguoi",
        "chuc",
        "tuong",
        "duong",
        "cung",
        "thong",
        "khong",
        "hinh",
        "thuc",
        "khac",
        "sach",
        "quan",
        "cac",
        "tac",
        "toi",
        "den",
        "hieu",
        "luc",
    }
    QUERY_STOPWORDS = {
        "cong",
        "ty",
        "hoac",
        "khong",
        "nhung",
        "vui",
        "long",
        "cung",
        "cap",
        "phan",
        "tra",
        "loi",
        "muc",
        "do",
        "thuc",
        "te",
        "rieng",
        "biet",
        "co",
        "duoc",
        "lien",
        "quan",
        "hieu",
        "luc",
        "bao",
        "gom",
        "thong",
        "hien",
        "khac",
        "dung",
        "cac",
        "voi",
        "theo",
        "nam",
        "cau",
        "hoi",
        "nguyen",
        "truong",
        "tac",
        "toi",
        "den",
    }
    OPTION_STOPWORDS = QUERY_STOPWORDS | {
        "bat",
        "ky",
        "rong",
        "rai",
        "mot",
        "nhieu",
        "noi",
        "dung",
    }

    def __init__(self):
        self.normalizer = TextNormalizer()
        self.phrase_library = [
            "ý kiến kiểm toán ngoại trừ",
            "ý kiến kiểm toán",
            "báo cáo tài chính",
            "đại hội đồng cổ đông",
            "tham dự trực tuyến",
            "bỏ phiếu điện tử",
            "phát biểu ý kiến",
            "hội đồng quản trị",
            "cân bằng giới",
            "thành viên nữ",
            "báo cáo phát triển bền vững",
            "trách nhiệm về môi trường và xã hội",
            "chính sách môi trường",
            "quản lý môi trường",
            "phúc lợi và đãi ngộ",
            "người lao động",
            "an toàn lao động",
            "đào tạo",
            "nguyên vật liệu",
            "vật liệu tái chế",
            "vật liệu tái tạo",
            "năng lượng",
            "tiêu thụ năng lượng",
            "tiết kiệm năng lượng",
            "nước",
            "nước thải",
            "tuần hoàn nước",
            "phát thải",
            "khí nhà kính",
            "scope 1",
            "scope 2",
            "scope 3",
            "co2",
            "chất thải",
            "đa dạng sinh học",
            "đóng gói",
            "vật liệu đóng gói",
            "thu hồi",
            "tái sử dụng",
            "tuyển dụng",
            "nhân viên mới",
            "tỷ lệ nghỉ việc",
            "giờ đào tạo",
            "thời gian đào tạo trung bình",
            "số giờ đào tạo trung bình",
            "tai nạn lao động",
            "cơ cấu nhân sự",
            "thu nhập bình quân",
            "giới tính",
            "độ tuổi",
            "cấp quản lý",
            "cấp nhân viên",
            "chương trình phát triển kỹ năng",
            "học tập liên tục",
            "trách nhiệm xã hội",
            "cộng đồng địa phương",
            "hoạt động cộng đồng",
            "thiện nguyện",
            "đóng góp cộng đồng",
            "tài trợ cộng đồng",
            "tài trợ",
            "quỹ sữa",
            "vươn cao việt nam",
            "cứu trợ thiên tai",
            "bão yagi",
            "gắn kết yêu thương",
            "lan tỏa giá trị",
            "xử lý chất thải",
            "chất thải nguy hại",
            "chất thải không nguy hại",
            "doanh thu",
            "tổng doanh thu",
            "doanh thu thuần",
            "lợi nhuận",
            "tài sản",
        ]
        self.specialized_terms = {
            "đhđcđ": ["tham dự trực tuyến", "bỏ phiếu điện tử", "biểu quyết điện tử", "tham dự và phát biểu"],
            "hđqt": ["thành viên độc lập", "thù lao", "cơ cấu hội đồng quản trị", "cân bằng giới", "trình độ chuyên môn", "kinh nghiệm"],
            "kiểm toán": ["ý kiến kiểm toán", "ngoại trừ", "báo cáo tài chính", "kiểm toán độc lập"],
            "môi trường": ["chính sách môi trường", "quản lý môi trường", "phát thải", "năng lượng", "nước", "chất thải"],
            "người lao động": ["phúc lợi", "đãi ngộ", "đào tạo", "an toàn lao động", "sức khỏe nghề nghiệp"],
            "ptbv": ["báo cáo phát triển bền vững", "báo cáo bền vững", "báo cáo ESG"],
            "nguyên vật liệu": ["nguyên vật liệu", "vật liệu tái chế", "vật liệu tái tạo"],
            "năng lượng": ["năng lượng", "tiêu thụ năng lượng", "tiết kiệm năng lượng", "điện năng", "kwh", "joules"],
            "nước": ["nước", "tiêu thụ nước", "nước thải", "tuần hoàn nước", "m3"],
            "phát thải": ["phát thải", "khí nhà kính", "scope 1", "scope 2", "scope 3", "co2"],
            "chất thải": ["chất thải", "nguy hại", "không nguy hại", "tái chế"],
            "đa dạng sinh học": ["đa dạng sinh học", "khu bảo tồn", "bảo tồn thiên nhiên"],
            "đóng gói": ["đóng gói", "vật liệu đóng gói", "thu hồi", "tái sử dụng", "tái chế"],
            "tuyển dụng": ["tuyển dụng", "nhân viên mới", "lao động mới"],
            "nghỉ việc": ["nghỉ việc", "thôi việc", "biến động nhân sự"],
            "đào tạo": ["đào tạo", "giờ đào tạo", "học tập", "nâng cao năng lực"],
            "tai nạn lao động": ["tai nạn lao động", "an toàn lao động", "thương tích"],
            "nhân sự": ["cơ cấu nhân sự", "nhân sự", "giới tính", "độ tuổi"],
            "thu nhập": ["thu nhập bình quân", "thu nhập", "lương bình quân"],
            "doanh thu": ["doanh thu", "tổng doanh thu", "doanh thu thuần", "doanh thu hợp nhất"],
            "quản lý": ["cấp quản lý", "quản lý", "phó phòng"],
            "nhân viên": ["cấp nhân viên", "nhân viên"],
            "trách nhiệm xã hội": ["trách nhiệm xã hội", "hoạt động cộng đồng", "thiện nguyện", "cộng đồng địa phương", "tài trợ cộng đồng", "đóng góp cộng đồng", "quỹ sữa", "vươn cao", "cứu trợ thiên tai"],
            "cộng đồng": ["hoạt động cộng đồng", "thiện nguyện", "đóng góp cộng đồng", "quỹ sữa", "vươn cao", "bão yagi", "hỗ trợ cộng đồng"],
            "thù lao": ["thù lao từng thành viên", "chi phí thù lao", "mức thù lao", "lương thưởng hđqt"],
            "đa dạng hđqt": ["trình độ chuyên môn", "kinh nghiệm", "học vấn", "chuyên ngành", "profile hđqt"],
            "chất thải": ["xử lý chất thải", "chất thải nguy hại", "chất thải không nguy hại", "rác thải nhựa"],
        }
        self.subcategory_terms = {
            "Quyền cổ đông": [
                "đại hội đồng cổ đông",
                "thư mời họp",
                "tài liệu họp",
                "ứng viên hội đồng quản trị",
                "bỏ phiếu điện tử",
                "tham dự trực tuyến",
            ],
            "Bên liên quan": [
                "quan hệ cổ đông",
                "website công ty",
                "công bố thông tin",
                "tiếng anh",
                "thành viên hội đồng quản trị độc lập",
                "báo cáo phát triển bền vững",
            ],
            "Công bố thông tin": [
                "ủy ban quản trị công ty",
                "ủy ban lương thưởng",
                "bổ nhiệm",
                "thành viên độc lập",
                "cân bằng giới",
                "đa dạng hội đồng quản trị",
            ],
            "Trách nhiệm HĐQT": [
                "ủy ban kiểm toán",
                "ủy ban quản lý rủi ro",
                "hội đồng quản trị độc lập",
                "chủ tịch ủy ban",
                "quản trị rủi ro",
            ],
            "Kiểm soát": [
                "kênh tương tác",
                "tiếp nhận phản hồi",
                "thu nhập của tổng giám đốc",
                "mục tiêu phát triển bền vững",
                "giám sát phát triển bền vững",
            ],
        }

    def build(self, rule: dict, corpus=None) -> RetrievalQuery:
        question = str(rule.get("question", "")).strip()
        qid = str(rule.get("id", "")).strip()
        exact_phrases = self._extract_exact_phrases(question, rule)
        primary_terms = self._extract_primary_terms(question, exact_phrases)
        secondary_terms = self._extract_secondary_terms(question, rule)
        intent_terms = self._extract_intent_terms(rule)
        preferred_doc_types = []
        if corpus:
            preferred_doc_types = corpus.choose_preferred_doc_types(q_id=qid, question=question) or []

        return RetrievalQuery(
            question_id=qid,
            question_text=question,
            exact_phrases=exact_phrases,
            primary_terms=primary_terms,
            secondary_terms=secondary_terms,
            intent_terms=intent_terms,
            preferred_document_types=preferred_doc_types,
        )

    def _extract_exact_phrases(self, question: str, rule: dict) -> list[str]:
        haystack = self.normalizer.normalize_for_search(
            " ".join([question, str(rule.get("options", ""))])
        )
        phrases = [
            phrase for phrase in self.phrase_library
            if self.normalizer.normalize_for_search(phrase) in haystack
        ]
        if "ngoai tru" in haystack and "kiem toan" in haystack:
            phrases.append("ý kiến kiểm toán ngoại trừ")
        if "kiem toan" in haystack:
            phrases.append("báo cáo kiểm toán độc lập")
        if "nguyen vat lieu" in haystack or "vat lieu tai che" in haystack:
            phrases.extend(["nguyên vật liệu", "vật liệu tái chế", "vật liệu tái tạo"])
        if "nang luong" in haystack:
            phrases.extend(["năng lượng", "tiêu thụ năng lượng"])
        if "nuoc thai" in haystack:
            phrases.extend(["nước", "nước thải"])
        elif "nuoc" in haystack:
            phrases.append("nước")
        if "phat thai" in haystack or "khi nha kinh" in haystack:
            phrases.extend(["phát thải", "khí nhà kính"])
        if "chat thai" in haystack:
            phrases.append("chất thải")
        if "da dang sinh hoc" in haystack:
            phrases.append("đa dạng sinh học")
        if "dong goi" in haystack:
            phrases.extend(["đóng gói", "vật liệu đóng gói"])
        if "tuyen dung" in haystack:
            phrases.extend(["tuyển dụng", "nhân viên mới"])
        if "nghi viec" in haystack:
            phrases.append("tỷ lệ nghỉ việc")
        if "dao tao" in haystack:
            phrases.extend(["đào tạo", "giờ đào tạo"])
        if "thoi gian dao tao trung binh" in haystack or "so gio dao tao trung binh" in haystack:
            phrases.extend(["thời gian đào tạo trung bình", "số giờ đào tạo trung bình"])
        if "tai nan lao dong" in haystack:
            phrases.append("tai nạn lao động")
        elif "tai nan" in haystack and "lao dong" in haystack:
            phrases.append("tai nạn lao động")
        if "su co nghiem trong" in haystack:
            phrases.append("sự cố nghiêm trọng")
        if "co cau nhan su" in haystack:
            phrases.append("cơ cấu nhân sự")
        if "thu nhap binh quan" in haystack:
            phrases.append("thu nhập bình quân")
        if "gioi tinh" in haystack:
            phrases.append("giới tính")
        if "do tuoi" in haystack:
            phrases.append("độ tuổi")
        if "cap quan ly" in haystack:
            phrases.append("cấp quản lý")
        if "cap nhan vien" in haystack:
            phrases.append("cấp nhân viên")
        if "chuong trinh phat trien ky nang" in haystack:
            phrases.extend(["chương trình phát triển kỹ năng", "học tập liên tục"])
        if "trach nhiem xa hoi" in haystack or "cong dong" in haystack or "thien nguyen" in haystack:
            phrases.extend([
                "trách nhiệm xã hội", "cộng đồng địa phương", "hoạt động cộng đồng",
                "thiện nguyện", "tài trợ cộng đồng", "đóng góp cộng đồng",
                "quỹ sữa", "vươn cao việt nam", "cứu trợ thiên tai", "bão yagi",
            ])
        if "thu lao" in haystack or "luong thuong" in haystack:
            phrases.extend(["thù lao", "thù lao từng thành viên", "lương thưởng hđqt"])
        if "da dang" in haystack and ("hdqt" in haystack or "hoi dong quan tri" in haystack):
            phrases.extend(["trình độ chuyên môn", "kinh nghiệm", "học vấn", "chuyên ngành"])
        if "xu ly chat thai" in haystack or "chat thai nguy hai" in haystack or "chat thai khong nguy hai" in haystack:
            phrases.extend(["xử lý chất thải", "chất thải nguy hại", "chất thải không nguy hại"])
        if "doanh thu" in haystack:
            phrases.extend(["doanh thu", "tổng doanh thu", "doanh thu thuần", "doanh thu hợp nhất"])
        phrases.extend(self.subcategory_terms.get(str(rule.get("sub_category", "")).strip(), [])[:4])
        return list(dict.fromkeys(self.normalizer.normalize_for_search(item) for item in phrases if item))

    def _extract_primary_terms(self, question: str, exact_phrases: list[str]) -> list[str]:
        question_clean = re.sub(r"^[A-Z]+\.[\d.]+\s*", "", question).strip()
        candidates = list(exact_phrases)

        tokens = re.findall(r"[A-Za-zÀ-ỹà-ỹĐđ]{4,}", question_clean)
        for token in tokens:
            lowered = self.normalizer.normalize_for_search(token)
            if lowered not in self.QUERY_STOPWORDS and len(lowered) >= 6:
                candidates.append(lowered)

        filtered_tokens = [
            self.normalizer.normalize_for_search(token)
            for token in tokens
            if self.normalizer.normalize_for_search(token) not in self.QUERY_STOPWORDS and len(token) >= 4
        ]
        for idx in range(len(filtered_tokens) - 1):
            bigram = f"{filtered_tokens[idx]} {filtered_tokens[idx + 1]}"
            if not self._valid_bigram(filtered_tokens[idx], filtered_tokens[idx + 1], bigram):
                continue
            candidates.append(bigram)

        expanded = []
        for candidate in candidates[:18]:
            expanded.extend(self.normalizer.expand_term(candidate))
        return list(dict.fromkeys(item for item in expanded if item))

    def _extract_secondary_terms(self, question: str, rule: dict) -> list[str]:
        secondary = []
        normalized_question = self.normalizer.normalize_for_search(question)
        for trigger, terms in self.specialized_terms.items():
            if self.normalizer.normalize_for_search(trigger) in normalized_question:
                secondary.extend(terms)
        secondary.extend(self.subcategory_terms.get(str(rule.get("sub_category", "")).strip(), []))

        options = str(rule.get("options", ""))
        for phrase in re.findall(r"[A-Z][\.\)]\s*([^\n]+)", options):
            cleaned = phrase.strip()
            if len(cleaned) <= 120:
                secondary.extend(self._option_terms(cleaned))

        expanded = []
        for candidate in secondary[:15]:
            expanded.extend(self.normalizer.expand_term(candidate))
        return list(dict.fromkeys(item for item in expanded if item))

    def _extract_intent_terms(self, rule: dict) -> list[str]:
        sub_category = str(rule.get("sub_category", "")).strip()
        factor = str(rule.get("factor", "")).strip()
        pillar = str(rule.get("pillar", "")).strip()

        intent_library = {
            "Chính sách": ["chính sách", "quy định", "quy chế", "cam kết"],
            "Quản lý": ["quản lý", "quy trình", "giám sát", "triển khai", "thực hiện"],
            "Hiệu quả": ["chỉ tiêu", "số liệu", "thống kê", "định lượng"],
            "Quyền cổ đông": ["đại hội đồng cổ đông", "cổ đông", "biểu quyết", "tham dự trực tuyến"],
            "Bên liên quan": ["bên liên quan", "giao dịch", "xung đột lợi ích"],
            "Công bố thông tin": ["công bố thông tin", "báo cáo", "minh bạch"],
            "Trách nhiệm HĐQT": ["hội đồng quản trị", "ủy ban", "lương thưởng", "bổ nhiệm"],
            "Kiểm soát": ["kiểm soát", "kiểm toán nội bộ", "quản trị rủi ro"],
        }
        factor_terms = {
            "E1": ["môi trường", "chính sách môi trường"],
            "E2": ["phát thải", "năng lượng", "nước", "chất thải"],
            "E3": ["nguyên vật liệu", "đóng gói", "năng lượng", "nước", "nước thải", "phát thải", "khí nhà kính", "chất thải", "đa dạng sinh học"],
            "S1": ["người lao động", "nhân viên", "phúc lợi"],
            "S2": ["quy trình", "đào tạo", "an toàn lao động"],
            "S3": ["nhân viên mới", "tuyển dụng", "nghỉ việc", "đào tạo", "tai nạn lao động", "cơ cấu nhân sự", "thu nhập bình quân", "giới tính", "độ tuổi", "thống kê nhân sự"],
            "G1": ["đại hội đồng cổ đông", "cổ đông"],
            "G2": ["bên liên quan"],
            "G3": ["công bố thông tin", "minh bạch"],
            "G4": ["hội đồng quản trị", "ủy ban"],
            "G5": ["kiểm soát", "kiểm toán", "rủi ro"],
        }
        pillar_terms = {
            "E": ["môi trường"],
            "S": ["xã hội", "người lao động"],
            "G": ["quản trị", "hội đồng quản trị", "cổ đông"],
        }

        terms = []
        terms.extend(intent_library.get(sub_category, []))
        question_text = self.normalizer.normalize_for_search(str(rule.get("question", "")))
        is_s3_csr = factor == "S3" and any(token in question_text for token in [
            "cong dong", "thien nguyen", "trach nhiem xa hoi", "tai tro", "ho tro", "truyen thong"
        ])
        if is_s3_csr:
            terms.extend(["hoạt động cộng đồng", "thiện nguyện", "trách nhiệm xã hội", "đóng góp cộng đồng", "quỹ sữa", "vươn cao", "cứu trợ thiên tai"])
        else:
            terms.extend(factor_terms.get(factor, []))
        terms.extend(pillar_terms.get(pillar, []))
        if factor == "S3":
            if "dao tao" in question_text:
                terms.extend(["thời gian đào tạo trung bình", "số giờ đào tạo trung bình", "chương trình phát triển kỹ năng"])
            if is_s3_csr:
                terms.extend(["trách nhiệm xã hội", "cộng đồng địa phương", "hoạt động cộng đồng", "thiện nguyện", "tài trợ cộng đồng", "đóng góp cộng đồng", "quỹ sữa", "vươn cao việt nam", "bão yagi"])
        if factor == "E3":
            question_text = self.normalizer.normalize_for_search(str(rule.get("question", "")))
            if "tiet kiem" in question_text and "nang luong" in question_text:
                terms.extend(["tiết kiệm năng lượng", "sáng kiến tiết kiệm điện"])
            if "chat thai" in question_text:
                terms.extend(["xử lý chất thải", "chất thải nguy hại", "chất thải không nguy hại"])
            if "doanh thu" in question_text:
                terms.extend(["doanh thu", "tổng doanh thu", "doanh thu hợp nhất"])

        expanded = []
        for candidate in terms:
            expanded.extend(self.normalizer.expand_term(candidate))
        return list(dict.fromkeys(item for item in expanded if item))

    def _valid_bigram(self, left: str, right: str, bigram: str) -> bool:
        if left in self.BIGRAM_STOPWORDS or right in self.BIGRAM_STOPWORDS:
            allowed = {
                "chinh sach",
                "moi truong",
                "phat thai",
                "nang luong",
                "nuoc thai",
                "chat thai",
                "doanh thu",
                "tong doanh",
            }
            return bigram in allowed
        return True

    def _option_terms(self, text: str) -> list[str]:
        normalized = self.normalizer.normalize_for_search(text)
        terms = []
        for phrase in self.phrase_library:
            phrase_norm = self.normalizer.normalize_for_search(phrase)
            if phrase_norm in normalized:
                terms.append(phrase_norm)
        if "cong khai" in normalized:
            terms.extend(["công khai", "minh bạch", "công bố thông tin"])
        if "doanh thu" in normalized:
            terms.extend(["doanh thu", "tổng doanh thu", "doanh thu hợp nhất"])
        tokens = [
            token for token in normalized.split()
            if len(token) >= 5 and token not in self.OPTION_STOPWORDS
        ]
        terms.extend(tokens[:8])
        return list(dict.fromkeys(terms))
