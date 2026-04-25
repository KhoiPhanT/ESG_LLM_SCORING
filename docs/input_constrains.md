# Ràng buộc Dữ liệu Đầu vào (Input Constraints)

Tài liệu này ghi nhận các quy tắc và ràng buộc đối với nguồn dữ liệu đầu vào của hệ thống Đánh giá ESG, dựa trên thiết kế hiện tại và giới hạn kỹ thuật. Cần tuân thủ các quy tắc này để hệ thống hoạt động ổn định và chính xác.

## 1. Báo cáo Phi cấu trúc (PDF)
- **Định dạng mặc định**: Tất cả các báo cáo Thường niên (Annual Reports) và Báo cáo Phát triển Bền vững (Sustainability Reports) phải ở định dạng `.pdf`.
- **Thư mục lưu trữ**: `inputs/[Tên_Doanh_Nghiệp]/reports/` (Ví dụ: `inputs/ACB/reports/`).
- **Xử lý Text & Hình ảnh**: 
  - Hệ thống mặc định dùng `PyMuPDF` (Fitz) để quét chữ có sẵn trong PDF.
  - Các trang chứa bảng biểu dạng ảnh (scan) sẽ được đẩy qua luồng OCR (Tesseract / Cloud Vision API). Để tối ưu chi phí và thời gian chạy, file PDF cần ưu tiên định dạng "Text Searchable" (bản gốc xuất từ Word/InDesign) thay vì bản in rồi scan lại.

## 2. Dữ liệu Cấu trúc & Chỉ số Tài chính (Excel/CSV)
- **Chuẩn hóa**: Dữ liệu tài chính (doanh thu, chi phí R&D, chi phí xử lý chất thải, v.v.) được trích xuất từ Bảng cân đối kế toán & Kết quả kinh doanh, nếu có sẵn dạng file, phải lưu dưới định dạng `.xlsx` hoặc `.csv`.
- **Chức năng**: Đóng vai trò là Ground Truth (dữ liệu đối soát) để đánh giá độ chính xác của Module NLP/NER khi hệ thống tự động bóc tách từ PDF.
- **Thư mục lưu trữ**: `inputs/[Tên_Doanh_Nghiệp]/chỉ số hiệu qủa tài chính/`.

## 3. Bộ câu hỏi VNSI 2025 (Ground Truth Rules)
- **Định dạng**: File `20250506 - VNSI - Bo cau hoi PTBV 2025.xlsx`.
- **Yêu cầu**: Không thay đổi tên sheet hoặc cấu trúc các cột (đặc biệt là các cột chứa logic tính điểm, câu hỏi SL1-SL5, và trọng số 20-30-50). Hệ thống sẽ parse trực tiếp từ file này để làm rule base cho module Chấm điểm đa tầng.

## 4. Dữ liệu Trung bình Ngành
- **Quy tắc**: Việc chấm điểm Hiệu quả (Performance Score - 50%) yêu cầu so sánh với trung bình ngành. 
- **Giải pháp Prototype**: Trong giai đoạn hiện tại (chưa có tập dữ liệu ngành thực tế), hệ thống sẽ sử dụng phương pháp Mocking Data (tạo bộ dữ liệu mô phỏng giả lập) hoặc thông số mặc định. Các biến mock này sẽ được lưu ở một file JSON/Excel cấu hình riêng trong hệ thống.

## 5. Dữ liệu Chuỗi thời gian (TFP & Analytics)
- **Malmquist TFP**: Yêu cầu bắt buộc phải có dữ liệu chuỗi thời gian liên tục từ **3 đến 5 năm** của doanh nghiệp để đo lường thay đổi công nghệ (TECH). Nếu dữ liệu dưới 3 năm, module phân tích này sẽ bị vô hiệu hóa hoặc bị đánh dấu thiếu độ tin cậy thống kê.

## 6. Bộ Từ khóa UI Index & Dictionary
- **Nguyên tắc**: Bộ từ khóa (ví dụ: "rủi ro", "biến động") và các chứng chỉ (ISO 14001, ISO 45001) được quản lý thông qua một từ điển tĩnh (Static Dictionary) dạng file cấu hình (`.json` hoặc `.yaml`) để đảm bảo tính minh bạch và dễ giải trình.
- **Cập nhật**: LLM có thể đề xuất từ khóa mới, nhưng phải được phê duyệt trước khi thêm vào bộ từ điển để dùng cho việc chấm điểm chính thức.
