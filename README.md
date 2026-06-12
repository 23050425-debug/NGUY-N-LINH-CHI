**AIDEOM-VN: Mô hình Tối ưu hóa Kinh tế Dẫn dắt bởi AI cho Việt Nam
**AIDEOM-VN (AI-Driven Economic Optimization Model for Vietnam)** là hệ thống Dashboard điều hành và hỗ trợ ra quyết định định lượng, nhằm hoạch định chiến lược phát triển kinh tế Việt Nam đến năm 2030 trong bối cảnh chuyển đổi số và kỷ nguyên Trí tuệ nhân tạo (AI). 

Hệ thống tích hợp 6 module tính toán cốt lõi (từ dự báo vĩ mô, đánh giá sẵn sàng số, tối ưu ngân sách, mô phỏng lao động đến kiểm soát rủi ro) và trực quan hóa qua giao diện web động, giúp lượng hóa các đánh đổi chính sách và đề xuất kịch bản phát triển bền vững.

---

## 📑 Mục lục
- [1. Tổng quan Hệ thống](#1-tổng-quan-hệ-thống)
- [2. Kiến trúc 6 Module cốt lõi (M1 - M6)](#2-kiến-trúc-6-module-cốt-lõi-m1---m6)
- [3. 5 Kịch bản Chính sách & Kết quả Nổi bật](#3-5-kịch-bản-chính-sách--kết-quả-nổi-bật)
- [4. Các Phát hiện & Hàm ý Chính sách Trọng tâm](#4-các-phát-hiện--hàm-ý-chính-sách-trọng-tâm)
- [5. Hướng dẫn Cài đặt & Chạy thử nghiệm](#5-hướng-dẫn-cài-đặt--chạy-thử-nghiệm)
- [6. Cấu trúc Thư mục](#6-cấu-trúc-thư-mục)
- [7. Nguồn Dữ liệu Chính thức](#7-nguồn-dữ-liệu-chính-thức)
- [8. Đề xuất Mở rộng Nghiên cứu](#8-đề-xuất-mở-rộng-nghiên-cứu)
- [9. Thông tin Tác giả](#9-thông-tin-tác-giả)

---

## 1. Tổng quan Hệ thống
AIDEOM-VN giải quyết bài toán đánh đổi phức tạp giữa **Tăng trưởng kinh tế (GDP)**, **Bất bình đẳng vùng miền (Gini)**, **Phát thải môi trường (CO2)** và **An ninh mạng (Cyber Risk)**. Hệ thống cho phép các nhà hoạch định chính sách thử nghiệm 5 kịch bản chiến lược và xem xét tác động đa chiều đến năm 2030, chuyển đổi dữ liệu thô thành các khuyến nghị định lượng.

---

## 2. Kiến trúc 6 Module cốt lõi (M1 - M6)

Hệ thống được thiết kế theo luồng xử lý dữ liệu khép kín:

| Module | Tên Module | Chức năng Chính | Phương pháp / Mô hình | Kết quả Nổi bật |
| :--- | :--- | :--- | :--- | :--- |
| **M1** | Dự báo Kinh tế | Ước lượng TFP, dự báo GDP 2026-2030 | Hàm Cobb-Douglas mở rộng (5 yếu tố) | TFP tăng từ 32.54 (2020) lên 40.94 (2025). MAPE dự báo chỉ 6.46%. |
| **M2** | Sẵn sàng Số | Xếp hạng 6 vùng kinh tế theo ưu tiên đầu tư AI | TOPSIS (Trọng số Chuyên gia & Entropy) | Đông Nam Bộ và Đồng bằng sông Hồng dẫn đầu. |
| **M3** | Phân bổ Ngân sách | Tối ưu hóa ngân sách số theo Ngành & Vùng | Linear Programming (LP), Stochastic Programming | Xác định cấu trúc phân bổ hiệu quả biên cao, kiểm soát sàn/trần. |
| **M4** | Tác động Lao động | Mô phỏng việc làm ròng (NetJob) trước tự động hóa | Mô hình dịch chuyển lao động & Đào tạo lại | Ngành Giáo dục - Đào tạo tạo ra 1,265,000 việc làm ròng lớn nhất. |
| **M5** | Đánh giá Rủi ro | Radar rủi ro đa chiều & Biên Pareto | NSGA-II (Đa mục tiêu), Minimax Regret | Cảnh báo rủi ro Cyber ở S3 và bất bình đẳng ở S2. |
| **M6** | Dashboard Ra QĐ | Trực quan hóa & Khuyến nghị chính sách | Streamlit, Plotly, Pandas | Tổng hợp thành giao diện điều hành 12 tab. |

---

## 3. 5 Kịch bản Chính sách & Kết quả Nổi bật

Hệ thống mô phỏng 5 kịch bản đầu tư đến năm 2030 với các chỉ số KPI chính:

| Kịch bản | GDP Gain (tỷ VND) | Gini Index (Bất bình đẳng) | Phát thải CO2 | Rủi ro Cyber |
| :--- | :---: | :---: | :---: | :---: |
| **S1. Truyền thống** | 51,730 | 0.6333 | 21.088 | Trung bình |
| **S2. Số hóa nhanh** | 55,458 | 0.6933 | 10.572 | Trung bình |
| **S3. AI dẫn dắt** | 58,786 *(Cao nhất)* | 0.8733 *(Cao nhất)* | 17.882 | **Cao nhất** ⚠️ |
| **S4. Bao trùm số** | 53,962 | **0.0957** *(Thấp nhất)* | **9.388** *(Thấp nhất)* | Thấp |
| **S5. Tối ưu cân bằng** | 53,660 | 0.3933 | 12.572 | Trung bình |

> **📊 Cảnh báo từ Module M5:**
> *   **Kịch bản S3 (AI dẫn dắt):** Có rủi ro An ninh mạng (Cyber) cao nhất, bắt buộc phải bổ sung ngân sách an ninh dữ liệu.
> *   **Kịch bản S4 (Bao trùm số):** Kiểm soát bất bình đẳng vùng miền tốt nhất, phù hợp với mục tiêu bao trùm của Chính phủ.
> *   **Vùng có AI Readiness thấp:** Cần ưu tiên hạ tầng số và nhân lực số trước khi mở rộng AI quy mô lớn.

---

## 4. Các Phát hiện & Hàm ý Chính sách Trọng tâm

1.  **Không có tăng trưởng miễn phí:** Kịch bản S5 (Tối ưu cân bằng) cho GDP 2030 cao nhất (16,238.4 nghìn tỷ VND), nhưng cần được kiểm soát bằng lớp ràng buộc rủi ro và an sinh. Theo đuổi tăng trưởng cực đoan (S3) sẽ để lại hệ quả bất bình đẳng nghiêm trọng.
2.  **Nhân lực số là điều kiện nền:** Kết quả M4 cho thấy đào tạo lại và nâng cấp kỹ năng tạo ra phần lớn NetJob ròng, đặc biệt ở các ngành dịch vụ công và tri thức (Giáo dục tạo ra >1.2 triệu việc làm). Nhân lực số là lớp đệm quan trọng khi tự động hóa tăng tốc.
3.  **Ràng buộc toán học định hình dòng ngân sách:** Các ràng buộc sàn/trần và đa dạng hóa trong M3 làm ngân sách lan tỏa hơn, giảm nguy cơ tập trung quá mức vào một số ít ngành/vùng có tốc độ tăng trưởng nhanh.
4.  **Kịch bản cân bằng là phương án điều hành tối ưu:** Theo điểm tổng hợp GDP - Gini - Cyber, **S4 (Bao trùm số)** là lựa chọn có hồ sơ rủi ro cân đối hơn để triển khai trong thực tế, đảm bảo phát triển bền vững.

---

## 5. Hướng dẫn Cài đặt & Chạy thử nghiệm
### Yêu cầu hệ thống
*   Python 3.9 hoặc cao hơn.
*   Git.
### Các bước thiết lập
Mở Terminal / Command Prompt và thực hiện các lệnh sau:

```bash
# 1. Clone repository về máy
git clone https://github.com/[TEN_GITHUB_CUA_BAN]/AIDEOM-VN.git
cd AIDEOM-VN

# 2. Tạo môi trường ảo (Khuyến nghị)
python -m venv venv
# Trên Windows:
venv\Scripts\activate
# Trên macOS/Linux:
source venv/bin/activate
# 3. Cài đặt các thư viện cần thiết
pip install -r requirements.txt
# 4. Chạy Dashboard
streamlit run AIDEOM_VN.py

# Cấu trúc thư mục
AIDEOM-VN/
├── AIDEOM_VN.py          # File điều hướng chính (Main App)
├── app_config.py         # Cấu hình giao diện, theme
├── data_loader.py        # Module nạp và làm sạch dữ liệu
├── ui_theme.py           # Tùy biến CSS, màu sắc Dashboard
├── requirements.txt      # Danh sách thư viện Python
├── tab1.py đến tab12.py  # 12 Module phân tích chi tiết
├── data/                 # Thư mục chứa dữ liệu thô (CSV)
└── README.md             # Tài liệu hướng dẫn này**
