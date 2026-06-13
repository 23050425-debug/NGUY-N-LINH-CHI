<p align="center">
  <h1 align="center">🇻🇳 AIDEOM-VN</h1>
  <p align="center"><i>Mô hình Tối ưu hóa Kinh tế Dẫn dắt bởi AI cho Việt Nam</i></p>
  <p align="center"><b>Dashboard điều hành phát triển kinh tế Việt Nam trong kỷ nguyên AI</b></p>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9+-blue?logo=python" alt="Python">
  <img src="https://img.shields.io/badge/Streamlit-Dashboard-red?logo=streamlit" alt="Streamlit">
  <img src="https://img.shields.io/badge/Optimization-PuLP%20%7C%20CVXPY%20%7C%20NSGA--II-green" alt="Optimization">
  <img src="https://img.shields.io/badge/License-MIT-yellow" alt="License">
</p>

<p align="center">
  🚀 <b>Trải nghiệm Dashboard trực tuyến tại:</b><br>
  <a href="https://nguy-n-linh-chi-t2zyyyanmmsk8xujes93rt.streamlit.app/"><b>https://nguy-n-linh-chi-t2zyyyanmmsk8xujes93rt.streamlit.app/</b></a>
</p>

---

### 🌟 TỔNG QUAN HỆ THỐNG (OVERVIEW)

**AIDEOM-VN** *(AI-Driven Economic Optimization Model for Vietnam)* không chỉ là một công cụ dự báo, mà là một **khung ra quyết định định lượng toàn diện**, được thiết kế để lượng hóa các đánh đổi chính sách phức tạp trong quá trình chuyển đổi số và phát triển kinh tế Việt Nam đến năm 2030. 

Thay vì phân tích rời rạc, AIDEOM-VN tích hợp **6 module tính toán cốt lõi (M1 - M6)** thành một luồng xử lý dữ liệu khép kín. Hệ thống cho phép các nhà hoạch định chính sách thử nghiệm **5 kịch bản chiến lược** (Truyền thống, Số hóa nhanh, AI dẫn dắt, Bao trùm số, Tối ưu cân bằng) và đánh giá tác động đa chiều lên 4 trụ cột: **Tăng trưởng (GDP) – Bất bình đẳng (Gini) – Phát thải (CO2) – An ninh mạng (Cyber Risk)**.

---

### 🏗️ KIẾN TRÚC 6 MODULE CỐT LÕI (M1 - M6)

Hệ thống được thiết kế theo kiến trúc module hóa, trong đó đầu ra của module trước là đầu vào của module sau, tạo thành một chuỗi logic chặt chẽ từ vĩ mô đến vi mô:

*   📈 **M1. Dự báo Kinh tế:** Sử dụng hàm Cobb-Douglas mở rộng với 5 yếu tố (Vốn, Lao động, Số hóa, AI, Nhân lực) để ước lượng Năng suất nhân tố tổng hợp (TFP) và dự báo quỹ đạo GDP 2026-2030.
*   🗺️ **M2. Đánh giá Sẵn sàng Số:** Xếp hạng 6 vùng kinh tế trọng điểm theo mức độ ưu tiên đầu tư AI bằng phương pháp TOPSIS, kết hợp đối chiếu giữa trọng số Chuyên gia và Entropy khách quan.
*   💰 **M3. Tối ưu Phân bổ Ngân sách:** Ứng dụng Quy hoạch tuyến tính (LP) và Quy hoạch ngẫu nhiên 2 giai đoạn để phân bổ ngân sách chuyển đổi số theo ma trận Ngành - Vùng dưới các ràng buộc công bằng và hiệu quả.
*   👥 **M4. Mô phỏng Lao động:** Lượng hóa việc làm ròng (NetJob) trước tác động của tự động hóa và đào tạo lại trên 8 ngành kinh tế, xác định các "điểm nghẽn" nhân lực.
*   🛡️ **M5. Kiểm soát Rủi ro:** Xây dựng Radar rủi ro 4 chiều và tìm nghiệm thỏa hiệp trên biên Pareto (sử dụng thuật toán di truyền NSGA-II) để cân bằng giữa tăng trưởng và bền vững.
*   💻 **M6. Dashboard Ra quyết định:** Tổng hợp và trực quan hóa toàn bộ kết quả từ M1-M5 thành giao diện điều hành tương tác, cung cấp các khuyến nghị chính sách tức thì.

---

### ⚖️ BỨC TRANH ĐÁNH ĐỔI: 5 KỊCH BẢN CHÍNH SÁCH

Hệ thống mô phỏng 5 kịch bản đầu tư đến năm 2030, phơi bày những đánh đổi sâu sắc giữa các mục tiêu kinh tế - xã hội - môi trường:

| Kịch bản | Tăng trưởng GDP (nghìn tỷ) | Bất bình đẳng (Gini) | Phát thải CO2 | Rủi ro An ninh mạng | GDP Dự báo 2030 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **S1. Truyền thống** | 51,730 | 0.633 | 21.088 | Trung bình | 14,800 |
| **S2. Số hóa nhanh** | 55,458 | 0.693 | 10.572 | Trung bình | 15,712 |
| **S3. AI dẫn dắt** | **58,786**  | **0.873**  | 17.882 | **Cao nhất**  | 16,236 |
| **S4. Bao trùm số** | 53,962 | **0.095**  | **9.388**  | Thấp | 15,635 |
| **S5. Tối ưu cân bằng**| 53,660 | 0.393 | 12.572 | Trung bình | **16,238**  |

> 💡 **Cảnh báo trực tiếp từ Hệ thống (M5):**
> *   **Kịch bản S3 (AI dẫn dắt):** Đạt GDP ngắn hạn cao nhất nhưng đi kèm bất bình đẳng nghiêm trọng và rủi ro an ninh dữ liệu ở mức báo động. Bắt buộc phải bổ sung ngân sách an ninh mạng tối thiểu 15%.
> *   **Kịch bản S4 (Bao trùm số):** Kiểm soát bất bình đẳng vùng miền và phát thải xuất sắc, là lựa chọn tối ưu cho các mục tiêu phát triển bền vững (SDGs) và cam kết COP26.
> *   **Vùng yếu kém:** Các vùng có *AI Readiness* thấp (Tây Nguyên, ĐBSCL, TD&MNPB) cần ưu tiên hạ tầng và nhân lực số trước khi mở rộng AI quy mô lớn.

---

### 🔑 CÁC PHÁT HIỆN & HÀM Ý CHÍNH SÁCH CỐT LÕI

Từ hàng nghìn kịch bản mô phỏng, AIDEOM-VN rút ra 4 nguyên tắc vàng cho hoạch định chính sách chuyển đổi số:

1.  🚫 **Không có tăng trưởng miễn phí:** 
    Kịch bản *Tối ưu cân bằng (S5)* cho GDP 2030 cao nhất (16,238 nghìn tỷ VND), nhưng cần được kiểm soát bằng lớp ràng buộc rủi ro. Theo đuổi tăng trưởng cực đoan (S3) sẽ để lại hệ quả bất bình đẳng nghiêm trọng, kìm hãm động lực dài hạn.
2.  🎓 **Nhân lực số là điều kiện nền, không phải chi phí:** 
    Kết quả từ M4 chỉ ra rằng đào tạo lại (đặc biệt trong khối Giáo dục - Đào tạo) tạo ra lượng việc làm ròng lớn gấp hàng chục lần so với đầu tư hạ tầng đơn thuần. Chính sách nhân lực số đóng vai trò là "lớp đệm" quan trọng nhất khi tự động hóa tăng tốc.
3.  📐 **Ràng buộc toán học định hình dòng ngân sách:** 
    Các ràng buộc công bằng vùng miền (sàn/trần) làm ngân sách lan tỏa hơn, giảm nguy cơ tập trung quá mức vào các vùng trọng điểm (Đông Nam Bộ, ĐBSH). Chi phí kinh tế của sự công bằng (~12% GDP) là hoàn toàn chấp nhận được so với lợi ích ổn định chính trị - xã hội.
4.  ⚖️ **Kịch bản cân bằng là phương án điều hành tối ưu:** 
    Theo điểm tổng hợp đa chiều (GDP - Gini - Cyber), **S4 (Bao trùm số)** và **S5 (Tối ưu cân bằng)** là hai phương án có hồ sơ rủi ro cân đối nhất để triển khai trong thực tế, đảm bảo Việt Nam không bị tụt hậu nhưng cũng không bỏ lại ai phía sau.

---

### 🛠️ CÔNG NGHỆ & CÀI ĐẶT

**Tech Stack:**
`Python` • `Streamlit` • `Pandas` • `NumPy` • `Plotly` • `PuLP` • `CVXPY` • `SciPy` • `Pymoo (NSGA-II)` • `Gymnasium (Q-Learning)`

**Chạy thử nghiệm trên máy local:**
```bash
# 1. Clone repository
git clone https://github.com/23050425-debug/NGUY-N-LINH-CHI.git
cd NGUY-N-LINH-CHI

# 2. Cài đặt thư viện
pip install -r requirements.txt

# 3. Khởi chạy Dashboard
streamlit run AIDEOM_VN.py
```

---

### 📚 DỮ LIỆU & NGUỒN THAM KHẢO

Hệ thống được xây dựng và chuẩn hóa dựa trên các nguồn dữ liệu chính thống giai đoạn 2020-2025:
*   **Trong nước:** Tổng cục Thống kê (GSO), Bộ KH&CN, Bộ TT&TT, Bộ KH&ĐT.
*   **Quốc tế:** World Bank (WDI), OECD.Stat, WIPO (GII 2025), Stanford HAI (AI Index), IMF, UN Comtrade, Codex.

---
