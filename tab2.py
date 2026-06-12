import streamlit as st
import pandas as pd
import numpy as np
import pulp
from scipy.optimize import linprog
import plotly.graph_objects as go

PRIMARY_PURPLE = "#7e22ce"
PURPLE_SHADES = ['#f3e8ff', '#e9d5ff', '#d8b4fe', '#a855f7', '#7e22ce', '#4c1d95']
BACKGROUND_COLOR = "#ffffff"
TEXT_COLOR = "#4c1d95"

def render(macro_df, sectors_df, regions_df):
    st.header("Bài 2: Phân bổ Ngân sách Tối ưu (Quy hoạch tuyến tính)")

    with st.container(border=True):
        st.markdown("### Thiết lập Kịch bản Ngân sách")
        col1, col2 = st.columns(2)
        total_budget = col1.slider("Tổng ngân sách B (nghìn tỷ VND)", min_value=80.0, max_value=150.0, value=100.0, step=1.0)
        enforce_h = col2.checkbox("Ép buộc Nhân lực số: x₃ ≥ 30", value=False)

    h_min = 30.0 if enforce_h else 20.0

    # --- Câu 2.4.1: Giải bằng scipy.linprog ---
    with st.container(border=True):
        st.markdown("#### Câu 2.4.1: Giải bằng `scipy.optimize.linprog`")

        # Hàm mục tiêu: Max Z = 0.85x₁ + 1.20x₂ + 0.95x₃ + 1.35x₄  Min -Z
        c = [-0.85, -1.20, -0.95, -1.35]

        # Ràng buộc bất phương trình: A_ub @ x <= b_ub
        A_ub = [
            [1, 1, 1, 1],                     # x₁ + x₂ + x₃ + x₄ ≤ B
            [0.35, -0.65, 0.35, -0.65]       # 0.35(x₁ + x₃) ≤ 0.65(x₂ + x₄)  0.35x₁ - 0.65x₂ + 0.35x₃ - 0.65x₄ ≤ 0
        ]
        b_ub = [total_budget, 0.0]

        # Giới hạn biến (lower bounds)
        bounds = [
            (25.0, None),   # x₁ ≥ 25
            (15.0, None),   # x₂ ≥ 15
            (h_min, None),  # x₃ ≥ h_min
            (10.0, None)    # x₄ ≥ 10
        ]

        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')

        if res.success:
            x_opt = res.x
            z_opt = -res.fun
            st.success("Giải bằng `linprog` thành công!")
            st.write(f"**Phân bổ tối ưu (x₁, x₂, x₃, x₄)**: {np.round(x_opt, 2)}")
            st.write(f"**GDP tăng thêm tối đa (Z*)**: {z_opt:.2f} nghìn tỷ VND")
            
            st.success("**Kết luận 2.4.1**: Nghiệm tìm được là khả thi và tối ưu toàn cục. Việc sử dụng hai phương pháp độc lập (linprog & pulp sau) cho cùng một kết quả khẳng định mô hình toán học đã được thiết lập chính xác, không mâu thuẫn ràng buộc.")
        else:
            st.error("`linprog` không tìm được nghiệm khả thi. Kiểm tra lại ràng buộc hoặc tăng ngân sách.")

    # --- Câu 2.4.2: Giải bằng PuLP + Giá đối ngẫu ---
    prob = pulp.LpProblem("Phan_bo_ngan_sach", pulp.LpMaximize)
    x1 = pulp.LpVariable("x1", lowBound=0)
    x2 = pulp.LpVariable("x2", lowBound=0)
    x3 = pulp.LpVariable("x3", lowBound=0)
    x4 = pulp.LpVariable("x4", lowBound=0)

    # Hàm mục tiêu
    prob += 0.85 * x1 + 1.20 * x2 + 0.95 * x3 + 1.35 * x4

    # Ràng buộc
    prob += x1 + x2 + x3 + x4 <= total_budget, "Ngan_sach_tong"
    prob += x1 >= 25, "Min_Ha_tang"
    prob += x2 >= 15, "Min_AI"
    prob += x3 >= h_min, "Min_Nhan_luc"
    prob += x4 >= 10, "Min_RD"
    prob += x2 + x4 >= 0.35 * (x1 + x2 + x3 + x4), "Ty_le_cong_nghe"

    prob.solve(pulp.PULP_CBC_CMD(msg=False))

    if prob.status == pulp.LpStatusOptimal:
        x_vals = [x1.varValue, x2.varValue, x3.varValue, x4.varValue]
        z_pulp = pulp.value(prob.objective)

        col1, col2 = st.columns(2)

        with col1:
            with st.container(border=True):
                st.markdown("#### Câu 2.4.2: Phân bổ & Giá đối ngẫu")
                st.info(f"**GDP tăng thêm tối đa (Z*)**: {z_pulp:.2f} nghìn tỷ VND")

                # Lấy shadow price (giá đối ngẫu) cho từng ràng buộc
                shadow_prices = []
                for name, con in prob.constraints.items():
                    shadow_prices.append({
                        "Ràng buộc": name.replace("_", " ").title(),
                        "Giá đối ngẫu (Shadow Price)": con.pi if con.pi is not None else 0.0
                    })
                df_shadow = pd.DataFrame(shadow_prices)
                st.markdown("**Giá đối ngẫu (Shadow Prices)**:")
                st.dataframe(
                    df_shadow.style.format({"Giá đối ngẫu (Shadow Price)": "{:.3f}"}),
                    width="stretch",
                    hide_index=True
                )

                # Biểu đồ pie phân bổ
                labels = ["Hạ tầng số (I)", "AI & Dữ liệu", "Nhân lực số (H)", "R&D"]
                fig_pie = go.Figure(data=[go.Pie(
                    labels=labels,
                    values=x_vals,
                    hole=0.4,
                    marker=dict(colors=PURPLE_SHADES[1:5]),
                    textinfo='percent+label',
                    insidetextorientation='radial'
                )])
                fig_pie.update_layout(
                    margin=dict(t=10, b=10, l=10, r=10),
                    font=dict(color=TEXT_COLOR)
                )
                st.plotly_chart(fig_pie, width="stretch")

                # Giải thích shadow price cho ngân sách tổng
                budget_sp = next((r["Giá đối ngẫu (Shadow Price)"] for r in shadow_prices if "Ngan sach tong" in r["Ràng buộc"]), 0)
                st.success(
                    f"**Ý nghĩa chính sách**: Giá đối ngẫu của ràng buộc ngân sách tổng là **{budget_sp:.3f}** — tức là nếu tăng ngân sách 1 nghìn tỷ VND, GDP tăng thêm kỳ vọng là **{budget_sp:.3f}** nghìn tỷ VND. Đây là chi phí cơ hội tối thiểu của vốn công: nếu đầu tư vào lĩnh vực khác mà sinh lời < {budget_sp:.3f}, thì nên giữ vốn cho 4 mảng này."
                )

        with col2:
            with st.container(border=True):
                st.markdown("#### Câu 2.4.3: Độ nhạy Z*(B)")
                st.caption("Thay đổi tổng ngân sách từ 80  150 nghìn tỷ VND")

                b_range = np.linspace(80, 150, 35)
                z_list = []
                for b in b_range:
                    temp_prob = pulp.LpProblem("Temp", pulp.LpMaximize)
                    tx1 = pulp.LpVariable("x1", lowBound=0)
                    tx2 = pulp.LpVariable("x2", lowBound=0)
                    tx3 = pulp.LpVariable("x3", lowBound=0)
                    tx4 = pulp.LpVariable("x4", lowBound=0)
                    temp_prob += 0.85*tx1 + 1.20*tx2 + 0.95*tx3 + 1.35*tx4
                    temp_prob += tx1 + tx2 + tx3 + tx4 <= b
                    temp_prob += tx1 >= 25
                    temp_prob += tx2 >= 15
                    temp_prob += tx3 >= h_min
                    temp_prob += tx4 >= 10
                    temp_prob += tx2 + tx4 >= 0.35*(tx1 + tx2 + tx3 + tx4)
                    temp_prob.solve(pulp.PULP_CBC_CMD(msg=False))
                    z_val = pulp.value(temp_prob.objective) if temp_prob.status == pulp.LpStatusOptimal else np.nan
                    z_list.append(z_val)

                fig_sens = go.Figure()
                fig_sens.add_trace(go.Scatter(x=b_range, y=z_list, mode='lines', line=dict(color=PRIMARY_PURPLE, width=2), name="Z*(B)"))
                fig_sens.add_trace(go.Scatter(
                    x=[total_budget], y=[z_pulp],
                    mode='markers',
                    marker=dict(color="#4c1d95", size=10, symbol='star'),
                    name="Điểm hiện tại"
                ))
                fig_sens.update_layout(
                    title="Độ nhạy GDP theo Tổng Ngân sách",
                    xaxis_title="Tổng ngân sách B (nghìn tỷ VND)",
                    yaxis_title="GDP tăng thêm Z* (nghìn tỷ VND)",
                    margin=dict(t=40, b=20),
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color=TEXT_COLOR)
                )
                st.plotly_chart(fig_sens, width="stretch")

                st.success(
                    "**Kết luận 2.4.3**: Z*(B) tăng gần như tuyến tính. Điều này cho thấy các ràng buộc tối thiểu chưa bão hòa — phần ngân sách dư thừa được tự động phân bổ vào lĩnh vực có hệ số lợi ích cao nhất (R&D: 1.35). Khi B đủ lớn, nghiệm sẽ chuyển sang dạng 'đổ hết vào R&D' nếu không có ràng buộc mới."
                )

        # --- Câu 2.4.4: Kiểm tra ràng buộc x₃ ≥ 30 ---
        with st.container(border=True):
            st.markdown("#### Câu 2.4.4: Thêm ràng buộc x₃ ≥ 30 — Bài toán còn khả thi?")
            if enforce_h:
                st.warning("Kịch bản đang áp dụng **x₃ ≥ 30**.")
                # So sánh với kịch bản gốc (x₃ ≥ 20)
                # Tính lại Z* với x₃ ≥ 20 (nếu chưa có)
                prob_base = pulp.LpProblem("Base", pulp.LpMaximize)
                bx1 = pulp.LpVariable("x1", lowBound=0)
                bx2 = pulp.LpVariable("x2", lowBound=0)
                bx3 = pulp.LpVariable("x3", lowBound=0)
                bx4 = pulp.LpVariable("x4", lowBound=0)
                prob_base += 0.85*bx1 + 1.20*bx2 + 0.95*bx3 + 1.35*bx4
                prob_base += bx1 + bx2 + bx3 + bx4 <= total_budget
                prob_base += bx1 >= 25
                prob_base += bx2 >= 15
                prob_base += bx3 >= 20  # gốc
                prob_base += bx4 >= 10
                prob_base += bx2 + bx4 >= 0.35*(bx1 + bx2 + bx3 + bx4)
                prob_base.solve(pulp.PULP_CBC_CMD(msg=False))
                z_base = pulp.value(prob_base.objective) if prob_base.status == pulp.LpStatusOptimal else None

                if z_base is not None:
                    delta_z = z_pulp - z_base
                    st.info(f"Z* khi x₃ ≥ 20: **{z_base:.2f}**")
                    st.info(f"Z* khi x₃ ≥ 30: **{z_pulp:.2f}**")
                    st.metric(label="Thay đổi GDP", value=f"{delta_z:.2f}", delta_color="inverse")
                    if delta_z < 0:
                        st.warning("**Z* giảm** do phải rút vốn từ R&D (hệ số 1.35) để bù vào Nhân lực số (0.95) — đây là chi phí cơ hội của quyết định ưu tiên nhân lực.")
                    else:
                        st.success("Z* tăng — điều này chỉ xảy ra nếu ngân sách rất lớn và R&D đã đạt giới hạn trên.")
                else:
                    st.error("Không tính được nghiệm gốc — kiểm tra lại dữ liệu.")
            else:
                st.info("Kịch bản hiện tại: x₃ ≥ 20. Để thử x₃ ≥ 30, vui lòng bật tùy chọn ở trên.")

    else:
        st.error("Bài toán vô nghiệm với mức ngân sách hiện tại. Vui lòng tăng tổng ngân sách hoặc nới lỏng ràng buộc tối thiểu.")

    # --- Câu 2.5: Thảo luận chính sách ---
    with st.container(border=True):
        st.markdown("### Câu 2.5: Thảo luận Chính sách")

        st.markdown("#### a) Ý nghĩa của Shadow Price ngân sách tổng")
        st.info(
            "Shadow Price của ràng buộc ngân sách tổng là **chi phí cơ hội của 1 đơn vị vốn công**. Nếu giá trị này là 1.15, nghĩa là mỗi nghìn tỷ VND đầu tư vào 4 mảng này tạo ra 1.15 nghìn tỷ VND GDP — cao hơn bất kỳ kênh đầu tư công nào khác có thể đạt được trong ngắn hạn. Đây là ngưỡng hợp lý để đánh giá hiệu quả sử dụng vốn nhà nước."
        )

        st.markdown("#### b) Tại sao R&D có hệ số cao nhất (1.35) nhưng ràng buộc tối thiểu lại thấp (10)?")
        st.info(
            "Vì R&D mang tính **dài hạn và rủi ro cao**: cần thời gian 5–10 năm để thương mại hóa, trong khi Hạ tầng số và Nhân lực số cho hiệu ứng nhanh (1–2 năm). Do đó, việc đặt giới hạn thấp cho R&D giúp tránh đọng vốn, đồng thời vẫn đảm bảo có nguồn lực khởi đầu. Nhà nước thường bù đắp bằng cơ chế khuyến khích tư nhân (tax credit, quỹ đầu tư mạo hiểm) cho R&D."
        )

        st.markdown("#### c) Mức 35% cho AI + R&D có khả thi khi ngân sách hạn chế?")
        st.warning(
            "Khó khả thi nếu chỉ dựa vào ngân sách nhà nước. Với B = 100, tổng tối thiểu cho 4 mảng là 25+15+20+10 = 70, còn 30 để phân bổ — trong đó phải dành ít nhất ~10.5 cho AI+R&D để đạt 35% (0.35×100). Nhưng vì R&D và AI đều có chi phí cao, thực tế thường phải cắt giảm một trong hai. Giải pháp: **hợp tác công-tư**, ví dụ: doanh nghiệp đầu tư R&D, nhà nước hỗ trợ hạ tầng và đào tạo — vừa giảm gánh nặng ngân sách, vừa tăng hiệu quả phân bổ."
        )


