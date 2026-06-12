import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import pulp

PRIMARY_PURPLE = "#7e22ce"
PURPLE_SHADES = ['#f3e8ff', '#e9d5ff', '#d4b8ff', '#d8b4fe', '#a855f7',  '#9333ea', '#7e22ce', '#4c1d95']
TEXT_COLOR = "#4c1d95"

def render(macro_df, sectors_df, regions_df):
    st.header("Bài 5: Lựa chọn Dự án Chuyển đổi số (MIP)")

    # --- Cấu hình ---
    with st.container(border=True):
        st.markdown("### Cấu hình Mô hình & Ràng buộc")
        col1, col2, col3 = st.columns(3)
        total_budget = col1.number_input("Tổng ngân sách 5 năm (tỷ VND)", value=80000, step=1000)
        budget_year12 = col2.number_input("Ngân sách năm 1-2 (tỷ VND)", value=40000, step=1000)
        min_proj = col3.number_input("Số dự án tối thiểu", value=7, min_value=1, max_value=15)

        st.markdown("#### Ràng buộc đặc biệt")
        col_r1, col_r2, col_r3 = st.columns(3)
        with col_r1:
            p1_p2_excl = st.selectbox("P1 vs P2 (loại trừ)", options=["Chỉ chọn 1", "Chọn cả 2", "Tự do"], index=0)
        with col_r2:
            req_ai = st.checkbox("P8 (AI) cần P12 (Đào tạo)", value=True)
        with col_r3:
            req_chip = st.checkbox("P13 (Bán dẫn) cần P12 (Đào tạo)", value=True)

        st.markdown("#### Mô hình hóa rủi ro")
        use_expected = st.checkbox("Tối đa hóa Lợi ích Kỳ vọng E[Z] (có xét xác suất hoàn thành)", value=False)

    # --- Dữ liệu dự án ---
    projects_data = [
        ("P1", "TT Dữ liệu quốc gia Hòa Lạc", "Hạ tầng", 12000, 21500, 8500, 3500),
        ("P2", "TT Dữ liệu quốc gia phía Nam", "Hạ tầng", 11500, 20800, 7500, 4000),
        ("P3", "Hệ thống 5G phủ sóng toàn quốc", "Hạ tầng", 18000, 32500, 12000, 6000),
        ("P4", "Định danh điện tử VNeID 2.0", "Chính phủ số", 4500, 9200, 3500, 1000),
        ("P5", "Cổng dịch vụ công quốc gia v3", "Chính phủ số", 3200, 6800, 2500, 700),
        ("P6", "Y tế số quốc gia", "Y tế số", 5800, 11400, 4000, 1800),
        ("P7", "Giáo dục số K-12 toàn quốc", "Giáo dục", 6500, 12200, 4500, 2000),
        ("P8", "TT AI quốc gia + siêu máy tính", "AI", 15000, 28500, 9000, 6000),
        ("P9", "Sandbox tài chính số", "Tài chính số", 2500, 5800, 1800, 700),
        ("P10", "Logistics thông minh + cảng biển số", "Logistics", 7200, 13800, 5000, 2200),
        ("P11", "Nông nghiệp số ĐBSCL", "Nông nghiệp", 4800, 8500, 3500, 1300),
        ("P12", "Đào tạo 50.000 kỹ sư AI/bán dẫn", "Nhân lực", 8500, 16200, 5500, 3000),
        ("P13", "Khu CN bán dẫn Bắc Ninh-BG", "Bán dẫn", 20000, 35000, 13000, 7000),
        ("P14", "An ninh mạng quốc gia (SOC)", "An ninh", 3800, 7500, 2800, 1000),
        ("P15", "Open Data + dữ liệu mở quốc gia", "Dữ liệu", 1500, 3800, 1200, 300)
    ]
    df_proj = pd.DataFrame(projects_data, columns=["ID", "Tên dự án", "Lĩnh vực", "Chi phí", "Lợi ích NPV", "Năm 1-2", "Năm 3-5"])
    # Xác suất hoàn thành theo gợi ý
    df_proj["Xác suất"] = df_proj["Lĩnh vực"].map({
        "Hạ tầng": 0.85, "Chính phủ số": 0.75, "AI": 0.65, "Bán dẫn": 0.65
    }).fillna(0.80)
    df_proj["Lợi ích E[Z]"] = df_proj["Lợi ích NPV"] * df_proj["Xác suất"]

    # --- Mô hình PuLP ---
    prob = pulp.LpProblem("Project_Selection_MIP", pulp.LpMaximize)
    y = pulp.LpVariable.dicts("y", df_proj.index, cat="Binary")

    # Hàm mục tiêu
    if use_expected:
        prob += pulp.lpSum(df_proj.loc[i, "Lợi ích E[Z]"] * y[i] for i in df_proj.index)
    else:
        prob += pulp.lpSum(df_proj.loc[i, "Lợi ích NPV"] * y[i] for i in df_proj.index)

    # Ràng buộc
    prob += pulp.lpSum(df_proj.loc[i, "Chi phí"] * y[i] for i in df_proj.index) <= total_budget
    prob += pulp.lpSum(df_proj.loc[i, "Năm 1-2"] * y[i] for i in df_proj.index) <= budget_year12
    prob += pulp.lpSum(y[i] for i in df_proj.index) >= min_proj
    prob += pulp.lpSum(y[i] for i in df_proj.index) <= 11  # Max 11 dự án
    prob += y[13] >= 1  # P14 bắt buộc
    prob += y[3] + y[4] >= 1  # P4 hoặc P5

    # Ràng buộc P1-P2
    if p1_p2_excl == "Chỉ chọn 1":
        prob += y[0] + y[1] <= 1
    elif p1_p2_excl == "Chọn cả 2":
        prob += y[0] == 1
        prob += y[1] == 1

    # Ràng buộc tiên quyết
    if req_ai:
        prob += y[7] <= y[11]  # P8 <= P12
    if req_chip:
        prob += y[12] <= y[11] # P13 <= P12

    prob.solve(pulp.PULP_CBC_CMD(msg=False))

    status = pulp.LpStatus[prob.status]

    if status != "Optimal":
        st.error(f"Mô hình không tìm được nghiệm khả thi: {status}")
        return

    # --- Trích xuất kết quả ---
    selected_indices = [i for i in df_proj.index if pulp.value(y[i]) > 0.5]
    df_selected = df_proj.loc[selected_indices].copy()

    total_cost = df_selected["Chi phí"].sum()
    total_npv = df_selected["Lợi ích NPV"].sum()
    total_ez = df_selected["Lợi ích E[Z]"].sum()
    marginal_ratio = total_npv / total_cost if total_cost > 0 else 0

    # --- Hiển thị kết quả ---
    with st.container(border=True):
        st.markdown("#### Câu 5.4.1: Danh sách Dự án Được Chọn")
        st.success(f"Z* (Lợi ích tối đa) = **{total_npv if not use_expected else total_ez:,.0f}** tỷ VND")
        st.metric(label="Tổng chi phí", value=f"{total_cost:,.0f} tỷ VND", delta=f"trên {total_budget:,.0f}")
        st.metric(label="Tổng lợi ích", value=f"{total_npv:,.0f} tỷ VND", delta=None)
        st.info(f"**NPV Biên = {marginal_ratio:.3f}** (Mỗi tỷ đồng đầu tư, kỳ vọng thu về {marginal_ratio:.3f} tỷ đồng lợi ích)")

        st.dataframe(
            df_selected[["ID", "Tên dự án", "Lĩnh vực", "Chi phí", "Lợi ích NPV", "Xác suất", "Lợi ích E[Z]"]]
            .style.format({
                "Chi phí": "{:,.0f}",
                "Lợi ích NPV": "{:,.0f}",
                "Lợi ích E[Z]": "{:,.0f}",
                "Xác suất": "{:.2f}"
            }),
            width="stretch",
            hide_index=True
        )
        st.success("**Kết luận 5.4.1**: Mô hình MIP đã chọn ra tập hợp dự án tối ưu, cân bằng giữa quy mô lợi ích và giới hạn ngân sách/nhiều ràng buộc.")

    # --- Câu 5.4.2: So sánh ngân sách 100k ---
    with st.container(border=True):
        st.markdown("#### Câu 5.4.2: So sánh khi nới ngân sách lên 100.000 tỷ VND")

        # Giải lại với ngân sách 100k
        prob_100k = pulp.LpProblem("PS_100k", pulp.LpMaximize)
        y_100k = pulp.LpVariable.dicts("y", df_proj.index, cat="Binary")
        prob_100k += pulp.lpSum(df_proj.loc[i, "Lợi ích NPV"] * y_100k[i] for i in df_proj.index)
        prob_100k += pulp.lpSum(df_proj.loc[i, "Chi phí"] * y_100k[i] for i in df_proj.index) <= 100000
        prob_100k += pulp.lpSum(df_proj.loc[i, "Năm 1-2"] * y_100k[i] for i in df_proj.index) <= budget_year12
        prob_100k += pulp.lpSum(y_100k[i] for i in df_proj.index) >= min_proj
        prob_100k += pulp.lpSum(y_100k[i] for i in df_proj.index) <= 15  # Có thể chọn nhiều hơn
        prob_100k += y_100k[13] >= 1
        prob_100k += y_100k[3] + y_100k[4] >= 1
        if p1_p2_excl == "Chỉ chọn 1":
            prob_100k += y_100k[0] + y_100k[1] <= 1
        elif p1_p2_excl == "Chọn cả 2":
            prob_100k += y_100k[0] == 1
            prob_100k += y_100k[1] == 1
        if req_ai:
            prob_100k += y_100k[7] <= y_100k[11]
        if req_chip:
            prob_100k += y_100k[12] <= y_100k[11]

        prob_100k.solve(pulp.PULP_CBC_CMD(msg=False))
        if pulp.LpStatus[prob_100k.status] == "Optimal":
            sel_100k = [i for i in df_proj.index if pulp.value(y_100k[i]) > 0.5]
            df_sel_100k = df_proj.loc[sel_100k].copy()
            z_100k = df_sel_100k["Lợi ích NPV"].sum()
            st.info(f"Z* với ngân sách 100.000 tỷ: **{z_100k:,.0f}** tỷ VND")
            st.metric(label="Tăng thêm", value=f"{z_100k - total_npv:,.0f} tỷ", delta_color="normal")
            st.success("**Kết luận 5.4.2**: Khi ngân sách dồi dào, mô hình chọn thêm các dự án có lợi ích cao, làm tăng đáng kể Z*. Tuy nhiên, sự tăng trưởng không hoàn toàn tuyến tính do tính rời rạc của bài toán nhị phân.")

    # --- Câu 5.4.3: Yêu cầu P1 và P2 ---
    with st.container(border=True):
        st.markdown("#### Câu 5.4.3: Yêu cầu chọn cả P1 và P2 (Redundancy)")
        if p1_p2_excl == "Chọn cả 2":
            z_with_both = total_npv if not use_expected else total_ez
            st.warning(f"Z* khi bắt buộc chọn cả P1 và P2: **{z_with_both:,.0f}** tỷ VND")
            st.success("**Kết luận 5.4.3**: Việc yêu cầu chọn cả P1 và P2 (để tăng tính dự phòng) có thể làm giảm Z* vì chiếm dụng ngân sách cho 2 dự án lớn nhưng không tăng hiệu quả biên. Đây là chi phí của sự an toàn (redundancy cost).")

    # --- Câu 5.4.4: Dùng E[Z] ---
    if use_expected:
        with st.container(border=True):
            st.markdown("#### Câu 5.4.4: Mô hình hóa rủi ro dự án (E[Z])")
            st.info("Mô hình hiện đang tối đa hóa Lợi ích Kỳ vọng (E[Z] = NPV * p).")
            df_selected_sorted_by_ez = df_selected.sort_values(by="Lợi ích E[Z]", ascending=False)
            st.dataframe(
                df_selected_sorted_by_ez[["ID", "Tên dự án", "Lợi ích NPV", "Xác suất", "Lợi ích E[Z]"]]
                .style.highlight_max(subset=["Lợi ích E[Z]"], color=PURPLE_SHADES[4]),
                width="stretch",
                hide_index=True
            )
            st.success("**Kết luận 5.4.4**: Khi tính đến rủi ro (xác suất), mô hình ưu tiên các dự án có xác suất hoàn thành cao hơn, dù NPV danh nghĩa có thể thấp hơn. Điều này phản ánh cách tiếp cận thận trọng trong đầu tư công.")

    # --- Biểu đồ ---
    col1, col2 = st.columns(2)
    with col1:
        with st.container(border=True):
            st.markdown("#### Phân bổ theo Lĩnh vực")
            df_grouped = df_selected.groupby("Lĩnh vực")[["Chi phí", "Lợi ích NPV"]].sum().reset_index()
            fig_bar = go.Figure(data=[
                go.Bar(name='Chi phí', x=df_grouped["Lĩnh vực"], y=df_grouped["Chi phí"], marker_color=PURPLE_SHADES[2]),
                go.Bar(name='Lợi ích', x=df_grouped["Lĩnh vực"], y=df_grouped["Lợi ích NPV"], marker_color=PURPLE_SHADES[4])
            ])
            fig_bar.update_layout(barmode='group', margin=dict(t=30), font=dict(color=TEXT_COLOR))
            st.plotly_chart(fig_bar, width="stretch")

    with col2:
        with st.container(border=True):
            st.markdown("#### Danh sách Dự án Chọn")
            fig_pie = px.pie(df_selected, values="Chi phí", names="Lĩnh vực", color_discrete_sequence=PURPLE_SHADES[1:8])
            fig_pie.update_layout(margin=dict(t=10, b=10, l=10, r=10), font=dict(color=TEXT_COLOR))
            st.plotly_chart(fig_pie, width="stretch")

    # --- Câu 5.5: Thảo luận chính sách ---
    with st.container(border=True):
        st.markdown("### Câu 5.5: Thảo luận Chính sách")

        st.markdown("#### a) Vì sao P15 (Open Data) bị bỏ qua?")
        st.info(
            "P15 có tỷ suất lợi nhuận (ROI) cao, nhưng tổng lợi ích tuyệt đối nhỏ. Trong mô hình Knapsack với ngân sách bị giới hạn, thuật toán ưu tiên các dự án có quy mô lớn để lấp đầy ngân sách, nhằm tối đa hóa tổng Z*. Điều này cho thấy mô hình toán học có thể bỏ sót các dự án đột phá nhỏ nhưng chiến lược."
        )

        st.markdown("#### b) Ràng buộc bắt buộc P14 (An ninh mạng) có hợp lý?")
        st.warning(
            "Ràng buộc bắt buộc P14 làm giảm Z*, nhưng điều này là hợp lý. An ninh mạng không thể được đánh giá chỉ bằng lợi ích kinh tế. Đây là dự án mang lại lợi ích phi kinh tế (an toàn, ổn định hệ thống), cần được đảm bảo đầu tư bất chấp hiệu quả tài chính."
        )

        st.markdown("#### c) Làm thế nào để mô hình hóa lợi ích cộng hưởng giữa P8 (AI) và P13 (Bán dẫn)?")
        st.info(
            "Để mô hình hóa hiệu ứng cộng hưởng, ta có thể thêm biến nhị phân phụ `y_p8_p13`, đại diện cho việc cả hai dự án P8 và P13 đều được chọn. Sau đó, thêm ràng buộc tuyến tính hóa: `y_p8_p13 <= y_P8`, `y_p8_p13 <= y_P13`, `y_p8_p13 >= y_P8 + y_P13 - 1`. Cuối cùng, cộng thêm `B_synergy * y_p8_p13` vào hàm mục tiêu. Điều này sẽ phản ánh đúng giá trị tổng hợp khi hai dự án này cùng được thực hiện."
        )


