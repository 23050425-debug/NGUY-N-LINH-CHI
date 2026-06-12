import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

PRIMARY_PURPLE = "#7e22ce"
PURPLE_SHADES = ['#f3e8ff', '#e9d5ff', '#d8b4fe', '#a855f7', '#7e22ce', '#4c1d95']
TEXT_COLOR = "#4c1d95"

def render(macro_df, sectors_df, regions_df):
    st.header("Bài 3: Chỉ số Ưu tiên Ngành (TOPSIS)")

    with st.container(border=True):
        st.markdown("### Cấu hình Trọng số Chính sách")
        cols = st.columns(7)
        w_growth = cols[0].number_input("Tăng trưởng (a₁)", value=0.15, min_value=0.0, max_value=1.0, step=0.05)
        w_productivity = cols[1].number_input("Năng suất (a₂)", value=0.15, min_value=0.0, max_value=1.0, step=0.05)
        w_spillover = cols[2].number_input("Lan tỏa (a₃)", value=0.20, min_value=0.0, max_value=1.0, step=0.05)
        w_export = cols[3].number_input("Xuất khẩu (a₄)", value=0.15, min_value=0.0, max_value=1.0, step=0.05)
        w_labor = cols[4].number_input("Việc làm (a₅)", value=0.10, min_value=0.0, max_value=1.0, step=0.05)
        w_ai = cols[5].number_input("AI (a₆)", value=0.20, min_value=0.0, max_value=1.0, step=0.05)
        w_risk = cols[6].number_input("Rủi ro (a₇)", value=0.15, min_value=0.0, max_value=1.0, step=0.05)

        weights = np.array([w_growth, w_productivity, w_spillover, w_export, w_labor, w_ai, w_risk])
        weights = weights / weights.sum() if weights.sum() != 0 else np.ones(7)/7
        st.caption(f"**Tổng trọng số**: {weights.sum():.3f} (đã chuẩn hóa)")

    # --- Câu 3.4.1: Chuẩn hóa dữ liệu ---
    cols_good = ['growth_rate_2024_pct', 'gdp_share_2024_pct', 'spillover_coef_0_1', 'export_billion_USD', 'labor_million', 'ai_readiness_0_100']
    col_bad = 'automation_risk_pct'

    def norm_good(x):
        x_min, x_max = x.min(), x.max()
        if x_max == x_min:
            return x * 0
        return (x - x_min) / (x_max - x_min)

    def norm_bad(x):
        x_min, x_max = x.min(), x.max()
        if x_max == x_min:
            return x * 0
        return (x_max - x) / (x_max - x_min)

    df3 = sectors_df.copy()
    Xg = df3[cols_good].apply(norm_good)
    Xb = norm_bad(df3[col_bad])

    with st.container(border=True):
        st.markdown("#### Câu 3.4.1: Ma trận Dữ liệu Đã Chuẩn hóa")
        df_norm = Xg.copy()
        df_norm['Rủi ro (Xb)'] = Xb
        df_norm.index = sectors_df['sector_name_vi']
        st.dataframe(
            df_norm.style.format("{:.3f}").background_gradient(cmap="Purples", axis=0),
            width="stretch"
        )
        st.success("**Kết luận 3.4.1**: Ma trận đã chuẩn hóa theo phương pháp min-max. Cột 'Rủi ro' được nghịch đảo để trở thành tiêu chí có lợi.")

    # --- Câu 3.4.2: Tính Priority ---
    priority = Xg.values @ weights[:6] + weights[6] * Xb.values  # Cộng vì Xb đã nghịch đảo
    df3['Priority'] = priority
    df_ranked = df3.sort_values('Priority', ascending=False)

    col1, col2 = st.columns(2)
    with col1:
        with st.container(border=True):
            st.markdown("#### Câu 3.4.2: Xếp hạng Ưu tiên 10 Ngành")
            top_n = 10
            top_sectors = df_ranked.head(top_n)
            fig_bar = go.Figure(go.Bar(
                x=top_sectors['Priority'],
                y=top_sectors['sector_name_vi'],
                orientation='h',
                marker_color=PURPLE_SHADES[4],
                text=top_sectors['Priority'].round(3),
                textposition='auto'
            ))
            fig_bar.update_layout(
                xaxis_title="Điểm Ưu tiên (Priority)",
                yaxis_title="Ngành",
                margin=dict(t=30, b=20, l=100, r=20),
                font=dict(color=TEXT_COLOR)
            )
            st.plotly_chart(fig_bar, width="stretch")
            st.success(f"**Kết luận 3.4.2**: Top 3 ngành theo bộ trọng số hiện tại là: **{top_sectors.iloc[0]['sector_name_vi']}**, **{top_sectors.iloc[1]['sector_name_vi']}**, **{top_sectors.iloc[2]['sector_name_vi']}**.")

    # --- Câu 3.4.3: Độ nhạy AI Readiness ---
    with col2:
        with st.container(border=True):
            st.markdown("#### Câu 3.4.3: Phân tích Độ nhạy Trọng số AI")
            ai_range = np.arange(0.05, 0.41, 0.05)
            heatmap_data = []

            for w_ai_val in ai_range:
                temp_weights = np.array([0.15, 0.15, 0.20, 0.15, 0.10, w_ai_val, 0.15])
                temp_weights = temp_weights / temp_weights.sum()
                temp_priority = Xg.values @ temp_weights[:6] + temp_weights[6] * Xb.values
                temp_rank = len(temp_priority) - temp_priority.argsort().argsort()
                heatmap_data.append(temp_rank)

            heatmap_matrix = np.array(heatmap_data).T
            sector_labels = sectors_df['sector_name_vi'].tolist()

            fig_heat = px.imshow(
                heatmap_matrix,
                x=[f"a₆={w:.2f}" for w in ai_range],
                y=sector_labels,
                color_continuous_scale='Purples_r',
                aspect="auto",
                labels=dict(x="Trọng số AI (a₆)", y="Ngành", color="Thứ hạng")
            )
            fig_heat.update_layout(
                title="Bản đồ độ nhạy theo trọng số AI",
                margin=dict(t=40, b=20),
                font=dict(color=TEXT_COLOR)
            )
            st.plotly_chart(fig_heat, width="stretch")
            st.success("**Kết luận 3.4.3**: Thay đổi trọng số AI từ 0.05 đến 0.40 có thể làm thay đổi thứ hạng của các ngành, đặc biệt là các ngành có chỉ số AI Readiness cao.")

    # --- Câu 3.4.4: So sánh kịch bản ---
    with st.container(border=True):
        st.markdown("#### Câu 3.4.4: So sánh Kịch bản Trọng số")

        # Kịch bản tăng trưởng
        w_growth_oriented = np.array([0.35, 0.20, 0.10, 0.20, 0.05, 0.05, 0.05])
        w_growth_oriented = w_growth_oriented / w_growth_oriented.sum()
        priority_growth = Xg.values @ w_growth_oriented[:6] + w_growth_oriented[6] * Xb.values

        # Kịch bản bao trùm
        w_inclusive = np.array([0.10, 0.10, 0.25, 0.05, 0.25, 0.05, 0.20])
        w_inclusive = w_inclusive / w_inclusive.sum()
        priority_inclusive = Xg.values @ w_inclusive[:6] + w_inclusive[6] * Xb.values

        df_comparison = pd.DataFrame({
            "Ngành": sectors_df['sector_name_vi'],
            "Tăng trưởng": priority_growth,
            "Bao trùm": priority_inclusive
        }).round(3)

        col_s1, col_s2 = st.columns(2)
        with col_s1:
            df_growth_sorted = df_comparison[['Ngành', 'Tăng trưởng']].sort_values('Tăng trưởng', ascending=False)
            st.markdown("** Kịch bản: Tăng trưởng (ưu tiên tăng trưởng, năng suất, xuất khẩu)**")
            st.dataframe(
                df_growth_sorted.head(5).style.format({"Tăng trưởng": "{:.3f}"}).background_gradient(cmap="Purples"),
                width="stretch",
                hide_index=True
            )
        with col_s2:
            df_inclusive_sorted = df_comparison[['Ngành', 'Bao trùm']].sort_values('Bao trùm', ascending=False)
            st.markdown("** Kịch bản: Bao trùm (ưu tiên việc làm, lan tỏa, an toàn)**")
            st.dataframe(
                df_inclusive_sorted.head(5).style.format({"Bao trùm": "{:.3f}"}).background_gradient(cmap="Greens"),
                width="stretch",
                hide_index=True
            )

        top3_growth = df_growth_sorted.head(3)['Ngành'].tolist()
        top3_inclusive = df_inclusive_sorted.head(3)['Ngành'].tolist()

        st.success(f"**Kết luận 3.4.4**: Top 3 theo 'Tăng trưởng' là: {', '.join(top3_growth)}. Top 3 theo 'Bao trùm' là: {', '.join(top3_inclusive)}. Có sự khác biệt rõ rệt, ví dụ: Ngành nông nghiệp vươn lên trong kịch bản bao trùm.")

    # --- Câu 3.5: Thảo luận chính sách ---
    with st.container(border=True):
        st.markdown("### Câu 3.5: Thảo luận Chính sách")

        st.markdown("#### a) Ba ngành nào nên được ưu tiên chuyển đổi số và AI?")
        st.info(
            "Ba ngành có điểm Priority cao nhất theo bộ trọng số mặc định là: **Công nghệ thông tin, truyền thông**, **Công nghiệp chế biến, chế tạo**, và **Tài chính, ngân hàng, bảo hiểm**. Những ngành này có tiềm năng lớn về năng suất, xuất khẩu và khả năng sẵn sàng AI — phù hợp với định hướng phát triển kinh tế số theo Chiến lược Quốc gia về Chuyển đổi số."
        )

        st.markdown("#### b) Tại sao Khai khoáng có năng suất cao nhưng không được ưu tiên?")
        st.info(
            "Mặc dù Khai khoáng có năng suất đầu ra cao, nhưng nó có điểm thấp ở các tiêu chí khác như: lan tỏa công nghệ, tạo việc làm quy mô lớn, khả năng thích ứng với AI và rủi ro tự động hóa thấp. TOPSIS đánh giá toàn diện nhiều khía cạnh, nên ngành này không được ưu tiên cao."
        )

        st.markdown("#### c) Ai nên quyết định trọng số?")
        st.warning(
            "Việc xác định trọng số không nên chỉ do chuyên gia kỹ thuật quyết định. Thay vào đó, nên có sự tham gia của: (1) Chuyên gia kỹ thuật (đề xuất dữ liệu, mô hình), (2) Chính sách (đại diện Chính phủ, xác định ưu tiên quốc gia), (3) Cộng đồng (qua đối thoại, tham vấn công khai). Điều này đảm bảo mô hình phản ánh đúng giá trị xã hội, tăng tính minh bạch và chính danh cho quyết sách."
        )


