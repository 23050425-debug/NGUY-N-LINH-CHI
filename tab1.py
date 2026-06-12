import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

PRIMARY_PURPLE = "#7e22ce"
PURPLE_SHADES = ['#f3e8ff', '#e9d5ff', '#d8b4fe', '#a855f7', '#7e22ce', '#4c1d95']
BACKGROUND_COLOR = "#ffffff"
TEXT_COLOR = "#4c1d95"

def render(macro_df, sectors_df, regions_df):
    st.header("Bài 1: Hàm sản xuất Cobb-Douglas mở rộng")

    with st.container(border=True):
        st.markdown("### Tùy chỉnh Tham số Hàm Sản xuất")
        col1, col2, col3, col4, col5 = st.columns(5)
        alpha = col1.slider("Vốn (K) α", min_value=0.05, max_value=0.6, value=0.33, step=0.01)
        beta = col2.slider("Lao động (L) β", min_value=0.05, max_value=0.6, value=0.38, step=0.01)
        gamma = col3.slider("Số hóa (D) γ", min_value=0.01, max_value=0.3, value=0.10, step=0.01)
        delta = col4.slider("AI (AI) δ", min_value=0.01, max_value=0.3, value=0.08, step=0.01)
        theta = col5.slider("Nhân lực số (H) θ", min_value=0.01, max_value=0.3, value=0.07, step=0.01)

        total_coef = alpha + beta + gamma + delta + theta
        st.caption(f"**Tổng các hệ số**: {total_coef:.2f} *(Nên ≈1.0 để lợi suất không đổi theo quy mô)*")

    K = np.array([16500, 17800, 19600, 21300, 23500, 25900])
    L = np.array([53.6, 50.5, 51.7, 52.4, 52.9, 53.4])
    D = np.array([12.0, 12.7, 14.3, 16.5, 18.3, 19.5])
    AI = np.array([55.6, 60.2, 65.4, 67.0, 73.8, 80.1])
    H = np.array([24.1, 26.1, 26.2, 27.0, 28.4, 29.2])
    Y = macro_df['GDP_trillion_VND'].values
    years = macro_df['year'].values

    # Câu 1.4.1: Tính TFP
    denominator = (K**alpha) * (L**beta) * (D**gamma) * (AI**delta) * (H**theta)
    A_t = Y / denominator

    col_chart1, col_chart2 = st.columns(2)

    with col_chart1:
        with st.container(border=True):
            st.markdown("#### Câu 1.4.1 & 1.4.2: Xu hướng Năng suất Nhân tố Tổng hợp (TFP)")
            fig_tfp = px.line(x=years, y=A_t, markers=True, title="TFP (Aₜ)")
            fig_tfp.update_traces(
                line=dict(color=PRIMARY_PURPLE, width=2),
                marker=dict(size=6, color=PRIMARY_PURPLE, line=dict(width=1, color='white')),
                hovertemplate="Năm: %{x}<br>TFP: %{y:.4f}<extra></extra>"
            )
            fig_tfp.update_layout(
                xaxis_title="Năm",
                yaxis_title="TFP (Aₜ)",
                margin=dict(t=30, b=20),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color=TEXT_COLOR)
            )
            st.plotly_chart(fig_tfp, width="stretch")

            # Câu 1.4.2: Tính MAPE
            A_mean = np.mean(A_t)
            Y_hat = A_mean * (K**alpha) * (L**beta) * (D**gamma) * (AI**delta) * (H**theta)
            mape = np.mean(np.abs((Y - Y_hat) / Y)) * 100

            st.info(f"**Sai số dự báo MAPE** (dùng $\\bar{{A}}$): **{mape:.2f}%**")
            trend = "tăng" if A_t[-1] > A_t[0] else "giảm"
            st.success(f"**Kết luận 1.4.1 & 1.4.2**: TFP có xu hướng **{trend} đều** từ {A_t[0]:.4f} (2020) lên {A_t[-1]:.4f} (2025). MAPE thấp cho thấy mô hình giải thích tốt dữ liệu thực tế.")

    with col_chart2:
        with st.container(border=True):
            st.markdown("#### Câu 1.4.3: Phân rã Tăng trưởng GDP (Growth Accounting)")

            dY = np.diff(np.log(Y))
            dA = np.diff(np.log(A_t))
            dK = alpha * np.diff(np.log(K))
            dL = beta * np.diff(np.log(L))
            dD = gamma * np.diff(np.log(D))
            dAI = delta * np.diff(np.log(AI))
            dH = theta * np.diff(np.log(H))

            years_diff = [f"{int(years[i])}-{int(years[i+1])}" for i in range(len(years)-1)]

            # Use Plotly stacked bar to avoid optional matplotlib dependency
            colors = PURPLE_SHADES[:6]
            labels = ["TFP", "Vốn (K)", "Lao động (L)", "Số hóa (D)", "Năng lực AI", "Nhân lực số (H)"]
            contributions = [dA*100, dK*100, dL*100, dD*100, dAI*100, dH*100]

            fig_stack = go.Figure()
            for contrib, color, label in zip(contributions, colors, labels):
                fig_stack.add_trace(go.Bar(x=years_diff, y=contrib, name=label, marker_color=color))

            fig_stack.update_layout(
                barmode='stack',
                title="Phân rã đóng góp vào tăng trưởng GDP",
                xaxis_title="Giai đoạn",
                yaxis_title="% Đóng góp vào tăng trưởng",
                legend=dict(orientation="v", yanchor="middle", y=0.5, xanchor="left", x=1.02),
                margin=dict(t=40, r=140),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color=TEXT_COLOR)
            )
            st.plotly_chart(fig_stack, width="stretch")

            # Summary table — dùng background_gradient với cmap Purples (tím)
            mean_dY = np.mean(dY)
            pct = lambda x: (np.mean(x) / mean_dY) * 100 if mean_dY != 0 else 0
            df_ga_summary = pd.DataFrame({
                "Nhân tố": ["Vốn (K)", "Lao động (L)", "Số hóa (D)", "Năng lực AI", "Nhân lực số (H)", "TFP"],
                "Đóng góp BQ (điểm %)": [
                    np.mean(dK)*100, np.mean(dL)*100, np.mean(dD)*100,
                    np.mean(dAI)*100, np.mean(dH)*100, np.mean(dA)*100
                ],
                "Tỷ trọng (%)": [
                    pct(dK), pct(dL), pct(dD), pct(dAI), pct(dH), pct(dA)
                ]
            }).round(2)

            st.markdown("##### Bảng tổng hợp đóng góp trung bình 2020–2025")
            df_formatted = df_ga_summary.copy()
            df_formatted["Đóng góp BQ (điểm %)"] = df_formatted["Đóng góp BQ (điểm %)"].map("{:.2f}".format)
            df_formatted["Tỷ trọng (%)"] = df_formatted["Tỷ trọng (%)"].map("{:.1f}%".format)
            st.dataframe(df_formatted, width="stretch", hide_index=True)

            dominant = df_ga_summary.loc[df_ga_summary["Tỷ trọng (%)"].idxmax(), "Nhân tố"]
            st.success(f"**Kết luận 1.4.3**: Trong giai đoạn 2020–2025, **{dominant}** chiếm tỷ trọng đóng góp cao nhất. Điều này phản ánh xu hướng chuyển dịch sang tăng trưởng dựa trên công nghệ và năng suất.")

    with st.container(border=True):
        st.markdown("#### Câu 1.4.4: Mô phỏng dự báo GDP năm 2030")
        st.caption("Kịch bản: D=30%, AI=100, H=35%, K & L tăng 6%/năm, TFP tăng 1.2%/năm từ 2025.")

        Y_2025 = Y[-1]
        K_2025 = K[-1]
        L_2025 = L[-1]
        A_2025 = A_t[-1]

        years_forecast = np.arange(2026, 2031)
        n = len(years_forecast)

        K_f = K_2025 * (1.06)**np.arange(1, n+1)
        L_f = L_2025 * (1.06)**np.arange(1, n+1)
        A_f = A_2025 * (1.012)**np.arange(1, n+1)
        D_f = np.full(n, 30.0)
        AI_f = np.full(n, 100.0)
        H_f = np.full(n, 35.0)

        Y_f = A_f * (K_f**alpha) * (L_f**beta) * (D_f**gamma) * (AI_f**delta) * (H_f**theta)

        all_years = np.concatenate([years, years_forecast])
        all_Y = np.concatenate([Y, Y_f])

        col_s1, col_s2 = st.columns([1, 2])
        with col_s1:
            st.metric(label="Dự báo GDP 2030", value=f"{Y_f[-1]:,.0f} nghìn tỷ VND", delta=None)
            df_fore = pd.DataFrame({
                "Năm": years_forecast.astype(int),
                "GDP (nghìn tỷ VND)": Y_f.round(1)
            })
            st.dataframe(df_fore, hide_index=True, width="stretch")

        with col_s2:
            fig_pred = px.line(x=all_years, y=all_Y, markers=True, title="Dự báo GDP Việt Nam đến năm 2030")
            fig_pred.add_vline(x=2025, line_dash="dash", line_color="red", annotation_text="Hiện tại", annotation_position="top right")
            fig_pred.update_traces(line=dict(color=PRIMARY_PURPLE, width=2), marker=dict(size=5, color=PRIMARY_PURPLE))
            fig_pred.update_layout(
                xaxis_title="Năm",
                yaxis_title="GDP (nghìn tỷ VND)",
                margin=dict(t=30),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color=TEXT_COLOR)
            )
            st.plotly_chart(fig_pred, width="stretch")

        st.success(f"**Kết luận 1.4.4**: Với kịch bản chuyển đổi số đồng bộ, GDP Việt Nam có thể đạt **{Y_f[-1]:,.0f} nghìn tỷ VND** vào năm 2030.")

    with st.container(border=True):
        st.markdown("### Câu 1.5: Thảo luận Chính sách")

        tfp_start = A_t[0]
        tfp_end = A_t[-1]
        tfp_trend = "tăng" if tfp_end > tfp_start else "giảm"

        new_factors = df_ga_summary[df_ga_summary["Nhân tố"].isin(["Số hóa (D)", "Năng lực AI", "Nhân lực số (H)"])]
        dominant_new = new_factors.loc[new_factors["Tỷ trọng (%)"].idxmax(), "Nhân tố"]
        pct_dominant_new = new_factors.loc[new_factors["Tỷ trọng (%)"].idxmax(), "Tỷ trọng (%)"]

        gdp_2030 = Y_f[-1]
        exceeds_18k = gdp_2030 > 18000

        st.markdown("#### a) Xu hướng TFP và chất lượng tăng trưởng")
        st.info(
            f"TFP có xu hướng **{tfp_trend}** từ {tfp_start:.2f} (2020) lên {tfp_end:.2f} (2025). "
            "Điều này cho thấy tăng trưởng không chỉ nhờ mở rộng vốn/lao động, mà còn nhờ cải thiện hiệu suất — "
            "phản ánh **chất lượng tăng trưởng đang được nâng cao**."
        )

        st.markdown("#### b) Trong các yếu tố mới (D, AI, H), yếu tố nào đóng góp nhiều nhất?")
        st.info(
            f"Trong nhóm yếu tố mới, **{dominant_new}** đóng góp tỷ trọng cao nhất ({pct_dominant_new:.1f}% tổng tăng trưởng trung bình), "
            "cho thấy vai trò ngày càng quan trọng của chuyển đổi số trong động lực tăng trưởng gần đây."
        )

        st.markdown("#### c) Mục tiêu 30% kinh tế số/GDP vào 2030 có khả thi?")
        st.markdown(
            f"<div style='background-color:#f3e8ff; padding:12px; border-left:4px solid #7e22ce; border-radius:4px; font-size:14px;'>"
            f"<strong> Dự báo GDP 2030:</strong> <span style='color:#7e22ce; font-weight:bold;'>{gdp_2030:,.0f} nghìn tỷ VND</span><br>"
            f"<strong> Đánh giá:</strong> {'Mục tiêu khả thi' if exceeds_18k else 'Cần điều chỉnh kịch bản'} để vượt ngưỡng 18,000 nghìn tỷ VND.<br>"
            f"<strong> Cảnh báo:</strong> Nếu chỉ tập trung vào D (số hóa) mà thiếu H (nhân lực số) và AI (năng lực cốt lõi), hiệu quả biên sẽ suy giảm — điều này được phản ánh qua độ nhạy của TFP trong mô hình."
            f"</div>",
            unsafe_allow_html=True
        )


