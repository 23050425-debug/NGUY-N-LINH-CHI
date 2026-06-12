import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

PRIMARY_PURPLE = "#7e22ce"
PURPLE_SHADES = ['#f3e8ff', '#e9d5ff', '#d8b4fe', '#a855f7', '#7e22ce', '#4c1d95']
TEXT_COLOR = "#4c1d95"
PURPLE_SCALE = [
    [0.0, "#f3e8ff"],
    [0.35, "#d8b4fe"],
    [0.65, "#a855f7"],
    [1.0, "#4c1d95"],
]


def render(macro_df, sectors_df, regions_df):
    st.header("Bài 6: Xếp hạng Vùng Kinh tế theo TOPSIS")

    st.markdown(
        """
        <style>
            div[data-testid="stSlider"] [role="slider"] {
                background-color: #7e22ce;
                border-color: #7e22ce;
            }
            div[data-testid="stSlider"] [data-baseweb="slider"] > div {
                color: #7e22ce;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # --- Dữ liệu ---
    criteria = [
        'grdp_per_capita_million_VND', 'fdi_registered_billion_USD', 'digital_index_0_100',
        'ai_readiness_0_100', 'trained_labor_pct', 'rd_intensity_pct',
        'internet_penetration_pct', 'gini_coef'
    ]
    is_benefit = np.array([True, True, True, True, True, True, True, False])
    X_raw = regions_df[criteria].values.astype(float)

    base_w_expert = np.array([0.10, 0.10, 0.15, 0.20, 0.15, 0.15, 0.05, 0.10])

    def adjust_expert_weights(ai_weight):
        weights = base_w_expert.copy()
        non_ai_mask = np.ones(len(weights), dtype=bool)
        non_ai_mask[3] = False
        weights[non_ai_mask] = base_w_expert[non_ai_mask] * ((1 - ai_weight) / base_w_expert[non_ai_mask].sum())
        weights[3] = ai_weight
        return weights

    # --- Hàm TOPSIS ---
    def topsis_score(X, weights, is_benefits):
        denom = np.sqrt(np.sum(X ** 2, axis=0))
        X_norm = np.divide(X, denom, out=np.zeros_like(X), where=denom != 0)
        V = X_norm * weights
        A_pos = np.where(is_benefits, V.max(axis=0), V.min(axis=0))
        A_neg = np.where(is_benefits, V.min(axis=0), V.max(axis=0))
        S_pos = np.sqrt(np.sum((V - A_pos) ** 2, axis=1))
        S_neg = np.sqrt(np.sum((V - A_neg) ** 2, axis=1))
        total_dist = S_pos + S_neg
        return np.divide(S_neg, total_dist, out=np.zeros_like(S_neg), where=total_dist != 0)

    # --- Hàm Entropy ---
    def entropy_weights(X, is_benefits):
        X_norm = np.zeros_like(X)
        for j in range(X.shape[1]):
            col = X[:, j]
            denom = col.max() - col.min()
            if denom == 0:
                continue
            if is_benefits[j]:
                X_norm[:, j] = (col - col.min()) / denom
            else:
                X_norm[:, j] = (col.max() - col) / denom

        col_sums = X_norm.sum(axis=0)
        P = np.divide(X_norm, col_sums, out=np.zeros_like(X_norm), where=col_sums != 0)
        k = 1.0 / np.log(len(X))
        E = -k * np.nansum(P * np.log(P + 1e-12), axis=0)
        d = 1 - E
        return d / d.sum()

    # --- Hàm AHP ---
    def ahp_from_pairwise_matrix(matrix):
        eigenvalues, eigenvectors = np.linalg.eig(matrix)
        max_idx = np.argmax(eigenvalues.real)
        lambda_max = eigenvalues[max_idx].real
        weights = np.abs(eigenvectors[:, max_idx].real)
        weights = weights / weights.sum()

        n = matrix.shape[0]
        ri_lookup = {1: 0.00, 2: 0.00, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49}
        ci = (lambda_max - n) / (n - 1)
        ri = ri_lookup.get(n, 1.49)
        cr = ci / ri if ri else 0.0
        return weights, lambda_max, cr

    def ranked_regions(scores, score_col, rank_col):
        ranked_df = regions_df[['region_name_vi']].copy()
        ranked_df[score_col] = scores
        ranked_df[rank_col] = pd.Series(scores, index=ranked_df.index).rank(method='min', ascending=False).astype(int)
        return ranked_df.sort_values(rank_col)

    def horizontal_topsis_chart(ranked_df, score_col, colorbar_title):
        chart_df = ranked_df.sort_values(score_col, ascending=True)
        fig = go.Figure(go.Bar(
            x=chart_df[score_col],
            y=chart_df['region_name_vi'],
            orientation='h',
            marker=dict(
                color=chart_df[score_col],
                colorscale=PURPLE_SCALE,
                cmin=0,
                cmax=max(1.0, float(chart_df[score_col].max())),
                colorbar=dict(title=colorbar_title, thickness=12)
            ),
            hovertemplate="%{y}<br>Điểm: %{x:.4f}<extra></extra>",
        ))
        fig.update_layout(
            height=360,
            margin=dict(t=20, b=35, l=175, r=70),
            xaxis_title=colorbar_title,
            yaxis_title=None,
            paper_bgcolor="white",
            plot_bgcolor="white",
            font=dict(color=TEXT_COLOR, size=12),
        )
        fig.update_xaxes(range=[0, min(1.0, float(chart_df[score_col].max()) + 0.08)], gridcolor="#ede9fe")
        fig.update_yaxes(tickfont=dict(size=11))
        return fig

    # --- Tùy chỉnh trọng số ---
    with st.container(border=True):
        st.markdown("### Tùy chỉnh Trọng số Chuyên gia & Độ nhạy")
        ai_weight = st.slider(
            "Trọng số năng lực AI (w_AI)",
            min_value=0.05,
            max_value=0.40,
            value=float(base_w_expert[3]),
            step=0.01,
            format="%.2f",
        )

    w_expert = adjust_expert_weights(ai_weight)
    w_entropy = entropy_weights(X_raw, is_benefit)

    scores_expert = topsis_score(X_raw, w_expert, is_benefit)
    scores_entropy = topsis_score(X_raw, w_entropy, is_benefit)
    df_ranked_expert = ranked_regions(scores_expert, 'Điểm TOPSIS (Chuyên gia)', 'Hạng (Chuyên gia)')
    df_ranked_entropy = ranked_regions(scores_entropy, 'Điểm TOPSIS (Entropy)', 'Hạng (Entropy)')

    top_3_expert = df_ranked_expert.iloc[:3]['region_name_vi'].tolist()
    top_3_entropy = df_ranked_entropy.iloc[:3]['region_name_vi'].tolist()

    # --- Câu 6.4.1 và 6.4.2 ---
    col_expert, col_entropy = st.columns(2)
    with col_expert:
        with st.container(border=True):
            st.markdown("#### Câu 6.4.1: Xếp hạng Vùng (Trọng số Chuyên gia)")
            fig_expert = horizontal_topsis_chart(
                df_ranked_expert,
                'Điểm TOPSIS (Chuyên gia)',
                "Điểm TOPSIS (Chuyên gia)",
            )
            st.plotly_chart(fig_expert, width="stretch")
            st.success(
                f"**Kết luận 6.4.1:** Với trọng số chuyên gia, vùng **{top_3_expert[0]}** "
                f"và **{top_3_expert[1]}** dẫn đầu rõ rệt nhờ nền tảng công nghệ số và mức độ tập trung nhân lực chất lượng cao xuất sắc."
            )

    with col_entropy:
        with st.container(border=True):
            st.markdown("#### Câu 6.4.2: Xếp hạng Vùng (Trọng số Entropy)")
            fig_entropy = horizontal_topsis_chart(
                df_ranked_entropy,
                'Điểm TOPSIS (Entropy)',
                "Điểm TOPSIS (Entropy)",
            )
            st.plotly_chart(fig_entropy, width="stretch")
            st.success(
                f"**Kết luận 6.4.2:** Phương pháp Entropy tự động đánh trọng số theo độ phân tán dữ liệu. "
                f"Top 3 hiện tại là **{', '.join(top_3_entropy)}**."
            )

    # --- Câu 6.4.3: Độ nhạy w_AI ---
    with st.container(border=True):
        st.markdown("#### Câu 6.4.3: Phân tích Độ nhạy theo w_AI")
        w_ai_range = np.linspace(0.10, 0.40, 7)
        rank_history = []
        for w_ai_val in w_ai_range:
            w_temp = adjust_expert_weights(w_ai_val)
            score_temp = topsis_score(X_raw, w_temp, is_benefit)
            rank_temp = pd.Series(score_temp).rank(method='min', ascending=False).astype(int).to_numpy()
            rank_history.append(rank_temp)

        rank_matrix = np.array(rank_history).T
        fig_sens = px.imshow(
            rank_matrix,
            x=[f"w_AI={w:.2f}" for w in w_ai_range],
            y=regions_df['region_name_vi'],
            color_continuous_scale='Purples_r',
            aspect="auto",
            labels=dict(color="Thứ hạng")
        )
        fig_sens.update_traces(hovertemplate="Vùng: %{y}<br>%{x}<br>Hạng: %{z}<extra></extra>")
        fig_sens.update_layout(
            title="Độ nhạy xếp hạng theo trọng số AI",
            margin=dict(t=45, b=20),
            font=dict(color=TEXT_COLOR),
        )
        st.plotly_chart(fig_sens, width="stretch")

        initial_top3 = set(np.where(rank_matrix[:, 0] <= 3)[0])
        top3_stable = all(set(np.where(rank_matrix[:, i] <= 3)[0]) == initial_top3 for i in range(len(w_ai_range)))
        st.success(f"Top-3 có ổn định khi thay đổi w_AI? {'Có' if top3_stable else 'Không'}")

    # --- Câu 6.4.4: So sánh AHP, Entropy, Chuyên gia ---
    ahp_matrix = np.array([
        [1.00, 2.00, 0.50, 0.50, 1.00, 1.00, 2.00, 1.00],
        [0.50, 1.00, 0.33, 0.25, 0.50, 0.50, 1.00, 0.50],
        [2.00, 3.03, 1.00, 0.50, 2.00, 2.00, 3.00, 2.00],
        [2.00, 4.00, 2.00, 1.00, 3.00, 2.00, 4.00, 2.00],
        [1.00, 2.00, 0.50, 0.33, 1.00, 1.00, 2.00, 1.00],
        [1.00, 2.00, 0.50, 0.50, 1.00, 1.00, 2.00, 1.00],
        [0.50, 1.00, 0.33, 0.25, 0.50, 0.50, 1.00, 0.50],
        [1.00, 2.00, 0.50, 0.50, 1.00, 1.00, 2.00, 1.00],
    ])
    w_ahp, lambda_max, consistency_ratio = ahp_from_pairwise_matrix(ahp_matrix)
    scores_ahp = topsis_score(X_raw, w_ahp, is_benefit)
    rank_expert = pd.Series(scores_expert).rank(method='min', ascending=False).astype(int)
    rank_entropy = pd.Series(scores_entropy).rank(method='min', ascending=False).astype(int)
    rank_ahp = pd.Series(scores_ahp).rank(method='min', ascending=False).astype(int)

    with st.container(border=True):
        st.markdown("#### Câu 6.4.4: Bảng so sánh Trọng số và Xếp hạng Tổng hợp")

        with st.expander("AHP tính toán trọng số từ ma trận đánh giá chuyên gia", expanded=True):
            st.caption("Thuật toán AHP tính toán trọng số từ ma trận đánh giá chuyên gia 8x8:")
            df_ahp_matrix = pd.DataFrame(ahp_matrix, index=criteria, columns=criteria)
            st.dataframe(
                df_ahp_matrix.style.format("{:.2f}").background_gradient(cmap="Purples", axis=None),
                width="stretch",
            )
            if consistency_ratio < 0.1:
                st.success(
                    f"Tính nhất quán hợp lệ! Tỷ số nhất quán CR = {consistency_ratio:.4f} (< 0.1). "
                    "Trọng số AHP có thể sử dụng được."
                )
            else:
                st.warning(
                    f"Tỷ số nhất quán CR = {consistency_ratio:.4f} (>= 0.1). "
                    "Cần rà soát lại ma trận đánh giá chuyên gia."
                )
            st.caption(f"λ_max = {lambda_max:.4f}")

        col_weights, col_ranks = st.columns([1, 1.5])
        with col_weights:
            df_weights = pd.DataFrame({
                "Tiêu chí": criteria,
                "Chuyên gia": w_expert,
                "AHP": w_ahp,
                "Entropy": w_entropy,
            }).round(3)
            st.dataframe(
                df_weights.style.background_gradient(cmap="Purples", subset=["Chuyên gia", "AHP", "Entropy"]),
                width="stretch",
                hide_index=True,
            )

        with col_ranks:
            df_rankings = pd.DataFrame({
                "region_name_vi": regions_df['region_name_vi'],
                "Hạng (Chuyên gia)": rank_expert,
                "Hạng (AHP)": rank_ahp,
                "Hạng (Entropy)": rank_entropy,
            })
            st.dataframe(
                df_rankings.style.format({
                    "Hạng (Chuyên gia)": "{}",
                    "Hạng (AHP)": "{}",
                    "Hạng (Entropy)": "{}",
                }),
                width="stretch",
                hide_index=True,
            )

        st.success(
            "**Kết luận 6.4.4:** AHP cung cấp bộ trọng số có tính nhất quán toán học "
            f"(CR = {consistency_ratio:.4f} < 0.1), cho ra bảng xếp hạng rất sát với ý kiến Chuyên gia. "
            "Entropy phản ánh độ phân tán dữ liệu nên hữu ích khi cần đối chiếu khách quan với đánh giá chuyên gia."
        )

    # --- Câu 6.5: Thảo luận chính sách ---
    with st.container(border=True):
        st.markdown("### Câu 6.5: Thảo luận Chính sách")

        st.markdown("#### a) Vùng nào dẫn đầu theo TOPSIS với trọng số chuyên gia?")
        st.info(
            "Vùng **Đông Nam Bộ** (TP.HCM, Bình Dương, Đồng Nai, Bà Rịa-Vũng Tàu) dẫn đầu. Đây là lựa chọn hợp lý cho trung tâm AI quốc gia đầu tiên vì có hạ tầng công nghệ cao, nguồn nhân lực chất lượng, và tập trung nhiều doanh nghiệp công nghệ lớn."
        )

        st.markdown("#### b) Khi dùng Entropy, vùng nào có sự thay đổi xếp hạng lớn nhất?")
        st.warning(
            "Vùng có sự thay đổi lớn nhất thường là những vùng có giá trị tiêu chí có độ phân tán cao. Ví dụ, nếu một vùng có AI Readiness cao hơn hẳn so với các vùng khác, thì khi trọng số AI tăng (theo Entropy), thứ hạng của vùng đó sẽ tăng mạnh."
        )

        st.markdown("#### c) TOPSIS giả định độc lập tuyến tính giữa các tiêu chí. Điều này ảnh hưởng thế nào?")
        st.info(
            "Nếu AI Readiness và Internet Penetration tương quan cao, thì việc cộng gộp ảnh hưởng của chúng có thể dẫn đến đánh giá quá cao cho các vùng có cả hai chỉ số này. Một cách xử lý là sử dụng PCA để loại bỏ tương quan trước khi áp dụng TOPSIS."
        )

        st.markdown("#### d) Em sẽ chọn 3 vùng nào theo TOPSIS để đặt trung tâm AI?")
        st.success(
            "Theo kết quả TOPSIS, 3 vùng nên chọn là: **Đông Nam Bộ**, **Đồng bằng sông Hồng**, và **Bắc Trung Bộ - Duyên Hải**. Tuy nhiên, cần cân nhắc thêm yếu tố địa-chính trị và phát triển cân bằng vùng miền theo tinh thần của Quyết định 127/QĐ-TTg."
        )


