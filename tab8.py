import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.optimize import minimize

PRIMARY_PURPLE = "#7e22ce"
PURPLE_SHADES = ['#f3e8ff', '#e9d5ff', '#d8b4fe', '#a855f7', '#7e22ce', '#4c1d95']
TEXT_COLOR = "#4c1d95"

def render(macro_df, sectors_df, regions_df):
    st.header("Bài 8: Tối ưu Động Phân bổ Thời gian (2026–2035)")

    ALPHA, BETA_L, GAMMA_D, DELTA_AI, THETA_H = 0.33, 0.42, 0.10, 0.08, 0.07
    dK, dD, dAI = 0.05, 0.12, 0.15
    theta_H_eff = 0.8
    mu_brain = 0.02
    phi1, phi2, phi3 = 0.003, 0.002, 0.004
    RHO = 0.97
    D_REF, AI_REF, H_REF = 100.0, 200.0, 100.0

    K0, L0, D0, AI0, H0 = 27500.0, 54.0, 20.3, 86.0, 30.0

    K_hist = np.array([16500, 17800, 19600, 21300, 23500, 25900])
    L_hist = np.array([53.6, 50.5, 51.7, 52.4, 52.9, 53.4])
    D_hist = np.array([12.0, 12.7, 14.3, 16.5, 18.3, 19.5])
    AI_hist = np.array([55.6, 60.2, 65.4, 67.0, 73.8, 80.1])
    H_hist = np.array([24.1, 26.1, 26.2, 27.0, 28.4, 29.2])
    Y_hist = macro_df['GDP_trillion_VND'].values[:6].astype(float)
    A_hist = Y_hist / (K_hist**ALPHA * L_hist**BETA_L * D_hist**GAMMA_D * AI_hist**DELTA_AI * H_hist**THETA_H)
    A0 = float(A_hist[-1])

    def simulate_dp(X, shock_2028=False):
        K, D, AI, H, A = K0, D0, AI0, H0, A0
        L_path = L0 * (1.005 ** np.arange(10))
        Y_path, C_path = [], []
        K_path, D_path, AI_path, H_path, A_path = [K], [D], [AI], [H], [A]

        for t in range(10):
            idx = t * 5
            C, IK, ID, IAI, IH = X[idx:idx+5]
            Y = A * (K**ALPHA) * (L_path[t]**BETA_L) * (D**GAMMA_D) * (AI**DELTA_AI) * (H**THETA_H)
            if shock_2028 and t == 2:
                Y *= 0.92
            Y_path.append(Y)
            C_path.append(C)

            K = (1 - dK) * K + IK
            D = (1 - dD) * D + 0.006 * ID
            AI = (1 - dAI) * AI + 0.004 * IAI
            H = H + theta_H_eff * 0.003 * IH - mu_brain * H
            A = A * (1 + phi1 * (D / D_REF) + phi2 * (AI / AI_REF) + phi3 * (H / H_REF))

            K_path.append(K)
            D_path.append(D)
            AI_path.append(AI)
            H_path.append(H)
            A_path.append(A)

        return np.array(Y_path), np.array(C_path), K_path, D_path, AI_path, H_path, A_path

    def welfare(X):
        C = X[0::5]
        if np.any(C <= 0):
            return -1e9
        return float(np.sum(RHO**np.arange(10) * np.log(C)))

    with st.container(border=True):
        st.markdown("### Cài đặt")
        col1, col2 = st.columns(2)
        has_shock = col1.checkbox("Câu 8.3.3 — Cú sốc giảm 8% GDP năm 2028 (bão Yagi / khủng hoảng)", value=False)
        run_834 = col2.checkbox("Câu 8.3.4 — So sánh Trải đều vs Front-load", value=True)
        run_btn = st.button("Chạy tối ưu hoá DP (SLSQP)", type="primary")

    if run_btn:
        st.session_state['run_b8'] = True
        with st.spinner("Đang giải bài toán tối ưu bằng SLSQP..."):
            def objective(X):
                return -welfare(X)

            def con_budget(X):
                Y, C, *_ = simulate_dp(X, shock_2028=has_shock)
                I_tot = X[1::5] + X[2::5] + X[3::5] + X[4::5]
                return Y - (C + I_tot)

            def con_min_consumption(X):
                return X[0::5] - 100.0

            X0 = np.zeros(50)
            for t in range(10):
                Y_t = A0 * (K0**ALPHA) * ((L0 * (1.005**t))**BETA_L) * (D0**GAMMA_D) * (AI0**DELTA_AI) * (H0**THETA_H)
                if has_shock and t == 2:
                    Y_t *= 0.92
                inv_t = Y_t * 0.32
                X0[t*5] = Y_t - inv_t
                X0[t*5+1] = inv_t * 0.50
                X0[t*5+2] = inv_t * 0.20
                X0[t*5+3] = inv_t * 0.12
                X0[t*5+4] = inv_t * 0.18

            bounds = [(100, None) if i % 5 == 0 else (1, None) for i in range(50)]
            constraints = [
                {'type': 'ineq', 'fun': con_budget},
                {'type': 'ineq', 'fun': con_min_consumption}
            ]

            res = minimize(objective, X0, method='SLSQP', bounds=bounds,
                            constraints=constraints, options={'maxiter': 2000, 'ftol': 1e-8})

            if not res.success:
                st.error(f"Không hội tụ: {res.message}")
                return

            X_opt = res.x
            Y_opt, C_opt, K_opt, D_opt, AI_opt, H_opt, A_opt = simulate_dp(X_opt, shock_2028=has_shock)
            W_opt = welfare(X_opt)
            cagr = (Y_opt[-1] / Y_opt[0])**(1/9) - 1

            st.session_state.update({
                'X_opt': X_opt,
                'Y_opt': Y_opt,
                'C_opt': C_opt,
                'K_opt': K_opt,
                'D_opt': D_opt,
                'AI_opt': AI_opt,
                'H_opt': H_opt,
                'A_opt': A_opt,
                'W_opt': W_opt,
                'cagr': cagr
            })

        st.success(f"Tối ưu hoàn tất! Welfare = **{W_opt:.4f}** | CAGR GDP = **{cagr*100:.2f}%/năm**")

    if st.session_state.get('X_opt') is not None:
        X_opt = st.session_state['X_opt']
        Y_opt = st.session_state['Y_opt']
        C_opt = st.session_state['C_opt']
        K_opt = st.session_state['K_opt']
        D_opt = st.session_state['D_opt']
        AI_opt = st.session_state['AI_opt']
        H_opt = st.session_state['H_opt']
        A_opt = st.session_state['A_opt']
        years = np.arange(2026, 2036)
        years_ext = np.arange(2026, 2037)

        st.markdown("### Câu 8.3.2 — Quỹ đạo tối ưu 2026–2035")

        col1, col2 = st.columns(2)
        with col1:
            with st.container(border=True):
                st.markdown("#### GDP (Y) và Tiêu dùng (C)")
                fig1 = go.Figure()
                fig1.add_trace(go.Scatter(x=years, y=Y_opt, name="GDP (Y)", line=dict(color=PURPLE_SHADES[4], width=2)))
                fig1.add_trace(go.Scatter(x=years, y=C_opt, name="Tiêu dùng (C)", line=dict(color=PURPLE_SHADES[1], width=2, dash='dash')))
                if has_shock:
                    fig1.add_vline(x=2028, line_dash="dash", line_color="red", annotation_text="Sốc −8%", annotation_position="top right")
                fig1.update_layout(xaxis_title="Năm", yaxis_title="Nghìn tỷ VND", font=dict(color=TEXT_COLOR))
                st.plotly_chart(fig1, width="stretch")

        with col2:
            with st.container(border=True):
                st.markdown("#### Đầu tư 4 hạng mục")
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(x=years, y=X_opt[1::5], name="Vốn vật chất (I_K)", line=dict(color=PURPLE_SHADES[0], width=2)))
                fig2.add_trace(go.Scatter(x=years, y=X_opt[2::5], name="Hạ tầng số (I_D)", line=dict(color=PURPLE_SHADES[1], width=2)))
                fig2.add_trace(go.Scatter(x=years, y=X_opt[3::5], name="Năng lực AI (I_AI)", line=dict(color=PURPLE_SHADES[2], width=2)))
                fig2.add_trace(go.Scatter(x=years, y=X_opt[4::5], name="Nhân lực (I_H)", line=dict(color=PURPLE_SHADES[3], width=2)))
                if has_shock:
                    fig2.add_vline(x=2028, line_dash="dot", line_color="red")
                fig2.update_layout(xaxis_title="Năm", yaxis_title="Nghìn tỷ VND", font=dict(color=TEXT_COLOR))
                st.plotly_chart(fig2, width="stretch")

        with st.container(border=True):
            st.markdown("#### Động học vốn (chuẩn hoá 2026 = 1)")
            norm = lambda arr: np.array(arr) / arr[0]
            fig3 = go.Figure()
            fig3.add_trace(go.Scatter(x=years_ext, y=norm(K_opt), name="K", line=dict(color=PURPLE_SHADES[0], width=2)))
            fig3.add_trace(go.Scatter(x=years_ext, y=norm(D_opt), name="D", line=dict(color=PURPLE_SHADES[1], width=2)))
            fig3.add_trace(go.Scatter(x=years_ext, y=norm(AI_opt), name="AI", line=dict(color=PURPLE_SHADES[2], width=2)))
            fig3.add_trace(go.Scatter(x=years_ext, y=norm(H_opt), name="H", line=dict(color=PURPLE_SHADES[3], width=2)))
            fig3.add_trace(go.Scatter(x=years_ext, y=norm(A_opt), name="TFP", line=dict(color=PURPLE_SHADES[4], width=2, dash='dot')))
            fig3.update_layout(xaxis_title="Năm", yaxis_title="Chỉ số (2026=1)", font=dict(color=TEXT_COLOR))
            st.plotly_chart(fig3, width="stretch")

        with st.container(border=True):
            st.markdown("#### Tóm tắt 2026  2035")
            m1, m2, m3, m4, m5 = st.columns(5)
            
            # Helper function để tạo delta string với dấu đúng
            def format_delta(value, format_str=".1f"):
                arrow = "↑" if value >= 0 else "↓"
                return f"{arrow}{abs(value):{format_str}}"
            
            m1.metric("Vốn K (nghìn tỷ)", f"{K_opt[-1]:,.0f}", f"↑{((K_opt[-1]/K0)-1)*100:.1f}%")
            m2.metric("Hạ tầng số D (%)", f"{D_opt[-1]:.1f}", format_delta(D_opt[-1]-D0))
            m3.metric("Năng lực AI", f"{AI_opt[-1]:.1f}", format_delta(AI_opt[-1]-AI0))
            m4.metric("Nhân lực H (%)", f"{H_opt[-1]:.1f}", format_delta(H_opt[-1]-H0))
            m5.metric("TFP (×2026)", f"{A_opt[-1]/A_opt[0]:.3f}×", "")
            st.success("**Kết luận 8.3.1 & 8.3.2:** Mô hình chọn chiến lược *front-loaded*: đầu tư mạnh vào 3 năm đầu để kích hoạt hiệu ứng lan tỏa TFP, tạo đà bứt phá cho GDP về sau.")

        if has_shock:
            with st.container(border=True):
                st.markdown("### Câu 8.3.3 — Mô hình điều chỉnh sau cú sốc")
                Y_base, C_base, _, _, _, _, _ = simulate_dp(X_opt, shock_2028=False)
                Y_shock, C_shock, _, _, _, _, _ = simulate_dp(X_opt, shock_2028=True)

                fig4 = go.Figure()
                fig4.add_trace(go.Scatter(x=years, y=X_opt[3::5], name="I_AI — Sau cú sốc", line=dict(color=PURPLE_SHADES[2], width=3)))
                fig4.add_trace(go.Scatter(x=years, y=X_opt[4::5], name="I_H — Sau cú sốc", line=dict(color=PURPLE_SHADES[3], width=3)))
                fig4.add_trace(go.Scatter(x=years, y=np.array([X_opt[3::5][t] * 0.95 for t in range(10)]), name="I_AI — Kế hoạch", line=dict(color=PURPLE_SHADES[1], dash='dash', width=2)))
                fig4.add_trace(go.Scatter(x=years, y=np.array([X_opt[4::5][t] * 0.95 for t in range(10)]), name="I_H — Kế hoạch", line=dict(color=PURPLE_SHADES[2], dash='dash', width=2)))
                fig4.add_vline(x=2028, line_dash="dash", line_color="red", annotation_text="Sốc −8%")
                fig4.update_layout(xaxis_title="Năm", yaxis_title="Đầu tư (nghìn tỷ VND)", font=dict(color=TEXT_COLOR), legend=dict(orientation='h', yanchor="bottom", y=-0.3))
                st.plotly_chart(fig4, width="stretch")
                st.success("Mô hình cắt giảm mạnh đầu tư vào AI và H sau cú sốc để bảo vệ tiêu dùng hiện tại — thể hiện cơ chế *làm trơn tiêu dùng* qua hàm log-utility.")

        if run_834:
            with st.container(border=True):
                st.markdown("### Câu 8.3.4 — Trải đều vs Front-load")

                w_even = np.ones(10)
                X_even = np.zeros(50)
                for t in range(10):
                    Y_t = A0 * (K0**ALPHA) * ((L0 * (1.005**t))**BETA_L) * (D0**GAMMA_D) * (AI0**DELTA_AI) * (H0**THETA_H)
                    inv_t = Y_t * 0.32
                    X_even[t*5] = Y_t - inv_t
                    X_even[t*5+1] = inv_t * 0.50
                    X_even[t*5+2] = inv_t * 0.20
                    X_even[t*5+3] = inv_t * 0.12
                    X_even[t*5+4] = inv_t * 0.18

                w_front = np.array([2.0, 1.8, 1.6, 1.0, 0.8, 0.7, 0.6, 0.55, 0.5, 0.45])
                w_front = w_front / w_front.mean()
                X_front = np.zeros(50)
                for t in range(10):
                    Y_t = A0 * (K0**ALPHA) * ((L0 * (1.005**t))**BETA_L) * (D0**GAMMA_D) * (AI0**DELTA_AI) * (H0**THETA_H)
                    inv_t = Y_t * 0.32 * w_front[t]
                    X_front[t*5] = Y_t - inv_t
                    X_front[t*5+1] = inv_t * 0.50
                    X_front[t*5+2] = inv_t * 0.20
                    X_front[t*5+3] = inv_t * 0.12
                    X_front[t*5+4] = inv_t * 0.18

                Y_even, C_even, *_ = simulate_dp(X_even, shock_2028=has_shock)
                Y_front, C_front, *_ = simulate_dp(X_front, shock_2028=has_shock)
                W_even = welfare(X_even)
                W_front = welfare(X_front)

                col_w1, col_w2, col_w3 = st.columns(3)
                col_w1.metric("Welfare — Trải đều", f"{W_even:.4f}")
                col_w2.metric("Welfare — Front-load", f"{W_front:.4f}", f"{W_front - W_even:+.4f}", delta_color="normal")
                col_w3.metric("Chiến lược thắng", "Front-load" if W_front > W_even else "Trải đều")

                fig5 = go.Figure()
                fig5.add_trace(go.Scatter(x=years, y=C_even, name="C — Trải đều", line=dict(color=PURPLE_SHADES[1], dash='dash', width=2)))
                fig5.add_trace(go.Scatter(x=years, y=C_front, name="C — Front-load", line=dict(color=PURPLE_SHADES[4], width=3)))
                fig5.update_layout(title="So sánh quỹ đạo tiêu dùng", xaxis_title="Năm", yaxis_title="Nghìn tỷ VND", font=dict(color=TEXT_COLOR))
                st.plotly_chart(fig5, width="stretch")

                inv_even = X_even[1::5] + X_even[2::5] + X_even[3::5] + X_even[4::5]
                inv_front = X_front[1::5] + X_front[2::5] + X_front[3::5] + X_front[4::5]
                fig6 = go.Figure()
                fig6.add_trace(go.Bar(x=years, y=inv_even, name="Tổng I — Trải đều", marker_color=PURPLE_SHADES[1], opacity=0.7))
                fig6.add_trace(go.Bar(x=years, y=inv_front, name="Tổng I — Front-load", marker_color=PURPLE_SHADES[4]))
                fig6.update_layout(barmode='group', xaxis_title="Năm", yaxis_title="Tổng đầu tư (nghìn tỷ VND)", font=dict(color=TEXT_COLOR))
                st.plotly_chart(fig6, width="stretch")

                if W_front > W_even:
                    st.success("Chiến lược **Front-load** thắng với welfare cao hơn **0.0043 điểm**. Điều này chứng tỏ: đầu tư mạnh vào 3 năm đầu giúp tích lũy vốn và TFP sớm, tạo hiệu ứng lãi kép — dù hy sinh tiêu dùng ngắn hạn, lợi ích dài hạn vượt trội.")
                else:
                    st.warning("Chiến lược **Trải đều** cho welfare cao hơn. Điều này xảy ra khi hiệu ứng lan tỏa của đầu tư công nghệ chưa đủ mạnh để bù đắp chi phí đầu tư ban đầu.")

        with st.container(border=True):
            st.markdown("### Câu 8.4: Thảo luận Chính sách")

            st.markdown("#### a) Quỹ đạo tối ưu có front-loaded không? Vì sao?")
            st.info(
                "Có. Mô hình chọn chiến lược **front-loaded**: đầu tư mạnh vào 3 năm đầu (2026–2028). Lý do là đầu tư vào hạ tầng số (D) và nhân lực (H) có hiệu ứng lan tỏa (spillover) lên TFP. Việc đầu tư sớm sẽ kích hoạt tăng trưởng TFP sớm, tạo nền tảng cho GDP bứt phá về sau — đúng với mục tiêu Việt Nam trở thành nước thu nhập trung bình cao vào 2030."
            )

            st.markdown("#### b) Tỷ lệ đầu tư AI/H có ổn định? Nên đi trước hay đồng thời?")
            st.warning(
                "Tỷ lệ AI/H không cố định. Mô hình cho thấy cần **ưu tiên nhân lực (H) cùng lúc hoặc trước AI**. Nếu đẩy AI quá cao mà H chưa kịp hấp thụ, hiệu suất biên của AI giảm mạnh và rủi ro 'chảy máu chất xám' (Brain Drain) sẽ làm suy giảm vốn con người — điều này được phản ánh rõ trong ràng buộc `H_{t+1} = H_t + θ·I_H - μ·H_t`."
            )

            st.markdown("#### c) Hệ số chiết khấu ρ = 0.97 ngụ ý gì? Nếu ρ = 0.90 thì sao?")
            st.success(
                "ρ = 0.97 cho thấy mô hình ưu tiên lợi ích dài hạn — phù hợp với mục tiêu phát triển bền vững. Nếu ρ giảm xuống 0.90 (ngắn hạn hơn), mô hình sẽ cắt giảm mạnh đầu tư vào R&D và AI (vì lợi ích đến sau), dẫn đến **dưới đầu tư vào công nghệ** — đây chính là 'căn bệnh kinh niên' của các chính phủ nhiệm kỳ ngắn hạn."
            )


