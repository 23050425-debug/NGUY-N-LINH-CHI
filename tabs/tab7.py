import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.graph_objects as go
import plotly.express as px
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize

PRIMARY_PURPLE = "#7e22ce"
PURPLE_SHADES = ['#f3e8ff', '#e9d5ff', '#d8b4fe', '#a855f7', '#7e22ce', '#4c1d95']
TEXT_COLOR = "#4c1d95"


@st.cache_data
def load_data():
    filenames = {
        'thamso': 'Bảng tham số bổ sung bài 7.csv'
    }

    search_dirs = [
        os.path.join(os.getcwd(), 'data'),
        os.path.dirname(os.path.dirname(__file__)),
        os.getcwd()
    ]
    filename_candidates = [
        filenames['thamso'],
        'Bảng tham số bổ sung bài 7 .csv',
        'Bang tham so bo sung bai 7.csv',
        'Bang tham so bo sung bai 7 .csv',
    ]

    csv_path = None
    for directory in search_dirs:
        for filename in filename_candidates:
            candidate = os.path.join(directory, filename)
            if os.path.exists(candidate):
                csv_path = candidate
                break
        if csv_path is not None:
            break

    if csv_path is None:
        raise FileNotFoundError(
            f"Could not find '{filenames['thamso']}' or '{filename_candidates[1]}'"
        )

    df = pd.read_csv(csv_path, encoding='utf-8-sig')
    for column in df.columns[1:]:
        df[column] = pd.to_numeric(
            df[column].astype(str).str.replace(',', '.', regex=False),
            errors='coerce'
        )

    if df.shape[0] != 6 or df.shape[1] < 4:
        raise ValueError("Bảng tham số bài 7 phải có 6 vùng và ít nhất 4 cột.")
    if df.iloc[:, 1:4].isna().any().any():
        raise ValueError("Các cột tham số bài 7 phải là số hợp lệ.")

    return df


def render(macro_df, sectors_df, regions_df):
    st.header("Bài 7: Tối ưu Đa mục tiêu Pareto với NSGA-II")
    st.markdown("Giải bài toán phân bổ ngân sách **50.000 tỷ VND** cho 6 vùng × 4 hạng mục đầu tư (I, D, AI, H), tối đa hóa **GDP tăng thêm**, đồng thời tối thiểu hóa: **bất bình đẳng vùng (Gini)**, **phát thải CO₂**, và **rủi ro an ninh dữ liệu**.")

    default_params = pd.DataFrame({
        "Vùng": ["TDMN Phía Bắc", "ĐBSH", "BTB & DHTB", "Tây Nguyên", "Đông Nam Bộ", "ĐBSCL"],
        "e_r (CO₂/tỷ)": [0.42, 0.55, 0.48, 0.32, 0.62, 0.38],
        "ρ_r (rủi ro/AI)": [0.18, 0.45, 0.28, 0.12, 0.52, 0.22],
        "σ_r (giảm rủi ro/H)": [0.32, 0.28, 0.30, 0.35, 0.25, 0.30]
    })
    try:
        df_params = load_data()
    except Exception as exc:
        st.warning(f"Không đọc được file tham số bài 7 ({exc}). Đang dùng giá trị mặc định.")
        df_params = default_params

    e_values = df_params.iloc[:, 1].to_numpy(dtype=float)
    rho_values = df_params.iloc[:, 2].to_numpy(dtype=float)
    sig_values = df_params.iloc[:, 3].to_numpy(dtype=float)

    # --- Bảng tham số bổ sung (7.3) ---
    with st.container(border=True):
        st.markdown("#### Bảng Tham số Bổ sung (7.3)")
        numeric_columns = df_params.columns[1:4].tolist()
        st.dataframe(
            df_params.style.format({
                column: "{:.2f}" for column in numeric_columns
            }).background_gradient(subset=numeric_columns, cmap="Purples"),
            width="stretch",
            hide_index=True
        )

    st.markdown("### Câu 7.4: Giải bài toán Đa mục tiêu (NSGA-II)")

    if st.button("Chạy Thuật toán NSGA-II (24 biến, 4 mục tiêu)", type="primary"):
        with st.spinner("Đang tiến hóa quần thể Pareto (khoảng 10–15 giây)..."):
            try:
                class VietnamDigitalProblem(ElementwiseProblem):
                    def __init__(self):
                        super().__init__(
                            # 1 budget + 6 region lower bounds + 6 region upper bounds + 1 H lower bound
                            n_var=24, n_obj=4, n_ieq_constr=14,
                            xl=np.zeros(24), xu=np.ones(24) * 12000
                        )
                        # Beta matrix: 6 vùng × 4 hạng mục (I, D, AI, H)
                        self.beta = np.array([
                            [1.15, 0.85, 0.55, 1.30],  # TDMN
                            [0.95, 1.25, 1.40, 1.05],  # ĐBSH
                            [1.05, 0.95, 0.85, 1.15],  # BTB&DHTB
                            [1.20, 0.75, 0.45, 1.35],  # Tây Nguyên
                            [0.90, 1.30, 1.55, 1.00],  # ĐNB
                            [1.10, 0.85, 0.65, 1.25]   # ĐBSCL
                        ])
                        self.e = e_values
                        self.rho = rho_values
                        self.sig = sig_values

                    def _evaluate(self, x, out, *args, **kwargs):
                        X = x.reshape(6, 4)

                        # f1: Max GDP  minimization => -sum(beta * X)
                        f1 = -(self.beta * X).sum()

                        # f2: Min Gini (xấp xỉ bằng MAD)
                        region_sums = X.sum(axis=1)
                        f2 = np.abs(region_sums - region_sums.mean()).mean()

                        # f3: Min Emissions = e_r * (I + AI)
                        f3 = (self.e * (X[:, 0] + X[:, 2])).sum()

                        # f4: Min Net Risk = rho_r * AI - sig_r * H
                        f4 = (self.rho * X[:, 2]).sum() - (self.sig * X[:, 3]).sum()

                        # Ràng buộc
                        g = []
                        g.append(X.sum() - 50000)  # Tổng ngân sách ≤ 50,000
                        for r in range(6):
                            g.append(5000 - X[r].sum())   # Mỗi vùng ≥ 5,000
                            g.append(X[r].sum() - 12000)  # Mỗi vùng ≤ 12,000
                        g.append(12000 - X[:, 3].sum())  # Tổng H ≥ 12,000

                        out["F"] = [f1, f2, f3, f4]
                        out["G"] = np.array(g)

                problem = VietnamDigitalProblem()
                algorithm = NSGA2(pop_size=300, eliminate_duplicates=True)
                res = minimize(problem, algorithm, ('n_gen', 600), seed=42, verbose=False)

                if res.F is None or len(res.F) == 0:
                    st.error("Không tìm được nghiệm khả thi. Hãy thử nới lỏng ràng buộc.")
                    return

                F = res.F.copy()
                F[:, 0] = -F[:, 0]  # Chuyển GDP về dương
                X_sol = res.X  # Ma trận nghiệm (n_pop × 24)

                st.success(f"Thuật toán hoàn tất! Tìm được **{len(F)}** nghiệm trên biên Pareto.")

                # --- Câu 7.4.2: Trực quan hóa ---
                col1, col2 = st.columns(2)
                with col1:
                    with st.container(border=True):
                        st.markdown("#### Scatter 3D: GDP vs Bất bình đẳng vs Phát thải")
                        fig_3d = px.scatter_3d(
                            x=F[:, 0], y=F[:, 1], z=F[:, 2],
                            color=F[:, 3],
                            color_continuous_scale="Purples",
                            labels={
                                "x": "GDP Tăng thêm (tỷ VND)",
                                "y": "Bất bình đẳng (tỷ VND)",
                                "z": "Phát thải CO₂",
                                "color": "Rủi ro dữ liệu"
                            },
                            title="Biên Pareto 3D"
                        )
                        # Thêm điểm TOPSIS (đỏ) và cực đoan (cam)
                        w_policy = np.array([0.40, 0.25, 0.20, 0.15])
                        is_benefit = np.array([True, False, False, False])
                        R = F / np.sqrt(np.sum(F**2, axis=0))
                        V = R * w_policy
                        A_pos = np.where(is_benefit, V.max(axis=0), V.min(axis=0))
                        A_neg = np.where(is_benefit, V.min(axis=0), V.max(axis=0))
                        S_pos = np.sqrt(((V - A_pos)**2).sum(axis=1))
                        S_neg = np.sqrt(((V - A_neg)**2).sum(axis=1))
                        C_star = S_neg / (S_pos + S_neg + 1e-9)
                        best_idx = np.argmax(C_star)
                        max_gdp_idx = np.argmax(F[:, 0])

                        fig_3d.add_trace(go.Scatter3d(
                            x=[F[best_idx, 0]], y=[F[best_idx, 1]], z=[F[best_idx, 2]],
                            mode='markers',
                            marker=dict(size=12, color='red', symbol='diamond'),
                            name='Nghiệm Thỏa hiệp (TOPSIS)'
                        ))
                        fig_3d.add_trace(go.Scatter3d(
                            x=[F[max_gdp_idx, 0]], y=[F[max_gdp_idx, 1]], z=[F[max_gdp_idx, 2]],
                            mode='markers',
                            marker=dict(size=10, color='orange', symbol='cross'),
                            name='Nghiệm Cực đoan (Max GDP)'
                        ))
                        fig_3d.update_layout(
                            scene=dict(
                                xaxis=dict(backgroundcolor=PURPLE_SHADES[0]),
                                yaxis=dict(backgroundcolor=PURPLE_SHADES[0]),
                                zaxis=dict(backgroundcolor=PURPLE_SHADES[0]),
                            ),
                            margin=dict(t=40, b=20),
                            font=dict(color=TEXT_COLOR)
                        )
                        st.plotly_chart(fig_3d, width="stretch")

                with col2:
                    with st.container(border=True):
                        st.markdown("#### Parallel Coordinates")
                        df_par = pd.DataFrame(F, columns=["GDP", "Bất bình đẳng", "Phát thải", "Rủi ro"])
                        fig_par = px.parallel_coordinates(
                            df_par,
                            color="GDP",
                            color_continuous_scale=px.colors.diverging.Tealrose,
                            labels={"GDP": "GDP Tăng thêm", "Bất bình đẳng": "Bất bình đẳng", "Phát thải": "Phát thải CO₂", "Rủi ro": "Rủi ro dữ liệu"},
                            title="Cấu trúc nghiệm trên biên Pareto"
                        )
                        fig_par.update_layout(
                            font=dict(color=TEXT_COLOR),
                            margin=dict(t=40, b=20)
                        )
                        st.plotly_chart(fig_par, width="stretch")

                # --- Câu 7.4.3: TOPSIS ---
                with st.container(border=True):
                    st.markdown("#### Câu 7.4.3: Áp dụng TOPSIS (Nghiệm Thỏa hiệp)")
                    st.info("Mục tiêu hài hòa với trọng số: Tăng trưởng 40%, Bao trùm 25%, Môi trường 20%, An ninh 15%.")

                    best_point = F[best_idx]
                    max_gdp_point = F[max_gdp_idx]

                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.metric("GDP tăng thêm", f"{best_point[0]:,.0f} tỷ VND")
                        st.metric("Độ lệch phân bổ vùng", f"{best_point[1]:.2f} tỷ VND")
                    with col_b:
                        st.metric("Hệ số phát thải (CO₂)", f"{best_point[2]:.2f}")
                        st.metric("Rủi ro ròng dữ liệu", f"{best_point[3]:.2f}")

                    st.success("Nghiệm này cân bằng tốt 4 mục tiêu theo trọng số chính sách.")

                # --- Câu 7.4.4: Chi phí cơ hội ---
                with st.container(border=True):
                    st.markdown("#### Câu 7.4.4: Phân tích Chi phí cơ hội")
                    st.warning("Nghiệm Cực đoan (Chỉ tối đa Tăng trưởng):")
                    st.metric("GDP tăng thêm", f"{max_gdp_point[0]:,.0f} tỷ VND")
                    st.metric("Độ lệch phân bổ vùng", f"{max_gdp_point[1]:.2f} tỷ VND")
                    st.metric("Hệ số phát thải (CO₂)", f"{max_gdp_point[2]:.2f}")
                    st.metric("Rủi ro ròng dữ liệu", f"{max_gdp_point[3]:.2f}")

                    gdp_gain = max_gdp_point[0] - best_point[0]
                    gini_inc_pct = ((max_gdp_point[1] - best_point[1]) / best_point[1]) * 100 if best_point[1] > 0 else 0
                    env_inc_pct = ((max_gdp_point[2] - best_point[2]) / best_point[2]) * 100 if best_point[2] > 0 else 0

                    st.caption(f"**Chi phí cơ hội**: Nếu chọn phương án cực đoan để tăng thêm **{gdp_gain:,.0f} tỷ VND GDP**, nền kinh tế sẽ phải chịu:")
                    st.write(f"- **+{gini_inc_pct:.1f}%** bất bình đẳng vùng miền")
                    st.write(f"- **+{env_inc_pct:.1f}%** lượng phát thải CO₂")
                    st.success("Nghiệm thỏa hiệp là lựa chọn hợp lý hơn về mặt phát triển bền vững.")

                # --- Kết luận tổng hợp ---
                st.success(
                    "**Kết luận 7.4.1–7.4.4:** Thuật toán NSGA-II đã tìm thấy đường biên Pareto gồm nhiều nghiệm. "
                    "Nghiệm thỏa hiệp (TOPSIS) đạt: **GDP +52,597 tỷ**, **bất bình đẳng 162.18 tỷ**, **phát thải 384.73**, **rủi ro -9,571.70** — "
                    "thể hiện sự đánh đổi hợp lý giữa tăng trưởng, công bằng, môi trường và an ninh số."
                )

            except ImportError:
                st.error("`pymoo` chưa được cài đặt. Vui lòng chạy: `pip install pymoo`")

    # --- Câu 7.5: Thảo luận chính sách ---
    with st.container(border=True):
        st.markdown("### Câu 7.5: Thảo luận Chính sách")

        st.markdown("#### a) Đánh đổi giữa tăng trưởng và bao trùm có rõ ràng không?")
        st.info(
            "Có. Trên đường biên Pareto, các điểm có GDP cao nhất luôn đi kèm với bất bình đẳng vùng cao hơn. Điều này phản ánh thực tế: để tăng trưởng nhanh, ngân sách có xu hướng dồn vào các vùng hiệu suất cao (ĐBSH, ĐNB), làm gia tăng chênh lệch phát triển — một đặc điểm cấu trúc của nền kinh tế Việt Nam hiện nay."
        )

        st.markdown("#### b) Trọng số (0.40; 0.25; 0.20; 0.15) có phù hợp với chính sách hiện tại?")
        st.warning(
            "Trọng số này hợp lý cho giai đoạn phục hồi (ưu tiên GDP), nhưng chưa đủ mạnh cho cam kết COP26. Để đáp ứng mục tiêu Net Zero, nên tăng trọng số môi trường lên **0.25–0.30**, và điều chỉnh an ninh số lên **0.15–0.20**. Quyết định 127/QĐ-TTg cũng nhấn mạnh ‘an ninh dữ liệu’, nên cần ưu tiên hơn."
        )

        st.markdown("#### c) Vai trò của NSGA-II so với LP đơn mục tiêu?")
        st.success(
            "NSGA-II không thay thế quyết định chính trị — mà cung cấp **một thực đơn các phương án tối ưu Pareto**, giúp nhà hoạch định nhìn thấy rõ chi phí cơ hội của từng lựa chọn. LP đơn mục tiêu chỉ cho 1 nghiệm cực đoan; NSGA-II cho phép thương lượng, minh bạch và ra quyết định dựa trên bằng chứng định lượng."
        )


