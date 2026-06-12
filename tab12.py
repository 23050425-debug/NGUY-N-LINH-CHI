import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.optimize import linprog

PRIMARY_PURPLE = "#7e22ce"
PURPLE_SHADES = ['#f3e8ff', '#e9d5ff', '#d8b4fe', '#a855f7', '#7e22ce', '#4c1d95']
TEXT_COLOR = "#4c1d95"
PURPLE_SCALE = [
    [0.0, "#f3e8ff"],
    [0.35, "#d8b4fe"],
    [0.65, "#a855f7"],
    [1.0, "#4c1d95"],
]

# ============================================================================
# MODULE M1: DỰ BÁO KINH TẾ - Cobb-Douglas (Bài 1)
# ============================================================================
def module_m1_forecast(macro_df):
    alpha, beta, gamma, delta, theta = 0.33, 0.42, 0.10, 0.08, 0.07
    years = np.array([2020, 2021, 2022, 2023, 2024, 2025])
    Y = macro_df['GDP_trillion_VND'].values
    K = np.array([16500, 17800, 19600, 21300, 23500, 25900])
    L = np.array([53.6, 50.5, 51.7, 52.4, 52.9, 53.4])
    D = np.array([12.0, 12.7, 14.3, 16.5, 18.3, 19.5])
    AI = np.array([55.6, 60.2, 65.4, 67.0, 73.8, 80.1])
    H = np.array([24.1, 26.1, 26.2, 27.0, 28.4, 29.2])

    A = Y / (K**alpha * L**beta * D**gamma * AI**delta * H**theta)
    A_mean = A.mean()
    Y_hat = A_mean * (K**alpha * L**beta * D**gamma * AI**delta * H**theta)
    mape = np.mean(np.abs((Y - Y_hat) / Y)) * 100

    dln_Y = np.diff(np.log(Y))
    dln_K = np.diff(np.log(K))
    dln_L = np.diff(np.log(L))
    dln_D = np.diff(np.log(D))
    dln_AI = np.diff(np.log(AI))
    dln_H = np.diff(np.log(H))
    contrib_K = alpha * dln_K
    contrib_L = beta * dln_L
    contrib_D = gamma * dln_D
    contrib_AI = delta * dln_AI
    contrib_H = theta * dln_H
    contrib_TFP = dln_Y - contrib_K - contrib_L - contrib_D - contrib_AI - contrib_H

    avg_growth = dln_Y.mean()
    decomposition = {
        'Vốn vật chất (K)': contrib_K.mean() / avg_growth * 100,
        'Lao động (L)': contrib_L.mean() / avg_growth * 100,
        'Số hóa (D)': contrib_D.mean() / avg_growth * 100,
        'Năng lực AI': contrib_AI.mean() / avg_growth * 100,
        'Nhân lực số (H)': contrib_H.mean() / avg_growth * 100,
        'TFP': contrib_TFP.mean() / avg_growth * 100,
    }

    K_base, L_base, D_base, AI_base, H_base = K[-1], L[-1], D[-1], AI[-1], H[-1]
    A_base = A[-1]

    scenario_params = {
        "S1. Truyền thống":    {"K_g": 0.08, "D_2030": 22.0, "AI_2030": 90,  "H_2030": 31, "tfp_g": 0.008},
        "S2. Số hóa nhanh":    {"K_g": 0.06, "D_2030": 30.0, "AI_2030": 100, "H_2030": 33, "tfp_g": 0.012},
        "S3. AI dẫn dắt":      {"K_g": 0.05, "D_2030": 28.0, "AI_2030": 120, "H_2030": 32, "tfp_g": 0.015},
        "S4. Bao trùm số":     {"K_g": 0.06, "D_2030": 25.0, "AI_2030": 95,  "H_2030": 35, "tfp_g": 0.010},
        "S5. Tối ưu cân bằng":  {"K_g": 0.06, "D_2030": 28.0, "AI_2030": 110, "H_2030": 35, "tfp_g": 0.012},
    }

    scenarios_forecast = {}
    for s_name, params in scenario_params.items():
        gdp_path = [Y[-1]]
        A_t = A_base
        for t in range(1, 6):
            frac = t / 5.0
            K_t = K_base * (1 + params["K_g"])**t
            L_t = L_base * (1 + 0.002)**t
            D_t = D_base + (params["D_2030"] - D_base) * frac
            AI_t = AI_base + (params["AI_2030"] - AI_base) * frac
            H_t = H_base + (params["H_2030"] - H_base) * frac
            A_t = A_t * (1 + params["tfp_g"])
            Y_t = A_t * K_t**alpha * L_t**beta * D_t**gamma * AI_t**delta * H_t**theta
            gdp_path.append(Y_t)
        scenarios_forecast[s_name] = gdp_path

    return {
        'years': years, 'Y': Y, 'A': A, 'A_mean': A_mean,
        'Y_hat': Y_hat, 'mape': mape, 'decomposition': decomposition,
        'scenarios_forecast': scenarios_forecast,
        'forecast_years': list(range(2025, 2031)),
    }

# ============================================================================
# MODULE M2: SẴN SÀNG SỐ - TOPSIS (Bài 6)
# ============================================================================
def module_m2_topsis(regions_df):
    regions = [
        "Trung du miền núi phía Bắc", "Đồng bằng sông Hồng",
        "Bắc Trung Bộ + DH Trung Bộ", "Tây Nguyên",
        "Đông Nam Bộ", "Đồng bằng sông Cửu Long"
    ]
    X = np.array([
        [57.0,  3.5,  38, 22, 21.5, 0.18, 72, 0.405],
        [152.3, 20.0,  78, 68, 36.8, 0.85, 92, 0.358],
        [87.5,  8.2,  55, 40, 27.5, 0.32, 84, 0.372],
        [68.9,  0.8,  32, 18, 18.2, 0.15, 68, 0.412],
        [158.9, 18.5, 82, 75, 42.5, 0.78, 94, 0.385],
        [80.5,  2.1,  48, 30, 16.8, 0.22, 78, 0.392],
    ])
    is_benefit = [True, True, True, True, True, True, True, False]

    w_expert = np.array([0.10, 0.10, 0.15, 0.20, 0.15, 0.15, 0.05, 0.10])

    def entropy_weights(X_raw):
        X_pos = X_raw - X_raw.min(axis=0) + 1e-10
        P = X_pos / X_pos.sum(axis=0)
        k = 1.0 / np.log(len(X_raw))
        E = -k * np.nansum(P * np.log(P + 1e-12), axis=0)
        d = 1 - E
        return d / d.sum()

    w_entropy = entropy_weights(X)

    def run_topsis(X_raw, w, is_benefit_list):
        R = X_raw / np.sqrt((X_raw**2).sum(axis=0))
        V = R * w
        A_star = np.where(is_benefit_list, V.max(axis=0), V.min(axis=0))
        A_neg = np.where(is_benefit_list, V.min(axis=0), V.max(axis=0))
        S_star = np.sqrt(((V - A_star)**2).sum(axis=1))
        S_neg = np.sqrt(((V - A_neg)**2).sum(axis=1))
        C_star = S_neg / (S_star + S_neg)
        return C_star

    scores_expert = run_topsis(X, w_expert, is_benefit)
    scores_entropy = run_topsis(X, w_entropy, is_benefit)

    ranking_expert = np.zeros(len(regions), dtype=int)
    ranking_entropy = np.zeros(len(regions), dtype=int)
    for i, idx in enumerate(np.argsort(-scores_expert)):
        ranking_expert[idx] = i + 1
    for i, idx in enumerate(np.argsort(-scores_entropy)):
        ranking_entropy[idx] = i + 1

    return {
        'regions': regions, 'X': X,
        'w_expert': w_expert, 'w_entropy': w_entropy,
        'scores_expert': scores_expert, 'scores_entropy': scores_entropy,
        'ranking_expert': ranking_expert, 'ranking_entropy': ranking_entropy,
    }

# ============================================================================
# MODULE M3: PHÂN BỔ - LP (Bài 2 & 4)
# ============================================================================
def module_m3_optimize():
    c = [-0.85, -1.20, -0.95, -1.35]
    A_ub = [
        [1, 1, 1, 1],
        [-1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, -1],
        [0.35, -0.65, 0.35, -0.65],
    ]
    b_ub = [100, -25, -15, -20, -10, 0]
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=[(0, None)]*4, method='highs')

    n_regions, n_items = 6, 4
    n_vars = n_regions * n_items
    beta_matrix = np.array([
        [1.15, 0.85, 0.55, 1.30],
        [0.95, 1.25, 1.40, 1.05],
        [1.05, 0.95, 0.85, 1.15],
        [1.20, 0.75, 0.45, 1.35],
        [0.90, 1.30, 1.55, 1.00],
        [1.10, 0.85, 0.65, 1.25],
    ])
    c_4 = -beta_matrix.flatten()

    A_ub_list = []
    b_ub_list = []
    A_ub_list.append(np.ones(n_vars))
    b_ub_list.append(50000)
    for r in range(n_regions):
        row = np.zeros(n_vars)
        for j in range(n_items):
            row[r * n_items + j] = -1
        A_ub_list.append(row)
        b_ub_list.append(-5000)
    for r in range(n_regions):
        row = np.zeros(n_vars)
        for j in range(n_items):
            row[r * n_items + j] = 1
        A_ub_list.append(row)
        b_ub_list.append(12000)
    row_h = np.zeros(n_vars)
    for r in range(n_regions):
        row_h[r * n_items + 3] = -1
    A_ub_list.append(row_h)
    b_ub_list.append(-12000)
    for r in range(n_regions):
        for j in range(n_items):
            row = np.zeros(n_vars)
            row[r * n_items + j] = -1
            A_ub_list.append(row)
            b_ub_list.append(-200)

    A_ub_4 = np.array(A_ub_list)
    b_ub_4 = np.array(b_ub_list)
    res_4 = linprog(c_4, A_ub=A_ub_4, b_ub=b_ub_4, bounds=[(0, None)] * n_vars, method='highs')

    alloc_matrix = res_4.x.reshape(n_regions, n_items) if res_4.success else None

    return {
        'optimal_x': res.x if res.success else None,
        'optimal_z': -res.fun if res.success else None,
        'alloc_matrix': alloc_matrix,
        'z_star_4': -res_4.fun if res_4.success else None,
    }

# ============================================================================
# MODULE M4: LAO ĐỘNG - LP (Bài 9)
# ============================================================================
def module_m4_labor():
    sectors = [
        "Nông-Lâm-Thủy sản", "CN chế biến chế tạo", "Xây dựng",
        "Bán buôn-bán lẻ", "Tài chính-Ngân hàng", "Logistics-Vận tải",
        "CNTT-Truyền thông", "Giáo dục-Đào tạo"
    ]
    N = 8
    risk = np.array([18, 42, 25, 38, 52, 35, 28, 22]) / 100
    a1 = np.array([8.5, 32.5, 12.8, 22.4, 45.8, 28.5, 62.5, 18.5])
    b1 = np.array([45, 28, 35, 32, 22, 30, 20, 55])
    c1 = np.array([5.2, 62.4, 18.5, 48.2, 72.5, 42.8, 32.5, 12.5])
    d1 = np.array([50, 32, 42, 38, 26, 36, 24, 62])

    net_coef_ai = a1 - c1 * risk
    net_coef_h = b1

    c_obj = np.zeros(2 * N)
    c_obj[:N] = -net_coef_ai
    c_obj[N:] = -net_coef_h

    A_ub_list = []
    b_ub_list = []
    A_ub_list.append(np.ones(2 * N))
    b_ub_list.append(30000)
    for i in range(N):
        row = np.zeros(2 * N)
        row[i] = -1
        row[N + i] = -1
        A_ub_list.append(row)
        b_ub_list.append(-1000)
    for i in range(N):
        row = np.zeros(2 * N)
        row[i] = -net_coef_ai[i]
        row[N + i] = -net_coef_h[i]
        A_ub_list.append(row)
        b_ub_list.append(0)
    for i in range(N):
        row = np.zeros(2 * N)
        row[i] = c1[i] * risk[i]
        row[N + i] = -d1[i]
        A_ub_list.append(row)
        b_ub_list.append(0)

    res = linprog(c_obj, A_ub=np.array(A_ub_list), b_ub=np.array(b_ub_list),
                  bounds=[(0, None)] * 2 * N, method='highs')

    if not res.success:
        return {}

    x_AI_opt = res.x[:N]
    x_H_opt = res.x[N:]
    new_jobs = a1 * x_AI_opt
    upgrade_jobs = b1 * x_H_opt
    displaced_jobs = c1 * risk * x_AI_opt
    net_jobs = new_jobs + upgrade_jobs - displaced_jobs

    return {
        'sectors': sectors,
        'x_AI': x_AI_opt, 'x_H': x_H_opt,
        'new_jobs': new_jobs, 'upgrade_jobs': upgrade_jobs,
        'displaced_jobs': displaced_jobs, 'net_jobs': net_jobs,
        'total_net': net_jobs.sum(),
        'total_budget_used': x_AI_opt.sum() + x_H_opt.sum(),
    }

# ============================================================================
# MODULE M5: RỦI RO - Đa mục tiêu (Bài 7 & 10)
# ============================================================================
def module_m5_risk():
    e_r = np.array([0.42, 0.55, 0.48, 0.32, 0.62, 0.38])
    rho_r = np.array([0.18, 0.45, 0.28, 0.12, 0.52, 0.22])
    sigma_r = np.array([0.32, 0.28, 0.30, 0.35, 0.25, 0.30])

    beta_matrix = np.array([
        [1.15, 0.85, 0.55, 1.30],
        [0.95, 1.25, 1.40, 1.05],
        [1.05, 0.95, 0.85, 1.15],
        [1.20, 0.75, 0.45, 1.35],
        [0.90, 1.30, 1.55, 1.00],
        [1.10, 0.85, 0.65, 1.25],
    ])

    alloc_scenarios = {
        "S1. Truyền thống":    np.array([70, 10, 10, 10]) / 100,
        "S2. Số hóa nhanh":    np.array([25, 45, 15, 15]) / 100,
        "S3. AI dẫn dắt":      np.array([20, 20, 45, 15]) / 100,
        "S4. Bao trùm số":     np.array([30, 20, 10, 40]) / 100,
        "S5. Tối ưu cân bằng":  np.array([35, 20, 15, 30]) / 100,
    }

    total_budget = 50000
    risk_results = {}

    region_weights = {
        "S1. Truyền thống":    np.array([0.08, 0.30, 0.12, 0.05, 0.35, 0.10]),
        "S2. Số hóa nhanh":    np.array([0.06, 0.28, 0.10, 0.04, 0.40, 0.12]),
        "S3. AI dẫn dắt":      np.array([0.04, 0.32, 0.08, 0.03, 0.45, 0.08]),
        "S4. Bao trùm số":     np.array([0.15, 0.18, 0.18, 0.14, 0.18, 0.17]),
        "S5. Tối ưu cân bằng":  np.array([0.10, 0.25, 0.14, 0.08, 0.28, 0.15]),
    }

    for s_name, alloc_pct in alloc_scenarios.items():
        rw = region_weights[s_name]
        X = np.outer(rw, alloc_pct) * total_budget
        f1_gdp = (beta_matrix * X).sum()
        sums_region = X.sum(axis=1)
        f2_gini = np.abs(sums_region - sums_region.mean()).mean() / (sums_region.mean() + 1e-10)
        f3_emission = (e_r * (X[:, 0] + X[:, 2])).sum()
        f4_cyber = (rho_r * X[:, 2]).sum() - (sigma_r * X[:, 3]).sum()

        risk_results[s_name] = {
            'gdp_gain': f1_gdp, 'gini': f2_gini,
            'emission': f3_emission, 'cyber_risk': f4_cyber,
        }

    return risk_results

# ============================================================================
# MODULE M6: DASHBOARD (Bài 12)
# ============================================================================
def _common_layout(fig, height=None, legend_bottom=False):
    fig.update_layout(
        height=height,
        margin=dict(t=45, b=55, l=40, r=35),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TEXT_COLOR),
    )
    fig.update_xaxes(gridcolor="#ede9fe", zerolinecolor="#ddd6fe")
    fig.update_yaxes(gridcolor="#ede9fe", zerolinecolor="#ddd6fe")
    if legend_bottom:
        fig.update_layout(legend=dict(orientation="h", y=-0.25, x=0))
    return fig


def _rank_frame(names, scores, rank_col="Xếp hạng"):
    df = pd.DataFrame({"Vùng": names, "TOPSIS Score": scores})
    df[rank_col] = df["TOPSIS Score"].rank(method="min", ascending=False).astype(int)
    return df.sort_values(rank_col)


def _source_frame():
    return pd.DataFrame({
        "Tên nguồn": [
            "Niên giám thống kê",
            "Cơ sở dữ liệu KH&CN",
            "Báo cáo kinh tế số",
            "FDI và đầu tư công",
            "World Development Indicators",
            "OECD.Stat",
            "Global Innovation Index 2025",
            "AI Index Report",
            "IMF World Economic Outlook",
            "UN Comtrade",
        ],
        "Cơ quan": [
            "Tổng cục Thống kê (GSO/NSO)",
            "Bộ Khoa học và Công nghệ (MoST)",
            "Bộ Thông tin và Truyền thông (MIC)",
            "Bộ Kế hoạch và Đầu tư (MPI)",
            "World Bank",
            "OECD",
            "WIPO",
            "Stanford HAI",
            "IMF",
            "Liên Hợp Quốc",
        ],
        "Địa chỉ truy cập": [
            "www.gso.gov.vn",
            "www.most.gov.vn",
            "www.mic.gov.vn",
            "www.mpi.gov.vn",
            "data.worldbank.org",
            "stats.oecd.org",
            "www.wipo.int/global_innovation_index",
            "aiindex.stanford.edu",
            "www.imf.org/weo",
            "comtrade.un.org",
        ],
    })


def _module_m3_region_lp(min_cell=200):
    beta_matrix = np.array([
        [1.15, 0.85, 0.55, 1.30],
        [0.95, 1.25, 1.40, 1.05],
        [1.05, 0.95, 0.85, 1.15],
        [1.20, 0.75, 0.45, 1.35],
        [0.90, 1.30, 1.55, 1.00],
        [1.10, 0.85, 0.65, 1.25],
    ])
    region_names = ["TDMNPB", "ĐBSH", "BTB+DHMT", "Tây Nguyên", "ĐNB", "ĐBSCL"]
    item_names = ["Hạ tầng số (I)", "CĐS DN (D)", "AI", "Nhân lực số (H)"]

    n_regions, n_items = beta_matrix.shape
    n_vars = n_regions * n_items
    c = -beta_matrix.flatten()
    a_ub, b_ub = [np.ones(n_vars)], [50000]

    for r in range(n_regions):
        row_min = np.zeros(n_vars)
        row_max = np.zeros(n_vars)
        for j in range(n_items):
            row_min[r * n_items + j] = -1
            row_max[r * n_items + j] = 1
        a_ub.append(row_min)
        b_ub.append(-5000)
        a_ub.append(row_max)
        b_ub.append(12000)

    row_h = np.zeros(n_vars)
    for r in range(n_regions):
        row_h[r * n_items + 3] = -1
    a_ub.append(row_h)
    b_ub.append(-12000)

    for r in range(n_regions):
        for j in range(n_items):
            row = np.zeros(n_vars)
            row[r * n_items + j] = -1
            a_ub.append(row)
            b_ub.append(-min_cell)

    res = linprog(c, A_ub=np.array(a_ub), b_ub=np.array(b_ub), bounds=[(0, None)] * n_vars, method="highs")
    alloc = res.x.reshape(n_regions, n_items) if res.success else np.zeros((n_regions, n_items))
    df = pd.DataFrame(alloc, index=region_names, columns=item_names)
    df["Tổng"] = df.sum(axis=1)
    return df, (-res.fun if res.success else 0)


def render(macro_df, sectors_df, regions_df):
    st.title("Bài 12: Đồ án Tích hợp - Hệ thống AIDEOM-VN")
    st.markdown("**Mô hình Ra quyết định Phát triển Kinh tế Việt Nam trong kỷ nguyên AI (AIDEOM-VN)**")
    st.caption(
        "Hệ thống tích hợp các module dự báo, đánh giá sẵn sàng số, tối ưu phân bổ ngân sách, "
        "mô phỏng lao động và kiểm soát rủi ro để lượng hóa các kịch bản chính sách đến năm 2030."
    )

    with st.spinner("Đang tổng hợp kết quả các module AIDEOM-VN..."):
        m1 = module_m1_forecast(macro_df)
        m2 = module_m2_topsis(regions_df)
        m3 = module_m3_optimize()
        m4 = module_m4_labor()
        m5 = module_m5_risk()

    scenarios = {
        "S1. Truyền thống":    [70, 10, 10, 10],
        "S2. Số hóa nhanh":    [25, 45, 15, 15],
        "S3. AI dẫn dắt":      [20, 20, 45, 15],
        "S4. Bao trùm số":     [30, 20, 10, 40],
        "S5. Tối ưu cân bằng":  [35, 20, 15, 30],
    }
    df_alloc = pd.DataFrame(scenarios, index=[
        "Vốn Vật chất (K)", "Chuyển đổi số (D)",
        "Trí tuệ nhân tạo (AI)", "Nhân lực số (H)"
    ]).T

    tab_pages = st.tabs([
        "Tổng quan Hệ thống",
        "M1: Dự báo Kinh tế",
        "M2: Sẵn sàng Số",
        "M3: Phân bổ Ngân sách",
        "M4: Lao động",
        "M5: Rủi ro",
        "Khuyến nghị Chính sách",
    ])

    # ==================== TAB 0: TỔNG QUAN ====================
    with tab_pages[0]:
        st.subheader("Tổng quan kiến trúc AIDEOM-VN")
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        kpi1.metric("Số module", "6")
        kpi2.metric("Giai đoạn dữ liệu", f"{int(m1['years'][0])}-{int(m1['years'][-1])}")
        kpi3.metric("Vùng đánh giá", f"{len(m2['regions'])}")
        kpi4.metric("Kịch bản 2030", f"{len(scenarios)}")

        module_data = {
            "Module": [
                "M1. Dự báo kinh tế", "M2. Đánh giá sẵn sàng số",
                "M3. Tối ưu phân bổ", "M4. Mô phỏng lao động",
                "M5. Đánh giá rủi ro", "M6. Dashboard ra QĐ"
            ],
            "Đầu vào": [
                "Macro 2020-2025", "Sectors, Regions",
                "Budget, ma trận beta", "Kế hoạch AI, H",
                "Tham số rủi ro (e, rho, sigma)", "Kết quả M1-M5"
            ],
            "Đầu ra": [
                "TFP, GDP dự báo 2026-2030",
                "TOPSIS scores, xếp hạng 6 vùng",
                "Phân bổ tối ưu ngành-vùng",
                "NetJob ròng từng ngành",
                "Rủi ro Cyber, CO2, Gini",
                "Trực quan kịch bản, Khuyến nghị"
            ],
            "Kỹ thuật": [
                "Cobb-Douglas (Bài 1)", "TOPSIS + Entropy (Bài 6)",
                "LP scipy/PuLP (Bài 2+4)", "LP lao động (Bài 9)",
                "Đa mục tiêu (Bài 7)", "Streamlit + Plotly (Bài 12)"
            ]
        }
        st.dataframe(
            pd.DataFrame(module_data).style.set_properties(
                subset=["Module"],
                **{"background-color": "#f3e8ff", "color": "#4c1d95", "font-weight": "600"},
            ),
            width="stretch",
            hide_index=True,
        )

        with st.container(border=True):
            st.markdown("#### Luồng xử lý dữ liệu")
            st.markdown(
                "**M1** dự báo trần tăng trưởng vĩ mô  **M2** xếp hạng vùng sẵn sàng số  "
                "**M3** tối ưu phân bổ vốn  **M4** lượng hóa việc làm ròng  "
                "**M5** kiểm tra rủi ro đa chiều  **M6** tổng hợp thành dashboard ra quyết định."
            )
            st.info(
                "AIDEOM-VN kết hợp mô hình toán tối ưu với ràng buộc chính sách để cân bằng giữa "
                "tăng trưởng, chuyển đổi số, việc làm và an toàn xã hội."
            )

        with st.container(border=True):
            st.markdown("#### F3.2. Nguồn dữ liệu chính thức")
            st.dataframe(
                _source_frame().style.set_properties(
                    subset=["Tên nguồn"],
                    **{"background-color": "#f3e8ff", "color": "#4c1d95", "font-weight": "600"},
                ),
                hide_index=True,
                width="stretch",
            )

        st.markdown("#### Trạng thái module")
        status_cols = st.columns(5)
        for label, col in zip(["M1", "M2", "M3", "M4", "M5"], status_cols):
            col.success(f"{label} đã tính toán")

    # ==================== TAB 1: M1 ====================
    with tab_pages[1]:
        st.subheader("Module M1: Dự báo Kinh tế - Hàm Cobb-Douglas mở rộng")
        st.latex(r"Y_t = A_t \cdot K^{0.33} \cdot L^{0.42} \cdot D^{0.10} \cdot AI^{0.08} \cdot H^{0.07}")

        col_chart, col_table = st.columns([1, 1.05])
        with col_chart:
            fig_tfp = px.line(
                x=m1["years"], y=m1["A"], markers=True,
                labels={"x": "Năm", "y": "TFP (A_t)"},
                title="Năng suất nhân tố tổng hợp (TFP) 2020-2025",
            )
            fig_tfp.update_traces(line=dict(color=PRIMARY_PURPLE, width=3), marker=dict(color=PRIMARY_PURPLE, size=7))
            st.plotly_chart(_common_layout(fig_tfp, height=360), width="stretch")

        with col_table:
            metric_cols = st.columns(2)
            metric_cols[0].metric("TFP trung bình (A_bar)", f"{m1['A_mean']:.4f}")
            metric_cols[1].metric("MAPE dự báo", f"{m1['mape']:.2f}%")
            df_fit = pd.DataFrame({
                "Năm": m1["years"],
                "GDP thực tế": m1["Y"],
                "GDP dự báo (A_bar)": m1["Y_hat"],
            })
            st.dataframe(
                df_fit.style.format({"GDP thực tế": "{:,.1f}", "GDP dự báo (A_bar)": "{:,.1f}"}),
                hide_index=True,
                width="stretch",
            )

        with st.container(border=True):
            st.markdown("#### Phân rã đóng góp tăng trưởng GDP")
            decomp_df = pd.DataFrame({
                "Yếu tố": list(m1["decomposition"].keys()),
                "Đóng góp (%)": list(m1["decomposition"].values()),
            })
            fig_decomp = px.bar(
                decomp_df,
                x="Yếu tố",
                y="Đóng góp (%)",
                color="Đóng góp (%)",
                color_continuous_scale=PURPLE_SCALE,
                title="Cơ cấu đóng góp trung bình giai đoạn 2020-2025",
            )
            fig_decomp.update_traces(texttemplate="%{y:.1f}%", textposition="outside")
            st.plotly_chart(_common_layout(fig_decomp, height=380), width="stretch")

        with st.container(border=True):
            st.markdown("#### Dự báo GDP Việt Nam 2025-2030 theo 5 kịch bản")
            forecast_df = pd.DataFrame({"Năm": m1["forecast_years"]})
            for name, path in m1["scenarios_forecast"].items():
                forecast_df[name] = path

            fig_forecast = go.Figure()
            line_styles = ["solid", "dot", "dash", "dashdot", "solid"]
            for idx, name in enumerate(m1["scenarios_forecast"]):
                fig_forecast.add_trace(go.Scatter(
                    x=m1["forecast_years"],
                    y=m1["scenarios_forecast"][name],
                    mode="lines+markers",
                    name=name,
                    line=dict(color=PURPLE_SHADES[min(idx + 1, 5)], width=3 if idx == 4 else 2, dash=line_styles[idx]),
                    marker=dict(size=6),
                ))
            fig_forecast.update_layout(title="Dự báo GDP Việt Nam 2025-2030", xaxis_title="Năm", yaxis_title="GDP (nghìn tỷ VND)")
            st.plotly_chart(_common_layout(fig_forecast, height=430), width="stretch")

            df_2030 = pd.DataFrame({
                "Kịch bản": list(m1["scenarios_forecast"].keys()),
                "GDP 2030": [values[-1] for values in m1["scenarios_forecast"].values()],
            })
            st.dataframe(df_2030.style.format({"GDP 2030": "{:,.1f}"}), hide_index=True, width="stretch")
            best = df_2030.sort_values("GDP 2030", ascending=False).iloc[0]
            st.success(
                f"**Nhận xét M1:** Kịch bản **{best['Kịch bản']}** cho GDP 2030 cao nhất "
                f"({best['GDP 2030']:,.1f} nghìn tỷ VND). TFP và các biến số D, AI, H là nhóm biến cần theo dõi để nâng chất lượng tăng trưởng."
            )

    # ==================== TAB 2: M2 ====================
    with tab_pages[2]:
        st.subheader("Module M2: Đánh giá Sẵn sàng Số - TOPSIS")
        col_a, col_b = st.columns(2)
        df_expert = _rank_frame(m2["regions"], m2["scores_expert"])
        df_entropy = _rank_frame(m2["regions"], m2["scores_entropy"])
        with col_a:
            st.markdown("#### Trọng số chuyên gia")
            st.dataframe(df_expert.style.format({"TOPSIS Score": "{:.4f}"}), hide_index=True, width="stretch")
        with col_b:
            st.markdown("#### Trọng số Entropy")
            st.dataframe(df_entropy.style.format({"TOPSIS Score": "{:.4f}"}), hide_index=True, width="stretch")

        criteria_names = ["GRDP/người", "FDI", "Digital Index", "AI Readiness", "LĐ đào tạo", "R&D/GRDP", "Internet", "Gini"]
        weight_df = pd.DataFrame({
            "Tiêu chí": criteria_names,
            "Chuyên gia": m2["w_expert"],
            "Entropy": m2["w_entropy"],
        })
        fig_weights = px.bar(
            weight_df.melt(id_vars="Tiêu chí", var_name="Phương pháp", value_name="Trọng số"),
            x="Tiêu chí", y="Trọng số", color="Phương pháp", barmode="group",
            color_discrete_sequence=[PURPLE_SHADES[4], PURPLE_SHADES[2]],
            title="So sánh trọng số hai phương pháp",
        )
        st.plotly_chart(_common_layout(fig_weights, height=380), width="stretch")

        score_df = pd.DataFrame({
            "Vùng": m2["regions"],
            "Chuyên gia": m2["scores_expert"],
            "Entropy": m2["scores_entropy"],
        })
        fig_scores = px.bar(
            score_df.melt(id_vars="Vùng", var_name="Phương pháp", value_name="Điểm TOPSIS"),
            x="Vùng", y="Điểm TOPSIS", color="Phương pháp", barmode="group",
            color_discrete_sequence=[PURPLE_SHADES[5], PURPLE_SHADES[3]],
            title="Điểm TOPSIS theo vùng",
        )
        fig_scores.update_xaxes(tickangle=-20)
        st.plotly_chart(_common_layout(fig_scores, height=430), width="stretch")
        st.info(
            f"**Nhận xét M2:** {df_expert.iloc[0]['Vùng']} và {df_expert.iloc[1]['Vùng']} dẫn đầu theo trọng số chuyên gia. "
            "Kết quả Entropy được dùng như phép kiểm tra khách quan dựa trên độ phân tán dữ liệu."
        )

    # ==================== TAB 3: M3 ====================
    with tab_pages[3]:
        st.subheader("Module M3: Tối ưu Phân bổ Ngân sách")
        col_lp, col_sens = st.columns([1, 1])
        with col_lp:
            st.markdown("#### Bài toán LP 4 hạng mục")
            item_df = pd.DataFrame({
                "Hạng mục": ["Hạ tầng số (I)", "AI & Dữ liệu", "Nhân lực số (H)", "R&D Công nghệ"],
                "Phân bổ tối ưu (nghìn tỷ)": m3["optimal_x"],
                "Hệ số tác động": [0.85, 1.20, 0.95, 1.35],
            })
            st.dataframe(
                item_df.style.format({"Phân bổ tối ưu (nghìn tỷ)": "{:,.1f}", "Hệ số tác động": "{:.2f}"}),
                hide_index=True,
                width="stretch",
            )
            st.metric("Z* (GDP gain tối ưu)", f"{m3['optimal_z']:.1f} nghìn tỷ VND")

        with col_sens:
            st.markdown("#### Phân tích độ nhạy ngân sách")
            budget_grid = np.array([100, 120, 140])
            z_grid = []
            for budget in budget_grid:
                c = [-0.85, -1.20, -0.95, -1.35]
                a_ub = [[1, 1, 1, 1], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, -1], [0.35, -0.65, 0.35, -0.65]]
                b_ub = [budget, -25, -15, -20, -10, 0]
                res = linprog(c, A_ub=a_ub, b_ub=b_ub, bounds=[(0, None)] * 4, method="highs")
                z_grid.append(-res.fun if res.success else np.nan)
            sens_df = pd.DataFrame({"Ngân sách (nghìn tỷ)": budget_grid, "Z* (GDP gain)": z_grid})
            st.dataframe(sens_df.style.format({"Z* (GDP gain)": "{:.1f}"}), hide_index=True, width="stretch")
            fig_sens = px.line(sens_df, x="Ngân sách (nghìn tỷ)", y="Z* (GDP gain)", markers=True, title="Đường cong Z*(B)")
            fig_sens.update_traces(line=dict(color=PRIMARY_PURPLE, width=3), marker=dict(color=PRIMARY_PURPLE, size=7))
            st.plotly_chart(_common_layout(fig_sens, height=300), width="stretch")

        region_alloc_df, z_region = _module_m3_region_lp(min_cell=200)
        st.markdown("#### LP phân bổ 50.000 tỷ theo 6 vùng x 4 hạng mục")
        st.caption("Cấu hình module tích hợp: mỗi cặp vùng-hạng mục nhận tối thiểu 200 tỷ để bảo đảm độ phủ phân bổ.")
        st.dataframe(
            region_alloc_df.style.format("{:,.0f}").background_gradient(cmap="Purples", subset=region_alloc_df.columns[:-1]),
            width="stretch",
        )
        st.metric("Z* (GDP gain vùng)", f"{z_region:,.1f} tỷ VND")

        fig_hm = px.imshow(
            region_alloc_df.iloc[:, :-1].values,
            x=region_alloc_df.columns[:-1],
            y=region_alloc_df.index,
            color_continuous_scale="Purples",
            text_auto=True,
            labels=dict(x="Hạng mục", y="Vùng", color="Tỷ VND"),
            title="Heatmap phân bổ tối ưu theo vùng",
        )
        fig_hm.update_xaxes(tickangle=-30)
        st.plotly_chart(_common_layout(fig_hm, height=420), width="stretch")

        st.markdown("#### So sánh cấu trúc phân bổ 5 kịch bản")
        fig_scenario = px.bar(
            df_alloc.reset_index().rename(columns={"index": "Kịch bản"}).melt(id_vars="Kịch bản", var_name="Hạng mục đầu tư", value_name="Tỷ trọng (%)"),
            x="Kịch bản", y="Tỷ trọng (%)", color="Hạng mục đầu tư",
            color_discrete_sequence=[PURPLE_SHADES[5], PURPLE_SHADES[4], PURPLE_SHADES[3], PURPLE_SHADES[2]],
            title="Tỷ trọng phân bổ ngân sách theo kịch bản",
        )
        fig_scenario.update_layout(barmode="stack")
        st.plotly_chart(_common_layout(fig_scenario, height=430, legend_bottom=True), width="stretch")
        st.success(
            "**Nhận xét M3:** LP xác định cấu trúc phân bổ có hiệu quả biên cao, đồng thời kiểm soát sàn/trần ngân sách theo vùng. "
            "Nhóm kịch bản cho phép so sánh rõ sự khác nhau giữa ưu tiên truyền thống, số hóa nhanh, AI dẫn dắt và bao trùm số."
        )

    with tab_pages[4]:
        st.subheader("M4: Tác động AI tới Thị trường Lao động (Bài 9)")

        labor_df = pd.DataFrame({
            "Ngành": m4["sectors"],
            "x_AI (tỷ)": m4["x_AI"],
            "x_H (tỷ)": m4["x_H"],
            "Việc mới": m4["new_jobs"],
            "Nâng cấp": m4["upgrade_jobs"],
            "Bị thay thế": m4["displaced_jobs"],
            "NetJob": m4["net_jobs"],
        })

        col_table, col_net = st.columns([1.1, 1])
        with col_table:
            st.dataframe(
                labor_df.style.format({
                    "x_AI (tỷ)": "{:,.0f}",
                    "x_H (tỷ)": "{:,.0f}",
                    "Việc mới": "{:,.0f}",
                    "Nâng cấp": "{:,.0f}",
                    "Bị thay thế": "{:,.0f}",
                    "NetJob": "{:,.0f}",
                }).background_gradient(cmap="Purples", subset=["NetJob"]),
                hide_index=True,
                width="stretch",
            )

        with col_net:
            st.metric("Tổng NetJob ròng", f"{m4['total_net']:,.0f} việc làm")
            st.metric("Ngân sách sử dụng", f"{m4['total_budget_used']:,.0f} tỷ VND")
            fig_net = px.bar(
                labor_df,
                x="Ngành",
                y="NetJob",
                text="NetJob",
                title="NetJob ròng theo ngành",
                color="NetJob",
                color_continuous_scale=PURPLE_SCALE,
            )
            fig_net.update_traces(texttemplate="%{text:,.0f}", textposition="outside")
            fig_net.update_xaxes(tickangle=-35)
            st.plotly_chart(_common_layout(fig_net, height=420), width="stretch")

        st.markdown("#### Phân rã tác động việc làm theo ngành")
        labor_breakdown = labor_df.melt(
            id_vars="Ngành",
            value_vars=["Việc mới", "Nâng cấp", "Bị thay thế"],
            var_name="Thành phần",
            value_name="Số việc làm",
        )
        fig_labor = px.bar(
            labor_breakdown,
            x="Ngành",
            y="Số việc làm",
            color="Thành phần",
            barmode="group",
            title="Tác động tạo mới, nâng cấp và thay thế việc làm",
            color_discrete_map={
                "Việc mới": PURPLE_SHADES[3],
                "Nâng cấp": PURPLE_SHADES[5],
                "Bị thay thế": PURPLE_SHADES[1],
            },
        )
        fig_labor.update_xaxes(tickangle=-35)
        st.plotly_chart(_common_layout(fig_labor, height=430), width="stretch")

        top_job = labor_df.sort_values("NetJob", ascending=False).iloc[0]
        retraining = labor_df.sort_values("x_H (tỷ)", ascending=False).iloc[0]
        st.success(
            f"**Nhận xét M4:** Ngành **{top_job['Ngành']}** tạo NetJob ròng lớn nhất "
            f"({top_job['NetJob']:,.0f} việc làm). Phân bổ đào tạo lại tập trung vào **{retraining['Ngành']}**, "
            "cho thấy chính sách nhân lực số là lớp đệm quan trọng khi tự động hóa tăng tốc."
        )

    with tab_pages[5]:
        st.subheader("M5: Radar Rủi ro Đa Chiều (Bài 7)")

        risk_df = pd.DataFrame([
            {
                "Kịch bản": scenario,
                "GDP Gain (tỷ VND)": values["gdp_gain"],
                "Gini Index": values["gini"],
                "Phát thải CO2": values["emission"],
                "Rủi ro Cyber": values["cyber_risk"],
            }
            for scenario, values in m5.items()
        ])

        radar_cols = ["GDP Gain (tỷ VND)", "Gini Index", "Phát thải CO2", "Rủi ro Cyber"]
        radar_labels = ["GDP Gain", "Bất bình đẳng (Gini)", "Phát thải CO2", "Rủi ro Cyber"]
        selected_scenarios = ["S1. Truyền thống", "S3. AI dẫn dắt", "S5. Tối ưu cân bằng"]

        radar_norm = risk_df.copy()
        for col in radar_cols:
            col_min = risk_df[col].min()
            col_max = risk_df[col].max()
            if col_max == col_min:
                radar_norm[col] = 50
            else:
                radar_norm[col] = (risk_df[col] - col_min) / (col_max - col_min) * 100

        fig_radar = go.Figure()
        radar_colors = {
            "S1. Truyền thống": PURPLE_SHADES[4],
            "S3. AI dẫn dắt": PURPLE_SHADES[2],
            "S5. Tối ưu cân bằng": PURPLE_SHADES[5],
        }
        for scenario in selected_scenarios:
            row = radar_norm.loc[radar_norm["Kịch bản"] == scenario, radar_cols].iloc[0].tolist()
            fig_radar.add_trace(go.Scatterpolar(
                r=row + [row[0]],
                theta=radar_labels + [radar_labels[0]],
                fill="toself",
                name=scenario,
                line=dict(color=radar_colors[scenario], width=2),
                opacity=0.65,
            ))
        fig_radar.update_layout(
            title="Radar Rủi ro Đa Chiều (chuẩn hóa 0-100)",
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 100], gridcolor="#ede9fe"),
                angularaxis=dict(gridcolor="#ddd6fe"),
            ),
            height=430,
            margin=dict(t=55, b=40, l=60, r=60),
            font=dict(color=TEXT_COLOR),
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_radar, width="stretch")

        st.markdown("#### Bảng tổng hợp KPI 5 kịch bản")
        st.dataframe(
            risk_df.style.format({
                "GDP Gain (tỷ VND)": "{:,.0f}",
                "Gini Index": "{:.4f}",
                "Phát thải CO2": "{:,.0f}",
                "Rủi ro Cyber": "{:,.0f}",
            }).background_gradient(cmap="Purples", subset=["GDP Gain (tỷ VND)", "Gini Index", "Phát thải CO2", "Rủi ro Cyber"]),
            hide_index=True,
            width="stretch",
        )

        high_cyber = risk_df.sort_values("Rủi ro Cyber", ascending=False).iloc[0]
        low_gini = risk_df.sort_values("Gini Index", ascending=True).iloc[0]
        st.markdown(
            f"""
            <div style="background:#f3e8ff;border-left:4px solid #7e22ce;border-radius:6px;padding:14px;color:#4c1d95;">
                <strong>Cảnh báo từ Module M5:</strong><br>
                • Kịch bản <strong>{high_cyber['Kịch bản']}</strong> có rủi ro Cyber cao nhất, cần bổ sung ngân sách an ninh dữ liệu.<br>
                • Kịch bản <strong>{low_gini['Kịch bản']}</strong> kiểm soát bất bình đẳng vùng tốt nhất, phù hợp với mục tiêu bao trùm.<br>
                • Với nhóm vùng có AI Readiness thấp, cần ưu tiên hạ tầng số và nhân lực số trước khi mở rộng AI quy mô lớn.
            </div>
            """,
            unsafe_allow_html=True,
        )

    with tab_pages[6]:
        st.subheader("Kết luận và Khuyến nghị Chính sách")
        st.success("Hệ thống AIDEOM-VN tổng hợp kết quả từ 6 module để chuyển dữ liệu thành khuyến nghị định lượng.")

        gdp_2030_df = pd.DataFrame({
            "Kịch bản": list(m1["scenarios_forecast"].keys()),
            "GDP 2030 (nghìn tỷ VND)": [values[-1] for values in m1["scenarios_forecast"].values()],
        })
        fig_gdp_2030 = px.bar(
            gdp_2030_df,
            x="Kịch bản",
            y="GDP 2030 (nghìn tỷ VND)",
            color="GDP 2030 (nghìn tỷ VND)",
            color_continuous_scale=PURPLE_SCALE,
            title="GDP dự báo năm 2030",
            text="GDP 2030 (nghìn tỷ VND)",
        )
        fig_gdp_2030.update_traces(texttemplate="%{text:,.0f}", textposition="outside")
        fig_gdp_2030.update_xaxes(tickangle=-10)
        st.plotly_chart(_common_layout(fig_gdp_2030, height=430), width="stretch")

        best_growth = gdp_2030_df.sort_values("GDP 2030 (nghìn tỷ VND)", ascending=False).iloc[0]
        balanced_risk = risk_df.assign(
            score=(
                risk_df["GDP Gain (tỷ VND)"].rank(ascending=False)
                + risk_df["Gini Index"].rank(ascending=True)
                + risk_df["Rủi ro Cyber"].rank(ascending=True)
            )
        ).sort_values("score").iloc[0]

        st.markdown(
            f"""
            <div style="background:#f3e8ff;border-left:4px solid #7e22ce;border-radius:6px;padding:16px;color:#4c1d95;line-height:1.65;">
                <strong>1. Không có tăng trưởng miễn phí:</strong><br>
                Kịch bản <strong>{best_growth['Kịch bản']}</strong> cho GDP 2030 cao nhất, nhưng cần được kiểm soát bằng lớp ràng buộc rủi ro và an sinh.<br><br>
                <strong>2. Nhân lực số là điều kiện nền:</strong><br>
                Kết quả M4 cho thấy đào tạo lại và nâng cấp kỹ năng tạo phần lớn NetJob ròng, đặc biệt ở các ngành dịch vụ công và tri thức.<br><br>
                <strong>3. Ràng buộc toán học định hình dòng ngân sách:</strong><br>
                M3 cho thấy các ràng buộc sàn/trần và đa dạng hóa làm ngân sách lan tỏa hơn, giảm nguy cơ tập trung quá mức vào một vài cực tăng trưởng.<br><br>
                <strong>4. Kịch bản cân bằng nên là phương án điều hành:</strong><br>
                Theo điểm tổng hợp GDP-Gini-Cyber, <strong>{balanced_risk['Kịch bản']}</strong> là lựa chọn có hồ sơ rủi ro cân đối hơn để triển khai chính sách thực tế.
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.divider()
        st.subheader("Đề xuất Mở rộng Nghiên cứu")
        st.markdown(
            """
            - **CGE/DSGE-AI:** Mở rộng sang mô hình cân bằng tổng thể động để mô phỏng tác động giá cả, lạm phát và năng suất dài hạn.
            - **Dữ liệu thời gian thực:** Kết nối Open Data Portal, Vietstock và dữ liệu hải quan để cập nhật ma trận beta theo quý.
            - **Multi-Agent RL:** Mô phỏng nhiều tác nhân tối ưu hóa lợi ích riêng để kiểm tra ổn định chính sách dưới cạnh tranh vùng/ngành.
            """
        )


