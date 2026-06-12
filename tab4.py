from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import pulp
import streamlit as st

PRIMARY_PURPLE = "#7e22ce"
PURPLE_SHADES = ["#f3e8ff", "#e9d5ff", "#d8b4fe", "#a855f7", "#7e22ce", "#4c1d95"]
TEXT_COLOR = "#4c1d95"

REGION_CODES = ["NMM", "RRD", "NCC", "CH", "SE", "MD"]
REGION_NAMES = [
    "Trung du miền núi phía Bắc",
    "Đồng bằng sông Hồng",
    "Bắc Trung Bộ + DH Trung Bộ",
    "Tây Nguyên",
    "Đông Nam Bộ",
    "ĐBSCL",
]
ITEMS = ["I", "D", "AI", "H"]
ITEM_NAMES = ["Hạ tầng số (I)", "CĐS DN (D)", "Năng lực AI", "Nhân lực số (H)"]

MIN_REGION_BUDGET = 5000
MAX_REGION_BUDGET = 12000
MIN_H_BUDGET = 12000
DEFAULT_MIN_CELL = 400

DEFAULT_BETA = {
    ("NMM", "I"): 1.15, ("NMM", "D"): 0.85, ("NMM", "AI"): 0.55, ("NMM", "H"): 1.30,
    ("RRD", "I"): 0.95, ("RRD", "D"): 1.25, ("RRD", "AI"): 1.40, ("RRD", "H"): 1.05,
    ("NCC", "I"): 1.05, ("NCC", "D"): 0.95, ("NCC", "AI"): 0.85, ("NCC", "H"): 1.15,
    ("CH", "I"): 1.20, ("CH", "D"): 0.75, ("CH", "AI"): 0.45, ("CH", "H"): 1.35,
    ("SE", "I"): 0.90, ("SE", "D"): 1.30, ("SE", "AI"): 1.55, ("SE", "H"): 1.00,
    ("MD", "I"): 1.10, ("MD", "D"): 0.85, ("MD", "AI"): 0.65, ("MD", "H"): 1.25,
}
DEFAULT_D0 = {"NMM": 38, "RRD": 78, "NCC": 55, "CH": 32, "SE": 82, "MD": 48}


def _to_float(value):
    if pd.isna(value):
        return np.nan
    return float(str(value).strip().replace(",", "."))


@st.cache_data
def _load_beta_coefficients():
    """Load beta from the supplied CSV; fall back to the embedded table if missing."""
    path = Path(__file__).resolve().parents[1] / "Bảng hệ số tác động biên bài 4.csv"
    if not path.exists():
        return DEFAULT_BETA.copy()

    raw = pd.read_csv(path, encoding="utf-8-sig").dropna(how="all")
    value_cols = [
        col for col in raw.columns
        if col != "Vùng" and not str(col).startswith("Unnamed")
    ][:4]

    if len(raw) < len(REGION_CODES) or len(value_cols) < len(ITEMS):
        return DEFAULT_BETA.copy()

    beta = {}
    for row_idx, region in enumerate(REGION_CODES):
        for item, col in zip(ITEMS, value_cols):
            beta[(region, item)] = _to_float(raw.iloc[row_idx][col])

    if any(pd.isna(v) for v in beta.values()):
        return DEFAULT_BETA.copy()
    return beta


def _load_d0(regions_df):
    if regions_df is None or "digital_index_0_100" not in regions_df.columns:
        return DEFAULT_D0.copy()
    values = pd.to_numeric(regions_df["digital_index_0_100"], errors="coerce").tolist()
    if len(values) < len(REGION_CODES) or any(pd.isna(v) for v in values[:len(REGION_CODES)]):
        return DEFAULT_D0.copy()
    return {region: float(values[idx]) for idx, region in enumerate(REGION_CODES)}


def _solve_budget_model(
    beta,
    d0,
    budget,
    fairness_lambda,
    gamma_coeff,
    use_fairness=True,
    use_diversity=True,
    min_cell=DEFAULT_MIN_CELL,
    use_region_cap=True,
):
    model = pulp.LpProblem("Regional_Digital_Budget", pulp.LpMaximize)
    x = pulp.LpVariable.dicts("x", (REGION_CODES, ITEMS), lowBound=0)

    model += pulp.lpSum(beta[(r, j)] * x[r][j] for r in REGION_CODES for j in ITEMS), "Expected_GDP_Gain"
    model += pulp.lpSum(x[r][j] for r in REGION_CODES for j in ITEMS) <= budget, "C1_Total_Budget"

    for r in REGION_CODES:
        model += pulp.lpSum(x[r][j] for j in ITEMS) >= MIN_REGION_BUDGET, f"C2_Min_Region_{r}"
        if use_region_cap:
            model += pulp.lpSum(x[r][j] for j in ITEMS) <= MAX_REGION_BUDGET, f"C3_Max_Region_{r}"
        if use_diversity:
            for j in ITEMS:
                model += x[r][j] >= min_cell, f"C6_Diversity_{r}_{j}"

    model += pulp.lpSum(x[r]["H"] for r in REGION_CODES) >= MIN_H_BUDGET, "C4_Min_Human_Capital"

    m_var = None
    if use_fairness:
        m_var = pulp.LpVariable("M", lowBound=0)
        for r in REGION_CODES:
            d_after = d0[r] + gamma_coeff * x[r]["D"]
            model += d_after <= m_var, f"C5a_Max_D_Index_{r}"
            model += d_after >= fairness_lambda * m_var, f"C5b_Min_D_Index_{r}"

    model.solve(pulp.PULP_CBC_CMD(msg=False))
    status = pulp.LpStatus[model.status]
    if status != "Optimal":
        return {
            "status": status,
            "objective": None,
            "allocation": None,
            "m_value": None,
            "d_after": None,
        }

    allocation = {
        r: {j: float(pulp.value(x[r][j]) or 0) for j in ITEMS}
        for r in REGION_CODES
    }
    d_after = {r: d0[r] + gamma_coeff * allocation[r]["D"] for r in REGION_CODES}
    m_value = float(pulp.value(m_var)) if m_var is not None else max(d_after.values())
    return {
        "status": status,
        "objective": float(pulp.value(model.objective)),
        "allocation": allocation,
        "m_value": m_value,
        "d_after": d_after,
    }


def _allocation_frame(allocation):
    matrix = [[allocation[r][j] for j in ITEMS] for r in REGION_CODES]
    df = pd.DataFrame(matrix, index=REGION_NAMES, columns=ITEM_NAMES)
    df["Tổng vùng"] = df.sum(axis=1)
    return df


def _format_heatmap_text(values):
    text = []
    for row in values:
        text.append([
            f"{v / 1000:.1f}k" if v >= 10000 else f"{v:,.0f}".replace(",", "")
            for v in row
        ])
    return text


def _constraint_summary(allocation, d0, d_after, m_value, budget, fairness_lambda, use_diversity, min_cell):
    values = np.array([[allocation[r][j] for j in ITEMS] for r in REGION_CODES])
    region_totals = values.sum(axis=1)
    total_used = values.sum()
    h_total = sum(allocation[r]["H"] for r in REGION_CODES)
    threshold = fairness_lambda * m_value
    tol = 1e-5

    rows = [
        ("C1 Tổng ngân sách", f"{total_used:,.0f}", f"≤ {budget:,.0f}", total_used <= budget + tol),
        ("C2 Sàn mỗi vùng", f"{region_totals.min():,.0f}", f"≥ {MIN_REGION_BUDGET:,.0f}", region_totals.min() + tol >= MIN_REGION_BUDGET),
        ("C3 Trần mỗi vùng", f"{region_totals.max():,.0f}", f"≤ {MAX_REGION_BUDGET:,.0f}", region_totals.max() <= MAX_REGION_BUDGET + tol),
        ("C4 Tổng nhân lực số H", f"{h_total:,.0f}", f"≥ {MIN_H_BUDGET:,.0f}", h_total + tol >= MIN_H_BUDGET),
        ("C5 Công bằng D-index", f"{min(d_after.values()):.2f} đến {max(d_after.values()):.2f}", f"≥ {threshold:.2f} và ≤ {m_value:.2f}", min(d_after.values()) + tol >= threshold),
    ]
    if use_diversity:
        rows.append(("C6 Đa dạng hóa", f"{values.min():,.0f}", f"≥ {min_cell:,.0f} mỗi ô", values.min() + tol >= min_cell))

    return pd.DataFrame(rows, columns=["Ràng buộc", "Giá trị kiểm tra", "Ngưỡng", "Kết quả"]).assign(
        **{"Kết quả": lambda df: df["Kết quả"].map(lambda ok: "Đạt" if ok else "Chưa đạt")}
    )


def _c5_detail_frame(allocation, d0, d_after, m_value, fairness_lambda):
    threshold = fairness_lambda * m_value
    return pd.DataFrame({
        "Vùng": REGION_NAMES,
        "D₀ ban đầu": [d0[r] for r in REGION_CODES],
        "x_D đầu tư": [allocation[r]["D"] for r in REGION_CODES],
        "D sau ĐT": [d_after[r] for r in REGION_CODES],
        f"Ngưỡng (λ·M={fairness_lambda:.1f}·{m_value:.1f})": [threshold] * len(REGION_CODES),
        "Thỏa C5?": ["Có" if d_after[r] + 1e-5 >= threshold else "Không" for r in REGION_CODES],
    })


def _solve_with_cvxpy(beta, d0, budget, fairness_lambda, gamma_coeff, use_diversity, min_cell):
    try:
        import cvxpy as cp
    except ImportError:
        return {"status": "not_installed", "message": "CVXPY chưa cài. Cài bằng: `pip install cvxpy`"}

    x = cp.Variable((len(REGION_CODES), len(ITEMS)), nonneg=True)
    m_var = cp.Variable(nonneg=True)
    beta_mat = np.array([[beta[(r, j)] for j in ITEMS] for r in REGION_CODES])
    d0_vec = np.array([d0[r] for r in REGION_CODES])

    constraints = [
        cp.sum(x) <= budget,
        cp.sum(x[:, ITEMS.index("H")]) >= MIN_H_BUDGET,
    ]
    for i in range(len(REGION_CODES)):
        constraints += [
            cp.sum(x[i, :]) >= MIN_REGION_BUDGET,
            cp.sum(x[i, :]) <= MAX_REGION_BUDGET,
            d0_vec[i] + gamma_coeff * x[i, ITEMS.index("D")] <= m_var,
            d0_vec[i] + gamma_coeff * x[i, ITEMS.index("D")] >= fairness_lambda * m_var,
        ]
        if use_diversity:
            constraints.append(x[i, :] >= min_cell)

    problem = cp.Problem(cp.Maximize(cp.sum(cp.multiply(beta_mat, x))), constraints)
    errors = []
    for solver, kwargs in [
        (getattr(cp, "SCIPY", None), {"scipy_options": {"method": "highs"}}),
        (getattr(cp, "CLARABEL", None), {}),
        (None, {}),
    ]:
        try:
            if solver is None:
                problem.solve()
            else:
                problem.solve(solver=solver, **kwargs)
            if problem.status in {"optimal", "optimal_inaccurate"}:
                break
        except Exception as exc:
            errors.append(str(exc))

    if problem.status not in {"optimal", "optimal_inaccurate"}:
        detail = f"({'; '.join(errors[:2])})" if errors else ""
        return {"status": problem.status, "message": f"CVXPY không hội tụ: {problem.status}{detail}"}

    values = np.maximum(np.asarray(x.value, dtype=float), 0)
    allocation = {
        r: {j: float(values[i, k]) for k, j in enumerate(ITEMS)}
        for i, r in enumerate(REGION_CODES)
    }
    return {
        "status": problem.status,
        "objective": float(problem.value),
        "allocation": allocation,
        "m_value": float(m_var.value),
    }


def _cvxpy_compare_frame(pulp_allocation, cvx_allocation):
    rows = []
    for r, region_name in zip(REGION_CODES, REGION_NAMES):
        for item, item_name in zip(ITEMS, ITEM_NAMES):
            pulp_value = pulp_allocation[r][item]
            cvx_value = cvx_allocation[r][item]
            rows.append({
                "Vùng": region_name,
                "Hạng mục": item_name,
                "PuLP": pulp_value,
                "CVXPY": cvx_value,
                "Chênh lệch": abs(pulp_value - cvx_value),
            })
    return pd.DataFrame(rows)


def _scenario_cost_rows(base_result, scenarios):
    rows = []
    base_z = base_result["objective"]
    for label, result in scenarios:
        if result["status"] != "Optimal":
            rows.append({
                "Loại chi phí": label,
                "Z* so sánh": "Không tối ưu",
                "GDP mất đi": "N/A",
                "Tỷ lệ": "N/A",
            })
            continue
        cost = result["objective"] - base_z
        pct = cost / result["objective"] * 100 if result["objective"] else 0
        rows.append({
            "Loại chi phí": label,
            "Z* so sánh": result["objective"],
            "GDP mất đi": cost,
            "Tỷ lệ": pct,
        })
    return pd.DataFrame(rows)


def render(macro_df, sectors_df, regions_df):
    st.header("Bài 4: Phân bổ Ngân sách Số theo Vùng (LP)")

    beta = _load_beta_coefficients()
    d0 = _load_d0(regions_df)

    with st.container(border=True):
        st.markdown("### Cấu hình Mô hình")
        col1, col2, col3, col4 = st.columns([1.15, 1.25, 1.15, 1.15])
        total_budget = col1.number_input("Tổng Ngân sách (tỷ VND)", value=50000, min_value=30000, step=1000)
        fairness_lambda = col2.slider("Hệ số công bằng λ", 0.0, 1.0, 0.60, 0.05)
        gamma_coeff = col3.number_input("γ (hệ số chuyển đổi D)", value=0.002, min_value=0.0, step=0.001, format="%.3f")
        use_diversity = col4.checkbox(
            f"Đa dạng hóa (Mỗi ô ≥ {DEFAULT_MIN_CELL})",
            value=True,
            help="Bật ràng buộc C6: mọi cặp vùng-hạng mục phải nhận tối thiểu 400 tỷ VND.",
        )

    result = _solve_budget_model(
        beta,
        d0,
        total_budget,
        fairness_lambda,
        gamma_coeff,
        use_fairness=True,
        use_diversity=use_diversity,
        min_cell=DEFAULT_MIN_CELL,
    )

    if result["status"] != "Optimal":
        st.error(f"PuLP/CBC không tìm được nghiệm tối ưu: {result['status']}")
        return

    allocation = result["allocation"]
    z_pulp = result["objective"]
    m_value = result["m_value"]
    d_after = result["d_after"]
    df_alloc = _allocation_frame(allocation)
    threshold = fairness_lambda * m_value

    col_left, col_right = st.columns([1.05, 1])
    with col_left:
        with st.container(border=True):
            st.markdown("### Câu 4.4.1 — Nghiệm tối ưu bằng PuLP/CBC")
            st.success(f"**GDP kỳ vọng tăng thêm Z* = {z_pulp:,.0f} tỷ VND**")
            st.success(f"C5 thỏa: M = {m_value:.1f}, ngưỡng tối thiểu = {threshold:.1f}")
            st.success(
                "**Nhận xét 4.4.1:** Nghiệm tối ưu vẫn ưu tiên các hạng mục có hệ số biên cao, "
                "nhưng ràng buộc C5 điều chỉnh thêm vốn cho CĐS DN tại các vùng có chỉ số số hóa ban đầu thấp."
            )

            st.dataframe(
                df_alloc.style.format("{:,.0f}").background_gradient(cmap="Purples", subset=ITEM_NAMES),
                width="stretch",
            )

            with st.expander("Kiểm tra tổng hợp các ràng buộc"):
                st.dataframe(
                    _constraint_summary(
                        allocation,
                        d0,
                        d_after,
                        m_value,
                        total_budget,
                        fairness_lambda,
                        use_diversity,
                        DEFAULT_MIN_CELL,
                    ),
                    hide_index=True,
                    width="stretch",
                )

            with st.expander("Kiểm tra chi tiết ràng buộc C5", expanded=True):
                st.dataframe(
                    _c5_detail_frame(allocation, d0, d_after, m_value, fairness_lambda).style.format({
                        "D₀ ban đầu": "{:.0f}",
                        "x_D đầu tư": "{:,.0f}",
                        "D sau ĐT": "{:.2f}",
                        f"Ngưỡng (λ·M={fairness_lambda:.1f}·{m_value:.1f})": "{:.2f}",
                    }),
                    hide_index=True,
                    width="stretch",
                )

    with col_right:
        with st.container(border=True):
            st.markdown("### Câu 4.4.3 — Bản đồ nhiệt phân bổ ngân sách")
            heatmap_values = df_alloc[ITEM_NAMES].values
            fig_hm = px.imshow(
                heatmap_values,
                x=ITEM_NAMES,
                y=REGION_NAMES,
                color_continuous_scale="Purples",
                aspect="auto",
                labels=dict(x="Hạng mục", y="Vùng", color="Tỷ VND"),
            )
            fig_hm.update_traces(
                text=_format_heatmap_text(heatmap_values),
                texttemplate="%{text}",
                hovertemplate="Vùng=%{y}<br>Hạng mục=%{x}<br>Ngân sách=%{z:,.0f} tỷ VND<extra></extra>",
            )
            fig_hm.update_layout(
                height=420,
                margin=dict(t=45, b=35, l=20, r=20),
                font=dict(color=TEXT_COLOR),
                xaxis=dict(tickangle=-30),
            )
            st.plotly_chart(fig_hm, width="stretch")
            st.success(
                "**Nhận xét 4.4.3:** Ma trận phân bổ cho thấy ngân sách tập trung theo lợi thế biên của từng vùng: "
                "AI nổi bật tại các vùng có β cao, trong khi CĐS DN được bổ sung cho nhóm có D-index thấp để bảo đảm C5."
            )

    with st.container(border=True):
        st.markdown("### Cơ cấu ngân sách theo vùng và hạng mục")
        df_melt = (
            df_alloc[ITEM_NAMES]
            .reset_index()
            .rename(columns={"index": "Vùng"})
            .melt(id_vars="Vùng", var_name="Hạng mục", value_name="Ngân sách (tỷ VND)")
        )
        fig_bar = px.bar(
            df_melt,
            x="Vùng",
            y="Ngân sách (tỷ VND)",
            color="Hạng mục",
            color_discrete_map={
                "Hạ tầng số (I)": PURPLE_SHADES[5],
                "CĐS DN (D)": PURPLE_SHADES[4],
                "Năng lực AI": PURPLE_SHADES[3],
                "Nhân lực số (H)": PURPLE_SHADES[2],
            },
        )
        fig_bar.update_layout(
            barmode="stack",
            height=430,
            margin=dict(t=30, b=60),
            font=dict(color=TEXT_COLOR),
            legend=dict(orientation="h", y=-0.22, x=0),
        )
        st.plotly_chart(fig_bar, width="stretch")

    with st.container(border=True):
        st.markdown("### Câu 4.4.2 — Kiểm chứng nghiệm bằng CVXPY")
        cvx_result = _solve_with_cvxpy(
            beta,
            d0,
            total_budget,
            fairness_lambda,
            gamma_coeff,
            use_diversity,
            DEFAULT_MIN_CELL,
        )
        if cvx_result["status"] in {"optimal", "optimal_inaccurate"}:
            z_cvx = cvx_result["objective"]
            diff_z = abs(z_pulp - z_cvx)
            st.success(f"CVXPY Z* = {z_cvx:,.0f} tỷ — khớp với PuLP (chênh lệch {diff_z:.2f} tỷ)")
            compare_df = _cvxpy_compare_frame(allocation, cvx_result["allocation"])
            st.dataframe(
                compare_df.style.format({"PuLP": "{:,.0f}", "CVXPY": "{:,.0f}", "Chênh lệch": "{:,.1f}"}),
                hide_index=True,
                width="stretch",
                height=430,
            )
        else:
            st.info(cvx_result.get("message", f"CVXPY trạng thái: {cvx_result['status']}"))

    with st.container(border=True):
        st.markdown("### Câu 4.4.4 — Đánh đổi hiệu quả khi áp dụng công bằng")
        no_c5 = _solve_budget_model(
            beta,
            d0,
            total_budget,
            fairness_lambda,
            gamma_coeff,
            use_fairness=False,
            use_diversity=use_diversity,
            min_cell=DEFAULT_MIN_CELL,
        )
        no_c6 = _solve_budget_model(
            beta,
            d0,
            total_budget,
            fairness_lambda,
            gamma_coeff,
            use_fairness=True,
            use_diversity=False,
            min_cell=DEFAULT_MIN_CELL,
        )
        no_c3 = _solve_budget_model(
            beta,
            d0,
            total_budget,
            fairness_lambda,
            gamma_coeff,
            use_fairness=True,
            use_diversity=use_diversity,
            min_cell=DEFAULT_MIN_CELL,
            use_region_cap=False,
        )

        if no_c5["status"] == "Optimal":
            fairness_cost = no_c5["objective"] - z_pulp
            fairness_pct = fairness_cost / no_c5["objective"] * 100 if no_c5["objective"] else 0
            col_a, col_b, col_c = st.columns(3)
            col_a.metric("Z* (có ràng buộc C5)", f"{z_pulp:,.0f} tỷ")
            col_b.metric("Z* (không C5)", f"{no_c5['objective']:,.0f} tỷ")
            col_c.metric("Chi phí công bằng", f"{fairness_cost:,.0f} tỷ", delta=f"-{fairness_pct:.2f}% GDP gain")

            st.info(
                f"Việc áp ràng buộc công bằng C5 làm giảm {fairness_cost:,.0f} tỷ VND GDP kỳ vọng "
                f"({fairness_pct:.2f}% so với kịch bản không C5). Đây là chi phí cơ hội của "
                "việc chuyển một phần hiệu quả kinh tế sang mục tiêu thu hẹp chênh lệch số giữa các vùng."
            )

            cost_df = _scenario_cost_rows(
                result,
                [
                    ("Chi phí công bằng C5", no_c5),
                    ("Chi phí đa dạng hóa C6", no_c6),
                    ("Chi phí trần vùng C3", no_c3),
                ],
            )
            st.dataframe(
                cost_df.style.format({
                    "Z* so sánh": lambda v: v if isinstance(v, str) else f"{v:,.0f}",
                    "GDP mất đi": lambda v: v if isinstance(v, str) else f"{v:,.0f}",
                    "Tỷ lệ": lambda v: v if isinstance(v, str) else f"{v:.2f}%",
                }),
                hide_index=True,
                width="stretch",
            )

            compare_cost_df = pd.DataFrame({
                "Vùng": REGION_NAMES,
                "Tổng (có C5)": [sum(allocation[r][j] for j in ITEMS) for r in REGION_CODES],
                "Tổng (không C5)": [sum(no_c5["allocation"][r][j] for j in ITEMS) for r in REGION_CODES],
                "D_index (có C5)": [d_after[r] for r in REGION_CODES],
                "D_index (không C5)": [no_c5["d_after"][r] for r in REGION_CODES],
            })
            st.dataframe(
                compare_cost_df.style.format({
                    "Tổng (có C5)": "{:,.0f}",
                    "Tổng (không C5)": "{:,.0f}",
                    "D_index (có C5)": "{:.1f}",
                    "D_index (không C5)": "{:.1f}",
                }),
                hide_index=True,
                width="stretch",
            )
            st.success(
                "**Nhận xét 4.4.4:** Khi bỏ C5, Z* tăng do ngân sách dịch chuyển nhiều hơn về các vùng có β cao. "
                "Phần chênh lệch GDP kỳ vọng phản ánh chi phí kinh tế của mục tiêu phát triển cân bằng."
            )
        else:
            st.error("Không thể giải kịch bản không có ràng buộc C5.")

    with st.container(border=True):
        st.markdown("### Câu 4.5 — Thảo luận chính sách")
        st.info(
            "**a) Nếu bỏ ràng buộc công bằng, vốn chảy về đâu?** "
            "Vốn ưu tiên chảy về Đồng bằng sông Hồng và Đông Nam Bộ do hệ số biên β cao, "
            "đặc biệt ở AI. Hậu quả là khoảng cách số giữa nhóm vùng dẫn đầu và vùng yếu sẽ tăng."
        )
        st.info(
            "**b) Ràng buộc trần ngân sách mỗi vùng (C3) ảnh hưởng thế nào?** "
            "C3 ngăn một vùng hấp thụ quá nhiều ngân sách và buộc phân bổ lan tỏa. Trong cấu hình hiện tại, "
            "chi phí C3 được lượng hóa ở bảng 4.4.4, nên có thể xem là chi phí để tránh tập trung quá mức."
        )
        st.warning(
            "**c) Tây Nguyên có hệ số AI thấp (0.45), nên ưu tiên gì?** "
            "Mô hình không dồn vốn vào AI tại Tây Nguyên. Khi có C5, vốn D tăng mạnh để nâng D-index; "
            "phần còn lại nên ưu tiên hạ tầng số và nhân lực số trước khi mở rộng các ứng dụng AI."
        )


