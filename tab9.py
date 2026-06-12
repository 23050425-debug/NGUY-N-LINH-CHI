from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pulp
import streamlit as st


PARAM_FILE_NAME = "Tham số 8 ngành Việt Nam bài 9.csv"

GREEN = "#16a34a"
GREEN_SOFT = "rgba(22, 163, 74, 0.48)"
ORANGE = "#f97316"
ORANGE_SOFT = "rgba(249, 115, 22, 0.48)"
RED = "#ef4444"
RED_SOFT = "rgba(239, 68, 68, 0.42)"
SLATE = "#334155"
SLATE_SOFT = "rgba(51, 65, 85, 0.28)"


@st.cache_data
def load_bai9_params():
    base_dir = Path(__file__).resolve().parents[1]
    search_paths = [
        base_dir / PARAM_FILE_NAME,
        Path.cwd() / PARAM_FILE_NAME,
        base_dir / "data" / PARAM_FILE_NAME,
    ]
    data_path = next((path for path in search_paths if path.exists()), None)
    if data_path is None:
        raise FileNotFoundError(f"Không tìm thấy file '{PARAM_FILE_NAME}' trong thư mục MÔ HÌNH.")

    raw = pd.read_csv(data_path, decimal=",")
    raw.columns = raw.columns.str.strip()
    required_cols = [
        "Ngành",
        "Lao động (triệu)",
        "Risk (%)",
        "a₁ (việc/tỷ)",
        "a₂ (việc/tỷ)",
        "b₁ (việc/tỷ)",
        "c₁ (việc/tỷ)",
        "d₁ (việc/tỷ)",
    ]
    missing = [col for col in required_cols if col not in raw.columns]
    if missing:
        raise ValueError("File tham số bài 9 thiếu cột: " + ", ".join(missing))

    df = raw.rename(
        columns={
            "Ngành": "sector",
            "Lao động (triệu)": "labor_million",
            "Risk (%)": "risk_pct",
            "a₁ (việc/tỷ)": "a1",
            "a₂ (việc/tỷ)": "a2",
            "b₁ (việc/tỷ)": "b1",
            "c₁ (việc/tỷ)": "c1",
            "d₁ (việc/tỷ)": "d1",
        }
    ).copy()

    for col in ["labor_million", "risk_pct", "a1", "a2", "b1", "c1", "d1"]:
        if not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = (
                df[col]
                .astype(str)
                .str.strip()
                .str.replace(".", "", regex=False)
                .str.replace(",", ".", regex=False)
                .astype(float)
            )

    df["sector_full"] = df["sector"].astype(str)
    df["sector"] = df["sector"].astype(str).str.replace(r"^\s*\d+\.\s*", "", regex=True)
    df["sector_id"] = np.arange(1, len(df) + 1)
    df["labor"] = df["labor_million"] * 1_000_000
    df["risk_rate"] = df["risk_pct"] / 100
    df["displaced_per_ai"] = df["c1"] * df["risk_rate"]
    df["retrain_ratio"] = np.divide(
        df["displaced_per_ai"],
        df["d1"],
        out=np.zeros(len(df), dtype=float),
        where=df["d1"].to_numpy() != 0,
    )
    df["ai_net_per_ty"] = df["a1"] - df["displaced_per_ai"]
    df["package_efficiency"] = np.divide(
        df["a1"] + df["b1"] * df["retrain_ratio"] - df["displaced_per_ai"],
        1 + df["retrain_ratio"],
        out=np.zeros(len(df), dtype=float),
        where=(1 + df["retrain_ratio"]).to_numpy() != 0,
    )
    return df, str(data_path)


def solve_netjob(params, total_budget, min_invest_enabled, min_invest_value, cap_displacement):
    n = len(params)
    labor = params["labor"].to_numpy(dtype=float)
    risk = params["risk_rate"].to_numpy(dtype=float)
    a1 = params["a1"].to_numpy(dtype=float)
    b1 = params["b1"].to_numpy(dtype=float)
    c1 = params["c1"].to_numpy(dtype=float)
    d1 = params["d1"].to_numpy(dtype=float)

    prob = pulp.LpProblem("NetJob_Maximization", pulp.LpMaximize)
    x_ai = [pulp.LpVariable(f"x_ai_{i}", lowBound=0) for i in range(n)]
    x_h = [pulp.LpVariable(f"x_h_{i}", lowBound=0) for i in range(n)]

    new_jobs = [a1[i] * x_ai[i] for i in range(n)]
    upgrade_jobs = [b1[i] * x_h[i] for i in range(n)]
    displaced_jobs = [c1[i] * risk[i] * x_ai[i] for i in range(n)]
    retrain_capacity = [d1[i] * x_h[i] for i in range(n)]
    net_jobs = [new_jobs[i] + upgrade_jobs[i] - displaced_jobs[i] for i in range(n)]

    prob += pulp.lpSum(net_jobs)
    prob += pulp.lpSum(x_ai[i] + x_h[i] for i in range(n)) <= float(total_budget), "TotalBudget"

    for i in range(n):
        prob += net_jobs[i] >= 0, f"NetJob_nonnegative_{i}"
        prob += displaced_jobs[i] <= retrain_capacity[i], f"Retrain_enough_{i}"
        prob += retrain_capacity[i] <= displaced_jobs[i], f"Retrain_not_excess_{i}"
        if min_invest_enabled:
            prob += x_ai[i] + x_h[i] >= float(min_invest_value), f"MinInvest_{i}"

    if cap_displacement:
        prob += pulp.lpSum(displaced_jobs) <= 0.05 * float(labor.sum()), "TotalDisplacementCap"

    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    status = pulp.LpStatus[prob.status]
    if status != "Optimal":
        return None, status

    x_ai_val = np.array([pulp.value(var) or 0 for var in x_ai], dtype=float)
    x_h_val = np.array([pulp.value(var) or 0 for var in x_h], dtype=float)
    new_val = a1 * x_ai_val
    upgrade_val = b1 * x_h_val
    disp_val = c1 * risk * x_ai_val
    retrain_val = d1 * x_h_val
    net_val = new_val + upgrade_val - disp_val
    total_invest = x_ai_val + x_h_val
    used_budget = total_invest.sum()

    alloc = pd.DataFrame(
        {
            "Mã ngành": params["sector_id"],
            "Ngành": params["sector"],
            "Đầu tư AI (tỷ VND)": x_ai_val,
            "Đào tạo lại (tỷ VND)": x_h_val,
            "Tổng đầu tư (tỷ VND)": total_invest,
            "Việc mới (AI)": new_val,
            "Việc nâng cấp (Đào tạo)": upgrade_val,
            "Mất việc (Tự động hóa)": disp_val,
            "Năng lực đào tạo (người)": retrain_val,
            "NetJob (Ròng)": net_val,
            "Tỷ lệ H/AI": np.divide(
                x_h_val,
                x_ai_val,
                out=np.zeros(n, dtype=float),
                where=x_ai_val != 0,
            ),
            "Tỷ trọng ngân sách (%)": np.divide(
                total_invest,
                used_budget,
                out=np.zeros(n, dtype=float),
                where=used_budget != 0,
            )
            * 100,
        }
    )
    return alloc, status


def format_param_table(params):
    table = params[
        [
            "sector_id",
            "sector",
            "labor_million",
            "risk_pct",
            "a1",
            "a2",
            "b1",
            "c1",
            "d1",
        ]
    ].rename(
        columns={
            "sector_id": "Mã",
            "sector": "Ngành",
            "labor_million": "Lao động (triệu)",
            "risk_pct": "Risk (%)",
            "a1": "a₁ (việc/tỷ)",
            "a2": "a₂ (việc/tỷ)",
            "b1": "b₁ (việc/tỷ)",
            "c1": "c₁ (việc/tỷ)",
            "d1": "d₁ (việc/tỷ)",
        }
    )
    return table


def format_constraint_table(params):
    table = params[
        [
            "sector_id",
            "sector",
            "displaced_per_ai",
            "retrain_ratio",
            "ai_net_per_ty",
            "package_efficiency",
        ]
    ].rename(
        columns={
            "sector_id": "Mã",
            "sector": "Ngành",
            "displaced_per_ai": "Mất việc / 1 tỷ AI",
            "retrain_ratio": "Tỷ lệ đào tạo/AI bắt buộc",
            "ai_net_per_ty": "Net AI trước đào tạo",
            "package_efficiency": "Net gói AI + đào tạo",
        }
    )
    table["Nhóm dễ tổn thương"] = np.where(table["Mã"].isin([1, 3, 4]), "Có", "Không")
    return table


def highlight_selected(row, selected_sector):
    color = "background-color: #fff7ed" if row["Ngành"] == selected_sector else ""
    return [color for _ in row]


def make_netjob_figure(alloc, selected_sector):
    plot_df = alloc.sort_values("NetJob (Ròng)", ascending=True)
    bar_colors = [
        ORANGE if sector == selected_sector else GREEN
        for sector in plot_df["Ngành"]
    ]
    investment = plot_df["Tổng đầu tư (tỷ VND)"].to_numpy(dtype=float)
    if investment.max() > investment.min():
        marker_sizes = np.interp(investment, (investment.min(), investment.max()), (8, 22))
    else:
        marker_sizes = np.full(len(plot_df), 14)

    customdata = np.stack(
        [
            plot_df["Tổng đầu tư (tỷ VND)"],
            plot_df["Việc mới (AI)"],
            plot_df["Việc nâng cấp (Đào tạo)"],
            plot_df["Mất việc (Tự động hóa)"],
        ],
        axis=-1,
    )
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            y=plot_df["Ngành"],
            x=plot_df["NetJob (Ròng)"],
            orientation="h",
            marker=dict(color=bar_colors, line=dict(color="white", width=1)),
            text=[f"{value:,.0f}" for value in plot_df["NetJob (Ròng)"]],
            textposition="outside",
            customdata=customdata,
            hovertemplate=(
                "<b>%{y}</b><br>"
                "NetJob: %{x:,.0f} người<br>"
                "Tổng đầu tư: %{customdata[0]:,.0f} tỷ VND<br>"
                "Việc mới: %{customdata[1]:,.0f}<br>"
                "Việc nâng cấp: %{customdata[2]:,.0f}<br>"
                "Mất việc: %{customdata[3]:,.0f}"
                "<extra></extra>"
            ),
        )
    )
    fig.add_trace(
        go.Scatter(
            y=plot_df["Ngành"],
            x=plot_df["NetJob (Ròng)"],
            mode="markers",
            marker=dict(size=marker_sizes, color=SLATE, opacity=0.55, line=dict(width=1, color="white")),
            hoverinfo="skip",
            showlegend=False,
        )
    )
    fig.update_layout(
        height=430,
        margin=dict(l=10, r=30, t=20, b=30),
        xaxis_title="Việc làm ròng (người)",
        yaxis_title="",
        showlegend=False,
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(color="#1f2937"),
    )
    fig.update_xaxes(tickformat=",.0f", gridcolor="#e5e7eb", zerolinecolor="#94a3b8")
    fig.update_yaxes(categoryorder="array", categoryarray=plot_df["Ngành"])
    return fig


def make_sector_waterfall(alloc, selected_sector):
    row = alloc.loc[alloc["Ngành"] == selected_sector].iloc[0]
    values = [
        row["Việc mới (AI)"],
        row["Việc nâng cấp (Đào tạo)"],
        -row["Mất việc (Tự động hóa)"],
        row["NetJob (Ròng)"],
    ]
    fig = go.Figure(
        go.Waterfall(
            x=["Việc mới", "Nâng cấp", "Mất việc", "NetJob"],
            y=values,
            measure=["relative", "relative", "relative", "total"],
            text=[f"{value:,.0f}" for value in values],
            textposition="outside",
            connector=dict(line=dict(color="#94a3b8", width=1)),
            increasing=dict(marker=dict(color=GREEN)),
            decreasing=dict(marker=dict(color=RED)),
            totals=dict(marker=dict(color=ORANGE)),
            hovertemplate="<b>%{x}</b><br>%{y:,.0f} người<extra></extra>",
        )
    )
    fig.update_layout(
        title=f"Phân rã NetJob: {selected_sector}",
        height=430,
        margin=dict(l=10, r=10, t=50, b=30),
        yaxis_title="Số lao động",
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(color="#1f2937"),
        showlegend=False,
    )
    fig.update_yaxes(tickformat=",.0f", gridcolor="#e5e7eb")
    return fig


def make_sankey_figure(flow_df):
    sectors = flow_df["Ngành"].tolist()
    labels = sectors + ["Việc mới (AI tạo ra)", "Việc nâng cấp (Đào tạo)", "Mất việc (Tự động hóa)"]
    target_offset = len(sectors)
    source_links = []
    target_links = []
    value_links = []
    color_links = []
    link_labels = []

    for idx, row in flow_df.iterrows():
        source_index = sectors.index(row["Ngành"])
        flows = [
            ("Việc mới (AI tạo ra)", row["Việc mới (AI)"], GREEN_SOFT, target_offset),
            ("Việc nâng cấp (Đào tạo)", row["Việc nâng cấp (Đào tạo)"], ORANGE_SOFT, target_offset + 1),
            ("Mất việc (Tự động hóa)", row["Mất việc (Tự động hóa)"], RED_SOFT, target_offset + 2),
        ]
        for target_name, value, color, target_index in flows:
            if value <= 0:
                continue
            source_links.append(source_index)
            target_links.append(target_index)
            value_links.append(float(value))
            color_links.append(color)
            link_labels.append(f"{row['Ngành']}  {target_name}")

    source_y = np.linspace(0.08, 0.88, len(sectors)).tolist() if sectors else []
    node_x = [0.02] * len(sectors) + [0.98, 0.98, 0.98]
    node_y = source_y + [0.12, 0.50, 0.86]
    node_colors = [SLATE] * len(sectors) + [GREEN, ORANGE, RED]

    fig = go.Figure(
        data=[
            go.Sankey(
                arrangement="fixed",
                node=dict(
                    pad=18,
                    thickness=18,
                    line=dict(color="#0f172a", width=0.4),
                    label=labels,
                    color=node_colors,
                    x=node_x,
                    y=node_y,
                    hovertemplate="<b>%{label}</b><br>Tổng luồng: %{value:,.0f} người<extra></extra>",
                ),
                link=dict(
                    source=source_links,
                    target=target_links,
                    value=value_links,
                    color=color_links,
                    customdata=link_labels,
                    hovertemplate="<b>%{customdata}</b><br>%{value:,.0f} người<extra></extra>",
                ),
            )
        ]
    )
    fig.update_layout(
        height=500,
        margin=dict(l=10, r=10, t=20, b=20),
        font=dict(size=12, color="#1f2937"),
        paper_bgcolor="white",
    )
    return fig


def render(macro_df, sectors_df, regions_df):
    st.header("Bài 9: Tác động AI đến Lao động (Mô hình NetJob)")
    st.caption(
        "Bài toán LP tối đa hóa tổng việc làm ròng, đồng thời buộc đầu tư đào tạo đi đúng với số lao động bị dịch chuyển bởi AI."
    )

    try:
        params, data_path = load_bai9_params()
    except Exception as exc:
        st.error(f"Lỗi đọc dữ liệu bài 9: {exc}")
        return

    sector_options = params["sector"].tolist()

    with st.container(border=True):
        st.markdown("#### Tham số đầu tư và ràng buộc")
        col1, col2, col3, col4 = st.columns([1.25, 1.2, 1.1, 1.1])
        total_budget = col1.number_input(
            "Tổng ngân sách AI & đào tạo (tỷ VND)",
            min_value=0.0,
            value=30000.0,
            step=1000.0,
            format="%.0f",
        )
        selected_sector = col2.selectbox(
            "Ngành đang xem chi tiết",
            sector_options,
            index=sector_options.index("CNTT-Truyền thông") if "CNTT-Truyền thông" in sector_options else 0,
        )
        min_invest_enabled = col3.checkbox("Bảo đảm tối thiểu mỗi ngành", value=True)
        min_invest_value = col3.number_input(
            "Mức tối thiểu (tỷ VND/ngành)",
            min_value=0.0,
            value=1500.0,
            step=100.0,
            format="%.0f",
        )
        cap_displacement = col4.checkbox("Mất việc ≤ 5% tổng lao động", value=False)
        col4.metric("Số ngành", f"{len(params)}")
        st.caption(f"Dữ liệu tham số: `{data_path}`")

        st.markdown("##### Bảng 1. Tham số gốc 8 ngành từ CSV")
        st.dataframe(
            format_param_table(params).style.format(
                {
                    "Lao động (triệu)": "{:,.2f}",
                    "Risk (%)": "{:,.0f}",
                    "a₁ (việc/tỷ)": "{:,.1f}",
                    "a₂ (việc/tỷ)": "{:,.1f}",
                    "b₁ (việc/tỷ)": "{:,.1f}",
                    "c₁ (việc/tỷ)": "{:,.1f}",
                    "d₁ (việc/tỷ)": "{:,.1f}",
                }
            ),
            width="stretch",
            hide_index=True,
        )

        st.markdown("##### Bảng 2. Ràng buộc suy ra theo ngành")
        st.dataframe(
            format_constraint_table(params).style.format(
                {
                    "Mất việc / 1 tỷ AI": "{:,.3f}",
                    "Tỷ lệ đào tạo/AI bắt buộc": "{:,.3f}",
                    "Net AI trước đào tạo": "{:,.3f}",
                    "Net gói AI + đào tạo": "{:,.3f}",
                }
            ),
            width="stretch",
            hide_index=True,
        )
        st.info(
            "Ràng buộc đào tạo đã được khóa hai chiều: "
            "`c1 * risk * xAI <= d1 * xH` và `d1 * xH <= c1 * risk * xAI`. "
            "Vì vậy đào tạo đủ hấp thụ lao động mất việc nhưng không bị dồn ngân sách vượt nhu cầu."
        )

    alloc, status = solve_netjob(params, total_budget, min_invest_enabled, min_invest_value, cap_displacement)
    if status != "Optimal":
        st.error(f"Không tìm được nghiệm tối ưu. Trạng thái solver: {status}")
        if min_invest_enabled:
            st.warning(
                f"Ngân sách tối thiểu đang yêu cầu là {len(params) * min_invest_value:,.0f} tỷ VND "
                f"cho {len(params)} ngành. Hãy tăng ngân sách hoặc giảm mức tối thiểu/ngành."
            )
        return

    total_net = alloc["NetJob (Ròng)"].sum()
    used_budget = alloc["Tổng đầu tư (tỷ VND)"].sum()
    total_displaced = alloc["Mất việc (Tự động hóa)"].sum()
    selected_row = alloc.loc[alloc["Ngành"] == selected_sector].iloc[0]

    st.success(
        f"Tối ưu hoàn tất: dùng {used_budget:,.0f}/{total_budget:,.0f} tỷ VND, "
        f"tổng việc làm ròng đạt {total_net:,.0f} người."
    )

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Tổng NetJob", f"{total_net:,.0f}")
    kpi2.metric("Ngân sách đã dùng", f"{used_budget:,.0f} tỷ")
    kpi3.metric("Tổng mất việc", f"{total_displaced:,.0f}")
    kpi4.metric(f"NetJob {selected_sector}", f"{selected_row['NetJob (Ròng)']:,.0f}")

    with st.container(border=True):
        st.markdown("#### Tác động Việc làm Tịnh (NetJob)")
        chart_col, detail_col = st.columns([1.55, 1])
        with chart_col:
            st.plotly_chart(make_netjob_figure(alloc, selected_sector), width="stretch")
        with detail_col:
            st.plotly_chart(make_sector_waterfall(alloc, selected_sector), width="stretch")

    with st.container(border=True):
        st.markdown("#### Câu 9.4.1: Phân bổ tối ưu & Tác động Việc làm")
        display_alloc = alloc[
            [
                "Mã ngành",
                "Ngành",
                "Đầu tư AI (tỷ VND)",
                "Đào tạo lại (tỷ VND)",
                "Tổng đầu tư (tỷ VND)",
                "Việc mới (AI)",
                "Việc nâng cấp (Đào tạo)",
                "Mất việc (Tự động hóa)",
                "Năng lực đào tạo (người)",
                "NetJob (Ròng)",
            ]
        ]
        st.dataframe(
            display_alloc.style.format(
                {
                    "Đầu tư AI (tỷ VND)": "{:,.0f}",
                    "Đào tạo lại (tỷ VND)": "{:,.0f}",
                    "Tổng đầu tư (tỷ VND)": "{:,.0f}",
                    "Việc mới (AI)": "{:,.0f}",
                    "Việc nâng cấp (Đào tạo)": "{:,.0f}",
                    "Mất việc (Tự động hóa)": "{:,.0f}",
                    "Năng lực đào tạo (người)": "{:,.0f}",
                    "NetJob (Ròng)": "{:,.0f}",
                }
            )
            .apply(highlight_selected, selected_sector=selected_sector, axis=1)
            .background_gradient(subset=["NetJob (Ròng)"], cmap="YlGn"),
            width="stretch",
            hide_index=True,
        )
        residual = (alloc["Năng lực đào tạo (người)"] - alloc["Mất việc (Tự động hóa)"]).abs().max()
        st.info(
            f"Kiểm tra 9.4.1: nghiệm đã loại lỗi dồn 100% vào đào tạo. "
            f"Sai lệch lớn nhất giữa năng lực đào tạo và số mất việc chỉ khoảng {residual:,.4f} người do làm tròn solver."
        )

    with st.container(border=True):
        st.markdown("#### Câu 9.4.2: Ngưỡng đào tạo tối thiểu cho ngành CN chế biến chế tạo")
        manufacturing = params.loc[params["sector"] == "CN chế biến chế tạo"].iloc[0]
        manufacturing_result = alloc.loc[alloc["Ngành"] == "CN chế biến chế tạo"].iloc[0]
        required_ratio = manufacturing["retrain_ratio"]
        required_training = required_ratio * manufacturing_result["Đầu tư AI (tỷ VND)"]
        st.info(
            f"Với ngành **CN chế biến chế tạo**, mỗi 1 tỷ VND đầu tư AI cần đi kèm ít nhất "
            f"**{required_ratio:.3f} tỷ VND đào tạo lại**. "
            f"Nghiệm hiện tại: AI = **{manufacturing_result['Đầu tư AI (tỷ VND)']:,.0f} tỷ**, "
            f"đào tạo = **{manufacturing_result['Đào tạo lại (tỷ VND)']:,.0f} tỷ**, "
            f"mức yêu cầu = **{required_training:,.0f} tỷ**."
        )

    with st.container(border=True):
        st.markdown("#### Câu 9.4.3: Luồng Dịch chuyển Lao động Nhóm Dễ tổn thương (Ngành 1, 3, 4)")
        default_vulnerable = params.loc[params["sector_id"].isin([1, 3, 4]), "sector"].tolist()
        flow_sectors = st.multiselect(
            "Chọn ngành hiển thị trong Sankey",
            sector_options,
            default=default_vulnerable,
            key="bai9_flow_sectors",
        )
        if not flow_sectors:
            st.warning("Hãy chọn ít nhất một ngành để vẽ luồng dịch chuyển.")
        else:
            flow_df = alloc.set_index("Ngành").loc[flow_sectors].reset_index()
            if (
                flow_df[["Việc mới (AI)", "Việc nâng cấp (Đào tạo)", "Mất việc (Tự động hóa)"]]
                .to_numpy()
                .sum()
                <= 0
            ):
                st.info("Nhóm đang chọn chưa có luồng việc làm để hiển thị với cấu hình hiện tại.")
            else:
                st.plotly_chart(make_sankey_figure(flow_df), width="stretch")
                st.dataframe(
                    flow_df[
                        [
                            "Ngành",
                            "Đầu tư AI (tỷ VND)",
                            "Đào tạo lại (tỷ VND)",
                            "Việc mới (AI)",
                            "Việc nâng cấp (Đào tạo)",
                            "Mất việc (Tự động hóa)",
                            "NetJob (Ròng)",
                        ]
                    ]
                    .style.format(
                        {
                            "Đầu tư AI (tỷ VND)": "{:,.0f}",
                            "Đào tạo lại (tỷ VND)": "{:,.0f}",
                            "Việc mới (AI)": "{:,.0f}",
                            "Việc nâng cấp (Đào tạo)": "{:,.0f}",
                            "Mất việc (Tự động hóa)": "{:,.0f}",
                            "NetJob (Ròng)": "{:,.0f}",
                        }
                    )
                    .apply(highlight_selected, selected_sector=selected_sector, axis=1),
                    width="stretch",
                    hide_index=True,
                )
                st.success(
                    "Kết luận 9.4.3: Sankey đang lấy trực tiếp từ nghiệm LP. "
                    "Luồng xanh là việc mới, luồng cam là việc nâng cấp, luồng đỏ là lao động bị thay thế."
                )

    if cap_displacement:
        cap_value = 0.05 * params["labor"].sum()
        st.success(
            f"Kết luận 9.4.4: ràng buộc mất việc ≤ 5% tổng lao động đang bật "
            f"({total_displaced:,.0f}/{cap_value:,.0f} người)."
        )

    with st.container(border=True):
        st.markdown("### Câu 9.5: Thảo luận Chính sách")
        top_training = alloc.loc[alloc["Đào tạo lại (tỷ VND)"].idxmax()]
        top_net = alloc.loc[alloc["NetJob (Ròng)"].idxmax()]
        finance = alloc.loc[alloc["Ngành"] == "Tài chính-Ngân hàng"].iloc[0]
        agriculture = alloc.loc[alloc["Ngành"] == "Nông-Lâm-Thủy sản"].iloc[0]

        st.markdown("#### a) Ngành nào cần đầu tư đào tạo lại nhiều nhất?")
        st.info(
            f"Ngành **{top_training['Ngành']}** nhận đào tạo lại cao nhất "
            f"(**{top_training['Đào tạo lại (tỷ VND)']:,.0f} tỷ VND**) trong nghiệm hiện tại. "
            f"Đây cũng là ngành cần theo dõi kỹ vì tác động ròng đạt **{top_training['NetJob (Ròng)']:,.0f}** việc làm."
        )

        st.markdown("#### b) Chiến lược cho ngành Tài chính-Ngân hàng")
        st.warning(
            f"Ngành Tài chính-Ngân hàng có rủi ro tự động hóa cao, nhưng nghiệm vẫn chỉ phân bổ "
            f"AI = **{finance['Đầu tư AI (tỷ VND)']:,.0f} tỷ** và đào tạo = "
            f"**{finance['Đào tạo lại (tỷ VND)']:,.0f} tỷ** theo đúng nhu cầu hấp thụ lao động bị dịch chuyển."
        )

        st.markdown("#### c) Có nên đầu tư AI vào Nông-Lâm-Thủy sản?")
        st.success(
            f"Nông-Lâm-Thủy sản đang nhận AI = **{agriculture['Đầu tư AI (tỷ VND)']:,.0f} tỷ** và "
            f"đào tạo = **{agriculture['Đào tạo lại (tỷ VND)']:,.0f} tỷ**. "
            "Với hệ số tạo việc làm AI thấp và quy mô lao động lớn, chiến lược hợp lý là giữ mức đầu tư có kiểm soát."
        )

        st.markdown("#### d) Ràng buộc đảm bảo an sinh xã hội")
        st.info(
            "Cặp ràng buộc đào tạo hai chiều là điểm chốt của mô hình: đào tạo phải đủ để hấp thụ số mất việc, "
            "nhưng không được vượt nhu cầu rồi biến thành kênh nhận ngân sách độc lập. "
            f"Ngành tạo NetJob lớn nhất hiện là **{top_net['Ngành']}** với **{top_net['NetJob (Ròng)']:,.0f}** việc làm ròng."
        )


