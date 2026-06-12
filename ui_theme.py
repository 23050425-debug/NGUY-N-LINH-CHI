import streamlit as st

from app_config import APP_TITLE


PRIMARY_PURPLE = "#7e22ce"
PURPLE_SHADES = ["#f3e8ff", "#e9d5ff", "#d8b4fe", "#a855f7", "#7e22ce", "#4c1d95"]
TEXT_COLOR = "#4c1d95"
PURPLE_SCALE = [
    [0.0, "#f3e8ff"],
    [0.35, "#d8b4fe"],
    [0.65, "#a855f7"],
    [1.0, "#4c1d95"],
]

THEMES = {
    "Sáng": {
        "bg": "#ffffff",
        "surface": "#ffffff",
        "surface_alt": "#f3e8ff",
        "text": "#241a36",
        "muted": "#64748b",
        "border": "#e9d5ff",
        "primary": "#7e22ce",
        "primary_soft": "#f3e8ff",
        "accent": "#0f766e",
        "warning": "#b45309",
        "shadow": "0 14px 36px rgba(76, 29, 149, 0.09)",
    },
    "Tối": {
        "bg": "#15121d",
        "surface": "#211a2f",
        "surface_alt": "#2b2140",
        "text": "#f8fafc",
        "muted": "#cbd5e1",
        "border": "#5b3f86",
        "primary": "#c084fc",
        "primary_soft": "#33234d",
        "accent": "#2dd4bf",
        "warning": "#fbbf24",
        "shadow": "0 16px 42px rgba(0, 0, 0, 0.26)",
    },
}


def configure_page():
    st.set_page_config(
        page_title=f"{APP_TITLE} | Dashboard điều hành",
        page_icon=":bar_chart:",
        layout="wide",
        initial_sidebar_state="expanded",
    )


def apply_theme(mode: str):
    theme = THEMES.get(mode, THEMES["Sáng"])
    st.markdown(
        f"""
        <style>
            :root {{
                --aideom-bg: {theme["bg"]};
                --aideom-surface: {theme["surface"]};
                --aideom-surface-alt: {theme["surface_alt"]};
                --aideom-text: {theme["text"]};
                --aideom-muted: {theme["muted"]};
                --aideom-border: {theme["border"]};
                --aideom-primary: {theme["primary"]};
                --aideom-primary-soft: {theme["primary_soft"]};
                --aideom-accent: {theme["accent"]};
                --aideom-warning: {theme["warning"]};
                --aideom-shadow: {theme["shadow"]};
            }}

            html, body, [class*="css"], .stApp, .stApp * {{
                letter-spacing: 0 !important;
            }}

            .stApp,
            [data-testid="stAppViewContainer"] {{
                background: var(--aideom-bg);
                color: var(--aideom-text);
            }}

            [data-testid="stHeader"] {{
                background: rgba(0, 0, 0, 0);
            }}

            section[data-testid="stSidebar"] > div {{
                background: linear-gradient(180deg, var(--aideom-surface), var(--aideom-surface-alt));
                border-right: 1px solid var(--aideom-border);
            }}

            h1, h2, h3, h4, h5, h6,
            [data-testid="stMarkdownContainer"] p,
            [data-testid="stMarkdownContainer"] li {{
                color: var(--aideom-text);
            }}

            small, .caption, [data-testid="stCaptionContainer"] {{
                color: var(--aideom-muted) !important;
            }}

            .aideom-brand {{
                padding: 14px 12px 16px;
                border: 1px solid var(--aideom-border);
                border-radius: 8px;
                background: var(--aideom-surface);
                box-shadow: var(--aideom-shadow);
                margin-bottom: 16px;
            }}

            .aideom-brand-title {{
                color: var(--aideom-primary);
                font-size: 1.25rem;
                font-weight: 800;
                letter-spacing: 0;
            }}

            .aideom-brand-subtitle {{
                color: var(--aideom-muted);
                font-size: 0.84rem;
                line-height: 1.35;
                margin-top: 6px;
            }}

            .aideom-sidebar-divider {{
                height: 1px;
                background: var(--aideom-border);
                margin: 16px 0 12px;
            }}

            section[data-testid="stSidebar"] [role="radiogroup"] {{
                gap: 4px;
            }}

            section[data-testid="stSidebar"] [role="radiogroup"] label {{
                border-radius: 8px;
                padding: 8px 10px;
                border-bottom: 3px solid transparent;
            }}

            section[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {{
                background: var(--aideom-primary-soft);
                border-bottom-color: var(--aideom-primary);
                color: var(--aideom-primary);
                font-weight: 800;
            }}

            .aideom-home {{
                min-height: calc(100vh - 96px);
                display: flex;
                align-items: center;
                justify-content: center;
                padding: 18px 10px 42px;
            }}

            .aideom-home-panel {{
                width: min(1120px, 100%);
                min-height: 520px;
                display: grid;
                grid-template-columns: minmax(420px, 1.1fr) minmax(340px, 0.9fr);
                gap: 24px;
                align-items: center;
                padding: clamp(28px, 5vw, 56px);
                border: 1px solid #fecdd3;
                border-radius: 4px;
                background:
                    radial-gradient(circle at 88% 12%, rgba(216, 180, 254, 0.42) 0 18%, transparent 19%),
                    linear-gradient(135deg, #ffffff 0%, #ffffff 58%, #fdf2f8 100%);
                box-shadow: 0 0 0 10px rgba(244, 63, 94, 0.13), 0 18px 42px rgba(126, 34, 206, 0.22);
                overflow: hidden;
                position: relative;
            }}

            .aideom-home-kicker {{
                margin: 0 0 14px 0;
                color: var(--aideom-muted) !important;
                font-size: 0.86rem;
                text-transform: uppercase;
                letter-spacing: 0 !important;
                font-weight: 700;
            }}

            .aideom-home h1 {{
                margin: 0;
                color: #241a36;
                font-size: clamp(1.9rem, 3.2vw, 2.65rem);
                line-height: 1.02;
                font-weight: 800;
                max-width: none;
                white-space: nowrap;
                word-break: keep-all;
            }}

            .aideom-home-subtitle {{
                color: #475569 !important;
                font-size: clamp(1rem, 2vw, 1.3rem);
                line-height: 1.55;
                max-width: none;
                margin: 22px 0 0;
                white-space: nowrap;
            }}

            .aideom-home-visual {{
                min-height: 360px;
                position: relative;
                display: flex;
                align-items: center;
                justify-content: center;
            }}

            .aideom-screen {{
                width: min(430px, 86%);
                aspect-ratio: 1.32;
                border-radius: 8px;
                background: #4c1d95;
                border: 12px solid #7e22ce;
                box-shadow: 0 26px 0 -12px #a855f7, 0 32px 34px rgba(76, 29, 149, 0.28);
                position: relative;
                padding: 32px;
            }}

            .aideom-screen::after {{
                content: "";
                position: absolute;
                left: 38%;
                bottom: -64px;
                width: 24%;
                height: 52px;
                background: linear-gradient(180deg, #7e22ce, #4c1d95);
                border-radius: 0 0 6px 6px;
            }}

            .aideom-chart {{
                position: absolute;
                bottom: 34px;
                width: 42px;
                border-radius: 6px 6px 0 0;
                background: linear-gradient(180deg, #f472b6, #fb923c);
            }}

            .aideom-chart-one {{ height: 112px; left: 42px; }}
            .aideom-chart-two {{ height: 168px; left: 100px; }}
            .aideom-chart-three {{ height: 82px; left: 158px; }}

            .aideom-line {{
                position: absolute;
                right: 42px;
                top: 70px;
                width: 132px;
                height: 10px;
                border-radius: 10px;
                background: #e9d5ff;
                box-shadow: 0 34px 0 #d8b4fe, 0 68px 0 #c084fc;
            }}

            .aideom-line.short {{
                top: 206px;
                width: 88px;
                background: #f9a8d4;
                box-shadow: none;
            }}

            .aideom-database {{
                position: absolute;
                left: 2%;
                bottom: 30px;
                width: 138px;
                padding: 12px;
                border-radius: 6px;
                background: linear-gradient(180deg, #581c87, #4c1d95);
                box-shadow: 0 18px 28px rgba(76, 29, 149, 0.22);
            }}

            .aideom-database span {{
                display: block;
                height: 18px;
                border-radius: 4px;
                background: #f8fafc;
                margin: 8px 0;
                box-shadow: inset 62px 0 0 #fb7185;
            }}

            .aideom-hero {{
                border: 1px solid var(--aideom-border);
                border-radius: 8px;
                background:
                    linear-gradient(135deg, var(--aideom-primary-soft), var(--aideom-surface) 62%),
                    var(--aideom-surface);
                padding: 24px 26px;
                box-shadow: var(--aideom-shadow);
                margin-bottom: 18px;
            }}

            .aideom-hero h1 {{
                margin: 0 0 8px 0;
                color: var(--aideom-primary);
                font-size: 2rem;
                line-height: 1.15;
                letter-spacing: 0;
            }}

            .aideom-hero p {{
                margin: 0;
                color: var(--aideom-text);
                max-width: 980px;
                line-height: 1.55;
            }}

            .aideom-pill {{
                display: inline-flex;
                align-items: center;
                border: 1px solid var(--aideom-border);
                border-radius: 999px;
                padding: 4px 10px;
                color: var(--aideom-primary);
                background: var(--aideom-surface);
                font-size: 0.82rem;
                font-weight: 700;
                margin: 0 8px 8px 0;
            }}

            .decision-card {{
                border-left: 4px solid var(--aideom-primary);
                border-radius: 8px;
                background: var(--aideom-surface);
                border-top: 1px solid var(--aideom-border);
                border-right: 1px solid var(--aideom-border);
                border-bottom: 1px solid var(--aideom-border);
                padding: 14px 16px;
                min-height: 124px;
                box-shadow: var(--aideom-shadow);
            }}

            .decision-card strong {{
                color: var(--aideom-primary);
            }}

            div[data-testid="stMetric"] {{
                background: var(--aideom-surface);
                border: 1px solid var(--aideom-border);
                border-radius: 8px;
                padding: 14px 16px;
                box-shadow: var(--aideom-shadow);
            }}

            div[data-testid="stMetricLabel"] p {{
                color: var(--aideom-muted) !important;
                font-size: 0.9rem;
            }}

            div[data-testid="stMetricValue"] {{
                color: var(--aideom-primary);
            }}

            div[data-testid="stVerticalBlockBorderWrapper"] {{
                border-color: var(--aideom-border) !important;
                background: var(--aideom-surface);
                border-radius: 8px !important;
                box-shadow: var(--aideom-shadow);
            }}

            div[data-testid="stDataFrame"] {{
                border-radius: 8px;
                overflow: hidden;
            }}

            .stTabs [data-baseweb="tab-list"] {{
                gap: 6px;
                border-bottom: 1px solid var(--aideom-border);
            }}

            .stTabs [data-baseweb="tab"] {{
                border-radius: 8px 8px 0 0;
                padding: 8px 12px;
                color: var(--aideom-muted);
            }}

            .stTabs [data-baseweb="tab"][aria-selected="true"] {{
                background: var(--aideom-primary-soft);
                border-bottom: 3px solid var(--aideom-primary);
            }}

            .stTabs [data-baseweb="tab"][aria-selected="true"] p {{
                color: var(--aideom-primary) !important;
                font-weight: 800;
            }}

            .stButton > button,
            div[data-testid="stDownloadButton"] > button {{
                border-radius: 8px;
                border: 1px solid var(--aideom-primary);
                color: var(--aideom-primary);
                background: var(--aideom-surface);
            }}

            div[role="radiogroup"] label {{
                border-radius: 8px;
            }}

            hr {{
                border-color: var(--aideom-border);
            }}

            @media (max-width: 900px) {{
                .aideom-home {{
                    align-items: flex-start;
                }}

                .aideom-home-panel {{
                    grid-template-columns: 1fr;
                    min-height: auto;
                }}

                .aideom-home h1 {{
                    font-size: clamp(1.28rem, 6vw, 2.2rem);
                    white-space: nowrap;
                }}

                .aideom-home-subtitle {{
                    white-space: normal;
                }}

                .aideom-home-visual {{
                    min-height: 300px;
                }}
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )
    return theme






