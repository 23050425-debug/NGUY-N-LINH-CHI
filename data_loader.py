from pathlib import Path

import pandas as pd
import streamlit as st

from app_config import DATA_FILES


PROJECT_ROOT = Path(__file__).resolve().parent


def _candidate_dirs():
    cwd = Path.cwd()
    return (
        PROJECT_ROOT / "data",
        PROJECT_ROOT,
        cwd / "data",
        cwd,
    )


def find_data_file(filename: str) -> Path:
    for directory in _candidate_dirs():
        candidate = directory / filename
        if candidate.exists():
            return candidate
    searched = ", ".join(str(path) for path in _candidate_dirs())
    raise FileNotFoundError(f"Không tìm thấy file dữ liệu '{filename}'. Đã tìm trong: {searched}")


def _read_csv(filename: str) -> pd.DataFrame:
    return pd.read_csv(find_data_file(filename), encoding="utf-8-sig")


@st.cache_data(show_spinner=False)
def load_data():
    macro_df = _read_csv(DATA_FILES["macro"]).sort_values("year").reset_index(drop=True)
    sectors_df = _read_csv(DATA_FILES["sectors"])
    regions_df = _read_csv(DATA_FILES["regions"])
    return macro_df, sectors_df, regions_df


def data_summary(macro_df: pd.DataFrame, sectors_df: pd.DataFrame, regions_df: pd.DataFrame) -> dict:
    return {
        "years": f"{int(macro_df['year'].min())}-{int(macro_df['year'].max())}",
        "sectors": len(sectors_df),
        "regions": len(regions_df),
        "latest_year": int(macro_df["year"].max()),
    }


