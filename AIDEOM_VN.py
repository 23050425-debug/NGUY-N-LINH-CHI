import streamlit as st

from app_config import APP_SUBTITLE, APP_TITLE
from data_loader import load_data
from tabs import tab1, tab10, tab11, tab12, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9
from ui_theme import apply_theme, configure_page


# ĐĂNG KÝ TẤT CẢ CÁC TAB TẠI ĐÂY
# Cấu trúc: ("Tên tab hiển thị trên menu", hàm_render_từ_module)
LESSONS = [
<<<<<<< HEAD
    ("Tab 1: Cobb-Douglas", tab1.render),
    ("Tab 2: Ngan sach so", tab2.render),
    ("Tab 3: Uu tien Nganh", tab3.render),
    ("Tab 4: Quy hoach Nganh-Vung", tab4.render),
    ("Tab 5: MIP Quy hoach", tab5.render),
    ("Tab 6: TOPSIS", tab6.render),
    ("Tab 7: Pareto", tab7.render),
    ("Tab 8: Toi uu Dong", tab8.render),
    ("Tab 9: Lao dong & AI", tab9.render),
    ("Tab 10: Quy hoach ngau nhien", tab10.render),
    ("Tab 11: Hoc Tang Cuong", tab11.render),
    ("Tab 12: Tich hop", tab12.render),
=======
>>>>>>> b776083181049d6df8fa0054462b01e4f9e6a7ef
]


def _render_sidebar() -> tuple[str, str]:
    with st.sidebar:
        st.markdown(
            f"""
            <div class="aideom-brand">
                <div class="aideom-brand-title">{APP_TITLE}</div>
                <div class="aideom-brand-subtitle">{APP_SUBTITLE}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        theme_mode = st.radio("Chế độ hiển thị", ["Sáng", "Tối"], horizontal=True, key="theme_mode")
        st.markdown('<div class="aideom-sidebar-divider"></div>', unsafe_allow_html=True)
        page = st.radio(
            "Điều hướng",
            ["Trang chủ"] + [label for label, _ in LESSONS],
            key="page_navigation",
        )
        return theme_mode, page


def _render_homepage():
    st.markdown(
        """
        <section class="aideom-home">
            <div class="aideom-home-panel">
                <div class="aideom-home-copy">
                    <p class="aideom-home-kicker">AIDEOM-VN</p>
                    <h1>AIDEOM_VN DASHBOARD</h1>
                    <p class="aideom-home-subtitle">
                        Dashboard điều hành phát triển kinh tế Việt Nam trong kỷ nguyên AI
                    </p>
                </div>
                <div class="aideom-home-visual" aria-hidden="true">
                    <div class="aideom-screen">
                        <div class="aideom-chart aideom-chart-one"></div>
                        <div class="aideom-chart aideom-chart-two"></div>
                        <div class="aideom-chart aideom-chart-three"></div>
                        <div class="aideom-line"></div>
                        <div class="aideom-line short"></div>
                    </div>
                    <div class="aideom-database">
                        <span></span><span></span><span></span>
                    </div>
                </div>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def main():
    configure_page()
    theme_mode, page = _render_sidebar()
    apply_theme(theme_mode)

    if page == "Trang chủ":
        _render_homepage()
        return

    try:
        macro_df, sectors_df, regions_df = load_data()
    except Exception as exc:
        st.error(f"Lỗi nạp dữ liệu: {exc}")
        st.stop()

    lesson_lookup = dict(LESSONS)
    lesson_lookup[page](macro_df, sectors_df, regions_df)


if __name__ == "__main__":
    main()
