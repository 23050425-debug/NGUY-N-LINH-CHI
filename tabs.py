import streamlit as st

# Import các module tab thực tế
import tab1
import tab2
import tab3
import tab4
import tab5
import tab6
import tab7
import tab8
import tab9
import tab10
import tab11
import tab12

# Dictionary mapping tên tab (không dấu) -> module tương ứng
TAB_MODULES = {
    "Tab 1: Cobb-Douglas": tab1,
    "Tab 2: Ngan sach so": tab2,
    "Tab 3: Uu tien Nganh": tab3,
    "Tab 4: Quy hoach Nganh-Vung": tab4,
    "Tab 5: MIP Quy hoach": tab5,
    "Tab 6: TOPSIS": tab6,
    "Tab 7: Pareto": tab7,
    "Tab 8: Toi uu Dong": tab8,
    "Tab 9: Lao dong & AI": tab9,
    "Tab 10: Quy hoach ngau nhien": tab10,
    "Tab 11: Hoc Tang Cuong": tab11,
    "Tab 12: Tich hop": tab12,
}

def render_tab(selected_tab, macro_df, sectors_df, regions_df):
    """
    Render tab được chọn
    
    Args:
        selected_tab: Tên tab được chọn
        macro_df: DataFrame dữ liệu vĩ mô
        sectors_df: DataFrame dữ liệu ngành
        regions_df: DataFrame dữ liệu vùng
    """
    if selected_tab in TAB_MODULES:
        module = TAB_MODULES[selected_tab]
        # Gọi hàm render() từ module tương ứng
        if hasattr(module, 'render'):
            module.render(macro_df, sectors_df, regions_df)
        else:
            st.error(f"Module {selected_tab} không có hàm render()")
    else:
        st.error(f"Tab '{selected_tab}' không tồn tại")

def get_tab_list():
    """Trả về danh sách tên các tab"""
    return list(TAB_MODULES.keys())
