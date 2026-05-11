import streamlit as st
from pathlib import Path

st.set_page_config(
    page_title="PSB — Мониторинг стресса ликвидности",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Убираем лишние отступы сверху и padding sidebar
st.markdown(
    """
    <style>
    /* Убираем верхний отступ в сайдбаре */
    [data-testid="stSidebarContent"] {
        padding-top: 0.5rem !important;
        overflow-x: hidden;
    }
    /* Убираем все горизонтальные разделители в сайдбаре */
    [data-testid="stSidebarContent"] hr,
    [data-testid="stSidebarNav"] + div hr,
    [data-testid="stSidebarContent"] [data-testid="stMarkdownContainer"] hr {
        display: none;
    }
    /* Streamlit nav добавляет border-bottom на обёртку */
    [data-testid="stSidebarNavSeparator"],
    [data-testid="stSidebarContent"] > div > div[style*="border"] {
        display: none !important;
    }
    section[data-testid="stSidebar"] > div:first-child {
        padding-top: 0 !important;
    }
    /* Убираем верхний отступ основного контента */
    .block-container {
        padding-top: 1.5rem !important;
    }
    /* Футер прибит к низу сайдбара */
    .sidebar-footer {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 21rem;
        padding: 0.6rem 1.2rem 0.8rem 1.2rem;
        font-size: 0.75rem;
        color: #888;
        border-top: none;
        background-color: inherit;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

pages_dir = Path(__file__).parent / "pages"

overview = st.Page(str(pages_dir / "01_overview.py"), title="Обзор системы", icon="🏠")
m1 = st.Page(str(pages_dir / "02_m1_reserves.py"), title="M1 — Резервы", icon="🏦")
m2 = st.Page(str(pages_dir / "03_m2_repo.py"), title="M2 — Репо ЦБ", icon="📋")
m3 = st.Page(str(pages_dir / "04_m3_ofz.py"), title="M3 — ОФЗ", icon="📜")
m4 = st.Page(str(pages_dir / "05_m4_tax.py"), title="M4 — Налоги", icon="📅")
m5 = st.Page(str(pages_dir / "06_m5_liquidity.py"), title="M5 — Ликвидность", icon="💧")
quality = st.Page(str(pages_dir / "07_data_quality.py"), title="Качество данных", icon="🔍")

pg = st.navigation(
    {
        "Главная": [overview],
        "Модули": [m1, m2, m3, m4, m5],
        "Инструменты": [quality],
    }
)

with st.sidebar:
    st.markdown(
        '<div class="sidebar-footer">'
        "<strong>PSB Case — Liquidity Stress Monitor</strong><br>"
        "LSI в разработке. Данные: ЦБ РФ, Минфин, ФНС."
        "</div>",
        unsafe_allow_html=True,
    )

pg.run()
