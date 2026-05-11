import streamlit as st
from pathlib import Path

st.set_page_config(
    page_title="PSB — Мониторинг стресса ликвидности",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
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
    st.markdown("---")
    st.caption(
        "**PSB Case — Liquidity Stress Monitor**  \n"
        "LSI в разработке. Данные: ЦБ РФ, Минфин, ФНС."
    )

pg.run()
