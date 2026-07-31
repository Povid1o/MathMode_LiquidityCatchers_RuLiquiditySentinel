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
    /* Отступ сверху с запасом под шапку разделов */
    .block-container {
        padding-top: 3.2rem !important;
    }
    /* Шапка: название слева, разделы справа */
    .app-brand {
        font-size: 1.1rem;
        font-weight: 600;
        letter-spacing: 0.01em;
        line-height: 2.4rem;
        white-space: nowrap;
        color: #d8dce3;
    }
    .app-header-rule {
        border-bottom: 1px solid rgba(250, 250, 250, 0.14);
        margin: 0.35rem 0 1.1rem 0;
    }
    /* Активный раздел в шапке подсвечен, а не выглядит выключенным */
    [data-testid="stPageLink"] a[disabled],
    [data-testid="stPageLink"] a[aria-disabled="true"] {
        opacity: 1 !important;
        background-color: rgba(31, 119, 180, 0.18);
        border-radius: 0.5rem;
        color: #ffffff !important;
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
signals = st.Page(str(pages_dir / "00_combined_signals.py"), title="Сводные сигналы", icon="📡")
m1 = st.Page(str(pages_dir / "02_m1_reserves.py"), title="M1 — Резервы", icon="🏦")
m2 = st.Page(str(pages_dir / "03_m2_repo.py"), title="M2 — Репо ЦБ", icon="📋")
m3 = st.Page(str(pages_dir / "04_m3_ofz.py"), title="M3 — ОФЗ", icon="📜")
m4 = st.Page(str(pages_dir / "05_m4_tax.py"), title="M4 — Налоги", icon="📅")
m5 = st.Page(str(pages_dir / "06_m5_liquidity.py"), title="M5 — Ликвидность", icon="💧")
quality = st.Page(str(pages_dir / "07_data_quality.py"), title="Качество данных", icon="🔍")
data_update = st.Page(str(pages_dir / "09_data_update.py"), title="Данные", icon="⚙️")
analyst = st.Page(str(pages_dir / "08_analyst.py"), title="Автокомментарий", icon="🧠")
analyst_chat = st.Page(str(pages_dir / "10_analyst_chat.py"), title="Беседы", icon="💬")

# Три самостоятельных раздела. Переключение между ними — в шапке, содержимое
# раздела — в сайдбаре. Штатный st.navigation умеет либо всё в сайдбаре, либо
# всё в шапке, поэтому двухуровневую схему собираем вручную: навигация скрыта,
# а ссылки рисуем сами через st.page_link.
SECTIONS: dict[str, dict] = {
    "Индекс": {
        "icon": "📊",
        "home": overview,
        "groups": {"Главная": [overview, signals], "Модули": [m1, m2, m3, m4, m5]},
    },
    "Аналитика": {
        "icon": "🧠",
        "home": analyst_chat,
        "groups": {"Аналитик": [analyst_chat, analyst]},
    },
    "Инструменты": {
        "icon": "🛠️",
        "home": quality,
        "groups": {"Данные": [quality, data_update]},
    },
}

pg = st.navigation(
    {name: [p for group in cfg["groups"].values() for p in group]
     for name, cfg in SECTIONS.items()},
    position="hidden",
)


def _current_section() -> str:
    """Определяет раздел по активной странице."""
    for name, cfg in SECTIONS.items():
        for group in cfg["groups"].values():
            if any(page.url_path == pg.url_path for page in group):
                return name
    return next(iter(SECTIONS))


active_section = _current_section()

# --- шапка: переключение разделов ---
header = st.columns([4, 1.3, 1.3, 1.3])
with header[0]:
    st.markdown(
        '<div class="app-brand">RU Liquidity Sentinel</div>',
        unsafe_allow_html=True,
    )
for column, (name, cfg) in zip(header[1:], SECTIONS.items()):
    with column:
        st.page_link(
            cfg["home"],
            label=name,
            icon=cfg["icon"],
            use_container_width=True,
            disabled=(name == active_section),
        )
st.markdown('<div class="app-header-rule"></div>', unsafe_allow_html=True)

# --- сайдбар: страницы текущего раздела ---
with st.sidebar:
    st.markdown(f"### {SECTIONS[active_section]['icon']} {active_section}")
    for group_name, group_pages in SECTIONS[active_section]["groups"].items():
        st.caption(group_name)
        for page in group_pages:
            st.page_link(page, label=page.title, icon=page.icon, use_container_width=True)

    st.markdown(
        '<div class="sidebar-footer">'
        "<strong>PSB Case — Liquidity Stress Monitor</strong><br>"
        "LSI Local/Global. Данные: ЦБ РФ, Минфин, ФНС, Росказна."
        "</div>",
        unsafe_allow_html=True,
    )

pg.run()
