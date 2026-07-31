"""Страница дашборда: Автокомментарий по LSI.

Быстрая сводка по текущему состоянию индекса одной кнопкой, без диалога.
Rule-based режим работает без LLM API — dashboard остаётся рабочим всегда.
Разбор периодов, модулей и отдельных признаков живёт на странице «Беседы».
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st

from backend.src.services import analyst_agent
from backend.src.services.lsi_commentary_service import (
    DEFAULT_THRESHOLD_PROFILE,
    build_rule_based_commentary,
    generate_llm_commentary,
    load_context,
)

st.set_page_config(page_title="Автокомментарий — LSI", layout="wide")

st.title("🧠 Автокомментарий")
st.markdown(
    "Сводка по текущему состоянию LSI одной кнопкой. Без API используется "
    "rule-based режим."
)

# ---------------------------------------------------------------------------
# Активный профиль порогов и доступность агента
# ---------------------------------------------------------------------------

active_profile: str = st.session_state.get("lsi_threshold_profile", DEFAULT_THRESHOLD_PROFILE)
agent_available = analyst_agent.is_available()

with st.sidebar:
    st.markdown("### Настройки аналитика")
    use_agent = st.toggle(
        "Агентный режим (инструменты + память)",
        value=agent_available,
        disabled=not agent_available,
        help="Требует OPENAI_API_KEY. Модель сама читает данные через инструменты.",
    )
    if agent_available:
        st.caption(f"Модель: **{os.environ.get('OPENAI_MODEL', 'gpt-4o-mini')}**")
    st.caption(f"Пороговый профиль: **{active_profile}**")
    st.caption("Сменить профиль можно на странице «Обзор системы».")

if not agent_available:
    st.warning(
        "⚠️ Агентный режим недоступен: не задан `OPENAI_API_KEY` или не установлен пакет "
        "`openai`. Работает rule-based комментарий."
    )
elif use_agent:
    _endpoint = os.environ.get("LLM_BASE_URL", "").strip() or "api.openai.com"
    st.success(
        f"✅ Агент подключён — `{os.environ.get('OPENAI_MODEL')}` через `{_endpoint}`, "
        f"{len(analyst_agent.TOOL_SCHEMAS)} инструментов"
    )
else:
    st.info("ℹ️ Агентный режим выключен: используется rule-based комментарий.")

st.markdown("---")

# ---------------------------------------------------------------------------
# Автокомментарий
# ---------------------------------------------------------------------------

st.subheader("📋 Автокомментарий")


@st.cache_data(ttl=300, show_spinner=False)
def _get_context(profile: str) -> dict:
    return load_context(threshold_profile=profile)


try:
    ctx = _get_context(active_profile)
    context_ok = True
except Exception as exc:
    st.error(f"Ошибка загрузки данных: {exc}")
    context_ok = False

if context_ok:
    cols = st.columns(3)
    with cols[0]:
        st.metric("Дата данных", ctx.get("data_date", "н/д"))

    _color_map = {"ЗЕЛЕНЫЙ": "normal", "ЖЕЛТЫЙ": "off", "КРАСНЫЙ": "inverse"}
    for column, key, label in (
        (cols[1], "lsi_local", "LSI Local"),
        (cols[2], "lsi_global", "LSI Global"),
    ):
        with column:
            value = ctx.get(key)
            if value is None:
                st.metric(label, "н/д")
                continue
            status = ctx.get("local_status" if key == "lsi_local" else "global_status", "")
            delta_color = next((v for k, v in _color_map.items() if k in status), "off")
            st.metric(label, f"{value:.2f}", delta=status, delta_color=delta_color)

    if st.button("🔄 Сгенерировать автокомментарий", type="primary"):
        with st.spinner("Генерация комментария..."):
            if use_agent:
                commentary = generate_llm_commentary(ctx)
                mode = "LLM"
            else:
                commentary = build_rule_based_commentary(ctx)
                mode = "Rule-based"
        st.session_state["last_commentary"] = commentary
        st.session_state["last_commentary_mode"] = mode

    if "last_commentary" in st.session_state:
        st.caption(f"Режим: {st.session_state.get('last_commentary_mode', 'Rule-based')}")
        st.text_area(
            "Комментарий",
            value=st.session_state["last_commentary"],
            height=320,
            label_visibility="collapsed",
        )
    else:
        st.caption("Нажмите кнопку выше, чтобы сгенерировать комментарий.")
else:
    st.warning("Не удалось загрузить контекст. Проверьте файлы в data/processed/")

st.markdown("---")

st.subheader("💬 Диалог по данным")
st.info(
    "Чат с аналитиком переехал на страницу **«Беседы»** — там можно вести несколько "
    "разговоров, возвращаться к прежним и держать контекст в рамках одной темы. "
    "Эта страница осталась для быстрого автокомментария по текущему состоянию.",
    icon="💬",
)

with st.expander("Ограничения автокомментария"):
    st.markdown("""
- Комментарий строится по последней дате данных, без вопросов и уточнений.
- LSI — модельный индикатор, не прогноз; финальное суждение за аналитиком.
- LSI Local и Global считаются на разных окнах и не смешиваются.
- Дата данных может отличаться от сегодняшней календарной даты.
- Для разбора периода, модуля или конкретной фичи используйте «Беседы».
""")
