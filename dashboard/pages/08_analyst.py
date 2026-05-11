"""Страница дашборда: Аналитик.

Автокомментарий и вопрос-ответ по данным LSI.
Rule-based fallback работает без LLM API.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st

from backend.src.services.lsi_thresholds import DEFAULT_THRESHOLD_PROFILE
from backend.src.services.lsi_commentary_service import (
    load_context,
    build_rule_based_commentary,
    generate_llm_commentary,
    answer_question,
    _is_llm_available,
)

st.set_page_config(page_title="Аналитик — LSI", layout="wide")

st.title("🧠 Аналитик")
st.markdown(
    "Автокомментарий и вопрос-ответ по данным LSI. "
    "Без API используется rule-based fallback — dashboard работает в любом случае."
)

# ---------------------------------------------------------------------------
# Активный профиль порогов
# ---------------------------------------------------------------------------

active_profile: str = st.session_state.get("lsi_threshold_profile", DEFAULT_THRESHOLD_PROFILE)

with st.sidebar:
    st.markdown("### Настройки аналитика")
    use_llm = st.toggle(
        "Использовать LLM API, если доступен",
        value=False,
        help="Требует OPENAI_API_KEY в переменных среды",
    )
    st.caption(f"Пороговый профиль: **{active_profile}**")
    st.caption("Сменить профиль можно на странице «Обзор системы».")

llm_available = _is_llm_available()

if use_llm and not llm_available:
    st.warning(
        "⚠️ LLM API запрошен, но OPENAI_API_KEY не задан. "
        "Будет использован rule-based комментарий. "
        "Для подключения LLM: `export OPENAI_API_KEY=sk-...`"
    )
elif use_llm and llm_available:
    st.success("✅ LLM API подключён (OPENAI_API_KEY найден)")
else:
    st.info("ℹ️ Режим rule-based: LLM API не используется.")

st.markdown("---")

# ---------------------------------------------------------------------------
# Загрузка контекста
# ---------------------------------------------------------------------------

@st.cache_data(ttl=300, show_spinner=False)
def _get_context(profile: str) -> dict:
    return load_context(threshold_profile=profile)


with st.spinner("Загрузка данных LSI..."):
    try:
        ctx = _get_context(active_profile)
        context_ok = True
    except Exception as exc:
        st.error(f"Ошибка загрузки данных: {exc}")
        context_ok = False

# ---------------------------------------------------------------------------
# Блок автокомментария
# ---------------------------------------------------------------------------

st.subheader("📋 Автокомментарий")

if context_ok:
    data_date = ctx.get("data_date", "н/д")
    lsi_local = ctx.get("lsi_local")
    lsi_global = ctx.get("lsi_global")

    # KPI-плашки
    cols = st.columns(3)
    with cols[0]:
        st.metric("Дата данных", data_date)
    with cols[1]:
        if lsi_local is not None:
            local_status = ctx.get("local_status", "")
            color_map = {
                "ЗЕЛЕНЫЙ": "normal",
                "ЖЕЛТЫЙ": "off",
                "КРАСНЫЙ": "inverse",
            }
            delta_color = next(
                (v for k, v in color_map.items() if k in local_status), "off"
            )
            st.metric("LSI Local", f"{lsi_local:.2f}", delta=local_status, delta_color=delta_color)
        else:
            st.metric("LSI Local", "н/д")
    with cols[2]:
        if lsi_global is not None:
            global_status = ctx.get("global_status", "")
            delta_color = next(
                (v for k, v in color_map.items() if k in global_status), "off"
            )
            st.metric("LSI Global", f"{lsi_global:.2f}", delta=global_status, delta_color=delta_color)
        else:
            st.metric("LSI Global", "н/д")

    st.markdown("")

    if st.button("🔄 Сгенерировать автокомментарий", type="primary"):
        with st.spinner("Генерация комментария..."):
            if use_llm:
                commentary = generate_llm_commentary(ctx)
            else:
                commentary = build_rule_based_commentary(ctx)
        st.session_state["last_commentary"] = commentary
        st.session_state["last_commentary_mode"] = "LLM" if (use_llm and llm_available) else "Rule-based"

    if "last_commentary" in st.session_state:
        mode = st.session_state.get("last_commentary_mode", "Rule-based")
        st.caption(f"Режим: {mode}")
        st.text_area(
            "Комментарий",
            value=st.session_state["last_commentary"],
            height=320,
            label_visibility="collapsed",
        )
    else:
        st.caption("Нажмите кнопку выше, чтобы сгенерировать комментарий.")
else:
    st.warning("Не удалось загрузить контекст. Проверьте наличие файлов в data/processed/")

st.markdown("---")

# ---------------------------------------------------------------------------
# Чат: вопрос-ответ
# ---------------------------------------------------------------------------

st.subheader("💬 Вопрос-ответ по данным")
st.caption(
    "Задайте вопрос о состоянии ликвидности, конкретном периоде или значении LSI. "
    "Примеры: «Что было в феврале 2022?», «Какой текущий статус?», «Есть ли налоговое давление?»"
)

# инициализируем историю
if "analyst_chat_history" not in st.session_state:
    st.session_state["analyst_chat_history"] = []

# показываем историю
for msg in st.session_state["analyst_chat_history"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# поле ввода
if question := st.chat_input("Введите вопрос по данным LSI..."):
    st.session_state["analyst_chat_history"].append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Анализирую..."):
            try:
                response = answer_question(
                    question,
                    threshold_profile=active_profile,
                    use_llm=use_llm,
                )
            except Exception as exc:
                response = (
                    f"Произошла ошибка при обработке вопроса: {exc}\n\n"
                    "Попробуйте переформулировать вопрос или обновить страницу."
                )
        st.markdown(response)

    st.session_state["analyst_chat_history"].append({"role": "assistant", "content": response})

# кнопка очистки истории
if st.session_state["analyst_chat_history"]:
    if st.button("🗑️ Очистить историю чата"):
        st.session_state["analyst_chat_history"] = []
        st.rerun()

st.markdown("---")

# ---------------------------------------------------------------------------
# Подсказки
# ---------------------------------------------------------------------------

with st.expander("📖 Подсказки и ограничения"):
    st.markdown("""
**Примеры вопросов:**
- Что было в феврале 2022?
- Какой текущий статус ликвидности?
- Есть ли налоговое давление?
- Что показывает LSI Global?
- Был ли стресс в декабре 2014?

**Как включить LLM API:**
```bash
export OPENAI_API_KEY=sk-...
# опционально:
export OPENAI_MODEL=gpt-4o-mini
```

**Ограничения:**
- LLM отвечает только по переданному контексту — без внешних новостей.
- Дата данных может отличаться от сегодняшней даты.
- LSI Local и LSI Global имеют разные обучающие окна.
- LSI — модельный индикатор; финальное суждение остаётся за аналитиком.
- Этот модуль не влияет на расчёт LSI.
""")
