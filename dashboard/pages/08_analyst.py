"""Страница дашборда: Аналитик.

Агентный чат по данным LSI: модель сама читает витрину через инструменты, помнит
предыдущие ходы диалога и ссылается на источники чисел.
Rule-based fallback работает без LLM API — dashboard остаётся рабочим всегда.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st

from backend.src.services import analyst_agent
from dashboard.components.agent_charts import render_charts
from backend.src.services.lsi_commentary_service import (
    DEFAULT_THRESHOLD_PROFILE,
    build_rule_based_commentary,
    generate_llm_commentary,
    load_context,
)

st.set_page_config(page_title="Аналитик — LSI", layout="wide")

st.title("🧠 Аналитик")
st.markdown(
    "Чат по данным LSI: аналитик сам читает витрину через инструменты, помнит контекст "
    "диалога и ссылается на источники. Без API работает rule-based режим."
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
    show_trace = st.toggle(
        "Показывать вызовы инструментов",
        value=True,
        help="Видно, какие данные аналитик поднял для ответа",
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

# ---------------------------------------------------------------------------
# Агентный чат
# ---------------------------------------------------------------------------

st.subheader("💬 Диалог по данным")
st.caption(
    "Аналитик помнит контекст беседы — можно уточнять: «а почему?», «сравни с прошлым годом», "
    "«покажи фичи этого модуля»."
)

# api_messages — история в формате OpenAI (включая вызовы инструментов), уходит в модель.
# display_messages — только то, что показываем пользователю.
st.session_state.setdefault("analyst_api_messages", [])
st.session_state.setdefault("analyst_display_messages", [])

for position, entry in enumerate(st.session_state["analyst_display_messages"]):
    with st.chat_message(entry["role"]):
        st.markdown(entry["content"])
        render_charts(entry.get("charts", []), key_prefix=f"hist{position}")
        for record in entry.get("trace", []):
            icon = "✅" if record["ok"] else "⚠️"
            with st.expander(f"{icon} {record['name']}", expanded=False):
                st.code(record["arguments"], language="json")
                st.caption(record["summary"])

if question := st.chat_input("Спросите о состоянии ликвидности, периоде, модуле или фичах..."):
    st.session_state["analyst_display_messages"].append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        if not use_agent:
            with st.spinner("Считаю rule-based ответ..."):
                try:
                    reply = build_rule_based_commentary(load_context(threshold_profile=active_profile))
                    reply = (
                        "Агентный режим выключен, поэтому это rule-based сводка, "
                        f"а не ответ на вопрос.\n\n{reply}"
                    )
                except Exception as exc:
                    reply = f"Не удалось собрать rule-based сводку: {exc}"
            st.markdown(reply)
            st.session_state["analyst_display_messages"].append(
                {"role": "assistant", "content": reply}
            )
        else:
            with st.spinner("Читаю данные и думаю..."):
                try:
                    result = analyst_agent.run_turn(
                        question,
                        st.session_state["analyst_api_messages"],
                    )
                    reply = result.reply
                    st.session_state["analyst_api_messages"] = result.messages
                    trace = [
                        {
                            "name": record.name,
                            "arguments": str(record.arguments),
                            "ok": record.ok,
                            "summary": record.summary,
                        }
                        for record in result.trace
                    ]
                    iterations = result.iterations
                    truncated = result.truncated_history
                    charts = result.charts
                except analyst_agent.AnalystAgentUnavailable as exc:
                    reply, trace, iterations, truncated, charts = (
                        f"Агент недоступен: {exc}", [], 0, False, []
                    )
                except Exception as exc:
                    reply = (
                        f"Ошибка при обработке вопроса: {exc}\n\n"
                        "Попробуйте переформулировать или обновить страницу."
                    )
                    trace, iterations, truncated, charts = [], 0, False, []

            st.markdown(reply)
            render_charts(charts, key_prefix="live")
            for record in trace:
                icon = "✅" if record["ok"] else "⚠️"
                with st.expander(f"{icon} {record['name']}", expanded=False):
                    st.code(record["arguments"], language="json")
                    st.caption(record["summary"])
            if iterations:
                st.caption(f"Обращений к модели: {iterations}, вызовов инструментов: {len(trace)}")
            if truncated:
                st.caption("⚠️ Ранние ходы диалога вытеснены из контекста по лимиту размера.")

            st.session_state["analyst_display_messages"].append({
                "role": "assistant",
                "content": reply,
                "trace": trace if show_trace else [],
                "charts": charts,
            })

if st.session_state["analyst_display_messages"]:
    if st.button("🗑️ Очистить историю чата"):
        st.session_state["analyst_api_messages"] = []
        st.session_state["analyst_display_messages"] = []
        st.rerun()

st.markdown("---")

# ---------------------------------------------------------------------------
# Подсказки
# ---------------------------------------------------------------------------

with st.expander("📖 Что умеет аналитик и чего не умеет"):
    st.markdown(f"""
**Инструменты, доступные аналитику ({len(analyst_agent.TOOL_SCHEMAS)}):**
- `list_tables` / `describe_table` — структура витрины
- `query_sql` — read-only SELECT по DuckDB (только SELECT/WITH, лимит строк, таймаут)
- `get_lsi_current` — текущий honest-LSI: значения, статусы, вклады, топ-драйверы
- `get_lsi_series` — LSI за период: статистика, пик, распределение по зонам
- `get_features` — фичи модуля M1–M5 со статистикой или сырыми значениями
- `get_data_freshness` — свежесть таблиц с поправкой на график публикации источника
- `plot_series` — график: line, signal (с полосами порога), bar, dual_axis, flag_timeline
- `plot_custom` — произвольная фигура Plotly, когда типового вида не хватает

**Примеры вопросов:**
- Какой сейчас статус и какой модуль даёт основной вклад?
- Сравни март 2022 с текущим состоянием.
- Покажи график LSI Global за последний год и объясни, что изменилось.
- Почему выросла ставка отсечения РЕПО в июле? Покажи динамику.
- Покажи фичи M2 за июль — что менялось?
- Есть ли в данных колонки-константы, которые нельзя трактовать как сигнал?
- Когда LSI Global последний раз был в красной зоне?

**Ограничения:**
- Аналитик читает посчитанный индекс и не пересчитывает LSI сам.
- Он не строит прогнозы и не заявляет причинность — только гипотезы и способ их проверки.
- LSI Local и Global считаются на разных окнах и не смешиваются.
- Дата данных может отличаться от сегодняшней календарной даты.
- Предел {analyst_agent.MAX_ITERATIONS} обращений к инструментам на вопрос: сложный
  запрос лучше разбить на несколько.
- Каждый вопрос — это несколько платных запросов к модели.
""")
