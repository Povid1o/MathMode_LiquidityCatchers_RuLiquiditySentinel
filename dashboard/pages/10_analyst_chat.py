"""Страница дашборда: Беседы с аналитиком.

Полноценный чат с AI-аналитиком: несколько бесед, переключение между ними,
история переживает перезапуск дашборда. Контекст держится в пределах беседы —
разбор одного эпизода не смешивается с разбором другого.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st

from backend.src.services import analyst_agent, analyst_chats
from dashboard.components.agent_charts import render_charts

st.set_page_config(page_title="Беседы с аналитиком", layout="wide")

agent_available = analyst_agent.is_available()


# ---------------------------------------------------------------------------
# Состояние: активная беседа
# ---------------------------------------------------------------------------

def _activate(chat_id: str) -> None:
    st.session_state["active_chat_id"] = chat_id


def _new_chat() -> None:
    chat = analyst_chats.create_chat()
    analyst_chats.save_chat(chat)
    _activate(chat.chat_id)


summaries = analyst_chats.list_chats()

if "active_chat_id" not in st.session_state:
    # Продолжаем самую свежую беседу, иначе заводим первую
    if summaries:
        _activate(summaries[0]["chat_id"])
    else:
        _new_chat()
        summaries = analyst_chats.list_chats()

chat = analyst_chats.load_chat(st.session_state["active_chat_id"])
if chat is None:
    _new_chat()
    summaries = analyst_chats.list_chats()
    chat = analyst_chats.load_chat(st.session_state["active_chat_id"])


# ---------------------------------------------------------------------------
# Сайдбар: список бесед и настройки
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("### Беседы")

    if st.button("➕ Новый чат", use_container_width=True, type="primary"):
        _new_chat()
        st.rerun()

    if summaries:
        st.caption(f"Всего бесед: {len(summaries)}")
        for item in summaries:
            is_active = item["chat_id"] == chat.chat_id
            label = item["title"] if item["message_count"] else "· пустая беседа"
            if st.button(
                ("▸ " if is_active else "") + label,
                key=f"open_{item['chat_id']}",
                use_container_width=True,
                disabled=is_active,
                help=f"Реплик: {item['message_count']}, изменена: {item['updated_at'][:16]}",
            ):
                _activate(item["chat_id"])
                st.rerun()

    st.markdown("---")
    st.markdown("### Текущая беседа")

    new_title = st.text_input("Название", value=chat.title, key="chat_title_input")
    columns = st.columns(2)
    if columns[0].button("Переименовать", use_container_width=True):
        analyst_chats.rename_chat(chat.chat_id, new_title)
        st.rerun()
    if columns[1].button("Удалить", use_container_width=True):
        analyst_chats.delete_chat(chat.chat_id)
        st.session_state.pop("active_chat_id", None)
        st.rerun()

    # Заполненность контекста: показываем, сколько истории уйдёт в модель
    used = analyst_chats.history_size(chat)
    budget = analyst_agent.HISTORY_CHAR_BUDGET
    st.progress(min(used / budget, 1.0), text=f"Контекст беседы: {used:,} / {budget:,} симв.")
    if used > budget:
        st.caption("⚠️ Самые ранние реплики не поместятся и будут отброшены.")

    if agent_available:
        st.markdown("---")
        st.caption(f"Модель: **{os.environ.get('OPENAI_MODEL', 'gpt-4o-mini')}**")


# ---------------------------------------------------------------------------
# Основная область
# ---------------------------------------------------------------------------

st.title("💬 " + (chat.title if not chat.is_empty else "Беседа с аналитиком"))

if not agent_available:
    st.error(
        "Агент недоступен: не задан `OPENAI_API_KEY` или не установлен пакет `openai`. "
        "Автокомментарий по LSI доступен на странице «Аналитик»."
    )
    st.stop()

if chat.is_empty:
    st.caption(
        "Аналитик сам читает витрину через инструменты, строит графики и помнит контекст "
        "беседы. Спросите о состоянии ликвидности, периоде, модуле или конкретной фиче."
    )
    with st.expander("С чего начать", expanded=True):
        st.markdown("""
- Какой сейчас статус ликвидности и какой модуль даёт основной вклад?
- Покажи график LSI Global за последний год и объясни, что изменилось.
- Сравни март 2022 с текущим состоянием.
- Что означает признак с наибольшим вкладом и как читать его рост?
- Когда LSI Global последний раз был в красной зоне?
        """)

for position, entry in enumerate(chat.display_messages):
    with st.chat_message(entry["role"]):
        st.markdown(entry["content"])
        render_charts(entry.get("charts", []), key_prefix=f"{chat.chat_id}_{position}")

if question := st.chat_input("Спросите о ликвидности, периоде, модуле или фиче..."):
    chat.display_messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # Первый вопрос задаёт название беседы — иначе список превратится в
    # десяток одинаковых «Новых бесед»
    if chat.title == analyst_chats.DEFAULT_TITLE:
        chat.title = analyst_chats.make_title(question)

    with st.chat_message("assistant"):
        with st.spinner("Читаю данные и думаю..."):
            try:
                result = analyst_agent.run_turn(question, chat.api_messages)
                reply = result.reply
                chat.api_messages = result.messages
                charts, truncated = result.charts, result.truncated_history
            except analyst_agent.AnalystAgentUnavailable as exc:
                reply, charts, truncated = f"Агент недоступен: {exc}", [], False
            except Exception as exc:
                reply = (
                    f"Ошибка при обработке вопроса: {exc}\n\n"
                    "Попробуйте переформулировать или начать новый чат."
                )
                charts, truncated = [], False

        st.markdown(reply)
        render_charts(charts, key_prefix="live")
        if truncated:
            st.caption("⚠️ Ранние реплики беседы вытеснены из контекста по лимиту размера.")

    # Трассу вызовов инструментов не храним и не показываем: она превращала
    # каждый ответ в простыню служебных блоков. Что именно аналитик поднял из
    # данных, видно из самого ответа — он обязан ссылаться на источники.
    chat.display_messages.append({
        "role": "assistant",
        "content": reply,
        "charts": charts,
    })
    analyst_chats.save_chat(chat)
    st.rerun()
