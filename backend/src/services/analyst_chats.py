"""analyst_chats — хранилище бесед с AI-аналитиком.

Беседы лежат на диске, а не в st.session_state: Streamlit теряет состояние при
перезапуске процесса и при обрыве сессии, а разбор эпизода ликвидности — это не
то, что хочется набирать заново. Один файл на беседу: они пишутся по одной,
конфликтов на запись нет, а повреждённый файл не утащит за собой остальные.

Хранится две ветки истории:
- api_messages    — формат OpenAI, включая вызовы инструментов; уходит в модель;
- display_messages — то, что видит пользователь: текст, трасса, спеки графиков.

Их специально держим раздельно. В api_messages лежат служебные tool-сообщения,
которые пользователю показывать незачем, а в display_messages — данные графиков,
которые незачем отправлять модели.
"""
from __future__ import annotations

import json
import re
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CHATS_DIR = PROJECT_ROOT / "data" / "analyst_chats"

# Потолок на размер файла беседы. Спеки графиков содержат точки рядов, и очень
# длинная беседа с десятком графиков может разрастись; при превышении срезаем
# данные графиков у самых старых ответов, сам текст остаётся.
MAX_CHAT_FILE_BYTES = 5 * 1024 * 1024

DEFAULT_TITLE = "Новая беседа"
TITLE_MAX_CHARS = 60


@dataclass
class Chat:
    """Одна беседа с аналитиком."""

    chat_id: str
    title: str = DEFAULT_TITLE
    created_at: str = ""
    updated_at: str = ""
    api_messages: list[dict[str, Any]] = field(default_factory=list)
    display_messages: list[dict[str, Any]] = field(default_factory=list)

    @property
    def message_count(self) -> int:
        """Число реплик, видимых пользователю."""
        return len(self.display_messages)

    @property
    def is_empty(self) -> bool:
        return not self.display_messages


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _chat_path(chat_id: str) -> Path:
    """Путь к файлу беседы. Идентификатор проверяем, чтобы не выйти из каталога."""
    if not re.fullmatch(r"[0-9a-f]{8,32}", chat_id):
        raise ValueError(f"Некорректный идентификатор беседы: {chat_id!r}")
    return CHATS_DIR / f"{chat_id}.json"


def make_title(question: str) -> str:
    """Собирает заголовок беседы из первого вопроса."""
    text = re.sub(r"\s+", " ", question).strip()
    if not text:
        return DEFAULT_TITLE
    if len(text) <= TITLE_MAX_CHARS:
        return text
    return text[:TITLE_MAX_CHARS].rstrip() + "…"


def create_chat() -> Chat:
    """Создаёт новую пустую беседу (на диск попадёт при первом сохранении)."""
    return Chat(chat_id=uuid.uuid4().hex[:16], created_at=_now(), updated_at=_now())


def save_chat(chat: Chat) -> None:
    """Записывает беседу на диск атомарно."""
    CHATS_DIR.mkdir(parents=True, exist_ok=True)
    chat.updated_at = _now()

    payload = asdict(chat)
    raw = json.dumps(payload, ensure_ascii=False, indent=1, default=str)

    if len(raw.encode("utf-8")) > MAX_CHAT_FILE_BYTES:
        payload = _drop_oldest_chart_data(payload)
        raw = json.dumps(payload, ensure_ascii=False, indent=1, default=str)

    path = _chat_path(chat.chat_id)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(raw, encoding="utf-8")
    temporary.replace(path)


def _drop_oldest_chart_data(payload: dict[str, Any]) -> dict[str, Any]:
    """Убирает точки графиков у старых ответов, оставляя текст и заголовок.

    Беседа не должна упираться в размер файла из-за визуализаций: текст разбора
    ценнее, чем возможность перерисовать график двухнедельной давности.
    """
    messages = payload.get("display_messages", [])
    for message in messages[:-6]:
        for chart in message.get("charts", []) or []:
            if chart.get("data"):
                chart["data"] = []
                chart["data_dropped"] = True
            if chart.get("figure"):
                chart["figure"] = {}
                chart["data_dropped"] = True
    return payload


def load_chat(chat_id: str) -> Chat | None:
    """Читает беседу с диска. Повреждённый файл не роняет страницу."""
    path = _chat_path(chat_id)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None

    return Chat(
        chat_id=payload.get("chat_id", chat_id),
        title=payload.get("title", DEFAULT_TITLE),
        created_at=payload.get("created_at", ""),
        updated_at=payload.get("updated_at", ""),
        api_messages=payload.get("api_messages", []),
        display_messages=payload.get("display_messages", []),
    )


def list_chats() -> list[dict[str, Any]]:
    """Список бесед, свежие сверху. Читает только заголовки, не всю историю."""
    if not CHATS_DIR.exists():
        return []

    summaries: list[dict[str, Any]] = []
    for path in CHATS_DIR.glob("*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        summaries.append({
            "chat_id": payload.get("chat_id", path.stem),
            "title": payload.get("title", DEFAULT_TITLE),
            "created_at": payload.get("created_at", ""),
            "updated_at": payload.get("updated_at", ""),
            "message_count": len(payload.get("display_messages", [])),
        })

    summaries.sort(key=lambda item: item.get("updated_at", ""), reverse=True)
    return summaries


def delete_chat(chat_id: str) -> bool:
    """Удаляет беседу с диска."""
    path = _chat_path(chat_id)
    if not path.exists():
        return False
    path.unlink()
    return True


def rename_chat(chat_id: str, title: str) -> bool:
    """Меняет заголовок беседы."""
    chat = load_chat(chat_id)
    if chat is None:
        return False
    chat.title = make_title(title)
    save_chat(chat)
    return True


def history_size(chat: Chat) -> int:
    """Объём истории в символах — для показа заполненности контекста."""
    return sum(
        len(json.dumps(message, ensure_ascii=False, default=str))
        for message in chat.api_messages
    )
