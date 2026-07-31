"""analyst_agent — агентный цикл AI-аналитика LSI.

Модель сама решает, какие инструменты вызвать, получает результаты и продолжает
рассуждение. От прежней схемы отличается тремя вещами:

- история диалога уходит в API, а не только рисуется в интерфейсе, поэтому
  «а почему?» после предыдущего ответа теперь осмысленно;
- вместо заранее собранного текстового контекста модель читает данные сама через
  analyst_tools, поэтому доступны все фичи, а не пять флагов;
- каждое число обязано сопровождаться источником, иначе агент с доступом к SQL
  звучит убедительнее, чем агент без него, не будучи точнее.

Пересчитывать LSI агенту запрещено промптом: он читает посчитанное. Иначе рядом
с официальным индексом появился бы второй, неофициальный.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Callable

PROJECT_ROOT = Path(__file__).resolve().parents[3]

try:
    from dotenv import load_dotenv as _load_dotenv

    _load_dotenv(dotenv_path=PROJECT_ROOT / ".env", override=False)
except ImportError:
    pass

from backend.src.services.analyst_tools import (
    TOOL_IMPLEMENTATIONS,
    TOOL_SCHEMAS,
    ToolError,
)


DEFAULT_MODEL = "gpt-4o-mini"

# Предел обращений к модели на один вопрос. Каждая итерация — отдельный платный
# запрос, поэтому цикл должен иметь потолок, а не надеяться на здравомыслие модели.
MAX_ITERATIONS = 6

# Бюджет истории в символах. Считаем грубо: точный подсчёт токенов требует
# токенизатора провайдера, а ошибка в два раза здесь ничего не меняет.
HISTORY_CHAR_BUDGET = 60_000

# Результат одного инструмента, который не должен вытеснить весь диалог.
MAX_TOOL_RESULT_CHARS = 12_000

# Reasoning-модели этого провайдера тратят часть бюджета вывода на размышление
# до появления текста, поэтому лимит должен быть с запасом: иначе ответ приходит
# пустым, хотя инструменты уже отработали.
MAX_OUTPUT_TOKENS = 8000
TEMPERATURE = 0.2


SYSTEM_PROMPT = """Ты — аналитик рублёвого денежного рынка в системе мониторинга стресса ликвидности (LSI).

ЯЗЫК: отвечай только по-русски. Никакого английского, включая заголовки.

КАК ТЫ РАБОТАЕШЬ
У тебя есть инструменты доступа к витрине данных. Не отвечай по памяти о состоянии
рынка — сначала возьми данные инструментом. Если для ответа нужны числа, которых
ты ещё не получил, вызови инструмент, а не оценивай на глаз.

Порядок, который почти всегда верен:
1. Для вопроса о текущем состоянии — get_lsi_current.
2. Перед выводом о текущем состоянии — get_data_freshness: если таблица модуля в
   статусе stale, её данные не обновились, и это надо оговорить в ответе.
3. Для периода в прошлом — get_lsi_series.
4. Для фич модуля — get_features (сначала без columns, чтобы увидеть перечень).
5. Для всего остального — query_sql. Не знаешь структуру — list_tables и describe_table.
6. Просят показать динамику, ряд, «построй график» — plot_series.
   Просят вклады модулей, «из чего сложился индекс» — plot_contributions.
   Нужного вида нет среди готовых — plot_custom с фигурой Plotly.

НИКАКИХ ПРЕАМБУЛ
Не пиши «сейчас построю», «уточняю», «давай посмотрю». Либо вызывай инструмент
прямо в этом шаге, либо давай готовый ответ. Текст без вызова инструмента
считается финальным ответом и показывается пользователю как есть — объявление
о будущем действии оставит его без ответа и без графика.

ГРАФИКИ
График не заменяет ответ. Построив его, обязательно опиши словами, что на нём
видно: направление, величину изменения, где перелом. Опирайся на series_summary
из результата инструмента — сами точки ряда тебе не возвращаются, и это нормально.
Не строй график там, где хватает одного числа.
Если пользователь попросил и объяснение, и график — сделай оба, в одном ответе.

ДИСЦИПЛИНА ССЫЛОК
Каждое число в ответе сопровождай источником: имя таблицы или инструмента, откуда
оно взято. Формат свободный, например «(honest_lsi_scores)». Если числа нет в
полученных данных — так и скажи, не заполняй пробел оценкой.

МЕТОДИКА — соблюдай строго
- LSI это модельный индикатор, а не прогноз. Не пиши «кризис будет».
- Не заявляй причинность. Связь вкладов и событий — это корреляция и гипотеза,
  формулируй как гипотезу и говори, чем её можно проверить.
- ЖЁЛТАЯ зона это «повышенное внимание», а не «стресс».
- КРАСНАЯ зона это «сигнал стресса, требует подтверждения аналитиком».
- LSI Local и LSI Global считаются на разных окнах и обучены на разных выборках.
  Не смешивай их и не усредняй.
- Не пересчитывай LSI сам. Ты читаешь посчитанные значения, а не производишь их.
- Дата последних данных не обязательно совпадает с сегодняшней календарной датой.
  Всегда указывай дату данных.

ЛОВУШКИ ДАННЫХ
- Основные колонки дат имеют тип TIMESTAMP и сравниваются с литералами DATE
  напрямую. Отдельные вспомогательные колонки *_date могли остаться VARCHAR:
  для них бери date_sql_expression из describe_table, иначе сравнение будет
  лексикографическим и даст неверный результат без сообщения об ошибке.
- Колонка, у которой на периоде одно значение (constant_columns), может быть не
  сигналом, а заполнением пропусков. Не трактуй такую колонку как «стресса нет».
- Если инструмент вернул ошибку, прочитай её и исправь запрос, а не повторяй тот же.

ФОРМА ОТВЕТА
Кратко и структурно. Заголовки и списки уместны. Не пересказывай сырые выгрузки —
делай вывод. Если аналитику стоит что-то проверить руками, скажи что именно.
"""


@dataclass
class ToolCallRecord:
    """Запись о вызове инструмента — для показа трассы в интерфейсе."""

    name: str
    arguments: dict[str, Any]
    ok: bool
    summary: str


@dataclass
class AgentResult:
    """Итог одного хода диалога."""

    reply: str
    messages: list[dict[str, Any]]
    trace: list[ToolCallRecord] = field(default_factory=list)
    charts: list[dict[str, Any]] = field(default_factory=list)
    iterations: int = 0
    truncated_history: bool = False


class AnalystAgentUnavailable(RuntimeError):
    """Агент не сконфигурирован (нет ключа или пакета openai)."""


def is_available() -> bool:
    """Проверяет, что агент можно запустить"""
    if not os.environ.get("OPENAI_API_KEY", "").strip():
        return False
    try:
        import openai  # noqa: F401,PLC0415
    except ImportError:
        return False
    return True


def _client():
    """Создаёт клиента OpenAI-совместимого провайдера"""
    if not os.environ.get("OPENAI_API_KEY", "").strip():
        raise AnalystAgentUnavailable("OPENAI_API_KEY не задан")
    try:
        import openai  # noqa: PLC0415
    except ImportError as error:
        raise AnalystAgentUnavailable(
            "Пакет openai не установлен: pip install openai"
        ) from error

    kwargs: dict[str, Any] = {"api_key": os.environ["OPENAI_API_KEY"]}
    base_url = os.environ.get("LLM_BASE_URL", "").strip()
    if base_url:
        kwargs["base_url"] = base_url
    return openai.OpenAI(**kwargs)


def _system_message() -> dict[str, str]:
    """Системное сообщение с актуальной календарной датой"""
    return {
        "role": "system",
        "content": (
            f"{SYSTEM_PROMPT}\n\nСегодняшняя календарная дата: {date.today().isoformat()}. "
            "Дата последних данных может быть раньше — уточняй инструментом."
        ),
    }


def _serialize_tool_result(payload: Any) -> str:
    """Готовит результат инструмента к отправке, ограничивая объём"""
    text = json.dumps(payload, ensure_ascii=False, default=str)
    if len(text) <= MAX_TOOL_RESULT_CHARS:
        return text
    return (
        text[:MAX_TOOL_RESULT_CHARS]
        + f'... [обрезано, всего {len(text)} символов. Сузьте запрос или агрегируйте]'
    )


def _dispatch_tool(
    name: str,
    arguments: dict[str, Any],
) -> tuple[str, bool, str, dict[str, Any] | None]:
    """Выполняет инструмент, превращая ошибку в сообщение для модели.

    Ошибку возвращаем модели как результат, а не бросаем наружу: агент должен
    иметь шанс исправить запрос сам — это дешевле, чем падение всего хода.

    Четвёртый элемент — спека графика, если инструмент её вернул. Она вырезается
    из payload до сериализации: точки ряда нужны интерфейсу для отрисовки, но не
    модели, и в контексте они заняли бы больше места, чем весь диалог.
    """
    implementation: Callable[..., Any] | None = TOOL_IMPLEMENTATIONS.get(name)
    if implementation is None:
        message = f"Инструмента '{name}' не существует"
        return json.dumps({"error": message}, ensure_ascii=False), False, message, None

    try:
        payload = implementation(**arguments)
    except ToolError as error:
        message = str(error)
        return json.dumps({"error": message}, ensure_ascii=False), False, message, None
    except TypeError as error:
        message = f"Неверные аргументы: {error}"
        return json.dumps({"error": message}, ensure_ascii=False), False, message, None
    except Exception as error:  # noqa: BLE001 — модель должна увидеть причину
        message = f"{type(error).__name__}: {error}"
        return json.dumps({"error": message}, ensure_ascii=False), False, message, None

    chart = None
    if isinstance(payload, dict) and "_chart" in payload:
        payload = dict(payload)
        chart = payload.pop("_chart")

    return _serialize_tool_result(payload), True, _describe_payload(payload), chart


def _describe_payload(payload: Any) -> str:
    """Короткое человекочитаемое описание результата для трассы в интерфейсе"""
    if not isinstance(payload, dict):
        return "получен результат"

    parts: list[str] = []
    if (source := payload.get("source")) is not None:
        parts.append(str(source))
    for key in ("row_count", "table_count", "column_count"):
        if (value := payload.get(key)) is not None:
            parts.append(f"{key}={value}")
    if payload.get("truncated"):
        parts.append("обрезано")
    return ", ".join(parts) if parts else "получен результат"


def _trim_history(messages: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], bool]:
    """Урезает историю по бюджету символов, не разрывая пары tool_calls → tool.

    Резать можно только по границам сообщений пользователя: если начать историю
    с результата инструмента, API отвергнет запрос — у tool-сообщения не окажется
    предшествующего вызова.
    """
    total = sum(len(json.dumps(m, ensure_ascii=False, default=str)) for m in messages)
    if total <= HISTORY_CHAR_BUDGET:
        return messages, False

    user_positions = [i for i, m in enumerate(messages) if m.get("role") == "user"]
    # Последний ход пользователя оставляем всегда, поэтому кандидаты — все кроме него
    for position in user_positions[:-1]:
        candidate = messages[position + 1:]
        # Начало истории не должно быть tool-сообщением
        while candidate and candidate[0].get("role") == "tool":
            candidate = candidate[1:]
        size = sum(len(json.dumps(m, ensure_ascii=False, default=str)) for m in candidate)
        if size <= HISTORY_CHAR_BUDGET:
            return candidate, True

    # Даже последний ход не влез — оставляем только его
    last = messages[user_positions[-1]:] if user_positions else messages[-1:]
    return last, True


def _finalize(client: Any, model: str, messages: list[dict[str, Any]]) -> str:
    """Добивает финальный текст, когда модель вернула пустой content.

    Вызов без tools: инструменты уже отработали, и повторный их перебор только
    сжёг бы ещё один платный запрос.
    """
    nudge = messages + [{
        "role": "user",
        "content": (
            "Сформулируй итоговый ответ по уже полученным данным. "
            "Только текст, инструменты вызывать не нужно."
        ),
    }]
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[_system_message()] + nudge,
            max_tokens=MAX_OUTPUT_TOKENS,
            temperature=TEMPERATURE,
        )
        text = (response.choices[0].message.content or "").strip()
    except Exception as error:  # noqa: BLE001
        return f"Не удалось получить текстовый ответ модели: {error}"

    return text or (
        "Модель не вернула текстовый ответ, хотя данные собраны. "
        "Попробуйте задать вопрос короче."
    )


def run_turn(
    user_message: str,
    history: list[dict[str, Any]] | None = None,
    *,
    max_iterations: int = MAX_ITERATIONS,
) -> AgentResult:
    """Проводит один ход диалога: от вопроса до финального ответа.

    history — сообщения в формате OpenAI без системного (он добавляется здесь).
    Возвращает обновлённую историю, которую вызывающий сохраняет для следующего хода.
    """
    client = _client()
    model = os.environ.get("OPENAI_MODEL", DEFAULT_MODEL)

    messages: list[dict[str, Any]] = list(history or [])
    messages.append({"role": "user", "content": user_message})
    messages, truncated = _trim_history(messages)

    trace: list[ToolCallRecord] = []
    charts: list[dict[str, Any]] = []
    iterations = 0

    while iterations < max_iterations:
        iterations += 1
        response = client.chat.completions.create(
            model=model,
            messages=[_system_message()] + messages,
            tools=TOOL_SCHEMAS,
            tool_choice="auto",
            max_tokens=MAX_OUTPUT_TOKENS,
            temperature=TEMPERATURE,
        )
        message = response.choices[0].message
        tool_calls = getattr(message, "tool_calls", None)

        # finish_reason у этого провайдера бывает None, поэтому ориентируемся
        # на наличие tool_calls, а не на код завершения
        if not tool_calls:
            reply = (message.content or "").strip()
            if not reply:
                # Инструменты отработали, а текста нет: модель израсходовала бюджет
                # вывода на размышление. Просим сформулировать ответ отдельным
                # вызовом без инструментов — данные для него уже в истории.
                reply = _finalize(client, model, messages)
            messages.append({"role": "assistant", "content": reply})
            return AgentResult(
                reply=reply,
                messages=messages,
                trace=trace,
                charts=charts,
                iterations=iterations,
                truncated_history=truncated,
            )

        messages.append({
            "role": "assistant",
            "content": message.content or "",
            "tool_calls": [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.function.name,
                        "arguments": call.function.arguments,
                    },
                }
                for call in tool_calls
            ],
        })

        for call in tool_calls:
            name = call.function.name
            try:
                arguments = json.loads(call.function.arguments or "{}")
                if not isinstance(arguments, dict):
                    raise ValueError("аргументы должны быть объектом JSON")
            except Exception as error:  # noqa: BLE001
                content = json.dumps(
                    {"error": f"Не удалось разобрать аргументы: {error}"}, ensure_ascii=False
                )
                trace.append(ToolCallRecord(name, {}, False, f"плохие аргументы: {error}"))
                messages.append({"role": "tool", "tool_call_id": call.id, "content": content})
                continue

            content, ok, summary, chart = _dispatch_tool(name, arguments)
            trace.append(ToolCallRecord(name, arguments, ok, summary))
            if chart is not None:
                charts.append(chart)
            messages.append({"role": "tool", "tool_call_id": call.id, "content": content})

    # Потолок итераций: просим модель ответить тем, что уже собрано
    messages.append({
        "role": "user",
        "content": (
            f"Достигнут предел в {max_iterations} обращений к инструментам. "
            "Сформулируй ответ по уже полученным данным и честно укажи, чего не хватило."
        ),
    })
    response = client.chat.completions.create(
        model=model,
        messages=[_system_message()] + messages,
        max_tokens=MAX_OUTPUT_TOKENS,
        temperature=TEMPERATURE,
    )
    reply = (response.choices[0].message.content or "").strip() or (
        "Не удалось собрать ответ в пределах лимита обращений к инструментам."
    )
    messages.append({"role": "assistant", "content": reply})
    return AgentResult(
        reply=reply,
        messages=messages,
        trace=trace,
        charts=charts,
        iterations=iterations,
        truncated_history=truncated,
    )


def main() -> None:
    """Быстрая проверка агента из консоли"""
    if not is_available():
        print("Агент недоступен: нет OPENAI_API_KEY или пакета openai")
        return

    question = "Какой сейчас статус ликвидности и какой модуль вносит основной вклад?"
    result = run_turn(question)
    print(f"Вопрос: {question}\n")
    print(f"Итераций: {result.iterations}")
    for record in result.trace:
        mark = "✓" if record.ok else "✗"
        print(f"  {mark} {record.name}({record.arguments}) — {record.summary}")
    print(f"\n{result.reply}")


if __name__ == "__main__":
    main()
