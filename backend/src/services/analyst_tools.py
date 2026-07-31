"""analyst_tools — инструменты AI-аналитика над витриной LSI.

Каждая функция здесь — это инструмент, который модель вызывает сама, вместо того
чтобы получать заранее собранный текстовый контекст. Так аналитик перестаёт быть
пересказчиком промпта и может дойти до конкретных фич, периодов и сравнений.

Три принципа, заложенные в конструкцию:

1. Каждый результат несёт поле `source` — таблицу или сервис, откуда взяты числа.
   Промпт требует ссылаться на источник, иначе агент с доступом к SQL врёт
   убедительнее, чем агент без него.

2. Объём ответа ограничен. Инструмент, вернувший 34 колонки на 3000 строк,
   съедает контекст и вытесняет сам диалог. Поэтому широкие запросы отдают
   describe-статистику, а не сырые строки.

3. Инструменты только ЧИТАЮТ посчитанное. Пересчитывать LSI своими формулами
   агенту нельзя — иначе рядом с официальным индексом появится второй,
   неофициальный, и разойтись они смогут незаметно.
"""
from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from typing import Any

import pandas as pd

from backend.src.db import warehouse as wh
from backend.src.services import data_freshness as freshness
from backend.src.services import feature_catalog as catalog
from backend.src.services.honest_lsi_prediction import (
    DEFAULT_HONEST_PROFILE,
    get_honest_lsi_response,
)
from backend.src.services.lsi_thresholds import get_threshold_profile


# --- лимиты объёма и времени -------------------------------------------------
MAX_SQL_ROWS = 200
MAX_RAW_SERIES_ROWS = 80
MAX_EXPLICIT_COLUMNS = 8
QUERY_TIMEOUT_SECONDS = 15

LSI_SCORES_TABLE = "honest_lsi_scores"
FEATURE_TABLES = {"final": "final_ml_dataset", "honest": "honest_ml_dataset"}
MODULES = ("m1", "m2", "m3", "m4", "m5")

# Запрещённые конструкции. Read-only соединение DuckDB и так отбивает запись, но
# полагаться на один слой нельзя: ATTACH/COPY/INSTALL умеют трогать файловую
# систему и в read-only режиме тоже.
_FORBIDDEN_KEYWORDS = (
    "attach", "detach", "copy", "install", "load", "pragma", "create", "insert",
    "update", "delete", "drop", "alter", "truncate", "export", "import", "set",
    "call", "grant", "revoke", "vacuum", "checkpoint",
)


class ToolError(RuntimeError):
    """Ошибка инструмента, которую можно безопасно показать модели."""


# ---------------------------------------------------------------------------
# SQL: валидация и выполнение
# ---------------------------------------------------------------------------

def _strip_sql_comments(sql: str) -> str:
    """Убирает комментарии, чтобы через них не проносили запрещённые слова"""
    without_block = re.sub(r"/\*.*?\*/", " ", sql, flags=re.DOTALL)
    without_line = re.sub(r"--[^\n]*", " ", without_block)
    return without_line


def _validate_sql(sql: str) -> str:
    """Проверяет, что запрос — одиночный read-only SELECT. Возвращает его же."""
    if not sql or not sql.strip():
        raise ToolError("Пустой SQL-запрос")

    cleaned = _strip_sql_comments(sql).strip()
    if not cleaned:
        raise ToolError("SQL состоит только из комментариев")

    # Несколько инструкций через ';' запрещены: вторая может быть чем угодно.
    statements = [part for part in cleaned.split(";") if part.strip()]
    if len(statements) > 1:
        raise ToolError("Разрешён только один SQL-оператор, точка с запятой не нужна")

    body = statements[0].strip()
    lowered = body.lower()

    if not (lowered.startswith("select") or lowered.startswith("with")):
        raise ToolError("Разрешены только запросы, начинающиеся с SELECT или WITH")

    for keyword in _FORBIDDEN_KEYWORDS:
        if re.search(rf"\b{keyword}\b", lowered):
            raise ToolError(
                f"Конструкция '{keyword}' запрещена: инструмент только читает данные"
            )

    return body


def _run_sql_with_timeout(sql: str) -> pd.DataFrame:
    """Выполняет SQL на read-only соединении с прерыванием по таймауту"""
    connection = wh.connect(read_only=True)
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(lambda: connection.execute(sql).fetch_df())
            try:
                return future.result(timeout=QUERY_TIMEOUT_SECONDS)
            except FutureTimeout:
                # interrupt() снимает выполняющийся запрос, иначе поток остался бы
                # висеть до конца процесса
                connection.interrupt()
                raise ToolError(
                    f"Запрос не уложился в {QUERY_TIMEOUT_SECONDS} с и был прерван. "
                    "Сузьте период или уберите соединения таблиц."
                ) from None
    finally:
        connection.close()


def query_sql(sql: str) -> dict[str, Any]:
    """Выполняет read-only SELECT по витрине и возвращает строки."""
    body = _validate_sql(sql)
    # Обёртка гарантирует предел строк даже если модель забыла LIMIT
    wrapped = f"SELECT * FROM ({body}) AS _agent_query LIMIT {MAX_SQL_ROWS + 1}"

    try:
        frame = _run_sql_with_timeout(wrapped)
    except ToolError:
        raise
    except Exception as error:
        raise ToolError(f"SQL не выполнился: {error}") from error

    truncated = len(frame) > MAX_SQL_ROWS
    if truncated:
        frame = frame.head(MAX_SQL_ROWS)

    return {
        "source": "DuckDB warehouse (read-only)",
        "sql": body,
        "row_count": int(len(frame)),
        "truncated": truncated,
        "rows": _frame_to_records(frame),
        "note": (
            f"Показаны первые {MAX_SQL_ROWS} строк — итог обрезан, агрегируйте в самом SQL"
            if truncated else None
        ),
    }


# ---------------------------------------------------------------------------
# Схема витрины
# ---------------------------------------------------------------------------

def list_tables() -> dict[str, Any]:
    """Перечисляет таблицы витрины с числом строк, диапазоном дат и свежестью."""
    manifest = wh.manifest()
    if manifest.empty:
        raise ToolError("Витрина пуста — данные ещё не синхронизированы")

    today = pd.Timestamp.today().normalize()
    tables: list[dict[str, Any]] = []
    for row in manifest.to_dict("records"):
        name = str(row.get("table_name"))
        date_max = row.get("date_max")
        lag = None
        if date_max and not pd.isna(date_max):
            lag = int((today - pd.Timestamp(str(date_max))).days)
        try:
            _, date_sql, stored_format = date_expression(name)
        except ToolError:
            date_sql, stored_format = None, None

        tables.append({
            "table": name,
            "rows": int(row.get("row_count") or 0),
            "date_min": str(row.get("date_min") or ""),
            "date_max": str(date_max or ""),
            "lag_days": lag,
            "freshness_status": freshness.classify(name, lag),
            "date_stored_format": stored_format,
            "date_sql_expression": date_sql,
        })

    return {
        "source": "DuckDB warehouse manifest",
        "table_count": len(tables),
        "tables": tables,
        "hint": (
            "Фичи модулей лежат в final_ml_dataset (сырые m1_..m5_) и "
            "honest_ml_dataset (honest-фичи, участвующие в LSI). "
            f"Значения LSI и вклады модулей — в {LSI_SCORES_TABLE}."
        ),
        "date_warning": (
            "Основные колонки дат в витрине имеют тип TIMESTAMP, их можно сравнивать "
            "с литералами DATE напрямую. Отдельные вспомогательные колонки *_date "
            "могли остаться VARCHAR там, где формат источника не разобрался — для них "
            "используй date_sql_expression из describe_table, иначе сравнение окажется "
            "лексикографическим и даст неверный результат без ошибки."
        ),
    }


def describe_table(table: str) -> dict[str, Any]:
    """Возвращает колонки таблицы с типами и примером значений."""
    available = wh.list_tables()
    if table not in available:
        raise ToolError(
            f"Таблицы '{table}' нет в витрине. Доступны: {', '.join(sorted(available))}"
        )

    frame = _run_sql_with_timeout(f'SELECT * FROM "{table}" LIMIT 3')
    full_count = _run_sql_with_timeout(f'SELECT count(*) AS n FROM "{table}"')

    columns = [
        {
            "name": str(name),
            "dtype": str(dtype),
            "label": catalog.label(str(name)),
            "label_status": (
                spec.status if (spec := catalog.lookup(str(name))) else "missing"
            ),
        }
        for name, dtype in zip(frame.columns, frame.dtypes)
    ]

    result = {
        "source": f"DuckDB warehouse: {table}",
        "table": table,
        "row_count": int(full_count["n"].iloc[0]),
        "column_count": len(columns),
        "columns": columns,
        "sample_rows": _frame_to_records(frame),
    }

    try:
        column, expression, stored_format = date_expression(table)
        result["date_column"] = column
        result["date_stored_format"] = stored_format
        result["date_sql_expression"] = expression
        result["date_warning"] = (
            f"В условиях по дате используй ровно {expression}. Сравнение "
            f"{column} со строкой напрямую лексикографично: ошибки не будет, "
            "но результат окажется неверным."
        )
    except ToolError:
        result["date_column"] = None

    return result


# ---------------------------------------------------------------------------
# LSI
# ---------------------------------------------------------------------------

def get_lsi_current(threshold_profile: str = DEFAULT_HONEST_PROFILE) -> dict[str, Any]:
    """Текущее состояние honest-LSI: значения, статусы, вклады и топ-драйверы."""
    try:
        response = get_honest_lsi_response(threshold_profile=threshold_profile)
    except Exception as error:
        raise ToolError(f"Не удалось посчитать honest-LSI: {error}") from error

    response["source"] = "honest_lsi_prediction.get_honest_lsi_response (honest-модели)"

    # Драйверы приходят техническими именами колонок. Подписи подставляем прямо
    # здесь, а не полагаемся на то, что модель дополнительно сходит в
    # lookup_features: на практике она отмечает, что справочник нужен, и всё
    # равно печатает имена колонок пользователю.
    for key in ("top_drivers", "global_top_drivers", "local_top_drivers"):
        drivers = response.get(key)
        if not isinstance(drivers, list):
            continue
        response[f"{key}_described"] = [catalog.describe(column) for column in drivers]

    response["naming_hint"] = (
        "В ответе называй признаки полем label. Имя из column пользователю не "
        "показывай — оно нужно только для запросов. Поля higher_means и status "
        "служебные: используй их для рассуждения, но не печатай в тексте."
    )
    return response


def get_lsi_series(
    date_from: str,
    date_to: str,
    threshold_profile: str = DEFAULT_HONEST_PROFILE,
) -> dict[str, Any]:
    """Значения honest-LSI за период: статистика, пик и вклады модулей."""
    start, end = _parse_period(date_from, date_to)
    profile = get_threshold_profile(threshold_profile)

    frame = _run_sql_with_timeout(
        f'SELECT * FROM "{LSI_SCORES_TABLE}" '
        f"WHERE {_period_filter(LSI_SCORES_TABLE, start, end)} "
        "ORDER BY date"
    )
    if frame.empty:
        return {
            "source": LSI_SCORES_TABLE,
            "period": [str(start.date()), str(end.date())],
            "row_count": 0,
            "note": "За этот период данных нет — проверьте диапазон через list_tables",
        }

    green_max = float(profile["green_max"])
    yellow_max = float(profile["yellow_max"])

    result: dict[str, Any] = {
        "source": LSI_SCORES_TABLE,
        "period": [str(start.date()), str(end.date())],
        "row_count": int(len(frame)),
        "threshold_profile": threshold_profile,
        "thresholds": {"green_max": green_max, "yellow_max": yellow_max},
        "metrics": {},
    }

    for metric in ("lsi_global", "lsi_local"):
        if metric not in frame.columns:
            continue
        series = pd.to_numeric(frame[metric], errors="coerce").dropna()
        if series.empty:
            continue
        peak_idx = series.idxmax()
        result["metrics"][metric] = {
            "min": round(float(series.min()), 2),
            "max": round(float(series.max()), 2),
            "mean": round(float(series.mean()), 2),
            "last": round(float(series.iloc[-1]), 2),
            "peak_date": str(pd.to_datetime(frame.loc[peak_idx, "date"]).date()),
            "days_red": int((series >= yellow_max).sum()),
            "days_yellow": int(((series >= green_max) & (series < yellow_max)).sum()),
            "days_green": int((series < green_max).sum()),
        }

    contrib_columns = [c for c in frame.columns if "_contrib_" in c]
    if contrib_columns:
        result["mean_module_contributions_pct"] = {
            column: round(float(pd.to_numeric(frame[column], errors="coerce").mean()), 1)
            for column in contrib_columns
        }

    # Сырые строки отдаём только для коротких периодов, иначе съедаем контекст
    if len(frame) <= MAX_RAW_SERIES_ROWS:
        keep = ["date"] + [
            c for c in ("lsi_global", "lsi_local", "lsi_global_status", "lsi_local_status")
            if c in frame.columns
        ]
        result["rows"] = _frame_to_records(frame[keep])
    else:
        monthly = frame.set_index(pd.to_datetime(frame["date"]))
        numeric = [c for c in ("lsi_global", "lsi_local") if c in monthly.columns]
        aggregated = monthly[numeric].resample("MS").agg(["mean", "max"]).round(2)
        aggregated.columns = [f"{a}_{b}" for a, b in aggregated.columns]
        aggregated = aggregated.reset_index()
        aggregated["date"] = aggregated["date"].dt.strftime("%Y-%m")
        result["monthly"] = _frame_to_records(aggregated)
        result["note"] = (
            f"Период длиннее {MAX_RAW_SERIES_ROWS} наблюдений — вместо дневных строк "
            "отдана месячная агрегация. Для дневных значений сузьте период."
        )

    return result


# ---------------------------------------------------------------------------
# Фичи модулей
# ---------------------------------------------------------------------------

def get_features(
    module: str,
    date_from: str,
    date_to: str,
    columns: list[str] | None = None,
    dataset: str = "final",
) -> dict[str, Any]:
    """Значения фич модуля за период.

    Без явного списка колонок возвращает не сырые строки, а перечень колонок со
    статистикой: у M5 их 34, и выгрузка всех значений вытеснила бы диалог.
    """
    module_key = module.strip().lower()
    if module_key not in MODULES:
        raise ToolError(f"Модуль должен быть одним из {', '.join(MODULES)}, получено: {module}")

    table = FEATURE_TABLES.get(dataset)
    if table is None:
        raise ToolError(f"dataset должен быть 'final' или 'honest', получено: {dataset}")

    start, end = _parse_period(date_from, date_to)
    frame = _run_sql_with_timeout(
        f'SELECT * FROM "{table}" '
        f"WHERE {_period_filter(table, start, end)} "
        "ORDER BY date"
    )
    if frame.empty:
        raise ToolError(f"В {table} нет строк за период {start.date()}..{end.date()}")

    module_columns = [
        c for c in frame.columns
        if c.lower().startswith(module_key) or c.lower().startswith(f"{module_key}x")
    ]
    if not module_columns:
        raise ToolError(f"В {table} не найдено колонок модуля {module_key}")

    if columns:
        unknown = [c for c in columns if c not in frame.columns]
        if unknown:
            raise ToolError(
                f"Нет колонок: {', '.join(unknown)}. "
                f"Доступны для {module_key}: {', '.join(module_columns)}"
            )
        if len(columns) > MAX_EXPLICIT_COLUMNS:
            raise ToolError(
                f"Не больше {MAX_EXPLICIT_COLUMNS} колонок за раз, запрошено {len(columns)}"
            )
        selected = frame[["date"] + list(columns)]
        if len(selected) > MAX_RAW_SERIES_ROWS:
            raise ToolError(
                f"{len(selected)} строк — слишком много для сырой выдачи. "
                f"Сузьте период до {MAX_RAW_SERIES_ROWS} наблюдений или агрегируйте через query_sql"
            )
        return {
            "source": f"{table} (модуль {module_key.upper()})",
            "period": [str(start.date()), str(end.date())],
            "row_count": int(len(selected)),
            "rows": _frame_to_records(selected),
        }

    statistics: list[dict[str, Any]] = []
    for column in module_columns:
        series = pd.to_numeric(frame[column], errors="coerce")
        non_null = series.dropna()
        spec = catalog.lookup(column)
        statistics.append({
            "column": column,
            "label": catalog.label(column),
            "means": spec.description if spec else "",
            "higher_means": spec.higher_means if spec else "unknown",
            "label_status": spec.status if spec else "missing",
            "non_null": int(len(non_null)),
            "nulls": int(series.isna().sum()),
            "min": round(float(non_null.min()), 4) if not non_null.empty else None,
            "max": round(float(non_null.max()), 4) if not non_null.empty else None,
            "mean": round(float(non_null.mean()), 4) if not non_null.empty else None,
            "last": round(float(non_null.iloc[-1]), 4) if not non_null.empty else None,
            "distinct": int(non_null.nunique()),
        })

    constant = [s["column"] for s in statistics if s["distinct"] <= 1]
    return {
        "source": f"{table} (модуль {module_key.upper()})",
        "period": [str(start.date()), str(end.date())],
        "row_count": int(len(frame)),
        "column_count": len(module_columns),
        "column_statistics": statistics,
        "constant_columns": constant,
        "note": (
            "Отдана статистика по колонкам, а не сырые значения. Для конкретных "
            "значений передайте columns (не больше "
            f"{MAX_EXPLICIT_COLUMNS}). Колонки из constant_columns не меняются на "
            "периоде — трактовать их как сигнал нельзя, это может быть заполнение пропусков."
        ),
    }


# ---------------------------------------------------------------------------
# Справочник признаков
# ---------------------------------------------------------------------------

def lookup_features(
    columns: list[str] | None = None,
    module: str | None = None,
) -> dict[str, Any]:
    """Возвращает человеческие названия и смысл признаков.

    Нужен, чтобы модель называла признаки понятно аналитику, а не техническими
    именами, и не догадывалась о смысле по имени колонки.
    """
    if columns:
        entries = [catalog.describe(column) for column in columns]
    elif module:
        key = module.strip().upper()
        entries = [
            catalog.describe(column)
            for column, spec in catalog.CATALOG.items()
            if spec.module.upper() == key
        ]
        if not entries:
            raise ToolError(
                f"В каталоге нет признаков модуля {key}. Доступны: "
                + ", ".join(sorted({s.module for s in catalog.CATALOG.values()}))
            )
    else:
        entries = [catalog.describe(column) for column in sorted(catalog.CATALOG)]

    return {
        "source": "feature_catalog",
        "count": len(entries),
        "features": entries,
        "hint": (
            "В ответе пользователю используй только поле label. Имя из column не "
            "показывай — оно нужно тебе для запросов, а не читателю. Поля "
            "higher_means и status служебные: направление трактовки выражай "
            "словами, сами названия полей в текст не выноси. Если status = "
            "needs_review или missing, скажи обычной фразой, что название уточняется."
        ),
    }


# ---------------------------------------------------------------------------
# Графики
# ---------------------------------------------------------------------------

# Спека рисуется существующими компонентами dashboard/components/charts.py:
# модель описывает, ЧТО показать, а не генерирует код. Стиль остаётся единым
# с остальным дашбордом, и исполнять произвольный код не требуется.
CHART_KINDS = ("line", "signal", "bar", "dual_axis", "flag_timeline")

MAX_CHART_SERIES = 4
MAX_CHART_POINTS = 2000


def plot_series(
    table: str,
    columns: list[str],
    date_from: str,
    date_to: str,
    kind: str = "line",
    title: str = "",
    yaxis_title: str = "",
) -> dict[str, Any]:
    """Строит график по колонкам таблицы за период.

    Данные для отрисовки возвращаются в служебном ключе `_chart`, который агент
    вырезает и НЕ отправляет модели: иначе сотни точек ряда съели бы контекст.
    Модели достаётся только сводка — чего достаточно, чтобы описать график словами.
    """
    if kind not in CHART_KINDS:
        raise ToolError(f"kind должен быть одним из {', '.join(CHART_KINDS)}, получено: {kind}")

    available = wh.list_tables()
    if table not in available:
        raise ToolError(f"Таблицы '{table}' нет в витрине. Доступны: {', '.join(sorted(available))}")

    if not columns:
        raise ToolError("Нужна хотя бы одна колонка для графика")
    if len(columns) > MAX_CHART_SERIES:
        raise ToolError(f"Не больше {MAX_CHART_SERIES} рядов на графике, запрошено {len(columns)}")
    if kind == "dual_axis" and len(columns) != 2:
        raise ToolError("kind='dual_axis' требует ровно две колонки")

    start, end = _parse_period(date_from, date_to)
    date_column, _, _ = date_expression(table)

    quoted = ", ".join(f'"{c}"' for c in columns)
    try:
        frame = _run_sql_with_timeout(
            f'SELECT "{date_column}", {quoted} FROM "{table}" '
            f"WHERE {_period_filter(table, start, end)} "
            f'ORDER BY "{date_column}" LIMIT {MAX_CHART_POINTS}'
        )
    except ToolError:
        raise
    except Exception as error:
        raise ToolError(
            f"Не удалось выбрать данные: {error}. "
            f"Проверьте имена колонок через describe_table('{table}')"
        ) from error

    if frame.empty:
        raise ToolError(f"За период {start.date()}..{end.date()} в {table} нет данных")

    summary: dict[str, Any] = {}
    for column in columns:
        series = pd.to_numeric(frame[column], errors="coerce").dropna()
        if series.empty:
            summary[column] = {"note": "нет числовых значений на периоде"}
            continue
        first, last = float(series.iloc[0]), float(series.iloc[-1])
        spec = catalog.lookup(column)
        summary[column] = {
            "label": catalog.label(column),
            "higher_means": spec.higher_means if spec else "unknown",
            "first": round(first, 4),
            "last": round(last, 4),
            "change": round(last - first, 4),
            "min": round(float(series.min()), 4),
            "max": round(float(series.max()), 4),
            "mean": round(float(series.mean()), 4),
        }

    return {
        "source": f"{table} (график)",
        "chart_rendered": True,
        "kind": kind,
        "table": table,
        "columns": columns,
        "period": [str(start.date()), str(end.date())],
        "point_count": int(len(frame)),
        "series_summary": summary,
        "note": (
            "График показан пользователю. Опиши словами, что на нём видно, опираясь "
            "на series_summary — сами точки ряда тебе не нужны."
        ),
        "_chart": {
            "kind": kind,
            "date_column": date_column,
            "columns": columns,
            # Легенда и заголовок — человеческими названиями, а не именами колонок
            "labels": {column: catalog.label(column) for column in columns},
            "title": title or f"{', '.join(catalog.label(c) for c in columns)}",
            "yaxis_title": yaxis_title,
            "data": _frame_to_records(frame),
        },
    }


def plot_contributions(
    date: str | None = None,
    metric: str = "lsi_global",
) -> dict[str, Any]:
    """Диаграмма вкладов модулей в LSI на конкретную дату.

    Отдельный инструмент, потому что вклады на одну дату — это не временной ряд:
    plot_series под них не подходит, а собирать фигуру Plotly руками ради частого
    запроса модель заставлять не стоит.
    """
    if metric not in ("lsi_global", "lsi_local"):
        raise ToolError("metric должен быть 'lsi_global' или 'lsi_local'")

    if date is None:
        frame = _run_sql_with_timeout(
            f'SELECT * FROM "{LSI_SCORES_TABLE}" ORDER BY date DESC LIMIT 1'
        )
    else:
        target, _ = _parse_period(date, date)
        frame = _run_sql_with_timeout(
            f'SELECT * FROM "{LSI_SCORES_TABLE}" '
            f"WHERE CAST(date AS DATE) <= DATE '{target.date()}' "
            "ORDER BY date DESC LIMIT 1"
        )

    if frame.empty:
        raise ToolError(f"Нет данных LSI на дату {date or 'последнюю'} или раньше неё")

    row = frame.iloc[0]
    actual_date = str(pd.to_datetime(row["date"]).date())
    marker = f"{metric}_contrib_"

    contributions = {
        str(column)[len(marker):].upper(): float(row[column])
        for column in frame.columns
        if str(column).startswith(marker) and pd.notna(row[column])
    }
    if not contributions:
        raise ToolError(f"В {LSI_SCORES_TABLE} нет колонок вкладов для {metric}")

    ordered = sorted(contributions.items(), key=lambda item: item[1], reverse=True)
    lsi_value = float(row[metric]) if pd.notna(row.get(metric)) else None

    return {
        "source": LSI_SCORES_TABLE,
        "chart_rendered": True,
        "date": actual_date,
        "metric": metric,
        "lsi_value": round(lsi_value, 2) if lsi_value is not None else None,
        "contributions_pct": {name: round(value, 2) for name, value in ordered},
        "note": (
            "Диаграмма показана пользователю. Вклады — это доли модулей в значении "
            "индекса, а не причины движения: трактуй их как указание, какой блок "
            "смотреть первым."
        ),
        "_chart": {
            "kind": "contributions",
            "title": f"Вклады модулей в {metric} на {actual_date}",
            "data": [
                {"module": name, "contribution": round(value, 2)} for name, value in ordered
            ],
        },
    }


def plot_custom(figure_json: dict[str, Any], title: str = "") -> dict[str, Any]:
    """Рисует произвольную фигуру Plotly по её JSON-описанию.

    Запасной путь для визуализаций, которых нет среди готовых видов. Принимается
    только описание фигуры (data/layout) — не код, поэтому исполнять нечего.
    """
    if not isinstance(figure_json, dict):
        raise ToolError("figure_json должен быть объектом с ключами data и layout")

    data = figure_json.get("data")
    if not isinstance(data, list) or not data:
        raise ToolError("figure_json.data должен быть непустым списком трейсов")
    if not all(isinstance(trace, dict) for trace in data):
        raise ToolError("каждый трейс в figure_json.data должен быть объектом")

    layout = figure_json.get("layout") or {}
    if not isinstance(layout, dict):
        raise ToolError("figure_json.layout должен быть объектом")

    return {
        "source": "plot_custom (фигура Plotly от модели)",
        "chart_rendered": True,
        "trace_count": len(data),
        "note": (
            "Фигура показана пользователю. Числа для неё ты должен был получить "
            "инструментами данных — не придумывай значения."
        ),
        "_chart": {
            "kind": "custom",
            "title": title,
            "figure": {"data": data, "layout": layout},
        },
    }


# ---------------------------------------------------------------------------
# Свежесть
# ---------------------------------------------------------------------------

def get_data_freshness() -> dict[str, Any]:
    """Свежесть таблиц витрины с поправкой на график публикации источника."""
    manifest = wh.manifest()
    if manifest.empty:
        raise ToolError("Витрина пуста")

    today = pd.Timestamp.today().normalize()
    rows: list[dict[str, Any]] = []
    for record in manifest.to_dict("records"):
        name = str(record.get("table_name"))
        date_max = record.get("date_max")
        lag = None
        if date_max and not pd.isna(date_max):
            lag = int((today - pd.Timestamp(str(date_max))).days)
        rows.append({
            "table": name,
            "date_max": str(date_max or ""),
            "lag_days": lag,
            "expected_lag_days": freshness.expected_lag(name),
            "overdue_days": freshness.overdue_days(name, lag),
            "status": freshness.classify(name, lag),
            "source_cadence": freshness.cadence(name),
        })

    stale = [r["table"] for r in rows if r["status"] == freshness.STATUS_STALE]
    return {
        "source": "data_freshness + warehouse manifest",
        "today": str(today.date()),
        "tables": rows,
        "stale_tables": stale,
        "hint": (
            "Отставание сравнивается с ожидаемым лагом публикации, а не с календарём: "
            "у месячных рядов ЦБ 50-70 дней это норма. Если таблица в статусе stale, "
            "выводы по её модулю нужно оговаривать — данные могли не обновиться."
        ),
    }


# ---------------------------------------------------------------------------
# Вспомогательное
# ---------------------------------------------------------------------------

_date_expression_cache: dict[str, tuple[str, str, str]] = {}


def date_expression(table: str) -> tuple[str, str, str]:
    """Возвращает (колонка даты, SQL-выражение приведения к DATE, описание формата).

    Витрина хранит даты неоднородно: часть таблиц в DD-MM-YYYY, часть в YYYY-MM-DD,
    honest_lsi_scores — в TIMESTAMP. Сравнение VARCHAR-даты со строкой лексикографично
    и НЕ выдаёт ошибки: '11-01-2010' >= '2022-01-01' просто ложно. Поэтому любое
    условие по дате обязано идти через приведение, иначе агент получит пустой или
    неверный результат и уверенно на нём построит вывод.
    """
    if table in _date_expression_cache:
        return _date_expression_cache[table]

    described = _run_sql_with_timeout(f'DESCRIBE "{table}"')
    columns = {str(r["column_name"]): str(r["column_type"]) for r in described.to_dict("records")}

    column = next((c for c in ("date", "auction_date", "dt") if c in columns), None)
    if column is None:
        raise ToolError(f"В таблице {table} нет колонки с датой")

    column_type = columns[column].upper()
    quoted = f'"{column}"'

    if "DATE" in column_type or "TIMESTAMP" in column_type:
        result = (column, f"CAST({quoted} AS DATE)", column_type)
    else:
        sample_frame = _run_sql_with_timeout(
            f'SELECT {quoted} AS v FROM "{table}" WHERE {quoted} IS NOT NULL LIMIT 1'
        )
        sample = str(sample_frame["v"].iloc[0]) if not sample_frame.empty else ""
        if re.match(r"^\d{4}-\d{2}-\d{2}", sample):
            result = (column, f"CAST({quoted} AS DATE)", "VARCHAR YYYY-MM-DD")
        elif re.match(r"^\d{2}-\d{2}-\d{4}", sample):
            result = (column, f"strptime({quoted}, '%d-%m-%Y')", "VARCHAR DD-MM-YYYY")
        else:
            raise ToolError(
                f"Не удалось распознать формат даты в {table}.{column}: пример {sample!r}"
            )

    _date_expression_cache[table] = result
    return result


def _period_filter(table: str, start: pd.Timestamp, end: pd.Timestamp) -> str:
    """Собирает корректное условие по периоду с учётом формата хранения даты"""
    _, expression, _ = date_expression(table)
    return (
        f"{expression} >= DATE '{start.date()}' AND {expression} <= DATE '{end.date()}'"
    )


def _parse_period(date_from: str, date_to: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Разбирает границы периода, отдавая понятную ошибку вместо стектрейса"""
    try:
        start = pd.Timestamp(date_from)
        end = pd.Timestamp(date_to)
    except Exception as error:
        raise ToolError(
            f"Даты должны быть в формате YYYY-MM-DD, получено: {date_from!r}, {date_to!r}"
        ) from error

    if start > end:
        raise ToolError(f"Начало периода {start.date()} позже конца {end.date()}")
    return start, end


def _frame_to_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Переводит датафрейм в JSON-совместимые записи"""
    prepared = frame.copy()
    for column in prepared.columns:
        if pd.api.types.is_datetime64_any_dtype(prepared[column]):
            prepared[column] = prepared[column].dt.strftime("%Y-%m-%d")
    prepared = prepared.where(pd.notna(prepared), None)
    return prepared.to_dict("records")


# ---------------------------------------------------------------------------
# Реестр для агента
# ---------------------------------------------------------------------------

TOOL_IMPLEMENTATIONS = {
    "list_tables": list_tables,
    "describe_table": describe_table,
    "query_sql": query_sql,
    "get_lsi_current": get_lsi_current,
    "get_lsi_series": get_lsi_series,
    "get_features": get_features,
    "get_data_freshness": get_data_freshness,
    "lookup_features": lookup_features,
    "plot_series": plot_series,
    "plot_contributions": plot_contributions,
    "plot_custom": plot_custom,
}

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "list_tables",
            "description": (
                "Перечислить таблицы витрины LSI с числом строк, диапазоном дат и "
                "статусом свежести. Вызывать первым, если не знаешь, где лежат данные."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "describe_table",
            "description": "Показать колонки таблицы с типами и примером строк.",
            "parameters": {
                "type": "object",
                "properties": {"table": {"type": "string", "description": "Имя таблицы витрины"}},
                "required": ["table"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_sql",
            "description": (
                "Выполнить read-only SELECT по витрине DuckDB. Разрешены только SELECT и WITH, "
                f"результат обрезается до {MAX_SQL_ROWS} строк — агрегируй в самом запросе. "
                "Используй для сравнений, корреляций, поиска эпизодов и всего, чего нет "
                "в остальных инструментах."
            ),
            "parameters": {
                "type": "object",
                "properties": {"sql": {"type": "string", "description": "Одиночный SELECT-запрос"}},
                "required": ["sql"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_lsi_current",
            "description": (
                "Текущее состояние honest-LSI на последнюю дату: LSI Global и Local, статусы, "
                "вклады модулей, топ-драйверы, налоговый overlay. Это официальные значения индекса."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_lsi_series",
            "description": (
                "Значения honest-LSI за период: min/max/среднее, дата пика, распределение по "
                "зонам светофора, средние вклады модулей. Для коротких периодов отдаёт дневные "
                "строки, для длинных — месячную агрегацию."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "date_from": {"type": "string", "description": "Начало периода, YYYY-MM-DD"},
                    "date_to": {"type": "string", "description": "Конец периода, YYYY-MM-DD"},
                },
                "required": ["date_from", "date_to"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_features",
            "description": (
                "Фичи модуля (m1-m5) за период. Без параметра columns отдаёт перечень колонок "
                "со статистикой и список колонок-константов; с columns — сырые значения "
                f"(не больше {MAX_EXPLICIT_COLUMNS} колонок и {MAX_RAW_SERIES_ROWS} строк). "
                "dataset='final' — сырые фичи модулей, 'honest' — фичи, входящие в LSI."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "module": {"type": "string", "enum": list(MODULES)},
                    "date_from": {"type": "string", "description": "YYYY-MM-DD"},
                    "date_to": {"type": "string", "description": "YYYY-MM-DD"},
                    "columns": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Конкретные колонки для сырой выдачи (необязательно)",
                    },
                    "dataset": {"type": "string", "enum": ["final", "honest"]},
                },
                "required": ["module", "date_from", "date_to"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lookup_features",
            "description": (
                "Узнать человеческие названия признаков и их смысл: что измеряет, в чём "
                "измеряется, куда трактовать рост значения. Вызывай ВСЕГДА перед тем, как "
                "упомянуть признак в ответе — технические имена вроде m3x_cover пользователю "
                "непонятны, а направление знака из имени не выводится. Без аргументов "
                "возвращает весь каталог."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "columns": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Конкретные технические имена признаков",
                    },
                    "module": {
                        "type": "string",
                        "description": "Модуль целиком: M1, M2, M3, M4, M5",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "plot_series",
            "description": (
                "Построить и показать пользователю график по колонкам таблицы за период. "
                "Виды: line (несколько рядов), signal (ряд с полосами порога стресса), "
                "bar, dual_axis (ровно две колонки на двух осях), flag_timeline (бинарные флаги). "
                "Вызывай, когда пользователь просит показать график, динамику или сравнение "
                "визуально. В ответе опиши словами, что на графике видно."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "table": {"type": "string", "description": "Таблица витрины"},
                    "columns": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": f"Колонки-ряды, не больше {MAX_CHART_SERIES}",
                    },
                    "date_from": {"type": "string", "description": "YYYY-MM-DD"},
                    "date_to": {"type": "string", "description": "YYYY-MM-DD"},
                    "kind": {"type": "string", "enum": list(CHART_KINDS)},
                    "title": {"type": "string", "description": "Заголовок графика"},
                    "yaxis_title": {"type": "string", "description": "Подпись оси Y"},
                },
                "required": ["table", "columns", "date_from", "date_to"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "plot_contributions",
            "description": (
                "Показать диаграмму вкладов модулей в LSI на дату. Используй для запросов "
                "вида «покажи вклады», «диаграмма по модулям», «из чего сложился индекс». "
                "Без параметра date берётся последняя доступная дата."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "date": {"type": "string", "description": "YYYY-MM-DD, необязательно"},
                    "metric": {"type": "string", "enum": ["lsi_global", "lsi_local"]},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "plot_custom",
            "description": (
                "Показать произвольную фигуру Plotly по её JSON-описанию (data + layout). "
                "Запасной путь: используй, только когда нужного вида нет в plot_series — "
                "например для наложения вкладов модулей, гистограммы распределения или "
                "диаграммы рассеяния. Числа бери из инструментов данных, не выдумывай."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "figure_json": {
                        "type": "object",
                        "description": "Объект фигуры Plotly: {\"data\": [...], \"layout\": {...}}",
                    },
                    "title": {"type": "string"},
                },
                "required": ["figure_json"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_data_freshness",
            "description": (
                "Свежесть таблиц витрины с поправкой на график публикации источника. "
                "Вызывай перед выводами о текущем состоянии: если таблица модуля в статусе "
                "stale, её данные не обновились и вывод нужно оговорить."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
]
