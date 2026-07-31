"""Отрисовка графиков, которые запросил AI-аналитик.

Модель возвращает спеку — описание того, ЧТО показать, — а рисуют существующие
компоненты дашборда. Кода модель не генерирует и не исполняет, поэтому стиль
остаётся единым с остальными страницами, а вариантов сломать отрисовку меньше.

Запасной путь — готовая фигура Plotly от модели (kind='custom'): нужен для
визуализаций, которых нет среди типовых видов.
"""
from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard.components.charts import (
    bar_chart,
    dual_axis_chart,
    flag_timeline,
    line_chart,
    signal_line,
)


def _spec_frame(spec: dict[str, Any]) -> pd.DataFrame:
    """Собирает датафрейм из данных спеки"""
    frame = pd.DataFrame(spec.get("data") or [])
    if frame.empty:
        return frame
    date_column = spec.get("date_column", "date")
    if date_column in frame.columns:
        frame[date_column] = pd.to_datetime(frame[date_column], errors="coerce")
    for column in spec.get("columns", []):
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def _build_figure(spec: dict[str, Any]) -> go.Figure | None:
    """Переводит спеку в фигуру Plotly существующими компонентами"""
    kind = spec.get("kind")

    if kind == "custom":
        figure = spec.get("figure") or {}
        return go.Figure(figure)

    if kind == "contributions":
        frame = pd.DataFrame(spec.get("data") or [])
        if frame.empty:
            return None
        return bar_chart(
            frame, "module", "contribution",
            title=spec.get("title", ""),
            yaxis_title="Вклад, %",
        )

    frame = _spec_frame(spec)
    if frame.empty:
        return None

    date_column = spec.get("date_column", "date")
    columns = [c for c in spec.get("columns", []) if c in frame.columns]
    if not columns:
        return None

    title = spec.get("title", "")
    yaxis_title = spec.get("yaxis_title", "")

    if kind == "signal":
        return signal_line(frame, date_column, columns[0], title=title)
    if kind == "bar":
        return bar_chart(frame, date_column, columns[0], title=title)
    if kind == "dual_axis" and len(columns) >= 2:
        return dual_axis_chart(
            frame, date_column, columns[0], columns[1],
            y1_label=columns[0], y2_label=columns[1], title=title,
        )
    if kind == "flag_timeline":
        return flag_timeline(frame, date_column, {c: c for c in columns}, title=title)

    # line — и он же разумный запасной вариант для неизвестного вида
    return line_chart(frame, date_column, columns, title=title, yaxis_title=yaxis_title)


def render_charts(specs: list[dict[str, Any]], *, key_prefix: str = "") -> None:
    """Рисует все графики, запрошенные аналитиком в рамках одного ответа"""
    for index, spec in enumerate(specs or []):
        try:
            figure = _build_figure(spec)
        except Exception as error:  # noqa: BLE001 — сбой графика не должен рушить чат
            st.warning(f"Не удалось построить график: {error}")
            continue

        if figure is None:
            st.caption("График запрошен, но данных для отрисовки не оказалось.")
            continue

        st.plotly_chart(
            figure,
            use_container_width=True,
            key=f"agent_chart_{key_prefix}_{index}",
        )
