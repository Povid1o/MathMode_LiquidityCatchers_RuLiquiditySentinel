"""Reusable metric cards and status badges for the dashboard."""
import streamlit as st
import pandas as pd
from dashboard.config import MAD_STRESS_THRESHOLD


def status_badge(ok: bool, label: str = "") -> str:
    color = "#2ca02c" if ok else "#d62728"
    icon = "✓" if ok else "✗"
    text = f"{icon} {label}" if label else icon
    return f'<span style="background:{color};color:white;padding:2px 8px;border-radius:4px;font-size:0.85em">{text}</span>'


def module_status_row(name: str, ok: bool, rows: int | None, date_max: pd.Timestamp | None) -> None:
    col1, col2, col3 = st.columns([2, 1, 2])
    with col1:
        st.markdown(status_badge(ok, name), unsafe_allow_html=True)
    with col2:
        st.markdown(f"**{rows:,}** строк" if rows is not None else "—")
    with col3:
        if date_max is not None:
            st.markdown(f"до **{date_max.strftime('%d.%m.%Y')}**")
        else:
            st.markdown("—")


def latest_value_metric(label: str, series: pd.Series, fmt: str = "{:.2f}", suffix: str = "") -> None:
    val = series.dropna().iloc[-1] if not series.dropna().empty else None
    prev = series.dropna().iloc[-2] if len(series.dropna()) >= 2 else None
    if val is not None:
        display = fmt.format(val) + suffix
        delta = None
        if prev is not None:
            delta = f"{val - prev:+.2f}{suffix}"
        st.metric(label, display, delta=delta)
    else:
        st.metric(label, "н/д")


def mad_status_metric(label: str, series: pd.Series, threshold: float = MAD_STRESS_THRESHOLD) -> None:
    val = series.dropna().iloc[-1] if not series.dropna().empty else None
    if val is None:
        st.metric(label, "н/д")
        return
    if abs(val) >= threshold:
        level = "🔴 Стресс"
    elif abs(val) >= 1.0:
        level = "🟡 Умеренный"
    else:
        level = "🟢 Норма"
    st.metric(label, f"{val:.2f}", delta=level, delta_color="off")


def lsi_stub_banner() -> None:
    st.warning(
        "**LSI (Индекс стресса ликвидности) в разработке.**  \n"
        "Финальная модель и индекс не рассчитаны. "
        "Ниже представлены сигналы по отдельным модулям. "
        "Они не являются итоговым индексом стресса.",
        icon="⚠️",
    )


def proxy_score_note() -> None:
    st.info(
        "**PROXY / DEMO:** Показанный составной балл рассчитан как упрощённое среднее "
        "нормализованных MAD-сигналов модулей. Это **не финальный LSI** и "
        "не должен использоваться как готовый индекс.",
        icon="ℹ️",
    )


def date_range_filter(df: pd.DataFrame, key: str) -> pd.DataFrame:
    """Sidebar date range filter that returns filtered df."""
    min_date = df["date"].min().date()
    max_date = df["date"].max().date()
    start, end = st.sidebar.date_input(
        "Период",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
        key=key,
    )
    mask = (df["date"].dt.date >= start) & (df["date"].dt.date <= end)
    return df[mask].copy()
