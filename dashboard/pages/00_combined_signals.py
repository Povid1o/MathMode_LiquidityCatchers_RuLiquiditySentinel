import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from dashboard.data.loader import load_final, load_lsi, load_threshold_profile
from backend.src.services.lsi_thresholds import DEFAULT_THRESHOLD_PROFILE
from dashboard.components.metrics import proxy_score_note, quick_period_filter, csv_download_button, freshness_header
from dashboard.config import COLORS, PLOTLY_TEMPLATE, MAD_STRESS_THRESHOLD

st.set_page_config(page_title="Сводные сигналы", layout="wide")
st.title("Сводные сигналы по модулям")

st.markdown(
    "Все MAD-сигналы M1–M4 на одной оси времени. "
    "Позволяет сравнить уровни стресса по модулям без переключения страниц."
)
proxy_score_note()

with st.spinner("Загрузка финального датасета и LSI..."):
    df = load_lsi()

# Читаем активный профиль из session_state (устанавливается на странице обзора).
# Если пользователь ещё не выбрал профиль, используем DEFAULT_THRESHOLD_PROFILE.
_active_profile: str = st.session_state.get("lsi_threshold_profile", DEFAULT_THRESHOLD_PROFILE)
_thr = load_threshold_profile(_active_profile)
_thr_green = float(_thr["green_max"])
_thr_yellow = float(_thr["yellow_max"])

freshness_header(df, "Final ML Dataset")
df = quick_period_filter(df, key="signals_period")

if df.empty:
    st.warning("Нет данных для выбранного периода.")
    st.stop()

as_of_date = df["date"].max()
st.caption(
    f"Сводные дневные показатели и LSI показаны на дату финального ML dataset: "
    f"{as_of_date.strftime('%d.%m.%Y')}. Это не обязательно сегодняшняя календарная дата."
)

# --- Compute proxy score ---
mad_cols = {
    "m1_spread_mad_score": "M1 Спред",
    "m2_MAD_score_cover": "M2 Cover",
    "m3_cover_stress_score": "M3 Cover",
    "m4_MAD_tax_pressure": "M4 Давление",
}
avail = [c for c in mad_cols if c in df.columns]
df["_proxy_score"] = df[avail].abs().mean(axis=1)


def signal_level(value: float) -> str:
    """Возвращает текстовый уровень сигнала"""
    if abs(value) >= MAD_STRESS_THRESHOLD:
        return "🔴 Стресс"
    if abs(value) >= 1.0:
        return "🟡 Умеренный"
    return "🟢 Норма"


def latest_event_value(column: str, flag_column: str) -> tuple[float | None, str]:
    """Возвращает последнее значение по событийному модулю"""
    if column not in df.columns or flag_column not in df.columns:
        return None, "нет данных"
    event_df = df[(df[flag_column] == 1) & df[column].notna()]
    if event_df.empty:
        return None, "нет события"
    row = event_df.iloc[-1]
    return float(row[column]), f"аукцион {row['date'].strftime('%d.%m.%Y')}"

# --- KPI row ---
st.subheader("Текущие сигналы")
st.caption(
    "M1 и M4 читаются на последнюю дневную дату. "
    "M2 и M3 разреженные, поэтому показывается последний аукцион, а не искусственный ноль в день без аукциона."
)

metric_cards = [
    ("M1 Спред", "m1_spread_mad_score", None),
    ("M2 Cover", "m2_MAD_score_cover", "m2_auction_flag"),
    ("M3 Cover", "m3_cover_stress_score", "m3_auction_flag"),
    ("M4 Давление", "m4_MAD_tax_pressure", None),
]

cols = st.columns(len(metric_cards) + 1)
for i, (label, col, flag_col) in enumerate(metric_cards):
    with cols[i]:
        if col not in df.columns:
            st.metric(label, "н/д", delta="нет колонки", delta_color="off")
            continue
        if flag_col:
            val, note = latest_event_value(col, flag_col)
        else:
            val = df[col].dropna().iloc[-1] if not df[col].dropna().empty else None
            note = "последняя дата"
            if label == "M4 Давление" and val == 0:
                note = "нет налогового окна"
        if val is None:
            st.metric(label, "н/д", delta=note, delta_color="off")
        else:
            st.metric(label, f"{val:.2f}", delta=f"{signal_level(val)} · {note}", delta_color="off")

with cols[-1]:
    proxy_val = df["_proxy_score"].dropna().iloc[-1] if not df["_proxy_score"].dropna().empty else None
    if proxy_val is not None:
        if proxy_val >= MAD_STRESS_THRESHOLD:
            level = "🔴 Стресс"
        elif proxy_val >= 1.0:
            level = "🟡 Умеренный"
        else:
            level = "🟢 Норма"
        st.metric(
            "Демо MAD-агрегат",
            f"{proxy_val:.2f}",
            delta=f"{level} · {as_of_date.strftime('%d.%m.%Y')}",
            delta_color="off",
        )

st.markdown("---")

# --- Combined MAD chart ---
st.subheader("MAD-сигналы по модулям")

fig = go.Figure()

palette = [COLORS["primary"], COLORS["secondary"], COLORS["success"], COLORS["warn"]]

for i, (col, label) in enumerate(mad_cols.items()):
    if col not in df.columns:
        continue
    series = df[["date", col]].dropna()
    fig.add_trace(go.Scatter(
        x=series["date"],
        y=series[col],
        name=label,
        mode="lines",
        line=dict(color=palette[i % len(palette)], width=1.5),
        connectgaps=False,
        opacity=0.85,
    ))

fig.add_hline(y=MAD_STRESS_THRESHOLD, line_dash="dash", line_color=COLORS["danger"],
              opacity=0.6, annotation_text="Стресс (+2σ)", annotation_position="right")
fig.add_hline(y=-MAD_STRESS_THRESHOLD, line_dash="dash", line_color=COLORS["danger"],
              opacity=0.6, annotation_text="Стресс (−2σ)", annotation_position="right")
fig.add_hline(y=0, line_dash="dot", line_color="white", opacity=0.2)

fig.update_layout(
    template=PLOTLY_TEMPLATE,
    height=420,
    yaxis_title="MAD score",
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    margin=dict(l=40, r=60, t=40, b=40),
)
st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# --- LSI chart (if available) ---
lsi_columns = {
    "lsi_local": ("LSI Local", COLORS["primary"]),
    "lsi_global": ("LSI Global", COLORS["secondary"]),
}
available_lsi_columns = {col: meta for col, meta in lsi_columns.items() if col in df.columns}
if available_lsi_columns or "lsi" in df.columns:
    st.subheader("LSI — Индекс стресса ликвидности")
    st.caption(
        "LSI Local обучается на последнем 365-дневном окне, LSI Global — на всей истории. "
        f"Пороговый профиль: **{_active_profile}** — "
        f"зелёный < {int(_thr_green)}, жёлтый {int(_thr_green)}–{int(_thr_yellow)}, "
        f"красный ≥ {int(_thr_yellow)}. "
        "Изменить профиль можно на странице «Обзор системы»."
    )

    fig_lsi = go.Figure()
    if not available_lsi_columns and "lsi" in df.columns:
        available_lsi_columns = {"lsi": ("LSI", COLORS["primary"])}

    for col, (label, color) in available_lsi_columns.items():
        lsi_df = df[["date", col]].dropna()
        fig_lsi.add_trace(go.Scatter(
            x=lsi_df["date"],
            y=lsi_df[col],
            mode="lines",
            fill="tozeroy" if col == "lsi_local" else None,
            fillcolor="rgba(31,119,180,0.12)",
            line=dict(color=color, width=2),
            name=label,
        ))
    fig_lsi.add_hline(y=_thr_yellow, line_dash="dash", line_color=COLORS["danger"],
                      opacity=0.7, annotation_text=f"Стресс ≥{int(_thr_yellow)}", annotation_position="right")
    fig_lsi.add_hline(y=_thr_green, line_dash="dot", line_color=COLORS["warn"],
                      opacity=0.7, annotation_text=f"Внимание ≥{int(_thr_green)}", annotation_position="left")

    fig_lsi.update_layout(
        template=PLOTLY_TEMPLATE,
        height=340,
        yaxis_title="LSI Score",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=40, r=60, t=40, b=40),
    )
    st.plotly_chart(fig_lsi, use_container_width=True)
    st.markdown("---")

# --- Proxy score chart ---
st.subheader("Демо MAD-агрегат")
st.caption(
    "Среднее абсолютных значений доступных MAD-сигналов. "
    "Не является финальным LSI. Используется только для иллюстрации."
)

proxy_df = df[["date", "_proxy_score"]].dropna()

fig_proxy = go.Figure()
fig_proxy.add_trace(go.Scatter(
    x=proxy_df["date"],
    y=proxy_df["_proxy_score"],
    mode="lines",
    fill="tozeroy",
    fillcolor="rgba(31,119,180,0.15)",
    line=dict(color=COLORS["primary"], width=2),
    name="Демо MAD-агрегат",
))
fig_proxy.add_hline(y=MAD_STRESS_THRESHOLD, line_dash="dash", line_color=COLORS["danger"],
                    opacity=0.7, annotation_text="Порог стресса", annotation_position="right")
fig_proxy.update_layout(
    template=PLOTLY_TEMPLATE,
    height=320,
    yaxis_title="Демо MAD-агрегат",
    hovermode="x unified",
    margin=dict(l=40, r=60, t=20, b=40),
)
st.plotly_chart(fig_proxy, use_container_width=True)

st.markdown("---")

# --- Flags timeline ---
st.subheader("Флаги событий по модулям")

flag_cols = {
    "m2_Flag_Demand": ("M2: Аномальный спрос РЕПО", COLORS["danger"]),
    "m3_Flag_Nedospros": ("M3: Недоспрос ОФЗ", COLORS["warn"]),
    "m3_Flag_Perespros": ("M3: Перес прос ОФЗ", COLORS["secondary"]),
    "m4_Tax_Day_Strict": ("M4: Налоговый день", COLORS["primary"]),
}

fig_flags = go.Figure()
for i, (col, (label, color)) in enumerate(flag_cols.items()):
    if col not in df.columns:
        continue
    active = df[df[col] == 1]
    if active.empty:
        continue
    fig_flags.add_trace(go.Scatter(
        x=active["date"],
        y=[i + 1] * len(active),
        mode="markers",
        marker=dict(symbol="line-ns", size=14, color=color,
                    line=dict(width=2, color=color)),
        name=label,
    ))

fig_flags.update_layout(
    template=PLOTLY_TEMPLATE,
    height=220,
    yaxis=dict(
        tickvals=list(range(1, len(flag_cols) + 1)),
        ticktext=[v[0] for v in flag_cols.values()],
        showgrid=False,
    ),
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    margin=dict(l=40, r=20, t=10, b=40),
)
st.plotly_chart(fig_flags, use_container_width=True)

# --- Export ---
with st.expander("Данные для экспорта"):
    export_cols = ["date"] + avail + ["_proxy_score"]
    for col in ["lsi_local", "lsi_global", "lsi"]:
        if col in df.columns:
            export_cols.append(col)
    export_cols = [c for c in export_cols if c in df.columns]
    export_df = df[export_cols].rename(
        columns={**mad_cols, "_proxy_score": "demo_mad_aggregate"}
    )
    st.dataframe(export_df.sort_values("date", ascending=False), use_container_width=True, hide_index=True)
    csv_download_button(export_df, "combined_signals.csv")
