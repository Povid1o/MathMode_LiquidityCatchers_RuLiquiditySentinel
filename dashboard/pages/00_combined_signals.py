import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from dashboard.data.loader import load_final, load_lsi
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
    df = load_lsi()  # Uses load_lsi which calls load_final + computes LSI

freshness_header(df, "Final ML Dataset")
df = quick_period_filter(df, key="signals_period")

if df.empty:
    st.warning("Нет данных для выбранного периода.")
    st.stop()

# --- Compute proxy score ---
mad_cols = {
    "m1_spread_mad_score": "M1 Спред",
    "m2_MAD_score_cover": "M2 Cover",
    "m3_MAD_score_cover": "M3 Cover",
    "m4_MAD_tax_pressure": "M4 Давление",
}
avail = [c for c in mad_cols if c in df.columns]
df["_proxy_score"] = df[avail].abs().mean(axis=1)

# --- KPI row ---
st.subheader("Текущие сигналы")
cols = st.columns(len(avail) + 1)
for i, col in enumerate(avail):
    with cols[i]:
        val = df[col].dropna().iloc[-1] if not df[col].dropna().empty else None
        label = mad_cols[col]
        if val is not None:
            if abs(val) >= MAD_STRESS_THRESHOLD:
                level = "🔴 Стресс"
            elif abs(val) >= 1.0:
                level = "🟡 Умеренный"
            else:
                level = "🟢 Норма"
            st.metric(label, f"{val:.2f}", delta=level, delta_color="off")
        else:
            st.metric(label, "н/д")

with cols[-1]:
    proxy_val = df["_proxy_score"].dropna().iloc[-1] if not df["_proxy_score"].dropna().empty else None
    if proxy_val is not None:
        if proxy_val >= MAD_STRESS_THRESHOLD:
            level = "🔴 Стресс"
        elif proxy_val >= 1.0:
            level = "🟡 Умеренный"
        else:
            level = "🟢 Норма"
        st.metric("PROXY Score ⚠️", f"{proxy_val:.2f}", delta=level, delta_color="off")

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
if "lsi" in df.columns:
    st.subheader("LSI — Индекс стресса ликвидности")
    st.caption("Рассчитан IsolationForest на основе 98 признаков (М1–М5). Шкала 0-100: <40 норма, 40-70 внимание, ≥70 стресс.")

    lsi_df = df[["date", "lsi"]].dropna()

    fig_lsi = go.Figure()
    fig_lsi.add_trace(go.Scatter(
        x=lsi_df["date"],
        y=lsi_df["lsi"],
        mode="lines",
        fill="tozeroy",
        fillcolor="rgba(31,119,180,0.15)",
        line=dict(color=COLORS["primary"], width=2),
        name="LSI",
    ))
    fig_lsi.add_hline(y=70, line_dash="dash", line_color=COLORS["danger"],
                      opacity=0.7, annotation_text="Стресс", annotation_position="right")
    fig_lsi.add_hline(y=40, line_dash="dot", line_color=COLORS["warn"],
                      opacity=0.7, annotation_text="Повышенное внимание", annotation_position="left")

    fig_lsi.update_layout(
        template=PLOTLY_TEMPLATE,
        height=340,
        yaxis_title="LSI Score",
        hovermode="x unified",
        margin=dict(l=40, r=60, t=20, b=40),
    )
    st.plotly_chart(fig_lsi, use_container_width=True)
    st.markdown("---")

# --- Proxy score chart ---
st.subheader("PROXY Stress Score (DEMO)")
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
    name="PROXY Score",
))
fig_proxy.add_hline(y=MAD_STRESS_THRESHOLD, line_dash="dash", line_color=COLORS["danger"],
                    opacity=0.7, annotation_text="Порог стресса", annotation_position="right")
fig_proxy.update_layout(
    template=PLOTLY_TEMPLATE,
    height=320,
    yaxis_title="PROXY Score",
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
    if "lsi" in df.columns:
        export_cols.append("lsi")
    export_cols = [c for c in export_cols if c in df.columns]
    export_df = df[export_cols].rename(columns={**mad_cols, "_proxy_score": "PROXY_score"})
    st.dataframe(export_df.sort_values("date", ascending=False), use_container_width=True, hide_index=True)
    csv_download_button(export_df, "combined_signals.csv")
