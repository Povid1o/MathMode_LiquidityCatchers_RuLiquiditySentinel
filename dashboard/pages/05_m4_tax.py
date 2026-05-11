import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from dashboard.data.loader import load_m4
from dashboard.components.charts import line_chart, mad_score_bar
from dashboard.components.metrics import latest_value_metric, quick_period_filter, freshness_header, csv_download_button
from dashboard.config import COLORS, PLOTLY_TEMPLATE

st.set_page_config(page_title="M4 — Налоги", layout="wide")
st.title("M4 — Налоговый календарь")

st.markdown(
    "Модуль описывает налоговую сезонность и давление на ликвидность. "
    "Данные **ежедневные** и детерминированные (основаны на налоговом календаре). "
    "**Внимание:** M4 — контекстный модуль, а не самостоятельный сигнал стресса."
)

with st.spinner("Загрузка данных M4..."):
    df = load_m4()

freshness_header(df, "M4 — Налоги")
df = quick_period_filter(df, key="m4_period")

if df.empty:
    st.warning("Нет данных для выбранного периода.")
    st.stop()

# --- KPI ---
st.subheader("Текущий статус")
latest = df.iloc[-1]
c1, c2, c3, c4 = st.columns(4)
with c1:
    tax_week = bool(latest.get("Tax_Week_Flag", 0))
    st.metric("Налоговая неделя", "ДА" if tax_week else "НЕТ")
with c2:
    tax_day = bool(latest.get("Tax_Day_Strict", 0))
    st.metric("Налоговый день (строгий)", "ДА" if tax_day else "НЕТ")
with c3:
    latest_value_metric("Налоговое давление", df["tax_pressure"], fmt="{:.3f}")
with c4:
    latest_value_metric("Налоговая близость", df["tax_proximity"], fmt="{:.3f}")

st.markdown("---")

# --- Tax pressure ---
st.subheader("Налоговое давление")
st.caption(
    "tax_pressure — агрегированный показатель налоговой нагрузки на ликвидность. "
    "tax_proximity — расстояние до ближайшего налогового дня."
)
fig_pressure = line_chart(
    df, x="date",
    y=["tax_pressure", "tax_pressure_smoothed"],
    labels={"tax_pressure": "Давление (raw)", "tax_pressure_smoothed": "Давление (smooth)"},
    title="Налоговое давление",
    yaxis_title="Индекс давления",
    height=360,
)
# Add Tax_Day_Strict markers
tax_days = df[df["Tax_Day_Strict"] == 1]
fig_pressure.add_trace(go.Scatter(
    x=tax_days["date"],
    y=tax_days["tax_pressure"],
    mode="markers",
    marker=dict(color=COLORS["danger"], size=6, symbol="x"),
    name="Налоговый день",
))
st.plotly_chart(fig_pressure, use_container_width=True)

# --- Tax proximity ---
st.subheader("Близость к налоговому дню")
fig_prox = line_chart(
    df, x="date",
    y=["tax_proximity"],
    labels={"tax_proximity": "Налоговая близость"},
    title="Proximity — расстояние до ближайшего налогового дня",
    yaxis_title="Proximity",
    height=300,
)
st.plotly_chart(fig_prox, use_container_width=True)

st.markdown("---")

# --- MAD scores ---
st.subheader("MAD-оценки налогового давления")
tab1, tab2 = st.tabs(["MAD Tax Pressure", "MAD Tax Proximity"])

with tab1:
    fig_mad_tp = mad_score_bar(df, x="date", y="MAD_tax_pressure", title="MAD-score налогового давления")
    st.plotly_chart(fig_mad_tp, use_container_width=True)

with tab2:
    fig_mad_tpr = mad_score_bar(df, x="date", y="MAD_tax_proximity", title="MAD-score налоговой близости")
    st.plotly_chart(fig_mad_tpr, use_container_width=True)

st.markdown("---")

# --- Seasonal factor ---
st.subheader("Сезонный фактор")
st.caption(
    "Seasonal_Factor_raw — контекстный показатель сезонности. "
    "Отражает исторические паттерны, а не стресс напрямую."
)
fig_seasonal = line_chart(
    df, x="date",
    y=["Seasonal_Factor_raw"],
    labels={"Seasonal_Factor_raw": "Сезонный фактор (raw)"},
    title="Сезонный фактор",
    yaxis_title="Seasonal Factor",
    height=300,
)
st.plotly_chart(fig_seasonal, use_container_width=True)

# --- Calendar heatmap ---
st.subheader("Тепловая карта налоговых дней")
st.caption("Количество налоговых дней (Tax_Day_Strict) по месяцам и годам.")

df_cal = df[df["Tax_Day_Strict"] == 1].copy()
df_cal["year"] = df_cal["date"].dt.year
df_cal["month"] = df_cal["date"].dt.month

pivot = df_cal.groupby(["year", "month"]).size().reset_index(name="count")
pivot_wide = pivot.pivot(index="year", columns="month", values="count").fillna(0)

month_names = ["Янв", "Фев", "Мар", "Апр", "Май", "Июн",
               "Июл", "Авг", "Сен", "Окт", "Ноя", "Дек"]

fig_heat = go.Figure(go.Heatmap(
    z=pivot_wide.values,
    x=[month_names[m - 1] for m in pivot_wide.columns],
    y=pivot_wide.index.astype(str),
    colorscale="Oranges",
    hovertemplate="Год: %{y}<br>Месяц: %{x}<br>Налоговых дней: %{z}<extra></extra>",
))
fig_heat.update_layout(
    title="Налоговые дни (Tax_Day_Strict) по месяцам",
    template=PLOTLY_TEMPLATE,
    height=400,
    margin=dict(l=40, r=20, t=40, b=40),
    xaxis_title="Месяц",
    yaxis_title="Год",
)
st.plotly_chart(fig_heat, use_container_width=True)

# --- Raw table ---
with st.expander("Таблица данных M4"):
    cols_show = ["date", "Tax_Week_Flag", "Tax_Day_Strict", "Tax_Pre_Flag", "Tax_Active_Flag",
                 "tax_pressure", "tax_pressure_smoothed", "tax_proximity",
                 "MAD_tax_pressure", "MAD_tax_proximity", "Seasonal_Factor_raw"]
    cols_show = [c for c in cols_show if c in df.columns]
    st.dataframe(df[cols_show].sort_values("date", ascending=False), use_container_width=True, hide_index=True)
    csv_download_button(df[cols_show], "m4_features.csv")
