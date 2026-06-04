import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dashboard.data.loader import load_m5, load_honest, load_module_contribution
from dashboard.components.charts import line_chart
from dashboard.components.honest import honest_driver_panel
from dashboard.components.metrics import (
    latest_value_metric, quick_period_filter, freshness_header, csv_download_button,
)
from dashboard.config import COLORS, PLOTLY_TEMPLATE

st.set_page_config(page_title="M5 — Ликвидность", layout="wide")
st.title("M5 — Ликвидность и операции ЦБ / Казначейства")

st.markdown(
    "Модуль охватывает баланс операций ЦБ с банками и средства Казначейства. В **honest-LSI** "
    "входят: **требования ЦБ** к банкам (`m5x_claims`), **обязательства ЦБ** (`m5x_liab`), "
    "постоянное РЕПО (`m5x_repostd`), обеспеченные кредиты (`m5x_secured`) — Global; "
    "и число заявителей Росказна (`m5x_rk_bidders`) — только Local. Рост требований ЦБ = "
    "банки активнее занимают у регулятора (классический признак дефицита ликвидности)."
)

df_native = load_m5()
df_honest = load_honest()

freshness_header(df_native, "M5 — Ликвидность")

df_h = quick_period_filter(df_honest, key="m5_period")
if df_h.empty:
    st.warning("Нет данных для выбранного периода.")
    st.stop()
cutoff = df_h["date"].min()
df_n = df_native[df_native["date"] >= cutoff].copy()

# --- KPI ---
st.subheader("Последние значения")
c1, c2, c3, c4 = st.columns(4)
with c1:
    latest_value_metric("Требования ЦБ к банкам", df_h["m5x_claims"], fmt="{:,.0f}")
with c2:
    latest_value_metric("Обязательства ЦБ", df_h["m5x_liab"], fmt="{:,.0f}")
with c3:
    latest_value_metric("Постоянное РЕПО", df_h["m5x_repostd"], fmt="{:,.0f}")
with c4:
    latest_value_metric("Заявители Росказна (Local)", df_h["m5x_rk_bidders"], fmt="{:.0f}")

# --- Live-вклад honest-фич в LSI ---
st.markdown("---")
st.subheader("Вклад M5 в текущий LSI")
honest_driver_panel(load_module_contribution("M5"), color=COLORS["primary"])

# --- Honest-драйверы (daily) ---
st.markdown("---")
st.subheader("Honest-признаки M5 (дневная шкала, вход в LSI)")
t1, t2, t3 = st.tabs(["Требования / обязательства ЦБ", "Standing facilities", "Заявители Росказна (Local)"])
with t1:
    st.plotly_chart(line_chart(df_h, x="date", y=["m5x_claims", "m5x_liab"], labels={"m5x_claims": "Требования ЦБ к банкам", "m5x_liab": "Обязательства ЦБ перед банками"}, title="Баланс операций ЦБ с банками", yaxis_title="млн руб.", height=340), use_container_width=True)
with t2:
    st.plotly_chart(line_chart(df_h, x="date", y=["m5x_repostd", "m5x_secured"], labels={"m5x_repostd": "Постоянное РЕПО", "m5x_secured": "Обеспеченные кредиты"}, title="Постоянные механизмы рефинансирования (standing)", yaxis_title="млн руб.", height=320), use_container_width=True)
with t3:
    rk = df_h.dropna(subset=["m5x_rk_bidders"])
    rk = rk[rk["m5x_rk_bidders"] > 0] if not rk.empty else rk
    if not rk.empty:
        st.plotly_chart(line_chart(rk, x="date", y=["m5x_rk_bidders"], labels={"m5x_rk_bidders": "Число заявителей Росказна"}, title="Активность заявителей на аукционах Росказны", yaxis_title="заявители", height=300), use_container_width=True)
    else:
        st.info("Данные по заявителям Росказны отсутствуют в окне (признак только для Local-модели).")

# --- Сырой контекст ---
st.markdown("---")
st.subheader("Сырой контекст: структурная позиция ликвидности")
st.caption("Положительные значения — профицит, отрицательные — дефицит (лаг 1 торговый день).")
liq_col = "liquidity_deficit_surplus_bln_rub_lag_1d"
liq_df = df_n.dropna(subset=[liq_col]) if liq_col in df_n else df_n.iloc[0:0]
if not liq_df.empty:
    colors = ["#22C55E" if v >= 0 else "#EF4444" for v in liq_df[liq_col]]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=liq_df["date"], y=liq_df[liq_col], marker=dict(color=colors), showlegend=False))
    fig.update_layout(title="Дефицит/профицит ликвидности банковского сектора (млрд руб.)",
                      template=PLOTLY_TEMPLATE, height=360, yaxis_title="млрд руб.",
                      hovermode="x unified", margin=dict(l=40, r=40, t=60, b=40))
    st.plotly_chart(fig, use_container_width=True)
bf = df_n.dropna(subset=["budget_funds_total_mln_rub_lagged"]) if "budget_funds_total_mln_rub_lagged" in df_n else df_n.iloc[0:0]
if not bf.empty:
    st.plotly_chart(line_chart(bf, x="date", y=["budget_funds_total_mln_rub_lagged"], labels={"budget_funds_total_mln_rub_lagged": "Бюджетные средства (млн руб.)"}, title="Бюджетные средства в банках", yaxis_title="млн руб.", height=300), use_container_width=True)

# --- Таблица ---
with st.expander("Таблица honest-признаков M5 (дневная шкала)"):
    cols = ["date", "m5x_claims", "m5x_liab", "m5x_repostd", "m5x_secured", "m5x_rk_bidders"]
    cols = [c for c in cols if c in df_h.columns]
    st.dataframe(df_h[cols].sort_values("date", ascending=False), use_container_width=True, hide_index=True)
    csv_download_button(df_h[cols], "m5_honest_features.csv")
