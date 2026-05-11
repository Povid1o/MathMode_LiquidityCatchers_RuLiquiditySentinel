import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st
import plotly.graph_objects as go
from dashboard.data.loader import load_m5
from dashboard.components.charts import line_chart, bar_chart
from dashboard.components.metrics import latest_value_metric, quick_period_filter, freshness_header, csv_download_button
from dashboard.config import COLORS, PLOTLY_TEMPLATE

st.set_page_config(page_title="M5 — Ликвидность", layout="wide")
st.title("M5 — Ликвидность и средства Казначейства")

st.markdown(
    "Модуль охватывает структурную позицию ликвидности банковского сектора, "
    "бюджетные средства в банках и операции Казначейства (Росказна). "
    "Данные **ежедневные**, ряд показателей — с лагом 1 день."
)

with st.spinner("Загрузка данных M5..."):
    df = load_m5()

freshness_header(df, "M5 — Ликвидность")
df = quick_period_filter(df, key="m5_period")

if df.empty:
    st.warning("Нет данных для выбранного периода.")
    st.stop()

# --- KPI ---
st.subheader("Последние значения")
c1, c2, c3, c4 = st.columns(4)
with c1:
    latest_value_metric(
        "Дефицит/профицит (млрд руб.)",
        df["liquidity_deficit_surplus_bln_rub_lag_1d"],
        fmt="{:.1f}",
    )
with c2:
    latest_value_metric(
        "Бюджетные средства, всего (млн руб.)",
        df["budget_funds_total_mln_rub_lagged"],
        fmt="{:,.0f}",
    )
with c3:
    latest_value_metric(
        "Чистый поток Росказны, 7д (млн руб.)",
        df["roskazna_net_flow_rolling_7d_mln_rub"],
        fmt="{:,.0f}",
    )
with c4:
    latest_value_metric(
        "Доля руб. бюдж. средств",
        df["budget_funds_rub_share_lagged"],
        fmt="{:.1%}",
    )

st.markdown("---")

# --- Liquidity deficit/surplus ---
st.subheader("Структурная позиция ликвидности")
st.caption(
    "Положительные значения — профицит ликвидности, отрицательные — дефицит. "
    "Данные с лагом 1 торговый день."
)

liq_col = "liquidity_deficit_surplus_bln_rub_lag_1d"
liq_df = df.dropna(subset=[liq_col])
if not liq_df.empty:
    fig_liq = go.Figure()
    positive = liq_df[liq_df[liq_col] >= 0]
    negative = liq_df[liq_df[liq_col] < 0]
    fig_liq.add_trace(go.Bar(
        x=positive["date"], y=positive[liq_col],
        name="Профицит", marker_color=COLORS["success"],
    ))
    fig_liq.add_trace(go.Bar(
        x=negative["date"], y=negative[liq_col],
        name="Дефицит", marker_color=COLORS["danger"],
    ))
    fig_liq.add_trace(go.Scatter(
        x=liq_df["date"], y=liq_df["liquidity_deficit_surplus_bln_rub_change_5d"],
        name="Изменение за 5д", mode="lines",
        line=dict(color=COLORS["secondary"]),
        yaxis="y2",
    ))
    fig_liq.update_layout(
        title="Дефицит/профицит ликвидности банковского сектора (млрд руб.)",
        template=PLOTLY_TEMPLATE,
        height=400,
        barmode="overlay",
        hovermode="x unified",
        yaxis_title="млрд руб.",
        yaxis2=dict(title="Изменение (5д), млрд руб.", overlaying="y", side="right"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=40, r=60, t=40, b=40),
    )
    st.plotly_chart(fig_liq, use_container_width=True)
else:
    st.info("Данные по ликвидности отсутствуют.")

st.markdown("---")

# --- Budget funds ---
st.subheader("Бюджетные средства в банках")
st.caption("Средства, размещённые Минфином/бюджетом в банковской системе (с лагом на дату публикации).")

tab1, tab2 = st.tabs(["Общий объём", "Доля рублёвых средств"])

with tab1:
    bfunds_df = df.dropna(subset=["budget_funds_total_mln_rub_lagged"])
    if not bfunds_df.empty:
        fig_bf = line_chart(
            bfunds_df, x="date",
            y=["budget_funds_total_mln_rub_lagged"],
            labels={"budget_funds_total_mln_rub_lagged": "Бюджетные средства (млн руб.)"},
            title="Бюджетные средства в банках",
            yaxis_title="млн руб.",
            height=320,
        )
        st.plotly_chart(fig_bf, use_container_width=True)
    else:
        st.info("Данные по бюджетным средствам отсутствуют.")

with tab2:
    share_df = df.dropna(subset=["budget_funds_rub_share_lagged"])
    if not share_df.empty:
        fig_share = line_chart(
            share_df, x="date",
            y=["budget_funds_rub_share_lagged"],
            labels={"budget_funds_rub_share_lagged": "Доля руб. средств"},
            title="Доля рублёвых бюджетных средств",
            yaxis_title="Доля (0–1)",
            height=280,
        )
        st.plotly_chart(fig_share, use_container_width=True)
    else:
        st.info("Данные по доле рублёвых средств отсутствуют.")

st.markdown("---")

# --- Roskazna flows ---
st.subheader("Операции Казначейства (Росказна)")

tab_a, tab_b, tab_c = st.tabs(["Чистые потоки (rolling)", "Первые / вторые ноги", "Аукционная активность"])

with tab_a:
    roll_cols = [
        "roskazna_net_flow_rolling_7d_mln_rub",
        "roskazna_net_flow_rolling_14d_mln_rub",
        "roskazna_net_flow_rolling_30d_mln_rub",
    ]
    roll_labels = {
        "roskazna_net_flow_rolling_7d_mln_rub": "Net flow 7д",
        "roskazna_net_flow_rolling_14d_mln_rub": "Net flow 14д",
        "roskazna_net_flow_rolling_30d_mln_rub": "Net flow 30д",
    }
    roll_df = df.dropna(subset=roll_cols, how="all")
    avail_cols = [c for c in roll_cols if c in roll_df.columns and roll_df[c].notna().any()]
    if avail_cols:
        horizon = st.selectbox(
            "Горизонт rolling",
            options=avail_cols,
            format_func=lambda c: roll_labels.get(c, c),
            key="m5_rolling_horizon",
        )
        fig_roll = line_chart(
            roll_df, x="date", y=[horizon],
            labels={horizon: roll_labels.get(horizon, horizon)},
            title=f"Чистый поток Росказны ({roll_labels.get(horizon, horizon)})",
            yaxis_title="млн руб.",
            height=320,
        )
        st.plotly_chart(fig_roll, use_container_width=True)
    else:
        st.info("Данные по rolling net flow отсутствуют.")

with tab_b:
    leg_cols = {
        "roskazna_first_leg_settled_volume_mln_rub": "Первая нога (размещение)",
        "roskazna_second_leg_return_volume_mln_rub": "Вторая нога (возврат)",
        "roskazna_net_flow_by_legs_mln_rub": "Чистый поток (ноги)",
    }
    avail_leg = [c for c in leg_cols if c in df.columns]
    if avail_leg:
        fig_legs = line_chart(
            df.dropna(subset=avail_leg, how="all"),
            x="date", y=avail_leg,
            labels=leg_cols,
            title="Объёмы размещения и возврата Росказны",
            yaxis_title="млн руб.",
            height=340,
        )
        st.plotly_chart(fig_legs, use_container_width=True)
    else:
        st.info("Данные по ногам отсутствуют.")

with tab_c:
    demand_col = "roskazna_demand_volume_mln_rub_lag_1d"
    auction_df = df[df.get("roskazna_auction_day_flag_lag_1d", df["date"].notna()) == 1].copy() \
        if "roskazna_auction_day_flag_lag_1d" in df.columns else df.dropna(subset=[demand_col])

    if not auction_df.empty and demand_col in auction_df.columns:
        fig_demand = bar_chart(
            auction_df.dropna(subset=[demand_col]),
            x="date", y=demand_col,
            title="Спрос на аукционах Росказны (млн руб.)",
            yaxis_title="млн руб.",
            height=300,
        )
        st.plotly_chart(fig_demand, use_container_width=True)
    else:
        st.info("Данные по спросу на аукционах Росказны отсутствуют.")

st.markdown("---")

# --- Days since last Roskazna auction ---
days_col = "days_since_last_roskazna_auction"
if days_col in df.columns:
    st.subheader("Активность аукционов Росказны")
    st.caption("Дней с последнего аукциона Росказны.")
    fig_days = line_chart(
        df.dropna(subset=[days_col]), x="date", y=[days_col],
        labels={days_col: "Дней с последнего аукциона"},
        title="Дней с последнего аукциона Росказны",
        yaxis_title="Дней",
        height=260,
    )
    st.plotly_chart(fig_days, use_container_width=True)

# --- Raw table ---
with st.expander("Таблица данных M5"):
    cols_show = [
        "date",
        "liquidity_deficit_surplus_bln_rub_lag_1d",
        "budget_funds_total_mln_rub_lagged",
        "budget_funds_rub_share_lagged",
        "roskazna_net_flow_by_legs_mln_rub",
        "roskazna_net_flow_rolling_7d_mln_rub",
        "roskazna_net_flow_rolling_30d_mln_rub",
        "days_since_last_roskazna_auction",
    ]
    cols_show = [c for c in cols_show if c in df.columns]
    st.dataframe(df[cols_show].sort_values("date", ascending=False), use_container_width=True, hide_index=True)
    csv_download_button(df[cols_show], "m5_features.csv")
