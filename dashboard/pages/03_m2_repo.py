import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st
from dashboard.data.loader import load_m2
from dashboard.components.charts import event_scatter, mad_score_bar, flag_timeline
from dashboard.components.metrics import latest_value_metric, mad_status_metric, date_range_filter

st.set_page_config(page_title="M2 — Репо ЦБ", layout="wide")
st.title("M2 — Аукционы РЕПО Банка России")

st.markdown(
    "Модуль анализирует 7-дневные аукционы РЕПО ЦБ. "
    "Данные **разреженные** (только в дни проведения аукционов). "
    "Ключевые сигналы: коэффициент покрытия и спред к ключевой ставке."
)

with st.spinner("Загрузка данных M2..."):
    df = load_m2()

df = date_range_filter(df, key="m2_date")

if df.empty:
    st.warning("Нет данных для выбранного периода.")
    st.stop()

# --- KPI ---
st.subheader("Последние значения")
c1, c2, c3, c4 = st.columns(4)
with c1:
    latest_value_metric("Cover ratio (последний аукцион)", df["cover_ratio"], fmt="{:.2f}")
with c2:
    latest_value_metric("Rate spread (%)", df["rate_spread"], fmt="{:.3f}", suffix="%")
with c3:
    mad_status_metric("MAD-score cover", df["MAD_score_cover"])
with c4:
    flag_count = int(df["Flag_Demand"].sum()) if "Flag_Demand" in df.columns else 0
    total = len(df)
    st.metric("Аукционов с флагом спроса", f"{flag_count} / {total}", delta=f"{flag_count/total*100:.1f}%")

st.markdown("---")

# --- Cover ratio scatter ---
st.subheader("Коэффициент покрытия (Cover Ratio)")
st.caption("Отношение спроса к объёму предложения. Значения > 1 — спрос превышает предложение.")

fig_cover = event_scatter(
    df, x="date", y="cover_ratio",
    flag_col="Flag_Demand",
    flag_label="Аномальный спрос",
    title="Cover Ratio по аукционам РЕПО",
    yaxis_title="Cover Ratio",
    height=360,
)
st.plotly_chart(fig_cover, use_container_width=True)

# --- Rate spread scatter ---
st.subheader("Спред ставки аукциона к ключевой ставке")
rate_df = df.dropna(subset=["rate_spread"])
if not rate_df.empty:
    fig_rate = event_scatter(
        rate_df, x="date", y="rate_spread",
        flag_col="Flag_Demand",
        flag_label="Аномальный спрос",
        title="Rate Spread (аукционная ставка − ключевая ставка)",
        yaxis_title="Спред, п.п.",
        height=340,
    )
    st.plotly_chart(fig_rate, use_container_width=True)
else:
    st.info("Данные по спреду ставки отсутствуют в выбранном периоде.")

st.markdown("---")

# --- MAD scores ---
st.subheader("MAD-оценки аномальности")
tab1, tab2 = st.tabs(["MAD Cover Ratio", "MAD Rate Spread"])

with tab1:
    fig_mad_cover = mad_score_bar(df, x="date", y="MAD_score_cover", title="MAD-score покрытия")
    st.plotly_chart(fig_mad_cover, use_container_width=True)

with tab2:
    rate_mad_df = df.dropna(subset=["MAD_score_rate_spread"])
    if not rate_mad_df.empty:
        fig_mad_rate = mad_score_bar(rate_mad_df, x="date", y="MAD_score_rate_spread", title="MAD-score спреда ставки")
        st.plotly_chart(fig_mad_rate, use_container_width=True)
    else:
        st.info("MAD-score по спреду ставки недоступен для выбранного периода.")

st.markdown("---")

# --- Flag timeline ---
st.subheader("Флаги аномального спроса")
st.caption("Каждая отметка — аукцион с зафиксированным сигналом аномалии спроса.")
fig_flags = flag_timeline(
    df, x="date",
    flags={"Flag_Demand": "Аномальный спрос (Flag_Demand)"},
    title="Аукционы РЕПО: флаги",
    height=180,
)
st.plotly_chart(fig_flags, use_container_width=True)

# --- Raw table ---
with st.expander("Таблица данных M2"):
    cols_show = ["date", "auction_type", "term_days", "cover_ratio", "rate_spread",
                 "key_rate", "Flag_Demand", "MAD_score_cover", "MAD_score_rate_spread"]
    cols_show = [c for c in cols_show if c in df.columns]
    st.dataframe(df[cols_show].sort_values("date", ascending=False), use_container_width=True, hide_index=True)
