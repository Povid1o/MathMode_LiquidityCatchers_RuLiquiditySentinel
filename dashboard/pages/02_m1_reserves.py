import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st
from dashboard.data.loader import load_m1
from dashboard.components.charts import line_chart, mad_score_bar, signal_line, dual_axis_chart
from dashboard.components.metrics import latest_value_metric, mad_status_metric, quick_period_filter, freshness_header, csv_download_button

st.set_page_config(page_title="M1 — Резервы", layout="wide")
st.title("M1 — Усреднение обязательных резервов")

st.markdown(
    "Модуль отслеживает выполнение банками требований по обязательным резервам. "
    "Ключевой сигнал — спред между фактическими и требуемыми резервами. "
    "Данные публикуются на конец каждого периода усреднения (~месяц)."
)

with st.spinner("Загрузка данных M1..."):
    df = load_m1()

freshness_header(df, "M1 — Резервы")
df = quick_period_filter(df, key="m1_period")

if df.empty:
    st.warning("Нет данных для выбранного периода.")
    st.stop()

# --- KPI row ---
st.subheader("Последние значения")
c1, c2, c3, c4 = st.columns(4)
with c1:
    latest_value_metric("Спред (млн руб.)", df["spread"], fmt="{:.1f}")
with c2:
    latest_value_metric("RUONIA (%)", df["ruonia_rate"], fmt="{:.2f}", suffix="%")
with c3:
    mad_status_metric("MAD-score спреда", df["spread_mad_score"])
with c4:
    reliable = df["m1_reliable"].iloc[-1] if "m1_reliable" in df.columns else None
    if reliable is not None:
        label = "✓ Достоверный" if reliable == 1 else "✗ Предварительный"
        color = "normal" if reliable == 1 else "inverse"
        st.metric("Статус сигнала", label)
    else:
        st.metric("Статус сигнала", "н/д")

st.markdown("---")

# --- Spread chart ---
st.subheader("Спред обязательных резервов")
fig_spread = line_chart(
    df, x="date",
    y=["spread", "spread_ma3"],
    labels={"spread": "Спред (млн руб.)", "spread_ma3": "MA-3"},
    title="Спред фактических и требуемых резервов",
    yaxis_title="млн руб.",
    height=380,
)
st.plotly_chart(fig_spread, use_container_width=True)

# --- RUONIA vs Key Rate (dual axis where key rate available) ---
st.subheader("Ставка RUONIA")
ruonia_df = df.dropna(subset=["ruonia_rate"])
if not ruonia_df.empty:
    fig_ruonia = line_chart(
        ruonia_df, x="date",
        y=["ruonia_rate", "ruonia_period_avg"],
        labels={"ruonia_rate": "RUONIA (%)", "ruonia_period_avg": "Среднее за период (%)"},
        title="Ставка RUONIA",
        yaxis_title="% годовых",
        height=340,
    )
    st.plotly_chart(fig_ruonia, use_container_width=True)
else:
    st.info("Данные по RUONIA отсутствуют в выбранном периоде.")

st.markdown("---")

# --- MAD scores ---
st.subheader("MAD-оценки аномальности")
tab1, tab2, tab3 = st.tabs(["Спред", "Относит. спред", "Нагрузка резервов"])

with tab1:
    fig_mad_spread = mad_score_bar(df, x="date", y="spread_mad_score", title="MAD-score спреда")
    st.plotly_chart(fig_mad_spread, use_container_width=True)

with tab2:
    fig_mad_rel = mad_score_bar(df, x="date", y="spread_relative_mad_score", title="MAD-score относительного спреда")
    st.plotly_chart(fig_mad_rel, use_container_width=True)

with tab3:
    fig_mad_load = mad_score_bar(df, x="date", y="reserve_load_mad_score", title="MAD-score нагрузки резервов")
    st.plotly_chart(fig_mad_load, use_container_width=True)

st.markdown("---")

# --- Signal ---
st.subheader("Итоговый сигнал M1")
st.caption(
    "m1_signal_final — нормализованный сигнал стресса по модулю. "
    "Пороги ±2 соответствуют зонам повышенного стресса."
)
signal_df = df.dropna(subset=["m1_signal_final"])
if not signal_df.empty:
    fig_signal = signal_line(
        signal_df, x="date", y="m1_signal_final",
        title="Сигнал M1 (m1_signal_final)",
        height=340,
    )
    st.plotly_chart(fig_signal, use_container_width=True)
else:
    st.info("Данные сигнала отсутствуют в выбранном периоде.")

# --- Raw table ---
with st.expander("Таблица данных M1"):
    cols_show = ["date", "spread", "spread_ma3", "ruonia_rate", "spread_mad_score",
                 "ruonia_mad_score", "m1_signal_final", "m1_reliable"]
    cols_show = [c for c in cols_show if c in df.columns]
    st.dataframe(df[cols_show].sort_values("date", ascending=False), use_container_width=True, hide_index=True)
    csv_download_button(df[cols_show], "m1_features.csv")
