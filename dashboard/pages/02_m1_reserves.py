import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st
from dashboard.data.loader import load_m1, load_honest, load_module_contribution
from dashboard.components.charts import line_chart, mad_score_bar
from dashboard.components.honest import honest_driver_panel
from dashboard.components.metrics import (
    latest_value_metric, mad_status_metric, quick_period_filter,
    freshness_header, csv_download_button,
)
from dashboard.config import COLORS

st.set_page_config(page_title="M1 — Резервы", layout="wide")
st.title("M1 — Усреднение обязательных резервов")

st.markdown(
    "Модуль отслеживает выполнение банками требований по обязательным резервам. "
    "В **honest-LSI** модуль входит пятью признаками: аномальность спреда (MAD), "
    "относительного спреда, нагрузки резервов, RUONIA и **волатильности спреда** "
    "(`spread_vol`). Старые `m1_signal_final` / `flag_end_of_period` в индекс **не входят**."
)

# Native (period-level) + honest (daily, как видит индекс)
df_native = load_m1()
df_honest = load_honest()

freshness_header(df_native, "M1 — Резервы (периоды усреднения)")

# Один фильтр периода (по дневной honest-шкале); тем же окном режем native
df_h = quick_period_filter(df_honest, key="m1_period")
if df_h.empty:
    st.warning("Нет данных для выбранного периода.")
    st.stop()
cutoff = df_h["date"].min()
df_n = df_native[df_native["date"] >= cutoff].copy()

# --- KPI ---
st.subheader("Последние значения")
c1, c2, c3, c4 = st.columns(4)
with c1:
    latest_value_metric("Спред (млн руб.)", df_n["spread"] if "spread" in df_n else df_h["m1_spread"], fmt="{:.1f}")
with c2:
    latest_value_metric("RUONIA (%)", df_n["ruonia_rate"] if "ruonia_rate" in df_n else df_h["m1_ruonia_rate"], fmt="{:.2f}", suffix="%")
with c3:
    mad_status_metric("MAD спреда (LSI-вход)", df_h["m1_spread_mad_score"])
with c4:
    mad_status_metric("Волатильность спреда", df_h["m1_spread_vol"])

# --- Live-вклад honest-фич в LSI ---
st.markdown("---")
st.subheader("Вклад M1 в текущий LSI")
honest_driver_panel(load_module_contribution("M1"), color=COLORS["primary"])

# --- Honest-драйверы (daily, как их потребляет индекс) ---
st.markdown("---")
st.subheader("Honest-признаки M1 (дневная шкала, вход в LSI)")
st.caption(
    "Признаки показаны на дневной шкале — именно так их видит индекс "
    "(месячные периоды усреднения forward-fill до дневной частоты)."
)
t1, t2, t3, t4, t5 = st.tabs([
    "Спред (MAD)", "Относит. спред (MAD)", "Нагрузка (MAD)", "RUONIA (MAD)", "Волатильность",
])
with t1:
    st.plotly_chart(mad_score_bar(df_h, x="date", y="m1_spread_mad_score", title="Аномальность спреда резервов"), use_container_width=True)
with t2:
    st.plotly_chart(mad_score_bar(df_h, x="date", y="m1_spread_relative_mad_score", title="Аномальность относительного спреда"), use_container_width=True)
with t3:
    st.plotly_chart(mad_score_bar(df_h, x="date", y="m1_reserve_load_mad_score", title="Аномальность нагрузки резервов"), use_container_width=True)
with t4:
    st.plotly_chart(mad_score_bar(df_h, x="date", y="m1_ruonia_mad_score", title="Аномальность ставки RUONIA"), use_container_width=True)
with t5:
    st.plotly_chart(line_chart(df_h, x="date", y=["m1_spread_vol"], labels={"m1_spread_vol": "Волатильность спреда |Δ| (MAD)"}, title="Волатильность спреда резервов", yaxis_title="MAD", height=320), use_container_width=True)

# --- Сырой контекст (нативная гранулярность периодов) ---
st.markdown("---")
st.subheader("Сырой контекст (периоды усреднения)")
if not df_n.empty:
    st.plotly_chart(
        line_chart(df_n, x="date", y=["spread", "spread_ma3"],
                   labels={"spread": "Спред (млн руб.)", "spread_ma3": "MA-3"},
                   title="Спред фактических и требуемых резервов", yaxis_title="млн руб.", height=360),
        use_container_width=True,
    )
    ruonia_df = df_n.dropna(subset=["ruonia_rate"]) if "ruonia_rate" in df_n else df_n.iloc[0:0]
    if not ruonia_df.empty:
        st.plotly_chart(
            line_chart(ruonia_df, x="date", y=["ruonia_rate", "ruonia_period_avg"],
                       labels={"ruonia_rate": "RUONIA (%)", "ruonia_period_avg": "Среднее за период (%)"},
                       title="Ставка RUONIA", yaxis_title="% годовых", height=320),
            use_container_width=True,
        )
else:
    st.info("Нет нативных данных периодов в выбранном окне.")

# --- Таблица ---
with st.expander("Таблица honest-признаков M1 (дневная шкала)"):
    cols = ["date", "m1_spread", "m1_ruonia_rate", "m1_spread_mad_score",
            "m1_spread_relative_mad_score", "m1_reserve_load_mad_score",
            "m1_ruonia_mad_score", "m1_spread_vol"]
    cols = [c for c in cols if c in df_h.columns]
    st.dataframe(df_h[cols].sort_values("date", ascending=False), use_container_width=True, hide_index=True)
    csv_download_button(df_h[cols], "m1_honest_features.csv")
