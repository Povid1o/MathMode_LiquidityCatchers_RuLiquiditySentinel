import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st
from dashboard.data.loader import load_m2, load_honest, load_module_contribution
from dashboard.components.charts import event_scatter, flag_timeline, line_chart, mad_score_bar
from dashboard.components.honest import honest_driver_panel
from dashboard.components.metrics import (
    latest_value_metric, mad_status_metric, quick_period_filter,
    freshness_header, csv_download_button,
)
from dashboard.config import COLORS

st.set_page_config(page_title="M2 — Репо ЦБ", layout="wide")
st.title("M2 — Аукционы РЕПО Банка России")

st.markdown(
    "Модуль анализирует аукционы РЕПО ЦБ. В **honest-LSI** входят: факт аукциона, "
    "флаг спроса, аномальность переподписки (`base_cover_mad`), спред отсечения к "
    "ключевой ставке (`cutoff_spread`), активность short-РЕПО и «дней с последнего short». "
    "Старые `MAD_score_rate_spread` / сырые `cover_ratio` в индекс **не входят** — они контекст."
)

df_native = load_m2()
df_honest = load_honest()

freshness_header(df_native, "M2 — Репо ЦБ (события аукционов)")

df_h = quick_period_filter(df_honest, key="m2_period")
if df_h.empty:
    st.warning("Нет данных для выбранного периода.")
    st.stop()
cutoff = df_h["date"].min()
df_n = df_native[df_native["date"] >= cutoff].copy()

# --- KPI ---
st.subheader("Последние значения")
c1, c2, c3, c4 = st.columns(4)
with c1:
    mad_status_metric("MAD переподписки (LSI-вход)", df_h["m2_base_cover_mad"])
with c2:
    latest_value_metric("Спред отсечения (п.п.)", df_h["m2_cutoff_spread"], fmt="{:.3f}")
with c3:
    latest_value_metric("Дней с последнего short-РЕПО", df_h["m2_days_since_short"], fmt="{:.0f}")
with c4:
    active = int(df_h["m2_short_active30"].iloc[-1]) if "m2_short_active30" in df_h else 0
    st.metric("Short-РЕПО активен (30д)", "ДА" if active else "НЕТ")

# --- Live-вклад honest-фич в LSI ---
st.markdown("---")
st.subheader("Вклад M2 в текущий LSI")
honest_driver_panel(load_module_contribution("M2"), color=COLORS["primary"])

# --- Honest-драйверы (daily) ---
st.markdown("---")
st.subheader("Honest-признаки M2 (дневная шкала, вход в LSI)")
st.caption("Сырьё разрежено по дням аукционов; honest-признаки доведены до дневной шкалы — так их видит индекс.")
t1, t2, t3 = st.tabs(["Переподписка (MAD)", "Спред отсечения", "Short-РЕПО"])
with t1:
    st.plotly_chart(mad_score_bar(df_h, x="date", y="m2_base_cover_mad", title="Аномальность переподписки РЕПО"), use_container_width=True)
with t2:
    st.plotly_chart(line_chart(df_h, x="date", y=["m2_cutoff_spread"], labels={"m2_cutoff_spread": "Спред отсечения к ключевой (п.п.)"}, title="Спред отсечения аукционов РЕПО", yaxis_title="п.п.", height=320), use_container_width=True)
with t3:
    st.plotly_chart(line_chart(df_h, x="date", y=["m2_short_active30", "m2_days_since_short"], labels={"m2_short_active30": "Short активен (30д)", "m2_days_since_short": "Дней с последнего short"}, title="Активность short-РЕПО", yaxis_title="флаг / дни", height=320), use_container_width=True)

# --- Сырой контекст (события аукционов) ---
st.markdown("---")
st.subheader("Сырой контекст (события аукционов)")
if not df_n.empty:
    st.plotly_chart(
        event_scatter(df_n, x="date", y="cover_ratio", flag_col="Flag_Demand",
                      flag_label="Аномальный спрос", title="Cover Ratio по аукционам РЕПО",
                      yaxis_title="Cover Ratio", height=340),
        use_container_width=True,
    )
    rate_df = df_n.dropna(subset=["rate_spread"]) if "rate_spread" in df_n else df_n.iloc[0:0]
    if not rate_df.empty:
        st.plotly_chart(
            event_scatter(rate_df, x="date", y="rate_spread", flag_col="Flag_Demand",
                          flag_label="Аномальный спрос", title="Спред ставки аукциона к ключевой",
                          yaxis_title="п.п.", height=320),
            use_container_width=True,
        )
    st.plotly_chart(
        flag_timeline(df_n, x="date", flags={"Flag_Demand": "Аномальный спрос"},
                      title="Аукционы РЕПО: флаги спроса", height=170),
        use_container_width=True,
    )
else:
    st.info("Нет событий аукционов в выбранном окне.")

# --- Таблица ---
with st.expander("Таблица honest-признаков M2 (дневная шкала)"):
    cols = ["date", "m2_auction_flag", "m2_Flag_Demand", "m2_base_cover_mad",
            "m2_cutoff_spread", "m2_cutoff_spread_available", "m2_short_active30", "m2_days_since_short"]
    cols = [c for c in cols if c in df_h.columns]
    st.dataframe(df_h[cols].sort_values("date", ascending=False), use_container_width=True, hide_index=True)
    csv_download_button(df_h[cols], "m2_honest_features.csv")
