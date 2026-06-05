import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st
from dashboard.data.loader import load_m3, load_honest, load_module_contribution
from dashboard.components.charts import line_chart, flag_timeline, event_scatter
from dashboard.components.honest import honest_driver_panel
from dashboard.components.metrics import (
    latest_value_metric, quick_period_filter, freshness_header, csv_download_button,
)
from dashboard.config import COLORS

st.set_page_config(page_title="M3 — ОФЗ", layout="wide")
st.title("M3 — Размещение ОФЗ")

st.markdown(
    "Модуль анализирует первичные аукционы ОФЗ Минфина. В **honest-LSI** входят "
    "**event-aware** признаки: переподписка (`m3x_cover`), доля размещения (`m3x_placement`), "
    "премия доходности к ключевой (`m3x_yield_to_key`), возраст/наличие аукциона, «дней с "
    "последнего» и признак несостоявшегося аукциона, плюс флаги недоспроса. Старые "
    "`MAD_score_*` в индекс **не входят**. M3 — крупнейший по вкладу канал (≈30% в среднем)."
)

df_native = load_m3()
df_honest = load_honest()

freshness_header(df_native, "M3 — ОФЗ (события аукционов)")

df_h = quick_period_filter(df_honest, key="m3_period")
if df_h.empty:
    st.warning("Нет данных для выбранного периода.")
    st.stop()
cutoff = df_h["date"].min()
df_n = df_native[df_native["date"] >= cutoff].copy()

# --- KPI (сырые значения последнего аукциона, не нормализованные) ---
st.subheader("Последние значения (по последнему аукциону)")
c1, c2, c3, c4 = st.columns(4)
with c1:
    latest_value_metric("Переподписка (cover ratio)", df_n["cover_ratio"], fmt="{:.2f}")
with c2:
    latest_value_metric("Средневзв. доходность ОФЗ-ПД, %", df_n["weighted_yield"], fmt="{:.2f}", suffix="%")
with c3:
    latest_value_metric("Дней с последнего аукциона", df_h["m3x_days_since"], fmt="{:.0f}")
with c4:
    failed = int(df_h["m3x_failed"].iloc[-1]) if "m3x_failed" in df_h else 0
    st.metric("Последний аукцион несостоялся", "ДА" if failed else "НЕТ")

# --- Live-вклад honest-фич в LSI ---
st.markdown("---")
st.subheader("Вклад M3 в текущий LSI")
honest_driver_panel(load_module_contribution("M3"), color=COLORS["primary"])

# --- Honest-драйверы (daily) ---
st.markdown("---")
st.subheader("Honest-признаки M3 (дневная шкала, вход в LSI)")
st.caption(
    "⚠️ Это **нормализованные MAD-аномалии (z-оценки)**, а НЕ сырые значения. "
    "Шкала безразмерная: 0 — норма, выше — отклонение в сторону стресса (для cover/placement "
    "знак инвертирован: рост = недоспрос/недоразмещение). Сырые значения (cover ratio, "
    "доходность в %) — ниже, в блоке «Сырой контекст»."
)
with st.expander("Что это за признаки M3? (расшифровка)"):
    st.markdown(
        "- **Аномалия переподписки** (`m3x_cover`) — отклонение спроса/предложения от нормы; рост = недоспрос.\n"
        "- **Аномалия размещения** (`m3x_placement`) — отклонение доли размещённого объёма; рост = Минфин не смог продать выпуск.\n"
        "- **Аномалия премии доходности** (`m3x_yield_to_key`) — отклонение премии «доходность ОФЗ-ПД − ключевая ставка»; рост = вынужденная премия за спрос.\n"
        "- **Возраст / дней с последнего** (`m3x_age`, `m3x_days_since`) — свежесть аукционных данных, в днях (с ограничением сверху).\n\n"
        "Все аномалии считаются по событиям (event-time MAD за ~3 года) и доводятся до дневной шкалы. "
        "Доходность и премия — только по номинальным **ОФЗ-ПД** (флоатеры ОФЗ-ПК и инфляционные ОФЗ-ИН исключены как несопоставимые)."
    )
t1, t2, t3 = st.tabs(["Аномалия спроса/размещения (MAD)", "Аномалия премии доходности (MAD)", "Возраст / дней с последнего (дни)"])
with t1:
    st.plotly_chart(line_chart(df_h, x="date", y=["m3x_cover", "m3x_placement"], labels={"m3x_cover": "Аномалия переподписки", "m3x_placement": "Аномалия размещения"}, title="Аномалии спроса и размещения ОФЗ (event-time MAD)", yaxis_title="MAD-оценка (z); вверх = стресс", height=320), use_container_width=True)
with t2:
    st.plotly_chart(line_chart(df_h, x="date", y=["m3x_yield_to_key"], labels={"m3x_yield_to_key": "Аномалия премии доходности"}, title="Аномалия премии доходности ОФЗ-ПД к ключевой (event-time MAD)", yaxis_title="MAD-оценка (z); вверх = стресс", height=320), use_container_width=True)
with t3:
    st.plotly_chart(line_chart(df_h, x="date", y=["m3x_age", "m3x_days_since"], labels={"m3x_age": "Возраст аукциона", "m3x_days_since": "Дней с последнего"}, title="Свежесть аукционных данных ОФЗ", yaxis_title="дни", height=320), use_container_width=True)

# --- Сырой контекст ---
st.markdown("---")
st.subheader("Сырой контекст (события аукционов)")
if not df_n.empty:
    st.plotly_chart(
        event_scatter(df_n, x="date", y="cover_ratio", flag_col="Flag_Nedospros",
                      flag_label="Недоспрос", title="Cover Ratio по аукционам ОФЗ",
                      yaxis_title="Cover Ratio", height=340),
        use_container_width=True,
    )
    yld_df = df_n.dropna(subset=["weighted_yield"]) if "weighted_yield" in df_n else df_n.iloc[0:0]
    if not yld_df.empty:
        st.plotly_chart(
            line_chart(yld_df, x="date", y=["weighted_yield"], labels={"weighted_yield": "Средневзв. доходность ОФЗ-ПД (%)"},
                       title="Средневзвешенная доходность размещений ОФЗ-ПД", yaxis_title="% годовых", height=300),
            use_container_width=True,
        )
    st.plotly_chart(
        flag_timeline(df_n, x="date", flags={"Flag_Nedospros": "Недоспрос", "Flag_Perespros": "Переспрос"},
                      title="Флаги аукционов ОФЗ", height=190),
        use_container_width=True,
    )
else:
    st.info("Нет событий аукционов в выбранном окне.")

# --- Таблица ---
with st.expander("Таблица honest-признаков M3 (дневная шкала)"):
    cols = ["date", "m3_auction_flag", "m3_Flag_Nedospros", "m3x_cover", "m3x_placement",
            "m3x_yield_to_key", "m3x_age", "m3x_available", "m3x_days_since", "m3x_failed"]
    cols = [c for c in cols if c in df_h.columns]
    st.dataframe(df_h[cols].sort_values("date", ascending=False), use_container_width=True, hide_index=True)
    csv_download_button(df_h[cols], "m3_honest_features.csv")
