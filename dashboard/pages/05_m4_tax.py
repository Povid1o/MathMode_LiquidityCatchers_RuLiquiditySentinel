import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st
import pandas as pd
from dashboard.data.loader import load_m4
from dashboard.components.metrics import latest_value_metric, quick_period_filter, freshness_header, csv_download_button

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

# --- Информационный блок ---
st.subheader("Официальный налоговый календарь")
st.markdown(
    "M4 использует календарь налоговых сроков ФНС как заранее известный источник "
    "**календарного давления** на ликвидность. "
    "В dashboard графики M4 не показываются, потому что календарные признаки дискретные "
    "и плохо читаются как длинные временные ряды.\n\n"
    "Важно: M4 отражает **календарное налоговое давление**, а не рублёвые объёмы налоговых платежей."
)
st.link_button("Открыть календарь ФНС", "https://www.nalog.gov.ru/rn77/calendar/")

st.markdown("---")

# --- Raw table ---
with st.expander("Таблица данных M4 (все признаки)"):
    cols_show = ["date", "Tax_Week_Flag", "Tax_Day_Strict", "Tax_Pre_Flag", "Tax_Active_Flag",
                 "tax_pressure", "tax_pressure_smoothed", "tax_proximity",
                 "MAD_tax_pressure", "MAD_tax_proximity", "Seasonal_Factor_raw"]
    cols_show = [c for c in cols_show if c in df.columns]
    st.dataframe(df[cols_show].sort_values("date", ascending=False), use_container_width=True, hide_index=True)
    csv_download_button(df[cols_show], "m4_features.csv")
