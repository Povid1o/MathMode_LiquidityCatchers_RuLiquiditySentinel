import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st
from dashboard.data.loader import load_m4, load_module_contribution
from dashboard.components.charts import line_chart, flag_timeline
from dashboard.components.honest import honest_driver_panel
from dashboard.components.metrics import (
    latest_value_metric, quick_period_filter, freshness_header, csv_download_button,
)

st.set_page_config(page_title="M4 — Налоги (overlay)", layout="wide")
st.title("M4 — Налоговый календарь (overlay)")

st.info(
    "**M4 — это overlay, а не модуль PCA.** В honest-LSI налоговый календарь намеренно "
    "**вынесен из индекса**: он детерминирован (известен заранее) и при включении в PCA "
    "давал ложную сезонную «подсветку». Теперь M4 отдаётся как **контекст** рядом с LSI "
    "(налоговая неделя / день, давление), но **не двигает** значение индекса.",
    icon="🪧",
)

df_native = load_m4()
freshness_header(df_native, "M4 — Налоги (overlay)")
df = quick_period_filter(df_native, key="m4_period")
if df.empty:
    st.warning("Нет данных для выбранного периода.")
    st.stop()

# --- KPI: текущий налоговый контекст ---
st.subheader("Текущий налоговый контекст")
latest = df.iloc[-1]
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Налоговая неделя", "ДА" if bool(latest.get("Tax_Week_Flag", 0)) else "НЕТ")
with c2:
    st.metric("Налоговый день (строгий)", "ДА" if bool(latest.get("Tax_Day_Strict", 0)) else "НЕТ")
with c3:
    latest_value_metric("Налоговое давление", df["tax_pressure"], fmt="{:.3f}")
with c4:
    latest_value_metric("Налоговая близость", df["tax_proximity"], fmt="{:.3f}")

# --- Подтверждение overlay: вклад в LSI = 0 ---
st.markdown("---")
st.subheader("Вклад M4 в текущий LSI")
honest_driver_panel(load_module_contribution("M4"))

# --- Налоговый контекст во времени ---
st.markdown("---")
st.subheader("Налоговое давление во времени")
st.caption("Календарные признаки ФНС: давление и близость к налоговым срокам. Это контекст для интерпретации LSI, не его вход.")
pr = df.dropna(subset=["tax_pressure"]) if "tax_pressure" in df else df.iloc[0:0]
if not pr.empty:
    ycols = [c for c in ["tax_pressure", "tax_pressure_smoothed"] if c in pr.columns]
    st.plotly_chart(
        line_chart(pr, x="date", y=ycols,
                   labels={"tax_pressure": "Налоговое давление", "tax_pressure_smoothed": "Сглаженное"},
                   title="Налоговое давление (календарь ФНС)", yaxis_title="индекс давления", height=320),
        use_container_width=True,
    )
mad = df.dropna(subset=["MAD_tax_pressure"]) if "MAD_tax_pressure" in df else df.iloc[0:0]
if not mad.empty:
    st.plotly_chart(
        line_chart(mad, x="date", y=["MAD_tax_pressure"], labels={"MAD_tax_pressure": "MAD налогового давления"},
                   title="Аномальность налогового давления (контекст)", yaxis_title="MAD", height=280),
        use_container_width=True,
    )

st.subheader("Налоговые периоды")
st.plotly_chart(
    flag_timeline(df, x="date",
                  flags={"Tax_Week_Flag": "Налоговая неделя", "Tax_Day_Strict": "Налоговый день",
                         "Tax_Pre_Flag": "Преднал. период"},
                  title="Налоговый календарь: периоды", height=210),
    use_container_width=True,
)

st.link_button("Открыть календарь ФНС", "https://www.nalog.gov.ru/rn77/calendar/")

# --- Таблица ---
with st.expander("Таблица данных M4 (overlay-контекст)"):
    cols = ["date", "Tax_Week_Flag", "Tax_Day_Strict", "Tax_Pre_Flag", "Tax_Active_Flag",
            "tax_pressure", "tax_pressure_smoothed", "tax_proximity",
            "MAD_tax_pressure", "MAD_tax_proximity"]
    cols = [c for c in cols if c in df.columns]
    st.dataframe(df[cols].sort_values("date", ascending=False), use_container_width=True, hide_index=True)
    csv_download_button(df[cols], "m4_overlay_context.csv")
