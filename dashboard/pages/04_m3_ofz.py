import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st
from dashboard.data.loader import load_m3
from dashboard.components.charts import event_scatter, bar_chart, mad_score_bar, flag_timeline
from dashboard.components.metrics import latest_value_metric, mad_status_metric, quick_period_filter, freshness_header, csv_download_button

st.set_page_config(page_title="M3 — ОФЗ", layout="wide")
st.title("M3 — Размещение ОФЗ")

st.markdown(
    "Модуль анализирует первичные аукционы ОФЗ Минфина. "
    "Данные **разреженные** — только в дни аукционов (обычно среды). "
    "Ключевые сигналы: недоспрос (Flag_Nedospros) и перес прос (Flag_Perespros)."
)

with st.spinner("Загрузка данных M3..."):
    df = load_m3()

freshness_header(df, "M3 — ОФЗ")
df = quick_period_filter(df, key="m3_period")

if df.empty:
    st.warning("Нет данных для выбранного периода.")
    st.stop()

# --- KPI ---
st.subheader("Последние значения")
c1, c2, c3, c4 = st.columns(4)
with c1:
    latest_value_metric("Cover Ratio (последний аукцион)", df["cover_ratio"], fmt="{:.2f}")
with c2:
    yld_df = df.dropna(subset=["yield_spread"])
    latest_value_metric("Yield Spread (прибл., п.п.)", yld_df["yield_spread"], fmt="{:.3f}")
with c3:
    count_ned = int(df["Flag_Nedospros"].sum()) if "Flag_Nedospros" in df.columns else 0
    st.metric("Аукционов: недоспрос", f"{count_ned} / {len(df)}", delta=f"{count_ned/len(df)*100:.1f}%")
with c4:
    count_per = int(df["Flag_Perespros"].sum()) if "Flag_Perespros" in df.columns else 0
    st.metric("Аукционов: перес прос", f"{count_per} / {len(df)}", delta=f"{count_per/len(df)*100:.1f}%")

st.markdown("---")

# --- Cover ratio ---
st.subheader("Коэффициент покрытия на аукционах ОФЗ")
st.caption(
    "Отношение суммарного спроса к объёму предложения. "
    "< 1 = недоспрос (красные маркеры), > 1.5 = перес прос."
)

# highlight by flag
import plotly.graph_objects as go
from dashboard.config import COLORS, PLOTLY_TEMPLATE

fig_cover = go.Figure()
for flag_val, label, color in [
    (0, "Норма", COLORS["primary"]),
    (1, "Недоспрос", COLORS["danger"]),
]:
    if "Flag_Nedospros" in df.columns:
        sub = df[df["Flag_Nedospros"] == flag_val]
    else:
        sub = df if flag_val == 0 else df.iloc[0:0]
    fig_cover.add_trace(go.Bar(
        x=sub["date"], y=sub["cover_ratio"],
        name=label,
        marker_color=color,
    ))

perespros_df = df[df["Flag_Perespros"] == 1] if "Flag_Perespros" in df.columns else df.iloc[0:0]
fig_cover.add_trace(go.Bar(
    x=perespros_df["date"], y=perespros_df["cover_ratio"],
    name="Перес прос", marker_color=COLORS["warn"],
))

fig_cover.add_hline(y=1.0, line_dash="dash", line_color="white", opacity=0.5, annotation_text="Cover=1")
fig_cover.update_layout(
    title="Cover Ratio по аукционам ОФЗ",
    template=PLOTLY_TEMPLATE, height=360,
    yaxis_title="Cover Ratio",
    barmode="overlay",
    hovermode="x unified",
    margin=dict(l=40, r=20, t=40, b=40),
)
st.plotly_chart(fig_cover, use_container_width=True)

# --- Yield spread ---
st.subheader("Доходность и спред")
st.caption("Yield spread — приближённый спред к кривой ОФЗ (не кривая нулевых купонов, используйте с осторожностью).")
tab1, tab2 = st.tabs(["Weighted Yield", "Yield Spread (прибл.)"])

with tab1:
    yld_df2 = df.dropna(subset=["weighted_yield"])
    if not yld_df2.empty:
        fig_yield = event_scatter(
            yld_df2, x="date", y="weighted_yield",
            title="Средневзвешенная доходность размещения ОФЗ",
            yaxis_title="Доходность, % годовых",
            height=320,
        )
        st.plotly_chart(fig_yield, use_container_width=True)
    else:
        st.info("Данные по доходности отсутствуют.")

with tab2:
    yld_sp_df = df.dropna(subset=["yield_spread"])
    if not yld_sp_df.empty:
        fig_ysp = event_scatter(
            yld_sp_df, x="date", y="yield_spread",
            title="Yield Spread (приближённый)",
            yaxis_title="Спред, п.п.",
            height=320,
        )
        st.plotly_chart(fig_ysp, use_container_width=True)
    else:
        st.info("Данные по спреду доходности отсутствуют.")

st.markdown("---")

# --- MAD scores ---
st.subheader("MAD-оценки")
tab_a, tab_b = st.tabs(["MAD Cover", "MAD Yield Spread"])

with tab_a:
    fig_mc = mad_score_bar(df, x="date", y="MAD_score_cover", title="MAD-score покрытия аукционов ОФЗ")
    st.plotly_chart(fig_mc, use_container_width=True)

with tab_b:
    ysp_mad_df = df.dropna(subset=["MAD_score_yield_spread"])
    if not ysp_mad_df.empty:
        fig_ms = mad_score_bar(ysp_mad_df, x="date", y="MAD_score_yield_spread", title="MAD-score yield spread")
        st.plotly_chart(fig_ms, use_container_width=True)
    else:
        st.info("MAD-score по yield spread недоступен.")

st.markdown("---")

# --- Flag timeline ---
st.subheader("Флаги на аукционах")
fig_flags = flag_timeline(
    df, x="date",
    flags={
        "Flag_Nedospros": "Недоспрос",
        "Flag_Perespros": "Перес прос",
    },
    title="Флаги аукционов ОФЗ",
    height=200,
)
st.plotly_chart(fig_flags, use_container_width=True)

# --- Raw table ---
with st.expander("Таблица данных M3"):
    cols_show = ["date", "demand_amount", "offered_amount", "placed_amount",
                 "cover_ratio", "weighted_yield", "yield_spread",
                 "Flag_Nedospros", "Flag_Perespros",
                 "MAD_score_cover", "MAD_score_yield_spread"]
    cols_show = [c for c in cols_show if c in df.columns]
    st.dataframe(df[cols_show].sort_values("date", ascending=False), use_container_width=True, hide_index=True)
    csv_download_button(df[cols_show], "m3_features.csv")
