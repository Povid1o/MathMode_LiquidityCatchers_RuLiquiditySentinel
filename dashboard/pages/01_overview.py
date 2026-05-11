import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from dashboard.data.loader import dataset_summary, load_lsi, load_lsi_response
from dashboard.components.metrics import module_status_row
from dashboard.config import COLORS, MODULE_LABELS, PLOTLY_TEMPLATE

st.set_page_config(page_title="Обзор системы", layout="wide")

st.title("Обзор системы мониторинга стресса ликвидности")

# Check if LSI is available
try:
    with st.spinner("Загрузка LSI..."):
        df_lsi = load_lsi()
        lsi_response = load_lsi_response()
    lsi_available = "lsi" in df_lsi.columns
except Exception as e:
    lsi_available = False
    lsi_response = {}
    st.warning(f"Ошибка при загрузке LSI: {e}", icon="⚠️")

st.markdown("---")

with st.spinner("Загрузка статусов модулей..."):
    summary = dataset_summary()

# --- Module status table ---
st.subheader("Статус модулей")
st.caption("Каждый модуль проверяется по наличию файла, количеству строк и свежести данных.")

header_cols = st.columns([2, 1, 2])
with header_cols[0]:
    st.markdown("**Модуль**")
with header_cols[1]:
    st.markdown("**Строк**")
with header_cols[2]:
    st.markdown("**Последняя дата**")

st.markdown("---")

module_keys = ["m1", "m2", "m3", "m4", "m5"]
for key in module_keys:
    info = summary.get(key, {})
    label = MODULE_LABELS.get(key, key.upper())
    ok = info.get("ok", False)
    rows = info.get("rows") if ok else None
    date_max = info.get("date_max") if ok else None
    module_status_row(label, ok, rows, date_max)
    if not ok:
        st.error(f"Ошибка загрузки {key}: {info.get('error', 'неизвестно')}")

st.markdown("---")

# --- Final dataset status ---
st.subheader("Финальный ML-датасет")
final = summary.get("final", {})
if final.get("ok"):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Строк", f"{final['rows']:,}")
    c2.metric("Колонок", f"{final['cols']}")
    c3.metric("Начало", pd.Timestamp(final["date_min"]).strftime("%d.%m.%Y"))
    c4.metric("Конец", pd.Timestamp(final["date_max"]).strftime("%d.%m.%Y"))
else:
    st.error(f"Финальный датасет недоступен: {final.get('error', 'неизвестно')}")

# --- LSI status ---
st.markdown("---")
st.subheader("Индекс стресса ликвидности (LSI)")

if lsi_available:
    st.success(
        "**✓ LSI рассчитан и доступен.**  \n"
        "LSI Local обучается на последнем 365-дневном окне, LSI Global — на всей истории. "
        "Обе версии используют StandardScaler + PCA + IsolationForest + EMA + MinMaxScaler.",
    )
    local_lsi = lsi_response.get("LSI_Local", lsi_response["LSI_Index"])
    global_lsi = lsi_response.get("LSI_Global")
    local_status = str(lsi_response.get("local_status", lsi_response["status"]))
    global_status = str(lsi_response.get("global_status", "н/д"))
    local_drivers = lsi_response.get("local_top_drivers", lsi_response.get("top_drivers", []))
    global_drivers = lsi_response.get("global_top_drivers", [])

    def status_value(label: str, status: str) -> None:
        st.markdown(f"**{label}**")
        st.markdown(
            f"""
            <div style="
                font-size: clamp(1.25rem, 2.0vw, 2.1rem);
                line-height: 1.12;
                color: #fafafa;
                white-space: normal;
                overflow-wrap: anywhere;
                padding-top: 0.15rem;
            ">
                {status}
            </div>
            """,
            unsafe_allow_html=True,
        )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("LSI Local", f"{float(local_lsi):.2f}")
    with c2:
        status_value("Статус Local", local_status)
    if global_lsi is not None:
        c3.metric("LSI Global", f"{float(global_lsi):.2f}")
    else:
        c3.metric("LSI Global", "н/д")
    with c4:
        status_value("Статус Global", global_status)

    st.caption("Пороги светофора: 40 / 70")
    if local_drivers:
        st.caption("Local drivers: " + ", ".join(local_drivers))
    if global_drivers:
        st.caption("Global drivers: " + ", ".join(global_drivers))
    if lsi_response.get("date"):
        st.caption(
            f"LSI рассчитан на последнюю дату финального ML dataset: {lsi_response['date']}. "
            "Это дата данных, а не обязательно сегодняшняя календарная дата."
        )

    def lsi_chart(column: str, title: str, color: str) -> go.Figure:
        """Строит график LSI с порогами светофора"""
        chart_df = df_lsi[["date", column]].dropna()
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=chart_df["date"],
            y=chart_df[column],
            mode="lines",
            fill="tozeroy",
            fillcolor="rgba(31,119,180,0.12)",
            line=dict(color=color, width=2),
            name=title,
        ))
        fig.add_hline(
            y=70,
            line_dash="dash",
            line_color=COLORS["danger"],
            opacity=0.7,
            annotation_text="Стресс",
            annotation_position="right",
        )
        fig.add_hline(
            y=40,
            line_dash="dot",
            line_color=COLORS["warn"],
            opacity=0.7,
            annotation_text="Повышенное внимание",
            annotation_position="left",
        )
        fig.update_layout(
            title=title,
            template=PLOTLY_TEMPLATE,
            height=320,
            yaxis_title="LSI",
            yaxis=dict(range=[0, 105]),
            hovermode="x unified",
            margin=dict(l=40, r=60, t=50, b=40),
            showlegend=False,
        )
        return fig

    st.markdown("---")
    st.subheader("Динамика LSI")
    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        if "lsi_local" in df_lsi.columns and df_lsi["lsi_local"].notna().any():
            st.plotly_chart(
                lsi_chart("lsi_local", "LSI Local", COLORS["primary"]),
                use_container_width=True,
            )
        else:
            st.info("LSI Local недоступен.")
    with chart_col2:
        if "lsi_global" in df_lsi.columns and df_lsi["lsi_global"].notna().any():
            st.plotly_chart(
                lsi_chart("lsi_global", "LSI Global", COLORS["secondary"]),
                use_container_width=True,
            )
        else:
            st.info("LSI Global недоступен.")
else:
    st.error(
        "**LSI недоступен.** Модели не найдены в `models/lsi_global_pipeline.joblib` "
        "и `models/lsi_local_pipeline.joblib`.  \n"
        "Датасет `final_ml_dataset` содержит 102 признака для её обучения.",
        icon="🚫",
    )

# --- Quick data freshness ---
st.markdown("---")
st.subheader("Свежесть данных")
st.caption(
    "Колонка показывает, насколько последняя дата источника отстоит от сегодняшнего дня. "
    "Для календаря ФНС будущая дата означает, что календарь загружен вперед."
)

freshness_data = []
for key in module_keys:
    info = summary.get(key, {})
    label = MODULE_LABELS.get(key, key.upper())
    if info.get("ok"):
        date_max = pd.Timestamp(info["date_max"])
        age = (pd.Timestamp.now().normalize() - date_max.normalize()).days
        if age < 0:
            age_label = f"через {abs(age)} дн."
        elif age == 0:
            age_label = "сегодня"
        else:
            age_label = f"{age} дн. назад"
        freshness_data.append({
            "Модуль": label,
            "Последняя дата": date_max.strftime("%d.%m.%Y"),
            "Относительно сегодня": age_label,
            "Строк": info["rows"],
            "Max пропусков, %": f"{info['missing_pct']:.1f}%",
        })

if freshness_data:
    df_fresh = pd.DataFrame(freshness_data)

    def highlight_age(val):
        if isinstance(val, str):
            if val.startswith("через") or val == "сегодня":
                return "color: #2ca02c"
            days_text = val.split(" ", maxsplit=1)[0]
            if not days_text.isdigit():
                return ""
            days = int(days_text)
            if days > 30:
                return "color: #d62728"
            if days > 14:
                return "color: #bcbd22"
            return "color: #2ca02c"
        return ""

    st.dataframe(
        df_fresh.style.map(highlight_age, subset=["Относительно сегодня"]),
        use_container_width=True,
        hide_index=True,
    )
