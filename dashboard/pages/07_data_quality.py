import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from dashboard.data.loader import (
    load_m1, load_m2, load_m3, load_m4, load_m5, load_final, dataset_summary
)
from dashboard.config import PLOTLY_TEMPLATE, MODULE_LABELS

st.set_page_config(page_title="Качество данных", layout="wide")
st.title("Качество данных")

st.markdown("Проверка наличия файлов, полноты данных, дублей и пропусков по каждому модулю.")

loaders = {
    "m1": load_m1,
    "m2": load_m2,
    "m3": load_m3,
    "m4": load_m4,
    "m5": load_m5,
    "final": load_final,
}

labels = {**MODULE_LABELS, "final": "Final ML Dataset"}

with st.spinner("Загрузка всех датасетов..."):
    summary = dataset_summary()

# --- Overview table ---
st.subheader("Сводка по датасетам")

rows = []
for key, info in summary.items():
    label = labels.get(key, key.upper())
    if info.get("ok"):
        rows.append({
            "Датасет": label,
            "Статус": "OK",
            "Строк": info["rows"],
            "Колонок": info["cols"],
            "Дата начала": pd.Timestamp(info["date_min"]).strftime("%d.%m.%Y"),
            "Дата конца": pd.Timestamp(info["date_max"]).strftime("%d.%m.%Y"),
            "Max пропуск, %": round(info["missing_pct"], 1),
            "Файл": info["path"],
        })
    else:
        rows.append({
            "Датасет": label,
            "Статус": f"ОШИБКА: {info.get('error', '?')}",
            "Строк": None, "Колонок": None,
            "Дата начала": None, "Дата конца": None,
            "Max пропуск, %": None,
            "Файл": info.get("path", "—"),
        })

df_summary = pd.DataFrame(rows)

def color_status(val):
    if val == "OK":
        return "color: #2ca02c"
    return "color: #d62728"

def color_missing(val):
    if val is None:
        return ""
    if val > 50:
        return "color: #d62728"
    elif val > 20:
        return "color: #bcbd22"
    return "color: #2ca02c"

st.dataframe(
    df_summary.style
    .map(color_status, subset=["Статус"])
    .map(color_missing, subset=["Max пропуск, %"]),
    use_container_width=True,
    hide_index=True,
)

st.markdown("---")

# --- Per-dataset detail ---
selected = st.selectbox(
    "Детальный анализ по датасету",
    options=list(loaders.keys()),
    format_func=lambda k: labels.get(k, k.upper()),
)

info = summary.get(selected, {})
if not info.get("ok"):
    st.error(f"Датасет недоступен: {info.get('error', 'неизвестно')}")
    st.stop()

with st.spinner(f"Загрузка {selected}..."):
    df = loaders[selected]()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Строк", f"{len(df):,}")
col2.metric("Колонок", len(df.columns))
col3.metric("Дат начала", df["date"].min().strftime("%d.%m.%Y"))
col4.metric("Дата конца", df["date"].max().strftime("%d.%m.%Y"))

# --- Duplicate dates ---
dup_count = df["date"].duplicated().sum()
if dup_count > 0:
    st.warning(f"Обнаружено **{dup_count}** дублирующихся дат.")
    st.dataframe(df[df["date"].duplicated(keep=False)].sort_values("date"), use_container_width=True)
else:
    st.success("Дублирующихся дат не обнаружено.")

st.markdown("---")

# --- Missing values heatmap ---
st.subheader("Пропуски по колонкам")

miss = df.isnull().mean() * 100
miss = miss[miss > 0].sort_values(ascending=False)

if miss.empty:
    st.success("Пропусков нет.")
else:
    fig_miss = go.Figure(go.Bar(
        x=miss.index.tolist(),
        y=miss.values,
        marker_color=[
            "#d62728" if v > 50 else "#bcbd22" if v > 20 else "#2ca02c"
            for v in miss.values
        ],
    ))
    fig_miss.update_layout(
        title="Доля пропусков по колонкам (%)",
        template=PLOTLY_TEMPLATE,
        height=350,
        margin=dict(l=40, r=20, t=40, b=120),
        xaxis_tickangle=-45,
        yaxis_title="Пропусков, %",
    )
    st.plotly_chart(fig_miss, use_container_width=True)

    st.caption("Топ-10 колонок с наибольшим числом пропусков:")
    top_miss = miss.head(10).reset_index()
    top_miss.columns = ["Колонка", "Пропусков, %"]
    st.dataframe(top_miss, use_container_width=True, hide_index=True)

st.markdown("---")

# --- Data types and basic stats ---
st.subheader("Базовая статистика числовых колонок")
numeric_cols = df.select_dtypes(include="number").columns.tolist()
if numeric_cols:
    desc = df[numeric_cols].describe().T
    desc.index.name = "Колонка"
    st.dataframe(desc.round(3), use_container_width=True)
else:
    st.info("Числовые колонки не найдены.")

st.markdown("---")

# --- Date coverage ---
st.subheader("Покрытие дат")
date_range = pd.date_range(df["date"].min(), df["date"].max(), freq="D")
coverage = len(df["date"].unique()) / len(date_range) * 100
st.metric(
    "Покрытие дат (уникальных дат / кол. календарных дней)",
    f"{coverage:.1f}%",
    delta=f"{len(df['date'].unique())} из {len(date_range)} дней",
)
