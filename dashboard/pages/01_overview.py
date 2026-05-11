import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st
import pandas as pd
from dashboard.data.loader import dataset_summary, load_lsi, load_lsi_response
from dashboard.components.metrics import module_status_row, proxy_score_note
from dashboard.config import MODULE_LABELS

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
        "Используется IsolationForest для обнаружения аномалий на финальном датасете "
        "(98 признаков из М1–М5, скейлировано + PCA).",
    )
    latest_lsi = float(lsi_response["LSI_Index"])
    lsi_status = str(lsi_response["status"])
    top_drivers = lsi_response.get("top_drivers", [])

    c1, c2, c3 = st.columns(3)
    c1.metric("Последний LSI", f"{latest_lsi:.2f}")
    c2.metric("Статус", lsi_status)
    c3.metric("Пороги светофора", "40 / 70")

    if top_drivers:
        st.caption("Top drivers: " + ", ".join(top_drivers))
else:
    st.error(
        "**LSI недоступен.** Модель не найдена в `models/lsi_pipeline.joblib`.  \n"
        "Датасет `final_ml_dataset` содержит 102 признака для её обучения.",
        icon="🚫",
    )

# --- Quick data freshness ---
st.markdown("---")
st.subheader("Свежесть данных")

freshness_data = []
for key in module_keys:
    info = summary.get(key, {})
    label = MODULE_LABELS.get(key, key.upper())
    if info.get("ok"):
        age = (pd.Timestamp.now() - pd.Timestamp(info["date_max"])).days
        freshness_data.append({
            "Модуль": label,
            "Последняя дата": pd.Timestamp(info["date_max"]).strftime("%d.%m.%Y"),
            "Дней назад": age,
            "Строк": info["rows"],
            "Max пропусков, %": f"{info['missing_pct']:.1f}%",
        })

if freshness_data:
    df_fresh = pd.DataFrame(freshness_data)

    def highlight_age(val):
        if isinstance(val, int):
            if val > 30:
                return "color: #d62728"
            elif val > 14:
                return "color: #bcbd22"
            return "color: #2ca02c"
        return ""

    st.dataframe(
        df_fresh.style.map(highlight_age, subset=["Дней назад"]),
        use_container_width=True,
        hide_index=True,
    )
