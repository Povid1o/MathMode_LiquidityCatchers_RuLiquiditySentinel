import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from dashboard.data.loader import (
    dataset_summary, load_lsi, load_lsi_response,
    load_threshold_metrics, load_threshold_profile,
)
from dashboard.components.metrics import module_status_row
from dashboard.config import COLORS, MODULE_LABELS, PLOTLY_TEMPLATE
from backend.src.services.lsi_thresholds import LSI_THRESHOLD_PROFILES
from backend.src.services.honest_lsi_prediction import DEFAULT_HONEST_PROFILE as DEFAULT_THRESHOLD_PROFILE

st.set_page_config(page_title="Обзор системы", layout="wide")

st.title("Обзор системы мониторинга стресса ликвидности")

# Загружаем числовые LSI-значения заранее — они не зависят от профиля
try:
    with st.spinner("Загрузка LSI..."):
        df_lsi = load_lsi()
    lsi_available = "lsi" in df_lsi.columns
except Exception as e:
    lsi_available = False
    df_lsi = pd.DataFrame()
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
    # ---------------------------------------------------------------
    # Selector порогового профиля
    # Выбор сохраняется в st.session_state["lsi_threshold_profile"]
    # и используется на всех страницах dashboard через session_state.
    # ---------------------------------------------------------------
    _profile_labels: dict[str, str] = {
        "honest":       "honest (40/60) — сбалансированный индекс, перекалиброванные пороги (p80/p95)",
        "conservative": "conservative (40/70) — меньше ложных тревог",
    }
    _profile_options = list(_profile_labels.keys())

    # Инициализируем session_state значением по умолчанию из backend.
    # Если в сессии остался профиль старого индекса — сбрасываем на honest.
    if st.session_state.get("lsi_threshold_profile") not in _profile_options:
        st.session_state["lsi_threshold_profile"] = DEFAULT_THRESHOLD_PROFILE

    selected_profile: str = st.radio(
        "Пороговый профиль",
        options=_profile_options,
        format_func=lambda p: _profile_labels[p],
        index=_profile_options.index(
            st.session_state.get("lsi_threshold_profile", DEFAULT_THRESHOLD_PROFILE)
        ),
        key="lsi_threshold_profile",
        horizontal=True,
        help=(
            "conservative — меньше ложных красных сигналов; "
            "backtest_sensitive — выше Event Recall Red, но больше False Red Alerts/year. "
            "Числовые значения LSI не меняются — только статус (ЗЕЛЕНЫЙ/ЖЕЛТЫЙ/КРАСНЫЙ)."
        ),
    )

    # Загружаем ответ модели с учётом выбранного профиля.
    # Кеш st.cache_data учитывает threshold_profile как ключ.
    with st.spinner("Применяем профиль..."):
        lsi_response = load_lsi_response(selected_profile)

    # --- Извлекаем данные из ответа ---
    thr_profile = str(lsi_response.get("threshold_profile", selected_profile))
    thr_green   = float(lsi_response.get("threshold_green_max", 40.0))
    thr_yellow  = float(lsi_response.get("threshold_yellow_max", 70.0))

    st.success(
        "**✓ LSI рассчитан и доступен.**  \n"
        "LSI Local обучается на последнем 365-дневном окне, LSI Global — на всей истории. "
        "Обе версии используют StandardScaler + PCA + IsolationForest + EMA + MinMaxScaler.",
    )

    local_lsi    = lsi_response.get("LSI_Local", lsi_response["LSI_Index"])
    global_lsi   = lsi_response.get("LSI_Global")
    local_status = str(lsi_response.get("local_status", lsi_response["status"]))
    global_status = str(lsi_response.get("global_status", "н/д"))
    local_drivers  = lsi_response.get("local_top_drivers", lsi_response.get("top_drivers", []))
    global_drivers = lsi_response.get("global_top_drivers", [])

    def status_value(label: str, status: str) -> None:
        status_upper = status.upper() if status else ""
        if "КРАСН" in status_upper:
            color = "#DC2626"
        elif "ЖЕЛТ" in status_upper or "ЖЁЛТ" in status_upper:
            color = "#B45309"
        elif "ЗЕЛЕН" in status_upper or "ЗЕЛЁН" in status_upper:
            color = "#059669"
        else:
            color = "inherit"
        st.markdown(f"**{label}**")
        st.markdown(
            f"""
            <div style="
                font-size: clamp(1.25rem, 2.0vw, 2.1rem);
                line-height: 1.12;
                color: {color};
                white-space: normal;
                overflow-wrap: anywhere;
                padding-top: 0.15rem;
                font-weight: 600;
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

    st.caption(
        f"Пороговый профиль: **{thr_profile}** — "
        f"зелёный < {int(thr_green)}, жёлтый {int(thr_green)}–{int(thr_yellow)}, красный ≥ {int(thr_yellow)}"
    )
    if local_drivers:
        st.caption("Local drivers: " + ", ".join(local_drivers))
    if global_drivers:
        st.caption("Global drivers: " + ", ".join(global_drivers))
    if lsi_response.get("date"):
        st.caption(
            f"LSI рассчитан на последнюю дату финального ML dataset: {lsi_response['date']}. "
            "Это дата данных, а не обязательно сегодняшняя календарная дата."
        )

    # -------------------------------------------------------------------
    # Expander: сравнение профилей и помощь в выборе
    # -------------------------------------------------------------------
    with st.expander("Как выбрать пороговый профиль?"):
        st.markdown(
            """
**`backtest_sensitive` (30 / 60) — чувствительный профиль:**
- ✅ Event Recall Red = **100%** по Global: все три стресс-эпизода детектированы красным
- ✅ Декабрь 2014: 1 красный день + 23 жёлтых (события видны заранее)
- ⚠️ Global FP rate ≈ **10.5%** (~25 ложных красных дней в год вне кризисов)
- ⚠️ Local FP rate ≈ **47%** (Local модель обучена в спокойный период)
- Красный сигнал требует ручного подтверждения аналитиком

**`conservative` (40 / 70) — консервативный профиль:**
- ✅ Global FP rate ≈ **3.95%** (~9 ложных красных дней в год)
- ⚠️ Event Recall Red = **33%** по Global: Декабрь 2014 и Август 2023 пропускаются красным
- Декабрь 2014: 0 красных, 19 жёлтых из 23 — событие видно, но не как стресс

**Вывод:** если приоритет — не пропустить кризис, выбирайте `backtest_sensitive`.
Если важнее снизить нагрузку на аналитика — `conservative`.

Числовые значения LSI не меняются при переключении — меняется только интерпретация.
"""
        )
        # Таблица метрик профилей
        threshold_metrics = load_threshold_metrics()
        if not threshold_metrics.empty:
            _profiles_df = pd.DataFrame([
                {"Профиль": "backtest_sensitive", "threshold_green": 30, "threshold_red": 60},
                {"Профиль": "conservative",       "threshold_green": 40, "threshold_red": 70},
            ])
            metrics_view = _profiles_df.merge(
                threshold_metrics,
                on=["threshold_green", "threshold_red"],
                how="left",
            )
            metrics_view = metrics_view[[
                "Профиль", "model",
                "event_recall_yellow_pct", "event_recall_red_pct",
                "lead_time_yellow_days", "lead_time_red_days",
                "false_red_alerts_per_year",
            ]].rename(columns={
                "model": "Модель",
                "event_recall_yellow_pct": "Recall Yellow, %",
                "event_recall_red_pct":    "Recall Red, %",
                "lead_time_yellow_days":   "Lead Yellow, дн.",
                "lead_time_red_days":      "Lead Red, дн.",
                "false_red_alerts_per_year": "False Red/year",
            })
            st.dataframe(metrics_view.round(2), use_container_width=True, hide_index=True)

    # -------------------------------------------------------------------
    # LSI-графики с линиями активного профиля
    # -------------------------------------------------------------------
    def lsi_chart(column: str, title: str, color: str) -> go.Figure:
        """Строит график LSI с линиями активного порогового профиля"""
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
            y=thr_yellow,
            line_dash="dash",
            line_color=COLORS["danger"],
            opacity=0.7,
            annotation_text=f"Стресс ≥{int(thr_yellow)}",
            annotation_position="right",
        )
        fig.add_hline(
            y=thr_green,
            line_dash="dot",
            line_color=COLORS["warn"],
            opacity=0.7,
            annotation_text=f"Внимание ≥{int(thr_green)}",
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

    # --- Вклад модулей (числа не зависят от профиля — только от модели PCA) ---
    local_contribs  = lsi_response.get("local_module_contributions", {})
    global_contribs = lsi_response.get("global_module_contributions", {})

    if local_contribs or global_contribs:
        st.markdown("---")
        st.subheader("Вклад модулей в LSI (последняя дата)")
        st.caption(
            "⚠️ Вклад рассчитан как **PCA-based approximation** по whitelist стресс-признаков "
            "(structural_weight[j] = Σₖ evr[k]·|components[k,j]|, нормировано до 100% по строке). "
            "Это **не SHAP** и не причинная декомпозиция — метрика показывает относительную "
            "нагрузку модуля на первые главные компоненты, а не его причинный вклад в LSI."
        )

        def contrib_bar(contribs: dict[str, float], title: str, color: str) -> go.Figure:
            modules = sorted(contribs.keys())
            values = [contribs[m] for m in modules]
            labels = [MODULE_LABELS.get(m.lower(), m) for m in modules]
            fig = go.Figure(go.Bar(
                x=modules,
                y=values,
                text=[f"{v:.1f}%" for v in values],
                textposition="outside",
                marker_color=color,
                customdata=labels,
                hovertemplate="%{customdata}<br>Вклад: %{y:.1f}%<extra></extra>",
            ))
            fig.update_layout(
                title=title,
                template=PLOTLY_TEMPLATE,
                height=300,
                yaxis_title="Вклад, %",
                yaxis=dict(range=[0, max(values) * 1.25 if values else 100]),
                margin=dict(l=40, r=20, t=50, b=40),
                showlegend=False,
            )
            return fig

        contrib_col1, contrib_col2 = st.columns(2)
        with contrib_col1:
            if local_contribs:
                st.plotly_chart(
                    contrib_bar(local_contribs, "Вклад модулей — LSI Local", COLORS["primary"]),
                    use_container_width=True,
                )
            else:
                st.info("Вклад модулей Local недоступен.")
        with contrib_col2:
            if global_contribs:
                st.plotly_chart(
                    contrib_bar(global_contribs, "Вклад модулей — LSI Global", COLORS["secondary"]),
                    use_container_width=True,
                )
            else:
                st.info("Вклад модулей Global недоступен.")

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
