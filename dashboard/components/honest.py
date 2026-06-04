"""Honest-LSI компоненты для страниц модулей: панель live-вклада фич в индекс.

Каждая вкладка модуля показывает, какие honest-признаки этого модуля и насколько
двигают LSI на последнюю дату (EVR-attribution), плюс человекочитаемые подписи.
"""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard.config import COLORS, PLOTLY_TEMPLATE

# Человекочитаемые подписи honest-признаков (входят в PCA honest-LSI).
HONEST_FEATURE_LABELS: dict[str, str] = {
    # M1 — резервы / RUONIA
    "m1_spread_mad_score": "Аномальность спреда резервов (MAD)",
    "m1_spread_relative_mad_score": "Аномальность относит. спреда (MAD)",
    "m1_reserve_load_mad_score": "Аномальность нагрузки резервов (MAD)",
    "m1_ruonia_mad_score": "Аномальность RUONIA (MAD)",
    "m1_spread_vol": "Волатильность спреда |Δ| (MAD)",
    # M2 — РЕПО-аукционы
    "m2_auction_flag": "Факт аукциона РЕПО",
    "m2_Flag_Demand": "Флаг высокого спроса",
    "m2_base_cover_mad": "Аномальность переподписки (MAD)",
    "m2_cutoff_spread": "Спред отсечения к ключевой ставке",
    "m2_cutoff_spread_available": "Доступность спреда отсечения",
    "m2_short_active30": "Активный short-РЕПО (30 дн.)",
    "m2_days_since_short": "Дней с последнего short-РЕПО",
    # M3 — ОФЗ-аукционы (event-aware)
    "m3_auction_flag": "Факт аукциона ОФЗ",
    "m3_Flag_Nedospros": "Флаг недоспроса",
    "m3x_cover": "Переподписка (event-aware)",
    "m3x_placement": "Доля размещения от предложения",
    "m3x_yield_to_key": "Премия доходности к ключевой",
    "m3x_age": "Возраст последнего аукциона",
    "m3x_available": "Наличие данных аукциона",
    "m3x_days_since": "Дней с последнего аукциона",
    "m3x_failed": "Признак несостоявшегося аукциона",
    # M5 — ликвидность ЦБ / ЕКС
    "m5x_claims": "Требования ЦБ к банкам",
    "m5x_liab": "Обязательства ЦБ перед банками",
    "m5x_repostd": "Постоянное РЕПО (standing facility)",
    "m5x_secured": "Обеспеченные кредиты (standing)",
    "m5x_rk_bidders": "Число заявителей Росказна (Local)",
}


def feature_label(feature: str) -> str:
    return HONEST_FEATURE_LABELS.get(feature, feature)


def honest_driver_panel(contrib: dict, *, color: str | None = None, height: int = 320) -> None:
    """Рендерит live-вклад honest-фич модуля в текущий LSI: bar + таблица.

    `contrib` — результат honest_module_feature_contributions (dict с features).
    """
    color = color or COLORS["primary"]
    feats = contrib.get("features", [])
    if not feats:
        st.info(
            "Модуль не входит в PCA honest-LSI — это **overlay** (контекст), "
            "который не двигает индекс напрямую.",
            icon="🪧",
        )
        return

    st.caption(
        f"Вклад honest-признаков модуля в LSI на **{contrib['date']}** "
        f"(модель: **{contrib['kind']}**, суммарно модуль ≈ **{contrib['module_total_pct']}%** индекса). "
        "Метрика — EVR-attribution: |scaled|·structural_weight, нормировано к 100% по всем "
        "признакам индекса. Это PCA-приближение нагрузки, не SHAP и не причинный вклад."
    )

    labels = [feature_label(f["feature"]) for f in feats]
    values = [f["contrib_pct"] for f in feats]
    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation="h",
        marker_color=color,
        text=[f"{v:.1f}%" for v in values],
        textposition="outside",
        hovertemplate="%{y}<br>Вклад: %{x:.2f}%<extra></extra>",
    ))
    fig.update_layout(
        template=PLOTLY_TEMPLATE, height=height,
        margin=dict(l=10, r=30, t=20, b=30),
        xaxis_title="Вклад в LSI, %",
        yaxis=dict(autorange="reversed"),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)

    table = pd.DataFrame([
        {
            "Признак": feature_label(f["feature"]),
            "Колонка": f["feature"],
            "Вклад, %": f["contrib_pct"],
            "z (отклонение)": f["z_scaled"],
            "Состояние": f["direction"],
        }
        for f in feats
    ])
    st.dataframe(table, use_container_width=True, hide_index=True)
