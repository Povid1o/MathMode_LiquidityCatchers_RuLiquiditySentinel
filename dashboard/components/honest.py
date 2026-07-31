"""Honest-LSI компоненты для страниц модулей: панель live-вклада фич в индекс.

Каждая вкладка модуля показывает, какие honest-признаки этого модуля и насколько
двигают LSI на последнюю дату (EVR-attribution), плюс человекочитаемые подписи.
"""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from backend.src.services import feature_catalog as fc
from dashboard.config import COLORS, PLOTLY_TEMPLATE

# Подписи признаков живут в едином каталоге backend/src/services/feature_catalog.py:
# их читают и страницы дашборда, и AI-аналитик, поэтому держать словарь здесь
# означало бы иметь два расходящихся источника правды. Здесь остаются только
# тонкие обёртки для обратной совместимости с существующими вызовами.
HONEST_FEATURE_LABELS: dict[str, str] = {
    column: spec.label for column, spec in fc.CATALOG.items()
}


def feature_label(feature: str) -> str:
    """Человекочитаемая подпись признака."""
    return fc.label(feature)


def feature_label_flagged(feature: str) -> str:
    """Подпись с пометкой ⚠︎, если формулировка ещё не подтверждена."""
    return fc.label_with_flag(feature)


def feature_description(feature: str) -> str:
    """Пояснение смысла признака для тултипа."""
    return fc.description(feature)


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

    labels = [feature_label_flagged(f["feature"]) for f in feats]
    values = [f["contrib_pct"] for f in feats]
    descriptions = [fc.description(f["feature"]) or "—" for f in feats]
    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation="h",
        marker_color=color,
        text=[f"{v:.1f}%" for v in values],
        textposition="outside",
        customdata=descriptions,
        hovertemplate="%{y}<br>Вклад: %{x:.2f}%<br><br>%{customdata}<extra></extra>",
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
            "Признак": feature_label_flagged(f["feature"]),
            "Что это": fc.description(f["feature"]) or "—",
            "Как читать рост": fc.HIGHER_MEANS_HINT[
                (fc.lookup(f["feature"]).higher_means if fc.lookup(f["feature"]) else "neutral")
            ],
            "Колонка": f["feature"],
            "Вклад, %": f["contrib_pct"],
            "z (отклонение)": f["z_scaled"],
            "Состояние": f["direction"],
        }
        for f in feats
    ])
    st.dataframe(table, use_container_width=True, hide_index=True)

    unverified = [f["feature"] for f in feats if (spec := fc.lookup(f["feature"])) and spec.needs_review]
    if unverified:
        st.caption(
            "⚠︎ Формулировка названия ещё не подтверждена предметным специалистом: "
            + ", ".join(feature_label(f) for f in unverified)
            + ". Само значение признака это не затрагивает — только его словесное описание."
        )
