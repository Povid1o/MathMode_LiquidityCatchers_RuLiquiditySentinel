"""Страница «Данные ⚙️» — обновление источников и витрины.

Одной кнопкой: загрузка сырья из источников до последней доступной даты →
пересчёт фич M1–M5 → сборка final → пересчёт honest-LSI → обновление витрины
DuckDB и графиков дашборда.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import streamlit as st

from backend.src.db import warehouse as wh
from backend.src.pipelines.refresh_pipeline import build_steps, _run_step

st.set_page_config(page_title="Данные — обновление", layout="wide")
st.title("⚙️ Данные — обновление и витрина")

st.markdown(
    "Конвейер тянет данные из источников (**ЦБ РФ, Минфин, ФНС, Росказна**) до последней "
    "доступной даты, пересчитывает признаки M1–M5, собирает `final_ml_dataset`, "
    "переобучает honest-LSI (Global/Local) и наполняет витрину **DuckDB**, после чего "
    "графики дашборда обновляются."
)

# ---------------------------------------------------------------------------
# 1. Свежесть витрины
# ---------------------------------------------------------------------------
st.subheader("Свежесть витрины (DuckDB warehouse)")

mani = wh.manifest()
if mani.empty:
    st.info("Витрина пуста. Запустите обновление или синхронизацию ниже.")
else:
    today = pd.Timestamp.today().normalize()

    def _age(d: object) -> object:
        if not d or pd.isna(d):
            return None
        return (today - pd.Timestamp(str(d))).days

    view = mani.copy()
    view["lag_days"] = view["date_max"].map(_age)
    view = view.rename(columns={
        "table_name": "Таблица", "row_count": "Строк",
        "date_min": "Дата с", "date_max": "Дата по",
        "lag_days": "Отставание, дн.", "updated_at": "Обновлено",
    })
    show_cols = ["Таблица", "Строк", "Дата с", "Дата по", "Отставание, дн.", "Обновлено"]

    def _hl_lag(val: object) -> str:
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return ""
        days = int(val)
        if days > 30:
            return "color: #d62728"
        if days > 7:
            return "color: #bcbd22"
        return "color: #2ca02c"

    st.dataframe(
        view[show_cols].style.map(_hl_lag, subset=["Отставание, дн."]),
        use_container_width=True, hide_index=True,
    )
    st.caption(
        "Отставание = разница между сегодняшней датой и последней датой в таблице. "
        "Сводные `final_ml_dataset` / `honest_ml_dataset` ограничены самым «коротким» "
        "ежедневным источником на дату стыковки."
    )

# ---------------------------------------------------------------------------
# 2. Запуск обновления
# ---------------------------------------------------------------------------
st.markdown("---")
st.subheader("Запуск")

_MODES = {
    "Полное обновление (сеть + пересчёт всего)": None,  # все шаги
    "Пересчёт индекса (final + honest), без загрузки из сети": ["final", "honest", "warehouse"],
    "Только синхронизация витрины из файлов": ["warehouse"],
}
mode_label = st.radio("Режим", list(_MODES.keys()), index=0)
keys = _MODES[mode_label]

if keys is None:
    st.warning(
        "⚠️ Полное обновление обращается к внешним источникам по сети и пересчитывает "
        "все признаки и модели — это может занять несколько минут. Не закрывайте вкладку "
        "до завершения.",
        icon="🌐",
    )

all_steps = build_steps()
steps = all_steps if keys is None else [s for s in all_steps if s.key in keys]

confirm = st.checkbox("Подтверждаю запуск выбранного режима", value=False)
run = st.button("▶️ Запустить обновление", type="primary", disabled=not confirm)

if run:
    results = []
    for step in steps:
        with st.status(f"{step.label}…", expanded=False) as status_box:
            res = _run_step(step)
            if res.status == "ok":
                status_box.update(label=f"✓ {step.label} · {res.seconds}s", state="complete")
            else:
                status_box.update(label=f"✗ {step.label} · {res.error}", state="error")
            if res.log_tail:
                st.code(res.log_tail[-4000:], language="text")
        results.append(res)

    ok = sum(r.status == "ok" for r in results)
    # Сбрасываем кеш дашборда, чтобы графики перечитали витрину
    st.cache_data.clear()

    if ok == len(results):
        st.success(
            f"Готово: {ok}/{len(results)} шагов успешно. Кеш дашборда очищен — "
            "графики на всех страницах перечитают обновлённую витрину."
        )
    else:
        failed = ", ".join(r.label for r in results if r.status != "ok")
        st.warning(
            f"Завершено с ошибками: {ok}/{len(results)} успешно. Проблемные шаги: {failed}. "
            "Витрина синхронизирована тем, что успешно пересчиталось — см. логи выше."
        )

    if st.button("🔄 Обновить страницу (показать новую свежесть)"):
        st.rerun()
