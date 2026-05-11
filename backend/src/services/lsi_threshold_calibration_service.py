"""Калибровка порогов светофора LSI.

Метод: проверяем несколько пар порогов (зелёный/красный) на размеченных
стресс-эпизодах из backtest и на полной истории lsi_scores для оценки
false positive rate. Результат — рекомендованная пара порогов с объяснением.

Важно: этот сервис не переобучает модель и не меняет features_list.
Пороги калибруются только по output-шкале 0-100, которую возвращают обе модели.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
DOCS_DIR = PROJECT_ROOT / "docs" / "backend"

BACKTEST_SCORES_FILE = DATA_DIR / "lsi_backtest_scores.csv"
LSI_SCORES_FILE = DATA_DIR / "lsi_scores.csv"

CALIBRATION_CSV = DATA_DIR / "lsi_threshold_calibration.csv"
CALIBRATION_PARQUET = DATA_DIR / "lsi_threshold_calibration.parquet"
CALIBRATION_METRICS_CSV = DATA_DIR / "lsi_threshold_metrics.csv"
CALIBRATION_METRICS_PARQUET = DATA_DIR / "lsi_threshold_metrics.parquet"
CALIBRATION_REPORT = DOCS_DIR / "lsi_threshold_calibration.md"
LEAD_TIME_LOOKBACK_DAYS = 30

# пары порогов для проверки (зелёный_макс, красный_мин)
THRESHOLD_PAIRS: list[tuple[int, int]] = [
    (30, 55),
    (30, 60),
    (35, 60),
    (35, 65),
    (40, 65),
    (40, 70),
    (45, 75),
    (50, 80),
]

# стресс-эпизоды (те же, что в backtest)
STRESS_EPISODES: dict[str, tuple[str, str]] = {
    "Декабрь 2014": ("2014-12-01", "2014-12-31"),
    "Февраль-март 2022": ("2022-02-01", "2022-03-31"),
    "Август 2023": ("2023-08-01", "2023-08-31"),
}


def _load_backtest_scores() -> pd.DataFrame:
    """Загружает backtest scores из CSV"""
    if not BACKTEST_SCORES_FILE.exists():
        raise FileNotFoundError(f"Файл backtest scores не найден: {BACKTEST_SCORES_FILE}")
    df = pd.read_csv(BACKTEST_SCORES_FILE, parse_dates=["date"])
    return df


def _load_lsi_scores() -> pd.DataFrame:
    """Загружает полную историю LSI scores"""
    if not LSI_SCORES_FILE.exists():
        raise FileNotFoundError(f"Файл lsi_scores не найден: {LSI_SCORES_FILE}")
    df = pd.read_csv(LSI_SCORES_FILE, parse_dates=["date"])
    return df


def _is_stress_date(date: pd.Timestamp) -> bool:
    """Проверяет, входит ли дата в один из размеченных стресс-эпизодов"""
    for start_str, end_str in STRESS_EPISODES.values():
        start = pd.Timestamp(start_str)
        end = pd.Timestamp(end_str)
        if start <= date <= end:
            return True
    return False


def _episode_stats(
    backtest: pd.DataFrame,
    episode_name: str,
    green_threshold: int,
    red_threshold: int,
) -> dict[str, object]:
    """Считает статистику detection для одного эпизода и одной пары порогов"""
    ep = backtest[backtest["episode"] == episode_name].copy()
    if ep.empty:
        return {
            "episode": episode_name,
            "threshold_green": green_threshold,
            "threshold_red": red_threshold,
            "period_days": 0,
            "global_max_lsi": None,
            "local_max_lsi": None,
            "global_yellow_days": 0,
            "global_red_days": 0,
            "local_yellow_days": 0,
            "local_red_days": 0,
        }

    global_vals = ep["lsi_global_backtest"].dropna()
    local_vals = ep["lsi_local_backtest"].dropna()

    return {
        "episode": episode_name,
        "threshold_green": green_threshold,
        "threshold_red": red_threshold,
        "period_days": len(ep),
        "global_max_lsi": round(float(global_vals.max()), 2) if not global_vals.empty else None,
        "local_max_lsi": round(float(local_vals.max()), 2) if not local_vals.empty else None,
        "global_yellow_days": int((global_vals >= green_threshold).sum()),
        "global_red_days": int((global_vals >= red_threshold).sum()),
        "local_yellow_days": int((local_vals >= green_threshold).sum()),
        "local_red_days": int((local_vals >= red_threshold).sum()),
    }


def _false_positive_rates(
    lsi_scores: pd.DataFrame,
    green_threshold: int,
    red_threshold: int,
) -> dict[str, object]:
    """Считает долю дней вне стресс-эпизодов с LSI >= red_threshold.

    Для LSI Local учитываем только строки, где он не NaN (последнее 365-дневное окно).
    """
    non_stress_mask = ~lsi_scores["date"].apply(_is_stress_date)
    non_stress = lsi_scores[non_stress_mask].copy()

    global_col = "lsi_global"
    local_col = "lsi_local"

    global_fp = None
    local_fp = None
    local_fp_note = "LSI Local недоступен или рассчитан только на ограниченном окне"

    if global_col in non_stress.columns:
        global_non_nan = non_stress[global_col].dropna()
        if not global_non_nan.empty:
            global_fp = round(float((global_non_nan >= red_threshold).mean() * 100), 2)

    if local_col in non_stress.columns:
        local_non_nan = non_stress[non_stress[local_col].notna()][local_col]
        if not local_non_nan.empty:
            local_fp = round(float((local_non_nan >= red_threshold).mean() * 100), 2)
            local_fp_note = f"рассчитан на {len(local_non_nan)} строках (только даты с доступным LSI Local)"

    return {
        "episode": "non_stress_fp",
        "threshold_green": green_threshold,
        "threshold_red": red_threshold,
        "period_days": len(non_stress),
        "global_max_lsi": None,
        "local_max_lsi": None,
        "global_yellow_days": None,
        "global_red_days": None,
        "local_yellow_days": None,
        "local_red_days": None,
        "global_fp_rate_pct": global_fp,
        "local_fp_rate_pct": local_fp,
        "local_fp_note": local_fp_note,
    }


def run_threshold_calibration(
    backtest: pd.DataFrame | None = None,
    lsi_scores: pd.DataFrame | None = None,
    *,
    threshold_pairs: list[tuple[int, int]] = THRESHOLD_PAIRS,
) -> pd.DataFrame:
    """Запускает калибровку порогов и возвращает сводную таблицу"""
    if backtest is None:
        backtest = _load_backtest_scores()
    if lsi_scores is None:
        lsi_scores = _load_lsi_scores()

    rows: list[dict[str, object]] = []

    for green_t, red_t in threshold_pairs:
        # результаты по стресс-эпизодам
        for episode_name in STRESS_EPISODES:
            stats = _episode_stats(backtest, episode_name, green_t, red_t)
            stats["global_fp_rate_pct"] = None
            stats["local_fp_rate_pct"] = None
            stats["local_fp_note"] = None
            rows.append(stats)

        # false positive на вне-стрессовой истории
        fp_row = _false_positive_rates(lsi_scores, green_t, red_t)
        rows.append(fp_row)

    result = pd.DataFrame(rows)

    # удобный порядок колонок
    col_order = [
        "threshold_green", "threshold_red", "episode", "period_days",
        "global_max_lsi", "local_max_lsi",
        "global_yellow_days", "global_red_days",
        "local_yellow_days", "local_red_days",
        "global_fp_rate_pct", "local_fp_rate_pct", "local_fp_note",
    ]
    result = result[[c for c in col_order if c in result.columns]]
    return result


def _event_recall(
    backtest: pd.DataFrame,
    *,
    model: str,
    threshold: int,
) -> float:
    """Считает долю стресс-эпизодов с хотя бы одним сигналом выше порога"""
    value_col = f"lsi_{model}_backtest"
    detected = 0
    total = 0

    for episode_name in STRESS_EPISODES:
        episode_scores = backtest[backtest["episode"] == episode_name]
        values = episode_scores[value_col].dropna() if value_col in episode_scores else pd.Series(dtype=float)
        if values.empty:
            continue
        total += 1
        detected += int((values >= threshold).any())

    if total == 0:
        return 0.0
    return round(detected / total * 100.0, 2)


def _average_lead_time(
    lsi_scores: pd.DataFrame,
    *,
    model: str,
    threshold: int,
) -> float | None:
    """Считает средний lead time сигнала в окне до начала стресс-эпизода"""
    value_col = f"lsi_{model}"
    if value_col not in lsi_scores.columns:
        return None

    lead_times: list[int] = []
    for start_str, _end_str in STRESS_EPISODES.values():
        start = pd.Timestamp(start_str)
        window_start = start - pd.Timedelta(days=LEAD_TIME_LOOKBACK_DAYS)
        pre_event = lsi_scores[
            (lsi_scores["date"] >= window_start) &
            (lsi_scores["date"] <= start) &
            lsi_scores[value_col].notna()
        ]
        signals = pre_event[pre_event[value_col] >= threshold]
        if signals.empty:
            continue
        first_signal_date = signals["date"].min()
        lead_times.append((start - first_signal_date).days)

    if not lead_times:
        return None
    return round(float(sum(lead_times) / len(lead_times)), 2)


def _false_red_alerts_per_year(
    lsi_scores: pd.DataFrame,
    *,
    model: str,
    red_threshold: int,
) -> tuple[float | None, int, int]:
    """Считает среднее число красных вне стресс-окон в год"""
    value_col = f"lsi_{model}"
    if value_col not in lsi_scores.columns:
        return None, 0, 0

    non_stress = lsi_scores[~lsi_scores["date"].apply(_is_stress_date)].copy()
    available = non_stress[non_stress[value_col].notna()].copy()
    if available.empty:
        return None, 0, 0

    red_days = int((available[value_col] >= red_threshold).sum())
    calendar_years = max(
        (available["date"].max() - available["date"].min()).days + 1,
        1,
    ) / 365.25
    return round(red_days / calendar_years, 2), red_days, int(len(available))


def build_threshold_metrics(
    backtest: pd.DataFrame | None = None,
    lsi_scores: pd.DataFrame | None = None,
    *,
    threshold_pairs: list[tuple[int, int]] = THRESHOLD_PAIRS,
) -> pd.DataFrame:
    """Считает Event Recall, Lead Time и False Alerts/year по парам порогов"""
    if backtest is None:
        backtest = _load_backtest_scores()
    if lsi_scores is None:
        lsi_scores = _load_lsi_scores()

    rows: list[dict[str, object]] = []
    for green_t, red_t in threshold_pairs:
        for model in ["global", "local"]:
            false_alerts, false_red_days, available_days = _false_red_alerts_per_year(
                lsi_scores,
                model=model,
                red_threshold=red_t,
            )
            rows.append(
                {
                    "threshold_green": green_t,
                    "threshold_red": red_t,
                    "model": model,
                    "event_recall_yellow_pct": _event_recall(
                        backtest,
                        model=model,
                        threshold=green_t,
                    ),
                    "event_recall_red_pct": _event_recall(
                        backtest,
                        model=model,
                        threshold=red_t,
                    ),
                    "lead_time_yellow_days": _average_lead_time(
                        lsi_scores,
                        model=model,
                        threshold=green_t,
                    ),
                    "lead_time_red_days": _average_lead_time(
                        lsi_scores,
                        model=model,
                        threshold=red_t,
                    ),
                    "false_red_alerts_per_year": false_alerts,
                    "false_red_days": false_red_days,
                    "non_stress_days_available": available_days,
                    "lead_time_note": (
                        f"lead time считается по lsi_scores в окне {LEAD_TIME_LOOKBACK_DAYS} дней "
                        "до начала стресс-эпизода"
                    ),
                }
            )

    return pd.DataFrame(rows)


def _recommend_threshold(calibration: pd.DataFrame) -> tuple[int, int, str]:
    """Выбирает рекомендованную пару порогов на основе результатов калибровки.

    Критерии (в порядке приоритета):
    1. Декабрь 2014 и Февраль-март 2022 должны давать хотя бы жёлтый сигнал (global_red_days > 0)
    2. Минимальный false positive rate по Global (global_fp_rate_pct)
    3. Из оставшихся — выбирается самый чувствительный (наименьший red_threshold)
    """
    threshold_pairs = calibration[["threshold_green", "threshold_red"]].drop_duplicates().values.tolist()
    candidates: list[tuple[float, int, int, str]] = []

    for green_t, red_t in threshold_pairs:
        pair_rows = calibration[
            (calibration["threshold_green"] == green_t) &
            (calibration["threshold_red"] == red_t)
        ]

        # проверяем detection крупных эпизодов
        ep2014 = pair_rows[pair_rows["episode"] == "Декабрь 2014"]
        ep2022 = pair_rows[pair_rows["episode"] == "Февраль-март 2022"]

        detected_2014 = int(ep2014["global_red_days"].iloc[0]) > 0 if not ep2014.empty else False
        detected_2022 = int(ep2022["global_red_days"].iloc[0]) > 0 if not ep2022.empty else False

        if not detected_2014 or not detected_2022:
            continue  # обязательное условие: оба главных эпизода должны давать красный

        fp_row = pair_rows[pair_rows["episode"] == "non_stress_fp"]
        fp_rate = float(fp_row["global_fp_rate_pct"].iloc[0]) if not fp_row.empty else 100.0

        candidates.append((fp_rate, green_t, red_t, ""))

    if not candidates:
        # если ни одна пара не детектирует оба эпизода — выбираем с наименьшим red_threshold
        fallback_green, fallback_red = THRESHOLD_PAIRS[0]
        reason = (
            f"Ни одна пара порогов не детектировала оба главных эпизода в виде красного сигнала. "
            f"Выбрана самая чувствительная пара ({fallback_green}/{fallback_red}) как запасная."
        )
        return fallback_green, fallback_red, reason

    # сортируем: сначала минимальный FP, затем минимальный red_threshold для чувствительности
    candidates.sort(key=lambda x: (x[0], x[2]))
    fp_rate, green_t, red_t, _ = candidates[0]
    reason = (
        f"Пара {green_t}/{red_t} детектирует оба главных стресс-эпизода красным сигналом "
        f"при global FP rate {fp_rate:.2f}% на вне-стрессовой истории."
    )
    return int(green_t), int(red_t), reason


def build_calibration_report(
    calibration: pd.DataFrame,
    metrics: pd.DataFrame | None = None,
    *,
    output_path: Path = CALIBRATION_REPORT,
) -> tuple[int, int, str]:
    """Строит markdown-отчёт по калибровке и возвращает рекомендацию"""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rec_green, rec_red, rec_reason = _recommend_threshold(calibration)
    if metrics is None:
        metrics = build_threshold_metrics()

    lines = [
        "# Калибровка порогов LSI",
        "",
        "## Метод",
        "",
        "Пороги светофора LSI (зелёный/красный) проверяются на трёх исторических стресс-эпизодах "
        "через уже рассчитанный rolling/expanding backtest (LSI Global и Local, "
        "point-in-time, без look-ahead). "
        "False positive rate считается на полной истории `lsi_scores.csv` "
        "за вычетом размеченных стресс-окон.",
        "",
        "**Стресс-эпизоды:**",
        "",
    ]

    for ep_name, (start_str, end_str) in STRESS_EPISODES.items():
        lines.append(f"- {ep_name}: {start_str} — {end_str}")
    lines.append("")

    lines.extend([
        "## Результаты по парам порогов",
        "",
        "| Зелёный | Красный | Эпизод | Дней | Global max | Local max | "
        "Global жёлтых | Global красных | Local жёлтых | Local красных |",
        "|---------|---------|--------|------|------------|-----------|"
        "--------------|----------------|--------------|--------------|",
    ])

    for _, row in calibration[calibration["episode"] != "non_stress_fp"].iterrows():
        lines.append(
            f"| {int(row['threshold_green'])} | {int(row['threshold_red'])} "
            f"| {row['episode']} | {int(row['period_days'])} "
            f"| {row['global_max_lsi'] if pd.notna(row['global_max_lsi']) else '—'} "
            f"| {row['local_max_lsi'] if pd.notna(row['local_max_lsi']) else '—'} "
            f"| {int(row['global_yellow_days']) if pd.notna(row['global_yellow_days']) else '—'} "
            f"| {int(row['global_red_days']) if pd.notna(row['global_red_days']) else '—'} "
            f"| {int(row['local_yellow_days']) if pd.notna(row['local_yellow_days']) else '—'} "
            f"| {int(row['local_red_days']) if pd.notna(row['local_red_days']) else '—'} |"
        )

    lines.extend([
        "",
        "## False Positive Rate вне стресс-эпизодов",
        "",
        "| Зелёный | Красный | Global FP % | Local FP % | Примечание |",
        "|---------|---------|-------------|------------|------------|",
    ])

    for _, row in calibration[calibration["episode"] == "non_stress_fp"].iterrows():
        global_fp = f"{row['global_fp_rate_pct']:.2f}%" if pd.notna(row["global_fp_rate_pct"]) else "—"
        local_fp = f"{row['local_fp_rate_pct']:.2f}%" if pd.notna(row["local_fp_rate_pct"]) else "—"
        note = str(row["local_fp_note"]) if pd.notna(row.get("local_fp_note")) else ""
        lines.append(
            f"| {int(row['threshold_green'])} | {int(row['threshold_red'])} "
            f"| {global_fp} | {local_fp} | {note} |"
        )

    lines.extend([
        "",
        "## Метрики качества порогов",
        "",
        "`Event Recall Yellow/Red` — доля стресс-эпизодов, где point-in-time backtest дал "
        "хотя бы один жёлтый или красный сигнал. `Lead Time` считается по полной линии "
        f"`lsi_scores.csv` в окне {LEAD_TIME_LOOKBACK_DAYS} дней до начала эпизода. "
        "`False Red Alerts/year` — среднее число красных дней в год вне размеченных стресс-окон.",
        "",
        "| Зелёный | Красный | Модель | Recall Yellow % | Recall Red % | "
        "Lead Yellow, дн. | Lead Red, дн. | False Red/year |",
        "|---------|---------|--------|-----------------|--------------|-------------------|----------------|----------------|",
    ])

    for _, row in metrics.iterrows():
        lead_y = f"{row['lead_time_yellow_days']:.2f}" if pd.notna(row["lead_time_yellow_days"]) else "—"
        lead_r = f"{row['lead_time_red_days']:.2f}" if pd.notna(row["lead_time_red_days"]) else "—"
        false_alerts = (
            f"{row['false_red_alerts_per_year']:.2f}"
            if pd.notna(row["false_red_alerts_per_year"])
            else "—"
        )
        lines.append(
            f"| {int(row['threshold_green'])} | {int(row['threshold_red'])} "
            f"| {row['model']} | {row['event_recall_yellow_pct']:.2f} "
            f"| {row['event_recall_red_pct']:.2f} | {lead_y} | {lead_r} | {false_alerts} |"
        )

    lines.extend([
        "",
        "## Рекомендация алгоритма",
        "",
        f"**Пороги по sensitivity-критерию: {rec_green} / {rec_red}**",
        "",
        rec_reason,
        "",
        "## Production default и обоснование выбора",
        "",
        "Алгоритмическая рекомендация (**sensitivity-критерий**) выбирает пару, "
        "которая детектирует оба ключевых эпизода красным и минимизирует FP rate.",
        "Production default подтверждён по бизнес-позиции: **приоритет раннего предупреждения**. "
        "Для системы раннего предупреждения ошибка пропуска стрессового эпизода дороже, "
        "чем ложная тревога.",
        "",
        "| Профиль | Зелёный | Красный | Global FP % | Local FP % | Декабрь 2014 красных дней |",
        "|---------|---------|---------|-------------|------------|--------------------------|",
        "| **backtest_sensitive** (**production default**) | < 30 | ≥ 60 | ~10.5% | ~47.1% | 1 |",
        "| conservative | < 40 | ≥ 70 | ~3.95% | ~35.6% | 0 (19 жёлтых) |",
        "",
        "**Production default = `backtest_sensitive` (30 / 60):**",
        "",
        "- Event Recall Red по Global = 100% на размеченных стресс-эпизодах",
        "- Декабрь 2014 детектируется красным сигналом, а не только жёлтым",
        "- Красный сигнал появляется раньше, что соответствует задаче раннего предупреждения",
        "- Высокий false alert rate считается осознанной бизнес-ценой чувствительности",
        "- Красный сигнал должен интерпретироваться как повод для ручной проверки аналитиком",
        "",
        "**`conservative` (40 / 70) как альтернатива:**",
        "",
        "- Даёт меньше ложных красных сигналов",
        "- Подходит, если бизнес хочет снизить нагрузку на аналитиков",
        "- Хуже подходит для раннего предупреждения: Декабрь 2014 остаётся жёлтым, "
        "а не красным",
        "",
        "Конфигурация профилей: `backend/src/services/lsi_thresholds.py`",
        "",
        "## Ограничения",
        "",
        "- Стресс-эпизоды размечены вручную: три кризисных периода. "
        "Другие периоды напряжённости (например, 2015–2016) не включены.",
        "- LSI Local доступен только на последнем 365-дневном окне, "
        "поэтому FP rate по Local считается на ограниченной выборке (261 строка).",
        "- Backtest scores и production scores нормированы на разных данных. "
        "Декабрь 2014 в production-модели: max global LSI ≈ 34 (ниже любого red threshold), "
        "в backtest-модели: max ≈ 63 (самый жёсткий кризис на виденных данных до той даты).",
        "- Backtest не учитывает нестационарность: модель переобучается на каждую дату "
        "point-in-time, но признаки whitelist фиксированы.",
        "- Пороги не оптимизированы по ROC/precision-recall и не являются статистически "
        "значимыми при трёх размеченных эпизодах. Это экспертная калибровка.",
        "- Окончательный выбор порогов должен быть подтверждён куратором / экспертом "
        "предметной области.",
    ])

    output_path.write_text("\n".join(lines), encoding="utf-8")
    return rec_green, rec_red, rec_reason


def save_calibration_outputs(
    calibration: pd.DataFrame,
    metrics: pd.DataFrame | None = None,
    *,
    csv_path: Path = CALIBRATION_CSV,
    parquet_path: Path = CALIBRATION_PARQUET,
    metrics_csv_path: Path = CALIBRATION_METRICS_CSV,
    metrics_parquet_path: Path = CALIBRATION_METRICS_PARQUET,
) -> None:
    """Сохраняет результаты калибровки в CSV и parquet"""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    calibration.to_csv(csv_path, index=False)
    calibration.to_parquet(parquet_path, index=False)
    if metrics is not None:
        metrics.to_csv(metrics_csv_path, index=False)
        metrics.to_parquet(metrics_parquet_path, index=False)
