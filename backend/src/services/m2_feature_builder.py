from __future__ import annotations

import csv
from datetime import datetime
from datetime import timedelta
from pathlib import Path
from statistics import median


PROJECT_ROOT = Path(__file__).resolve().parents[3]
INPUT_FILE = PROJECT_ROOT / "data/processed/m2_dataset.csv"
OUTPUT_FILE = PROJECT_ROOT / "data/processed/m2_features.csv"
PARQUET_FILE = PROJECT_ROOT / "data/processed/m2_features.parquet"
DAILY_PROFILE_FILE = PROJECT_ROOT / "data/processed/m2_daily_profile.csv"
DAILY_PROFILE_PARQUET = PROJECT_ROOT / "data/processed/m2_daily_profile.parquet"

START_DATE = datetime(2010, 1, 1)
MIN_DEALS_VOLUME = 1000.0
COVER_RATIO_LIMIT = 10.0
DEMAND_THRESHOLD = 2.0
ROLLING_WINDOW_DAYS = 365 * 3
MAD_MIN_VALUE = 0.05

# --- ПАРАМЕТРЫ срочностной структуры (term-aware) ---------------------------
# 2 яруса (по решению: плотный split). Границы по term_days включительно.
TERM_TIERS = {
    "short": (0, 5),       # O/N и сверхкороткие — дефицит ликвидности "прямо сейчас"
    "base": (6, 9999),     # 7d weekly + месячные + длинные (доминирующий инструмент)
}
# long — НЕ отдельный непрерывный ярус (всего ~40 аукционов с 2014),
# только событийные флаги (active / age / available) внутри base.
LONG_TERM_THRESHOLD_DAYS = 92

# Окна дневного term-профиля: структурный (Global) и тактический (Local).
ROLLING_WINDOW_GLOBAL_DAYS = 252
ROLLING_WINDOW_LOCAL_DAYS = 63
# Горизонт "свежести": ярус считается доступным, если аукцион был не дальше этого.
AVAILABILITY_HORIZON_DAYS = 90
# Потолок для age_days, чтобы не плодить огромные значения на старте истории.
AGE_CAP_DAYS = 365

OUTPUT_COLUMNS = [
    "date",
    "auction_type",
    "term_days",
    "tier",
    "auction_time",
    "total_deals_volume",
    "weighted_average_rate",
    "settlement_code",
    "demand_volume",
    "cutoff_rate",
    "min_rate",
    "max_rate",
    "limit_deals_volume",
    "weighted_average_limit_rate",
    "first_leg_date",
    "second_leg_date",
    "cover_ratio",
    "key_rate",
    "rate_for_spread",
    "rate_spread",
    "Flag_Demand",
    "MAD_score_cover",
    "MAD_score_rate_spread",
    "MAD_score_cover_tier",
    "MAD_score_rate_spread_tier",
]


def _read_csv(path: Path) -> list[dict[str, str]]:
    """Читает CSV-файл в список словарей"""
    with path.open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def _date_sort_key(date_text: str) -> datetime:
    """Готовит строковую дату DD-MM-YYYY для сортировки"""
    return datetime.strptime(date_text, "%d-%m-%Y")


def _to_float(value: str | object) -> float | None:
    """Преобразует значение в float"""
    if value is None or value == "":
        return None
    return float(value)


def _to_int(value: str | object) -> int | None:
    """Преобразует значение в int"""
    if value is None or value == "":
        return None
    return int(float(value))


def _median(values: list[float]) -> float | None:
    """Считает медиану непустого списка"""
    if not values:
        return None
    return float(median(values))


def _mad(values: list[float]) -> float | None:
    """Считает median absolute deviation"""
    center = _median(values)
    if center is None:
        return None
    deviations = [abs(value - center) for value in values]
    return _median(deviations)


def _tier_of(term_days: object) -> str | None:
    """Возвращает ярус срочности по term_days согласно TERM_TIERS."""
    value = _to_int(term_days)
    if value is None:
        return None
    for tier_name, (low, high) in TERM_TIERS.items():
        if low <= value <= high:
            return tier_name
    return None


def _add_tier_mad_scores(
    rows: list[dict[str, object]],
    source_column: str,
    output_column: str,
) -> None:
    """MAD-score по скользящему окну 3 года ВНУТРИ яруса срочности.

    В отличие от _add_mad_scores (смешивает все срочности в одном окне и из-за
    этого даёт белый шум), здесь окно ограничено строками того же tier — статистика
    однородна, и MAD имеет смысл.
    """
    window_start = timedelta(days=ROLLING_WINDOW_DAYS)

    by_tier: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        by_tier.setdefault(str(row.get("tier")), []).append(row)

    for row in rows:
        current_value = _to_float(row.get(source_column))
        if current_value is None or row.get("tier") is None:
            row[output_column] = None
            continue

        current_date = row["_date"]
        peers = by_tier.get(str(row["tier"]), [])
        window_values = [
            _to_float(candidate.get(source_column))
            for candidate in peers
            if current_date - window_start <= candidate["_date"] <= current_date
        ]
        window_values = [value for value in window_values if value is not None]

        rolling_median = _median(window_values)
        rolling_mad = _mad(window_values)
        if rolling_median is None or rolling_mad is None:
            row[output_column] = None
            continue

        rolling_mad = max(rolling_mad, MAD_MIN_VALUE)
        row[output_column] = (current_value - rolling_median) / rolling_mad


def _choose_rate_for_spread(row: dict[str, object]) -> float | None:
    """Выбирает ставку репо для расчета спреда"""
    cutoff_rate = _to_float(row.get("cutoff_rate"))
    if cutoff_rate is not None:
        return cutoff_rate
    return _to_float(row.get("weighted_average_rate"))


def _calculate_cover_ratio(row: dict[str, object]) -> float:
    """Пересчитывает cover ratio по методике аналитика"""
    demand_volume = _to_float(row.get("demand_volume"))
    total_deals_volume = _to_float(row.get("total_deals_volume"))

    if (
        demand_volume is None
        or total_deals_volume is None
        or total_deals_volume < MIN_DEALS_VOLUME
    ):
        return 1.0

    return min(demand_volume / total_deals_volume, COVER_RATIO_LIMIT)


def _add_mad_scores(rows: list[dict[str, object]], source_column: str, output_column: str) -> None:
    """Добавляет MAD-score по скользящему окну в 3 года"""
    window_start = timedelta(days=ROLLING_WINDOW_DAYS)

    for row in rows:
        current_date = row["_date"]
        current_value = _to_float(row.get(source_column))
        if current_value is None:
            row[output_column] = None
            continue

        window_values = [
            _to_float(candidate.get(source_column))
            for candidate in rows
            if current_date - window_start <= candidate["_date"] <= current_date
        ]
        window_values = [value for value in window_values if value is not None]

        rolling_median = _median(window_values)
        rolling_mad = _mad(window_values)
        if rolling_median is None or rolling_mad is None:
            row[output_column] = None
            continue

        rolling_mad = max(rolling_mad, MAD_MIN_VALUE)
        row[output_column] = (current_value - rolling_median) / rolling_mad


def build_m2_features(input_path: Path = INPUT_FILE) -> list[dict[str, object]]:
    """Собирает feature dataset М2 по всем срочностям репо"""
    rows = _read_csv(input_path)
    prepared_rows: list[dict[str, object]] = []
    last_key_rate: float | None = None

    for source_row in sorted(rows, key=lambda item: _date_sort_key(item["date"])):
        row_date = _date_sort_key(source_row["date"])
        term_days = _to_int(source_row["term_days"])
        key_rate = _to_float(source_row.get("key_rate"))

        if key_rate is not None:
            last_key_rate = key_rate

        if row_date < START_DATE:
            continue

        row: dict[str, object] = {
            "date": source_row["date"],
            "auction_type": source_row["auction_type"],
            "term_days": term_days,
            "tier": _tier_of(term_days),
            "auction_time": source_row["auction_time"],
            "total_deals_volume": _to_float(source_row["total_deals_volume"]),
            "weighted_average_rate": _to_float(source_row["weighted_average_rate"]),
            "settlement_code": source_row["settlement_code"],
            "demand_volume": _to_float(source_row.get("demand_volume")),
            "cutoff_rate": _to_float(source_row.get("cutoff_rate")),
            "min_rate": _to_float(source_row.get("min_rate")),
            "max_rate": _to_float(source_row.get("max_rate")),
            "limit_deals_volume": _to_float(source_row.get("limit_deals_volume")),
            "weighted_average_limit_rate": _to_float(
                source_row.get("weighted_average_limit_rate")
            ),
            "first_leg_date": source_row.get("first_leg_date", ""),
            "second_leg_date": source_row.get("second_leg_date", ""),
            "key_rate": last_key_rate,
            "_date": row_date,
        }
        row["cover_ratio"] = _calculate_cover_ratio(row)
        row["rate_for_spread"] = _choose_rate_for_spread(row)

        rate_for_spread = _to_float(row["rate_for_spread"])
        if rate_for_spread is None or last_key_rate is None:
            row["rate_spread"] = None
        else:
            row["rate_spread"] = rate_for_spread - last_key_rate

        row["Flag_Demand"] = int(row["cover_ratio"] > DEMAND_THRESHOLD)
        prepared_rows.append(row)

    # существующие (mixed-term) MAD оставляем для обратной совместимости...
    _add_mad_scores(prepared_rows, "cover_ratio", "MAD_score_cover")
    _add_mad_scores(prepared_rows, "rate_spread", "MAD_score_rate_spread")
    # ...и добавляем tier-aware (однородные внутри яруса) — это "почин" фич
    _add_tier_mad_scores(prepared_rows, "cover_ratio", "MAD_score_cover_tier")
    _add_tier_mad_scores(prepared_rows, "rate_spread", "MAD_score_rate_spread_tier")

    for row in prepared_rows:
        del row["_date"]

    return prepared_rows


def save_csv(rows: list[dict[str, object]], output_path: Path = OUTPUT_FILE) -> None:
    """Сохраняет feature dataset М2 в CSV-файл"""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def save_parquet(
    rows: list[dict[str, object]],
    output_path: Path = PARQUET_FILE,
) -> None:
    """Сохраняет feature dataset М2 в Parquet-файл"""
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ModuleNotFoundError as error:
        raise RuntimeError("Для сохранения parquet нужен пакет pyarrow") from error

    output_path.parent.mkdir(parents=True, exist_ok=True)

    schema = pa.schema(
        [
            ("date", pa.string()),
            ("auction_type", pa.string()),
            ("term_days", pa.int64()),
            ("tier", pa.string()),
            ("auction_time", pa.string()),
            ("total_deals_volume", pa.float64()),
            ("weighted_average_rate", pa.float64()),
            ("settlement_code", pa.string()),
            ("demand_volume", pa.float64()),
            ("cutoff_rate", pa.float64()),
            ("min_rate", pa.float64()),
            ("max_rate", pa.float64()),
            ("limit_deals_volume", pa.float64()),
            ("weighted_average_limit_rate", pa.float64()),
            ("first_leg_date", pa.string()),
            ("second_leg_date", pa.string()),
            ("cover_ratio", pa.float64()),
            ("key_rate", pa.float64()),
            ("rate_for_spread", pa.float64()),
            ("rate_spread", pa.float64()),
            ("Flag_Demand", pa.int64()),
            ("MAD_score_cover", pa.float64()),
            ("MAD_score_rate_spread", pa.float64()),
            ("MAD_score_cover_tier", pa.float64()),
            ("MAD_score_rate_spread_tier", pa.float64()),
        ]
    )
    ordered_rows = [
        {column: row.get(column) for column in OUTPUT_COLUMNS}
        for row in rows
    ]

    table = pa.Table.from_pylist(ordered_rows, schema=schema)
    pq.write_table(table, output_path)


# ----------------------------------------------------------------------------
# Daily term profile (Phase A: отдельный артефакт, final_dataset не трогаем)
# ----------------------------------------------------------------------------
def build_m2_daily_profile(
    input_path: Path = OUTPUT_FILE,
    *,
    global_window: int = ROLLING_WINDOW_GLOBAL_DAYS,
    local_window: int = ROLLING_WINDOW_LOCAL_DAYS,
):
    """Собирает ДНЕВНОЙ срочностной профиль M2 из per-auction фич.

    Разреженность аукционов обрабатывается явно (по решению):
    - tier-MAD и rate_spread сворачиваются в дневной ряд как last-known (ffill);
    - age_days = дней с последнего аукциона яруса (staleness), capped;
    - active = был ли аукцион яруса сегодня; available = свежесть в пределах горизонта;
    - shares/term_slope считаются в двух окнах (_w{global}/_w{local}) с флагом
      *_available; пропуски НЕ интерполируются (NaN -> 0 + availability-флаг).

    Возвращает pandas.DataFrame по календарю дней. Пишется в отдельный файл,
    текущий пайплайн и модель не затрагиваются.
    """
    import numpy as np
    import pandas as pd

    df = pd.read_csv(input_path)
    if df.empty:
        return pd.DataFrame()
    df["date"] = pd.to_datetime(df["date"], format="%d-%m-%Y", errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date")
    for col in ["total_deals_volume", "rate_spread", "MAD_score_cover_tier",
                "MAD_score_rate_spread_tier", "term_days"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    calendar = pd.DataFrame(
        {"date": pd.date_range(df["date"].min(), df["date"].max(), freq="D")}
    )

    def _tier_mask(name: str) -> pd.Series:
        if name == "long":
            return df["term_days"] >= LONG_TERM_THRESHOLD_DAYS
        low, high = TERM_TIERS[name]
        return (df["term_days"] >= low) & (df["term_days"] <= high)

    def _last_known(sub: pd.DataFrame, value_col: str, out_col: str) -> pd.DataFrame:
        """merge_asof: на каждый день — значение последнего прошедшего аукциона."""
        s = sub.dropna(subset=[value_col]).sort_values("date")[["date", value_col]]
        if s.empty:
            return pd.DataFrame({"date": calendar["date"], out_col: np.nan})
        return pd.merge_asof(calendar, s.rename(columns={value_col: out_col}),
                             on="date", direction="backward")

    out = calendar.copy()

    # --- по ярусам (short, base) + событийный long --------------------------
    for tier in ["short", "base", "long"]:
        sub = df[_tier_mask(tier)].copy()
        # активность сегодня
        active_dates = set(sub["date"])
        out[f"m2_{tier}_active"] = out["date"].isin(active_dates).astype(int)
        # last-known age (дней с последнего аукциона яруса)
        last = _last_known(sub.assign(_d=sub["date"]), "_d", "_last_date")
        age = (out["date"] - last["_last_date"]).dt.days
        out[f"m2_{tier}_age_days"] = age.clip(upper=AGE_CAP_DAYS).fillna(AGE_CAP_DAYS).astype(int)
        out[f"m2_{tier}_available"] = (age <= AVAILABILITY_HORIZON_DAYS).fillna(False).astype(int)
        # tier-MAD и rate_spread как last-known (только для непрерывных ярусов)
        if tier in ("short", "base"):
            cov = _last_known(sub, "MAD_score_cover_tier", f"m2_{tier}_cover_mad")
            rsp = _last_known(sub, "MAD_score_rate_spread_tier", f"m2_{tier}_ratespread_mad")
            out[f"m2_{tier}_cover_mad"] = cov[f"m2_{tier}_cover_mad"]
            out[f"m2_{tier}_ratespread_mad"] = rsp[f"m2_{tier}_ratespread_mad"]

    # --- оконные структурные фичи: short_share и term_slope -----------------
    df_idx = df.set_index("date").sort_index()
    short_vol = df_idx[_tier_mask("short").values]["total_deals_volume"]
    base_vol = df_idx[_tier_mask("base").values]["total_deals_volume"]

    def _rolling_sum_daily(series: pd.Series, window: int) -> pd.Series:
        if series.empty:
            return pd.Series(0.0, index=calendar["date"])
        daily = series.groupby(series.index).sum().reindex(
            calendar["date"], fill_value=0.0)
        return daily.rolling(window, min_periods=1).sum()

    # last-known rate_spread по ярусам (для slope)
    short_rsp = _last_known(df[_tier_mask("short")], "rate_spread", "v")["v"]
    base_rsp = _last_known(df[_tier_mask("base")], "rate_spread", "v")["v"]
    short_age = out["m2_short_age_days"]
    base_age = out["m2_base_age_days"]

    for tag, window in [("w%d" % global_window, global_window),
                        ("w%d" % local_window, local_window)]:
        sv = _rolling_sum_daily(short_vol, window).to_numpy()
        bv = _rolling_sum_daily(base_vol, window).to_numpy()
        total = sv + bv
        share = np.where(total > 0, sv / total, 0.0)
        out[f"m2_short_share_{tag}"] = np.round(share, 6)
        # term_slope = base_ratespread - short_ratespread (last-known)
        slope = (base_rsp - short_rsp)
        avail = ((short_age <= window) & (base_age <= window)).astype(int)
        out[f"m2_term_slope_{tag}"] = slope.fillna(0.0).round(6)
        out[f"m2_term_slope_available_{tag}"] = avail.to_numpy()

    out["date"] = out["date"].dt.strftime("%d-%m-%Y")
    return out


def save_daily_profile(
    profile,
    *,
    csv_path: Path = DAILY_PROFILE_FILE,
    parquet_path: Path = DAILY_PROFILE_PARQUET,
) -> None:
    """Сохраняет дневной term-профиль M2 в CSV и parquet."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    profile.to_csv(csv_path, index=False)
    try:
        profile.to_parquet(parquet_path, index=False)
    except Exception:
        pass


def main() -> None:
    """Запускает сборку feature dataset М2 и сохраняет результат"""
    rows = build_m2_features()
    save_csv(rows)
    save_parquet(rows)
    profile = build_m2_daily_profile()
    save_daily_profile(profile)
    print(f"Сохранено строк: {len(rows)}")
    print(f"Файл: {OUTPUT_FILE}")
    print(f"Файл parquet: {PARQUET_FILE}")
    print(f"Дневной term-профиль: {DAILY_PROFILE_FILE} ({len(profile)} строк)")


if __name__ == "__main__":
    main()
