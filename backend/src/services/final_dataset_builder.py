from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]

M1_FEATURES_FILE = PROJECT_ROOT / "data/processed/m1_features.csv"
M2_FEATURES_FILE = PROJECT_ROOT / "data/processed/m2_features.csv"
M3_FEATURES_FILE = PROJECT_ROOT / "data/processed/m3_features.csv"
M4_FEATURES_FILE = PROJECT_ROOT / "data/processed/m4_features.csv"
M5_FEATURES_FILE = PROJECT_ROOT / "data/processed/m5_features.csv"

OUTPUT_FILE = PROJECT_ROOT / "data/processed/final_ml_dataset.csv"
PARQUET_FILE = PROJECT_ROOT / "data/processed/final_ml_dataset.parquet"

M1_COLUMNS = [
    "averaging_period_days",
    "actual_balances",
    "required_reserves_avg",
    "accounting_reserves",
    "full_reserves",
    "spread",
    "spread_relative",
    "spread_delta",
    "spread_ma3",
    "reserve_load",
    "ruonia_rate",
    "ruonia_period_avg",
    "ruonia_start",
    "flag_end_of_period",
    "spread_mad_score",
    "spread_relative_mad_score",
    "spread_delta_mad_score",
    "reserve_load_mad_score",
    "ruonia_mad_score",
    "m1_signal",
    "m1_signal_final",
    "m1_reliable",
]

M2_AGGREGATIONS = {
    "total_deals_volume": "sum",
    "demand_volume": "sum",
    "cover_ratio": "max",
    "key_rate": "last",
    "rate_for_spread": "max",
    "rate_spread": "max",
    "Flag_Demand": "max",
    "MAD_score_cover": "max",
    "MAD_score_rate_spread": "max",
}

M3_AGGREGATIONS = {
    "demand_amount": "sum",
    "offered_amount": "sum",
    "placed_amount": "sum",
    "weighted_yield": "max",
    "cover_ratio": "min",
    "yield_spread": "max",
    "Flag_Nedospros": "max",
    "Flag_Perespros": "max",
    "MAD_score_cover": "min",
    "MAD_score_yield_spread": "max",
}

M4_COLUMNS = [
    "Tax_Pre_Flag",
    "Tax_Active_Flag",
    "Tax_Post_Flag",
    "Tax_Week_Flag",
    "Tax_Day_Strict",
    "is_quarter_end",
    "is_year_end",
    "is_month_end",
    "Regime_Post_ENP",
    "tax_pressure",
    "tax_proximity",
    "MAD_tax_pressure",
    "MAD_tax_proximity",
    "Seasonal_Factor_raw",
]

M2_NO_AUCTION_ZERO_COLUMNS = [
    "m2_auction_count",
    "m2_auction_flag",
    "m2_total_deals_volume",
    "m2_demand_volume",
    "m2_Flag_Demand",
    "m2_MAD_score_cover",
    "m2_MAD_score_rate_spread",
]

M3_NO_AUCTION_ZERO_COLUMNS = [
    "m3_auction_flag",
    "m3_demand_amount",
    "m3_offered_amount",
    "m3_placed_amount",
    "m3_Flag_Nedospros",
    "m3_Flag_Perespros",
    "m3_MAD_score_cover",
    "m3_MAD_score_yield_spread",
    "m3_cover_stress_score",
    "m3_yield_stress_score",
]


def _read_csv(path: Path) -> pd.DataFrame:
    """Читает CSV и проверяет наличие файла"""
    if not path.exists():
        raise FileNotFoundError(f"Файл не найден: {path}")
    return pd.read_csv(path)


def _parse_dates(values: pd.Series, column_name: str) -> pd.Series:
    """Парсит даты DD-MM-YYYY и YYYY-MM-DD в нормализованный datetime"""
    parsed = pd.to_datetime(values, format="%d-%m-%Y", errors="coerce")
    missing_mask = parsed.isna()
    if missing_mask.any():
        parsed.loc[missing_mask] = pd.to_datetime(
            values.loc[missing_mask],
            format="%Y-%m-%d",
            errors="coerce",
        )

    if parsed.isna().any():
        bad_values = values.loc[parsed.isna()].drop_duplicates().head(5).tolist()
        raise ValueError(
            f"Не удалось прочитать даты в колонке {column_name}: {bad_values}"
        )

    return parsed.dt.normalize().astype("datetime64[ns]")


def _rename_with_prefix(
    frame: pd.DataFrame,
    prefix: str,
    columns: list[str],
) -> pd.DataFrame:
    """Оставляет нужные колонки и добавляет префикс модуля"""
    result = frame[["date"] + columns].copy()
    return result.rename(columns={column: f"{prefix}_{column}" for column in columns})


def _prepare_base_calendar(m5_features_path: Path) -> pd.DataFrame:
    """Готовит дневную сетку финального датасета по датам М5"""
    m5 = _read_csv(m5_features_path)
    m5["date"] = _parse_dates(m5["date"].astype(str), "m5_features.date")
    return pd.DataFrame({"date": sorted(m5["date"].drop_duplicates())})


def _prepare_m1(m1_features_path: Path) -> pd.DataFrame:
    """Готовит М1 для as-of join по окончанию периода усреднения"""
    m1 = _read_csv(m1_features_path)
    m1["m1_period_start"] = _parse_dates(m1["date"].astype(str), "m1_features.date")
    m1["m1_available_date"] = _parse_dates(
        m1["averaging_period_end"].astype(str),
        "m1_features.averaging_period_end",
    )
    m1["m1_averaging_period_end"] = m1["m1_available_date"]

    selected_columns = [
        "m1_available_date",
        "m1_period_start",
        "m1_averaging_period_end",
    ] + M1_COLUMNS

    result = m1[selected_columns].copy()
    result = result.rename(
        columns={
            column: column if column.startswith("m1_") else f"m1_{column}"
            for column in M1_COLUMNS
        }
    )
    return result.sort_values("m1_available_date")


def _prepare_m2(m2_features_path: Path) -> pd.DataFrame:
    """Готовит дневные признаки М2 без протягивания аукционов вперед"""
    m2 = _read_csv(m2_features_path)
    m2["date"] = _parse_dates(m2["date"].astype(str), "m2_features.date")

    m2_daily = (
        m2.sort_values("date")
        .groupby("date", as_index=False)
        .agg(M2_AGGREGATIONS)
    )
    auction_counts = m2.groupby("date").size().rename("m2_auction_count")
    m2_daily = m2_daily.merge(auction_counts, on="date", how="left")
    m2_daily["m2_auction_flag"] = 1
    m2_daily["m2_rate_spread_available"] = m2_daily["rate_spread"].notna().astype(int)
    m2_daily["m2_cover_available"] = m2_daily["cover_ratio"].notna().astype(int)

    return m2_daily.rename(
        columns={
            column: f"m2_{column}"
            for column in M2_AGGREGATIONS
        }
    )


def _prepare_m3(m3_features_path: Path) -> pd.DataFrame:
    """Готовит дневные признаки М3 без протягивания аукционов вперед"""
    m3 = _read_csv(m3_features_path)
    m3["date"] = _parse_dates(m3["date"].astype(str), "m3_features.date")

    m3_daily = (
        m3.sort_values("date")
        .groupby("date", as_index=False)
        .agg(M3_AGGREGATIONS)
    )
    m3_daily["m3_auction_flag"] = 1
    m3_daily["m3_cover_available"] = m3_daily["cover_ratio"].notna().astype(int)
    m3_daily["m3_yield_spread_available"] = (
        m3_daily["yield_spread"].notna().astype(int)
    )

    m3_daily = m3_daily.rename(
        columns={
            column: f"m3_{column}"
            for column in M3_AGGREGATIONS
        }
    )
    m3_daily["m3_cover_stress_score"] = -m3_daily["m3_MAD_score_cover"]
    m3_daily["m3_yield_stress_score"] = m3_daily["m3_MAD_score_yield_spread"]
    return m3_daily


def _prepare_m4(m4_features_path: Path) -> pd.DataFrame:
    """Готовит календарные признаки М4 exact-date"""
    m4 = _read_csv(m4_features_path)
    m4["date"] = _parse_dates(m4["date"].astype(str), "m4_features.date")
    m4 = _rename_with_prefix(m4, "m4", M4_COLUMNS)
    m4["m4_calendar_available"] = 1
    return m4


def _prepare_m5(m5_features_path: Path) -> pd.DataFrame:
    """Готовит дневные признаки М5 exact-date"""
    m5 = _read_csv(m5_features_path)
    m5["date"] = _parse_dates(m5["date"].astype(str), "m5_features.date")
    feature_columns = [column for column in m5.columns if column != "date"]
    m5 = m5.rename(columns={column: f"m5_{column}" for column in feature_columns})
    m5["m5_features_available"] = 1
    return m5


def _fill_event_absence(final_dataset: pd.DataFrame) -> pd.DataFrame:
    """Заполняет нулями только дни без аукционных событий М2 и М3"""
    result = final_dataset.copy()

    m2_no_auction = result["m2_auction_flag"].isna()
    for column in M2_NO_AUCTION_ZERO_COLUMNS:
        result.loc[m2_no_auction, column] = 0
    result.loc[m2_no_auction, "m2_rate_spread_available"] = 0
    result.loc[m2_no_auction, "m2_cover_available"] = 0

    m3_no_auction = result["m3_auction_flag"].isna()
    for column in M3_NO_AUCTION_ZERO_COLUMNS:
        result.loc[m3_no_auction, column] = 0
    result.loc[m3_no_auction, "m3_cover_available"] = 0
    result.loc[m3_no_auction, "m3_yield_spread_available"] = 0

    return result


def _add_availability_columns(final_dataset: pd.DataFrame) -> pd.DataFrame:
    """Добавляет технические признаки доступности модулей"""
    result = final_dataset.copy()

    result["m1_days_since_available"] = (
        result["date"] - result["m1_available_date"]
    ).dt.days
    result["m1_features_available"] = result["m1_available_date"].notna().astype(int)
    result["m3_history_available"] = (
        result["date"] >= result.loc[result["m3_auction_flag"].eq(1), "date"].min()
    ).astype(int)

    return result


def _format_dates_for_output(final_dataset: pd.DataFrame) -> pd.DataFrame:
    """Форматирует даты в ISO-строки перед сохранением"""
    result = final_dataset.copy()
    for column in result.columns:
        if column == "date" or column.endswith("_date") or column.endswith("_start"):
            if pd.api.types.is_datetime64_any_dtype(result[column]):
                result[column] = result[column].dt.strftime("%Y-%m-%d")
    return result


def build_final_ml_dataset(
    m1_features_path: Path = M1_FEATURES_FILE,
    m2_features_path: Path = M2_FEATURES_FILE,
    m3_features_path: Path = M3_FEATURES_FILE,
    m4_features_path: Path = M4_FEATURES_FILE,
    m5_features_path: Path = M5_FEATURES_FILE,
) -> pd.DataFrame:
    """Собирает финальный дневной ML dataset из признаков М1-М5"""
    base = _prepare_base_calendar(m5_features_path)

    final_dataset = pd.merge_asof(
        base.sort_values("date"),
        _prepare_m1(m1_features_path),
        left_on="date",
        right_on="m1_available_date",
        direction="backward",
    )
    final_dataset = final_dataset.merge(_prepare_m2(m2_features_path), on="date", how="left")
    final_dataset = final_dataset.merge(_prepare_m3(m3_features_path), on="date", how="left")
    final_dataset = final_dataset.merge(_prepare_m4(m4_features_path), on="date", how="left")
    final_dataset = final_dataset.merge(_prepare_m5(m5_features_path), on="date", how="left")

    final_dataset = _fill_event_absence(final_dataset)
    final_dataset = _add_availability_columns(final_dataset)

    return final_dataset.sort_values("date").reset_index(drop=True)


def save_csv(
    frame: pd.DataFrame,
    output_path: Path = OUTPUT_FILE,
) -> None:
    """Сохраняет финальный ML dataset в CSV"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _format_dates_for_output(frame).to_csv(output_path, index=False)


def save_parquet(
    frame: pd.DataFrame,
    output_path: Path = PARQUET_FILE,
) -> None:
    """Сохраняет финальный ML dataset в parquet"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _format_dates_for_output(frame).to_parquet(output_path, index=False)


def main() -> None:
    """Запускает сборку финального ML dataset"""
    final_dataset = build_final_ml_dataset()
    save_csv(final_dataset)
    save_parquet(final_dataset)
    print(f"Сохранено строк: {len(final_dataset)}")
    print(f"Сохранено колонок: {len(final_dataset.columns)}")
    print(f"Файл: {OUTPUT_FILE}")
    print(f"Файл parquet: {PARQUET_FILE}")


if __name__ == "__main__":
    main()
