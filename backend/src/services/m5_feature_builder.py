from __future__ import annotations

import csv
from datetime import date
from datetime import datetime
from datetime import timedelta
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
M5_DATASET_FILE = PROJECT_ROOT / "data/processed/m5_dataset.csv"
CBR_BUDGET_FUNDS_FILE = PROJECT_ROOT / "data/processed/cbr_budget_funds.csv"
ROSKAZNA_DEPOSITS_FILE = PROJECT_ROOT / "data/processed/roskazna_treasury_deposits.csv"
OUTPUT_FILE = PROJECT_ROOT / "data/processed/m5_features.csv"
PARQUET_FILE = PROJECT_ROOT / "data/processed/m5_features.parquet"

BUDGET_PUBLICATION_LAG_DAYS = 30
ROLLING_WINDOWS = (7, 14, 30)

OUTPUT_COLUMNS = [
    "date",
    "liquidity_deficit_surplus_bln_rub_lag_1d",
    "liquidity_deficit_surplus_bln_rub_change_1d",
    "liquidity_deficit_surplus_bln_rub_change_5d",
    "budget_funds_total_mln_rub_lagged",
    "budget_funds_total_mln_rub_change_lagged",
    "budget_funds_total_mln_rub_pct_change_lagged",
    "budget_funds_rub_mln_rub_lagged",
    "budget_funds_rub_mln_rub_change_lagged",
    "budget_funds_rub_mln_rub_pct_change_lagged",
    "budget_funds_rub_share_lagged",
    "roskazna_auction_day_flag_lag_1d",
    "roskazna_demand_volume_mln_rub_lag_1d",
    "roskazna_cover_ratio_lag_1d",
    "roskazna_bidders_count_lag_1d",
    "roskazna_first_leg_settled_volume_mln_rub",
    "roskazna_second_leg_return_volume_mln_rub",
    "roskazna_net_flow_by_legs_mln_rub",
    "roskazna_first_leg_auctions_count",
    "roskazna_second_leg_auctions_count",
    "roskazna_net_flow_rolling_7d_mln_rub",
    "roskazna_net_flow_rolling_14d_mln_rub",
    "roskazna_net_flow_rolling_30d_mln_rub",
    "roskazna_first_leg_rolling_7d_mln_rub",
    "roskazna_first_leg_rolling_14d_mln_rub",
    "roskazna_first_leg_rolling_30d_mln_rub",
    "roskazna_second_leg_rolling_7d_mln_rub",
    "roskazna_second_leg_rolling_14d_mln_rub",
    "roskazna_second_leg_rolling_30d_mln_rub",
    "days_since_last_roskazna_auction",
]

FLOAT_COLUMNS = {
    "liquidity_deficit_surplus_bln_rub_lag_1d",
    "liquidity_deficit_surplus_bln_rub_change_1d",
    "liquidity_deficit_surplus_bln_rub_change_5d",
    "budget_funds_total_mln_rub_lagged",
    "budget_funds_total_mln_rub_change_lagged",
    "budget_funds_total_mln_rub_pct_change_lagged",
    "budget_funds_rub_mln_rub_lagged",
    "budget_funds_rub_mln_rub_change_lagged",
    "budget_funds_rub_mln_rub_pct_change_lagged",
    "budget_funds_rub_share_lagged",
    "roskazna_demand_volume_mln_rub_lag_1d",
    "roskazna_cover_ratio_lag_1d",
    "roskazna_bidders_count_lag_1d",
    "roskazna_first_leg_settled_volume_mln_rub",
    "roskazna_second_leg_return_volume_mln_rub",
    "roskazna_net_flow_by_legs_mln_rub",
    "roskazna_net_flow_rolling_7d_mln_rub",
    "roskazna_net_flow_rolling_14d_mln_rub",
    "roskazna_net_flow_rolling_30d_mln_rub",
    "roskazna_first_leg_rolling_7d_mln_rub",
    "roskazna_first_leg_rolling_14d_mln_rub",
    "roskazna_first_leg_rolling_30d_mln_rub",
    "roskazna_second_leg_rolling_7d_mln_rub",
    "roskazna_second_leg_rolling_14d_mln_rub",
    "roskazna_second_leg_rolling_30d_mln_rub",
}

INTEGER_COLUMNS = {
    "roskazna_auction_day_flag_lag_1d",
    "roskazna_first_leg_auctions_count",
    "roskazna_second_leg_auctions_count",
    "days_since_last_roskazna_auction",
}


def _read_csv(path: Path) -> list[dict[str, str]]:
    """Читает CSV-файл в список словарей"""
    with path.open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def _parse_date(value: str) -> date:
    """Преобразует строковую дату в объект date"""
    for date_format in ("%d-%m-%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(value, date_format).date()
        except ValueError:
            continue
    raise ValueError(f"Не удалось прочитать дату: {value}")


def _to_float(value: str | object | None) -> float | None:
    """Преобразует значение в float"""
    if value is None or value == "":
        return None
    return float(value)


def _safe_ratio(numerator: float | None, denominator: float | None) -> float | None:
    """Считает отношение с защитой от пустых и нулевых значений"""
    if numerator is None or denominator in {None, 0}:
        return None
    return numerator / denominator


def _stable_value(column: str, value: object) -> object:
    """Приводит значение признака к стабильному виду для CSV и parquet"""
    if value is None:
        return None
    if column in FLOAT_COLUMNS:
        return round(float(value), 12)
    if column in INTEGER_COLUMNS:
        return int(value)
    return value


def _sum_values(rows: list[dict[str, object]], field_name: str) -> float | None:
    """Суммирует числовое поле по строкам"""
    values = [_to_float(row.get(field_name)) for row in rows]
    values = [value for value in values if value is not None]
    if not values:
        return None
    return sum(values)


def _deduplicate_roskazna_rows(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    """Удаляет дубли аукционов Росказны по auction_id"""
    selected_by_id: dict[str, dict[str, object]] = {}

    for row in rows:
        prepared_row: dict[str, object] = dict(row)
        result_fields = [
            "demand_volume_mln_rub",
            "accepted_volume_mln_rub",
            "settled_volume_mln_rub",
            "weighted_average_accepted_rate",
            "bidders_count",
            "accepted_bidders_count",
        ]
        result_non_nulls = sum(1 for field in result_fields if row.get(field) != "")
        settled_volume = _to_float(row.get("settled_volume_mln_rub")) or -1.0
        prepared_row["_dedup_score"] = (
            result_non_nulls,
            settled_volume,
            str(row.get("source_file", "")),
        )

        auction_id = str(row.get("auction_id", "")).strip()
        if not auction_id:
            auction_id = f"{row.get('auction_date', '')}:{row.get('source_file', '')}"

        previous_row = selected_by_id.get(auction_id)
        if previous_row is None or prepared_row["_dedup_score"] > previous_row["_dedup_score"]:
            selected_by_id[auction_id] = prepared_row

    result_rows: list[dict[str, object]] = []
    for row in selected_by_id.values():
        row.pop("_dedup_score", None)
        result_rows.append(row)

    return sorted(
        result_rows,
        key=lambda row: (
            _parse_date(str(row["auction_date"])),
            int(str(row["auction_id"])),
        ),
    )


def _aggregate_roskazna_by_date(
    rows: list[dict[str, object]],
    date_field: str,
) -> dict[date, dict[str, object]]:
    """Агрегирует аукционы Росказны по выбранному полю даты"""
    rows_by_date: dict[date, list[dict[str, object]]] = {}

    for row in rows:
        date_value = str(row.get(date_field, "")).strip()
        if not date_value:
            continue
        rows_by_date.setdefault(_parse_date(date_value), []).append(row)

    result: dict[date, dict[str, object]] = {}
    for row_date, daily_rows in rows_by_date.items():
        max_volume = _sum_values(daily_rows, "max_volume_mln_rub")
        demand_volume = _sum_values(daily_rows, "demand_volume_mln_rub")
        accepted_volume = _sum_values(daily_rows, "accepted_volume_mln_rub")
        settled_volume = _sum_values(daily_rows, "settled_volume_mln_rub")

        result[row_date] = {
            "auctions_count": len(daily_rows),
            "max_volume_mln_rub": max_volume,
            "demand_volume_mln_rub": demand_volume,
            "accepted_volume_mln_rub": accepted_volume,
            "settled_volume_mln_rub": settled_volume,
            "cover_ratio": _safe_ratio(demand_volume, max_volume),
            "bidders_count": _sum_values(daily_rows, "bidders_count"),
        }

    return result


def _budget_rows_by_currency(rows: list[dict[str, str]]) -> dict[str, list[dict[str, object]]]:
    """Готовит месячные бюджетные ряды по типу валюты"""
    result: dict[str, list[dict[str, object]]] = {}

    for row in rows:
        currency_type = row["currency_type"]
        prepared_row: dict[str, object] = dict(row)
        prepared_row["_date"] = _parse_date(row["date"])
        prepared_row["_available_date"] = (
            prepared_row["_date"] + timedelta(days=BUDGET_PUBLICATION_LAG_DAYS)
        )
        prepared_row["_level"] = _to_float(row["budget_funds_total_mln_rub"])
        result.setdefault(currency_type, []).append(prepared_row)

    for currency_rows in result.values():
        currency_rows.sort(key=lambda row: row["_date"])
        previous_level: float | None = None
        for row in currency_rows:
            level = _to_float(row["_level"])
            row["_change"] = None if level is None or previous_level is None else level - previous_level
            row["_pct_change"] = _safe_ratio(row["_change"], previous_level)
            previous_level = level

    return result


def _latest_available_budget_row(
    rows: list[dict[str, object]],
    row_date: date,
) -> dict[str, object] | None:
    """Находит последнюю месячную бюджетную запись с учетом лага публикации"""
    selected_row: dict[str, object] | None = None

    for row in rows:
        if row["_available_date"] <= row_date:
            selected_row = row
        else:
            break

    return selected_row


def _rolling_sum_by_date(
    values_by_date: dict[date, float],
    row_date: date,
    window_days: int,
) -> float:
    """Считает скользящую сумму за прошлое daily-окно включая текущую дату"""
    start_date = row_date - timedelta(days=window_days - 1)
    return sum(
        value
        for value_date, value in values_by_date.items()
        if start_date <= value_date <= row_date
    )


def _days_since_last_event(event_dates: list[date], row_date: date) -> int | None:
    """Считает число дней с последнего события"""
    previous_dates = [event_date for event_date in event_dates if event_date <= row_date]
    if not previous_dates:
        return None
    return (row_date - previous_dates[-1]).days


def build_m5_features(
    m5_dataset_path: Path = M5_DATASET_FILE,
    budget_funds_path: Path = CBR_BUDGET_FUNDS_FILE,
    roskazna_deposits_path: Path = ROSKAZNA_DEPOSITS_FILE,
) -> list[dict[str, object]]:
    """Собирает feature dataset М5 по результатам аналитики"""
    m5_rows = sorted(
        _read_csv(m5_dataset_path),
        key=lambda row: _parse_date(row["date"]),
    )
    budget_rows = _read_csv(budget_funds_path)
    roskazna_rows = _deduplicate_roskazna_rows(_read_csv(roskazna_deposits_path))

    budget_by_currency = _budget_rows_by_currency(budget_rows)
    total_budget_rows = budget_by_currency.get("total", [])
    rub_budget_rows = budget_by_currency.get("rub", [])

    auction_index = _aggregate_roskazna_by_date(roskazna_rows, "auction_date")
    first_leg_index = _aggregate_roskazna_by_date(roskazna_rows, "first_leg_date")
    second_leg_index = _aggregate_roskazna_by_date(roskazna_rows, "second_leg_date")

    auction_dates = sorted(auction_index)
    first_leg_values = {
        row_date: _to_float(row.get("settled_volume_mln_rub")) or 0.0
        for row_date, row in first_leg_index.items()
    }
    second_leg_values = {
        row_date: _to_float(row.get("settled_volume_mln_rub")) or 0.0
        for row_date, row in second_leg_index.items()
    }
    net_flow_values = {
        row_date: first_leg_values.get(row_date, 0.0) - second_leg_values.get(row_date, 0.0)
        for row_date in set(first_leg_values) | set(second_leg_values)
    }

    liquidity_values = [
        _to_float(row["liquidity_deficit_surplus_bln_rub"])
        for row in m5_rows
    ]

    result_rows: list[dict[str, object]] = []
    for index, row in enumerate(m5_rows):
        row_date = _parse_date(row["date"])
        previous_day = row_date - timedelta(days=1)
        auction_previous_day = auction_index.get(previous_day, {})
        first_leg_row = first_leg_index.get(row_date, {})
        second_leg_row = second_leg_index.get(row_date, {})
        total_budget_row = _latest_available_budget_row(total_budget_rows, row_date)
        rub_budget_row = _latest_available_budget_row(rub_budget_rows, row_date)

        liquidity_lag_1d = liquidity_values[index - 1] if index >= 1 else None
        liquidity_lag_2d = liquidity_values[index - 2] if index >= 2 else None
        liquidity_lag_6d = liquidity_values[index - 6] if index >= 6 else None

        first_leg_value = _to_float(first_leg_row.get("settled_volume_mln_rub")) or 0.0
        second_leg_value = _to_float(second_leg_row.get("settled_volume_mln_rub")) or 0.0

        feature_row = {
            "date": row_date.isoformat(),
            "liquidity_deficit_surplus_bln_rub_lag_1d": liquidity_lag_1d,
            "liquidity_deficit_surplus_bln_rub_change_1d": (
                None
                if liquidity_lag_1d is None or liquidity_lag_2d is None
                else liquidity_lag_1d - liquidity_lag_2d
            ),
            "liquidity_deficit_surplus_bln_rub_change_5d": (
                None
                if liquidity_lag_1d is None or liquidity_lag_6d is None
                else liquidity_lag_1d - liquidity_lag_6d
            ),
            "budget_funds_total_mln_rub_lagged": (
                total_budget_row["_level"] if total_budget_row else None
            ),
            "budget_funds_total_mln_rub_change_lagged": (
                total_budget_row["_change"] if total_budget_row else None
            ),
            "budget_funds_total_mln_rub_pct_change_lagged": (
                total_budget_row["_pct_change"] if total_budget_row else None
            ),
            "budget_funds_rub_mln_rub_lagged": (
                rub_budget_row["_level"] if rub_budget_row else None
            ),
            "budget_funds_rub_mln_rub_change_lagged": (
                rub_budget_row["_change"] if rub_budget_row else None
            ),
            "budget_funds_rub_mln_rub_pct_change_lagged": (
                rub_budget_row["_pct_change"] if rub_budget_row else None
            ),
            "budget_funds_rub_share_lagged": _safe_ratio(
                _to_float(rub_budget_row["_level"]) if rub_budget_row else None,
                _to_float(total_budget_row["_level"]) if total_budget_row else None,
            ),
            "roskazna_auction_day_flag_lag_1d": int(previous_day in auction_index),
            "roskazna_demand_volume_mln_rub_lag_1d": auction_previous_day.get(
                "demand_volume_mln_rub"
            ),
            "roskazna_cover_ratio_lag_1d": auction_previous_day.get("cover_ratio"),
            "roskazna_bidders_count_lag_1d": auction_previous_day.get("bidders_count"),
            "roskazna_first_leg_settled_volume_mln_rub": first_leg_value,
            "roskazna_second_leg_return_volume_mln_rub": second_leg_value,
            "roskazna_net_flow_by_legs_mln_rub": first_leg_value - second_leg_value,
            "roskazna_first_leg_auctions_count": int(
                first_leg_row.get("auctions_count") or 0
            ),
            "roskazna_second_leg_auctions_count": int(
                second_leg_row.get("auctions_count") or 0
            ),
            "days_since_last_roskazna_auction": _days_since_last_event(
                auction_dates,
                row_date,
            ),
        }

        for window_days in ROLLING_WINDOWS:
            feature_row[f"roskazna_net_flow_rolling_{window_days}d_mln_rub"] = (
                _rolling_sum_by_date(net_flow_values, row_date, window_days)
            )
            feature_row[f"roskazna_first_leg_rolling_{window_days}d_mln_rub"] = (
                _rolling_sum_by_date(first_leg_values, row_date, window_days)
            )
            feature_row[f"roskazna_second_leg_rolling_{window_days}d_mln_rub"] = (
                _rolling_sum_by_date(second_leg_values, row_date, window_days)
            )

        result_rows.append(
            {
                column: _stable_value(column, feature_row.get(column))
                for column in OUTPUT_COLUMNS
            }
        )

    removed_duplicates = len(_read_csv(roskazna_deposits_path)) - len(roskazna_rows)
    print(f"Удалено дублей auction_id Росказны при сборке признаков: {removed_duplicates}")

    return result_rows


def save_csv(rows: list[dict[str, object]], output_path: Path = OUTPUT_FILE) -> None:
    """Сохраняет feature dataset М5 в CSV-файл"""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=OUTPUT_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def save_parquet(
    rows: list[dict[str, object]],
    output_path: Path = PARQUET_FILE,
) -> None:
    """Сохраняет feature dataset М5 в Parquet-файл"""
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ModuleNotFoundError as error:
        raise RuntimeError("Для сохранения parquet нужен пакет pyarrow") from error

    output_path.parent.mkdir(parents=True, exist_ok=True)

    schema_fields = [("date", pa.string())]
    for column in OUTPUT_COLUMNS:
        if column == "date":
            continue
        if column in INTEGER_COLUMNS:
            schema_fields.append((column, pa.int64()))
        else:
            schema_fields.append((column, pa.float64()))

    schema = pa.schema(schema_fields)
    ordered_rows = [
        {column: row.get(column) for column in OUTPUT_COLUMNS}
        for row in rows
    ]

    table = pa.Table.from_pylist(ordered_rows, schema=schema)
    pq.write_table(table, output_path)


def main() -> None:
    """Запускает сборку feature dataset М5 и сохраняет результат"""
    rows = build_m5_features()
    save_csv(rows)
    save_parquet(rows)
    print(f"Сохранено строк: {len(rows)}")
    print(f"Файл: {OUTPUT_FILE}")
    print(f"Файл parquet: {PARQUET_FILE}")


if __name__ == "__main__":
    main()
