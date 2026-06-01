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

START_DATE = datetime(2010, 1, 1)
MIN_DEALS_VOLUME = 1000.0
COVER_RATIO_LIMIT = 10.0
DEMAND_THRESHOLD = 2.0
ROLLING_WINDOW_DAYS = 365 * 3
MAD_MIN_VALUE = 0.05

OUTPUT_COLUMNS = [
    "date",
    "auction_type",
    "term_days",
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

    _add_mad_scores(prepared_rows, "cover_ratio", "MAD_score_cover")
    _add_mad_scores(prepared_rows, "rate_spread", "MAD_score_rate_spread")

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
        ]
    )
    ordered_rows = [
        {column: row.get(column) for column in OUTPUT_COLUMNS}
        for row in rows
    ]

    table = pa.Table.from_pylist(ordered_rows, schema=schema)
    pq.write_table(table, output_path)


def main() -> None:
    """Запускает сборку feature dataset М2 и сохраняет результат"""
    rows = build_m2_features()
    save_csv(rows)
    save_parquet(rows)
    print(f"Сохранено строк: {len(rows)}")
    print(f"Файл: {OUTPUT_FILE}")
    print(f"Файл parquet: {PARQUET_FILE}")


if __name__ == "__main__":
    main()
