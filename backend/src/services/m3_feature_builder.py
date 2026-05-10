from __future__ import annotations

import csv
from collections import defaultdict
from datetime import datetime
from datetime import timedelta
from pathlib import Path
from statistics import median


PROJECT_ROOT = Path(__file__).resolve().parents[3]
INPUT_FILE = PROJECT_ROOT / "data/processed/m3_dataset.csv"
OUTPUT_FILE = PROJECT_ROOT / "data/processed/m3_features.csv"
PARQUET_FILE = PROJECT_ROOT / "data/processed/m3_features.parquet"

NEDOSPROS_THRESHOLD = 1.2
PERESPROS_THRESHOLD = 2.0
ROLLING_WINDOW_DAYS = 365 * 3
MAD_MIN_VALUE = 0.05

OUTPUT_COLUMNS = [
    "date",
    "demand_amount",
    "offered_amount",
    "placed_amount",
    "weighted_yield",
    "cover_ratio",
    "yield_spread",
    "Flag_Nedospros",
    "Flag_Perespros",
    "MAD_score_cover",
    "MAD_score_yield_spread",
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


def _sum_values(rows: list[dict[str, str]], column: str) -> float:
    """Суммирует числовые значения колонки"""
    values = [_to_float(row.get(column)) for row in rows]
    return sum(value for value in values if value is not None)


def _weighted_average_yield(rows: list[dict[str, str]]) -> float | None:
    """Считает средневзвешенную доходность за день"""
    numerator = 0.0
    denominator = 0.0

    for row in rows:
        value = _to_float(row.get("weighted_average_yield"))
        weight = _to_float(row.get("placed_amount"))
        if value is None or weight is None or weight <= 0:
            continue

        numerator += value * weight
        denominator += weight

    if denominator == 0:
        return None

    return numerator / denominator


def _calculate_cover_ratio(
    demand_amount: float,
    offered_amount: float,
    placed_amount: float,
) -> float | None:
    """Считает дневной cover ratio по методике аналитика"""
    if offered_amount > 0:
        return demand_amount / offered_amount
    if placed_amount > 0:
        return demand_amount / placed_amount
    return None


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


def _add_mad_scores(rows: list[dict[str, object]], source_column: str, output_column: str) -> None:
    """Добавляет MAD-score по скользящему окну в 3 года"""
    window_delta = timedelta(days=ROLLING_WINDOW_DAYS)

    for row in rows:
        current_date = row["_date"]
        current_value = _to_float(row.get(source_column))
        if current_value is None:
            row[output_column] = None
            continue

        window_values = [
            _to_float(candidate.get(source_column))
            for candidate in rows
            if current_date - window_delta <= candidate["_date"] <= current_date
        ]
        window_values = [value for value in window_values if value is not None]

        rolling_median = _median(window_values)
        rolling_mad = _mad(window_values)
        if rolling_median is None or rolling_mad is None:
            row[output_column] = None
            continue

        rolling_mad = max(rolling_mad, MAD_MIN_VALUE)
        row[output_column] = (current_value - rolling_median) / rolling_mad


def build_m3_features(input_path: Path = INPUT_FILE) -> list[dict[str, object]]:
    """Собирает feature dataset М3 по требованиям аналитика"""
    source_rows = _read_csv(input_path)
    rows_by_date: dict[str, list[dict[str, str]]] = defaultdict(list)

    for row in source_rows:
        if row.get("auction_date"):
            rows_by_date[row["auction_date"]].append(row)

    result_rows: list[dict[str, object]] = []
    previous_weighted_yield: float | None = None

    for auction_date in sorted(rows_by_date, key=_date_sort_key):
        daily_rows = rows_by_date[auction_date]
        demand_amount = _sum_values(daily_rows, "demand_amount")
        offered_amount = _sum_values(daily_rows, "offered_amount")
        placed_amount = _sum_values(daily_rows, "placed_amount")
        weighted_yield = _weighted_average_yield(daily_rows)
        cover_ratio = _calculate_cover_ratio(
            demand_amount,
            offered_amount,
            placed_amount,
        )

        if weighted_yield is None or previous_weighted_yield is None:
            yield_spread = None
        else:
            yield_spread = weighted_yield - previous_weighted_yield

        if weighted_yield is not None:
            previous_weighted_yield = weighted_yield

        row: dict[str, object] = {
            "date": auction_date,
            "demand_amount": demand_amount,
            "offered_amount": offered_amount,
            "placed_amount": placed_amount,
            "weighted_yield": weighted_yield,
            "cover_ratio": cover_ratio,
            "yield_spread": yield_spread,
            "Flag_Nedospros": int(
                cover_ratio is not None and cover_ratio < NEDOSPROS_THRESHOLD
            ),
            "Flag_Perespros": int(
                cover_ratio is not None and cover_ratio > PERESPROS_THRESHOLD
            ),
            "_date": _date_sort_key(auction_date),
        }
        result_rows.append(row)

    _add_mad_scores(result_rows, "cover_ratio", "MAD_score_cover")
    _add_mad_scores(result_rows, "yield_spread", "MAD_score_yield_spread")

    for row in result_rows:
        del row["_date"]

    return result_rows


def save_csv(rows: list[dict[str, object]], output_path: Path = OUTPUT_FILE) -> None:
    """Сохраняет feature dataset М3 в CSV-файл"""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def save_parquet(
    rows: list[dict[str, object]],
    output_path: Path = PARQUET_FILE,
) -> None:
    """Сохраняет feature dataset М3 в Parquet-файл"""
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ModuleNotFoundError as error:
        raise RuntimeError("Для сохранения parquet нужен пакет pyarrow") from error

    output_path.parent.mkdir(parents=True, exist_ok=True)

    schema = pa.schema(
        [
            ("date", pa.string()),
            ("demand_amount", pa.float64()),
            ("offered_amount", pa.float64()),
            ("placed_amount", pa.float64()),
            ("weighted_yield", pa.float64()),
            ("cover_ratio", pa.float64()),
            ("yield_spread", pa.float64()),
            ("Flag_Nedospros", pa.int64()),
            ("Flag_Perespros", pa.int64()),
            ("MAD_score_cover", pa.float64()),
            ("MAD_score_yield_spread", pa.float64()),
        ]
    )
    ordered_rows = [
        {column: row.get(column) for column in OUTPUT_COLUMNS}
        for row in rows
    ]

    table = pa.Table.from_pylist(ordered_rows, schema=schema)
    pq.write_table(table, output_path)


def main() -> None:
    """Запускает сборку feature dataset М3 и сохраняет результат"""
    rows = build_m3_features()
    save_csv(rows)
    save_parquet(rows)
    print(f"Сохранено строк: {len(rows)}")
    print(f"Файл: {OUTPUT_FILE}")
    print(f"Файл parquet: {PARQUET_FILE}")


if __name__ == "__main__":
    main()
