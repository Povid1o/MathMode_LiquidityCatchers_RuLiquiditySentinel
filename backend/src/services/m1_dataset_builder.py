from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parents[3]
REQUIRED_RESERVES_FILE = PROJECT_ROOT / "data/processed/required_reserves.csv"
RUONIA_FILE = PROJECT_ROOT / "data/processed/ruonia.csv"
CSV_OUTPUT_FILE = PROJECT_ROOT / "data/processed/m1_dataset.csv"
PARQUET_OUTPUT_FILE = PROJECT_ROOT / "data/processed/m1_dataset.parquet"

OUTPUT_COLUMNS = [
    "date",
    "averaging_period_end",
    "actual_balances",
    "required_reserves_avg",
    "accounting_reserves",
    "averaging_period_days",
    "spread",
    "ruonia_rate",
    "ruonia_transactions_volume",
    "ruonia_transactions_count",
    "ruonia_participants_count",
]


def _to_float(value: str) -> float | None:
    """Преобразует строку в число с плавающей точкой"""
    if value == "":
        return None
    return float(value)


def _to_int(value: str) -> int | None:
    """Преобразует строку в целое число"""
    if value == "":
        return None
    return int(float(value))


def _read_csv(path: Path) -> list[dict[str, str]]:
    """Читает CSV-файл в список словарей"""
    with path.open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def _date_sort_key(date_text: str) -> datetime:
    """Готовит строковую дату DD-MM-YYYY для сортировки"""
    return datetime.strptime(date_text, "%d-%m-%Y")


def _build_ruonia_index(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    """Создает словарь строк RUONIA по датам"""
    return {row["date"]: row for row in rows}


def build_m1_dataset(
    required_reserves_path: Path = REQUIRED_RESERVES_FILE,
    ruonia_path: Path = RUONIA_FILE,
) -> list[dict[str, object]]:
    """Собирает датасет М1 из обязательных резервов и RUONIA"""
    required_reserves_rows = _read_csv(required_reserves_path)
    ruonia_rows = _read_csv(ruonia_path)
    ruonia_by_date = _build_ruonia_index(ruonia_rows)

    result_rows: list[dict[str, object]] = []
    for row in required_reserves_rows:
        period_days = _to_int(row["averaging_period_days"])
        if period_days is None:
            continue

        ruonia_row = ruonia_by_date.get(row["date"], {})

        result_rows.append(
            {
                "date": row["date"],
                "averaging_period_end": row["averaging_period_end"],
                "actual_balances": _to_float(row["actual_balances"]),
                "required_reserves_avg": _to_float(row["required_reserves_avg"]),
                "accounting_reserves": _to_float(row["accounting_reserves"]),
                "averaging_period_days": period_days,
                "spread": _to_float(row["spread"]),
                "ruonia_rate": _to_float(ruonia_row.get("ruonia_rate", "")),
                "ruonia_transactions_volume": _to_float(
                    ruonia_row.get("transactions_volume", "")
                ),
                "ruonia_transactions_count": _to_int(
                    ruonia_row.get("transactions_count", "")
                ),
                "ruonia_participants_count": _to_int(
                    ruonia_row.get("participants_count", "")
                ),
            }
        )

    return sorted(result_rows, key=lambda item: _date_sort_key(str(item["date"])))


def save_csv(
    rows: list[dict[str, object]],
    output_path: Path = CSV_OUTPUT_FILE,
) -> None:
    """Сохраняет датасет М1 в CSV-файл"""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def save_parquet(
    rows: list[dict[str, object]],
    output_path: Path = PARQUET_OUTPUT_FILE,
) -> None:
    """Сохраняет датасет М1 в Parquet-файл"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    schema = pa.schema(
        [
            pa.field("date", pa.string()),
            pa.field("averaging_period_end", pa.string()),
            pa.field("actual_balances", pa.float64()),
            pa.field("required_reserves_avg", pa.float64()),
            pa.field("accounting_reserves", pa.float64()),
            pa.field("averaging_period_days", pa.int64()),
            pa.field("spread", pa.float64()),
            pa.field("ruonia_rate", pa.float64()),
            pa.field("ruonia_transactions_volume", pa.float64()),
            pa.field("ruonia_transactions_count", pa.int64()),
            pa.field("ruonia_participants_count", pa.int64()),
        ]
    )
    table = pa.Table.from_pylist(rows, schema=schema)
    pq.write_table(table, output_path)


def main() -> None:
    """Запускает сборку датасета М1 и сохраняет результат"""
    rows = build_m1_dataset()
    save_csv(rows)
    save_parquet(rows)
    print(f"Сохранено строк: {len(rows)}")
    print(f"CSV-файл: {CSV_OUTPUT_FILE}")
    print(f"Parquet-файл: {PARQUET_OUTPUT_FILE}")


if __name__ == "__main__":
    main()
