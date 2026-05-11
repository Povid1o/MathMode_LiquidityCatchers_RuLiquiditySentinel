from __future__ import annotations

import csv
import math
from datetime import date
from datetime import datetime
from pathlib import Path
from statistics import median


PROJECT_ROOT = Path(__file__).resolve().parents[3]
INPUT_FILE = PROJECT_ROOT / "data/processed/m4_dataset.csv"
OUTPUT_FILE = PROJECT_ROOT / "data/processed/m4_features.csv"
PARQUET_FILE = PROJECT_ROOT / "data/processed/m4_features.parquet"

ENP_REFORM_DATE = date(2023, 1, 1)
DAILY_WINDOW = 365 * 3
MIN_MAD_PERIODS = DAILY_WINDOW // 4

TAX_WEIGHTS = {
    "profit_quarterly": 1.5,
    "profit_annual": 2.0,
    "quarter_end": 0.3,
    "year_end": 0.5,
}

OUTPUT_COLUMNS = [
    "date",
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
    "tax_pressure_smoothed",
    "tax_proximity",
    "MAD_tax_pressure",
    "MAD_tax_proximity",
    "Seasonal_Factor",
    "Seasonal_Factor_raw",
]

FLOAT_COLUMNS = {
    "tax_pressure",
    "tax_pressure_smoothed",
    "tax_proximity",
    "MAD_tax_pressure",
    "MAD_tax_proximity",
    "Seasonal_Factor",
    "Seasonal_Factor_raw",
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


def _to_float(value: str | object) -> float | None:
    """Преобразует значение в float"""
    if value is None or value == "":
        return None
    return float(value)


def _to_int(value: str | object) -> int:
    """Преобразует значение в int"""
    if value is None or value == "":
        return 0
    return int(float(value))


def _clip(value: float, lower: float, upper: float) -> float:
    """Ограничивает значение заданным диапазоном"""
    return min(max(value, lower), upper)


def _stable_value(column: str, value: object) -> object:
    """Приводит значение признака к стабильному виду для CSV и parquet"""
    if column in FLOAT_COLUMNS and value is not None:
        return round(float(value), 12)
    return value


def _centered_mean(values: list[float], index: int, half_window: int) -> float:
    """Считает центрированное скользящее среднее"""
    start = max(0, index - half_window)
    end = min(len(values), index + half_window + 1)
    window = values[start:end]
    return sum(window) / len(window)


def _median_abs_deviation(values: list[float]) -> float:
    """Считает медианное абсолютное отклонение"""
    center = median(values)
    return median([abs(value - center) for value in values])


def _mad_scores(values: list[float]) -> list[float | None]:
    """Считает daily MAD-score за скользящее окно 3 года"""
    result: list[float | None] = []

    for index, value in enumerate(values):
        window = values[max(0, index - DAILY_WINDOW + 1) : index + 1]
        if len(window) < MIN_MAD_PERIODS:
            result.append(None)
            continue

        rolling_median = median(window)
        rolling_mad = _median_abs_deviation(window)
        mad_floor = max(abs(rolling_median) * 0.01, 1e-6)
        safe_mad = max(rolling_mad, mad_floor)
        result.append(_clip((value - rolling_median) / safe_mad, -5, 5))

    return result


def build_m4_features(input_path: Path = INPUT_FILE) -> list[dict[str, object]]:
    """Собирает feature dataset М4 по налоговому календарю"""
    source_rows = sorted(_read_csv(input_path), key=lambda row: _parse_date(row["date"]))
    rows: list[dict[str, object]] = []

    for source_row in source_rows:
        row_date = _parse_date(source_row["date"])
        rows.append(
            {
                "_date": row_date,
                "is_tax_payment_day": _to_int(source_row["is_tax_payment_day"]),
                "tax_payment_events_count": _to_int(
                    source_row["tax_payment_events_count"]
                ),
                "days_to_next_tax_payment": _to_float(
                    source_row["days_to_next_tax_payment"]
                ),
                "days_since_prev_tax_payment": _to_float(
                    source_row["days_since_prev_tax_payment"]
                ),
                "is_month_end": _to_int(source_row["is_month_end"]),
                "is_quarter_end": _to_int(source_row["is_quarter_end"]),
                "is_year_end": _to_int(source_row["is_year_end"]),
                "is_weekend": int(row_date.weekday() >= 5),
            }
        )

    payment_flags = [int(row["is_tax_payment_day"]) for row in rows]
    tax_pressures: list[float] = []
    tax_proximities: list[float] = []
    seasonal_raw_values: list[float] = []

    for index, row in enumerate(rows):
        current_payment = payment_flags[index]
        next_payments = payment_flags[index + 1 : index + 4]
        previous_payments = payment_flags[max(0, index - 3) : index]

        tax_pre_flag = int(current_payment == 0 and max(next_payments or [0]) == 1)
        tax_active_flag = current_payment
        tax_post_flag = int(current_payment == 0 and max(previous_payments or [0]) == 1)
        tax_week_flag = max(tax_pre_flag, tax_active_flag, tax_post_flag)

        previous_payment = payment_flags[index - 1] if index > 0 else 0
        next_payment = payment_flags[index + 1] if index + 1 < len(payment_flags) else 0
        tax_day_strict = int(
            current_payment == 1 or previous_payment == 1 or next_payment == 1
        )

        days_to = min(max(_to_float(row["days_to_next_tax_payment"]) or 0.0, 0.0), 30.0)
        days_from = min(
            max(_to_float(row["days_since_prev_tax_payment"]) or 0.0, 0.0),
            30.0,
        )
        tax_proximity = max(math.exp(-days_to / 3.0), math.exp(-days_from / 3.0))

        tax_payment_events_count = int(row["tax_payment_events_count"])
        quarterly_bonus = (
            int(tax_payment_events_count >= 15) * TAX_WEIGHTS["profit_quarterly"] / 3
        )
        annual_bonus = (
            int(tax_payment_events_count >= 25)
            * (TAX_WEIGHTS["profit_annual"] - TAX_WEIGHTS["profit_quarterly"])
            / 3
        )
        quarter_end_bonus = int(row["is_quarter_end"]) * TAX_WEIGHTS["quarter_end"]
        year_end_bonus = int(row["is_year_end"]) * TAX_WEIGHTS["year_end"]
        tax_pressure = _clip(
            float(current_payment)
            + quarterly_bonus
            + annual_bonus
            + quarter_end_bonus
            + year_end_bonus,
            0,
            3,
        )

        seasonal_factor_raw = _clip(
            1.0
            + 0.15 * tax_week_flag
            + 0.10 * int(row["is_quarter_end"])
            + 0.20 * int(row["is_year_end"])
            + 0.05 * int(current_payment == 1 and int(row["is_weekend"]) == 1),
            1.0,
            1.4,
        )

        row.update(
            {
                "date": row["_date"].isoformat(),
                "Tax_Pre_Flag": tax_pre_flag,
                "Tax_Active_Flag": tax_active_flag,
                "Tax_Post_Flag": tax_post_flag,
                "Tax_Week_Flag": tax_week_flag,
                "Tax_Day_Strict": tax_day_strict,
                "Regime_Post_ENP": int(row["_date"] >= ENP_REFORM_DATE),
                "tax_pressure": tax_pressure,
                "tax_proximity": tax_proximity,
                "Seasonal_Factor_raw": seasonal_factor_raw,
            }
        )

        tax_pressures.append(tax_pressure)
        tax_proximities.append(tax_proximity)
        seasonal_raw_values.append(seasonal_factor_raw)

    tax_pressure_smoothed = [
        _centered_mean(tax_pressures, index, 3)
        for index in range(len(tax_pressures))
    ]
    seasonal_factors = [
        _centered_mean(seasonal_raw_values, index, 2)
        for index in range(len(seasonal_raw_values))
    ]
    mad_tax_pressure = _mad_scores(tax_pressures)
    mad_tax_proximity = _mad_scores(tax_proximities)

    result_rows: list[dict[str, object]] = []
    for index, row in enumerate(rows):
        row.update(
            {
                "tax_pressure_smoothed": tax_pressure_smoothed[index],
                "MAD_tax_pressure": mad_tax_pressure[index],
                "MAD_tax_proximity": mad_tax_proximity[index],
                "Seasonal_Factor": seasonal_factors[index],
            }
        )
        result_rows.append(
            {
                column: _stable_value(column, row.get(column))
                for column in OUTPUT_COLUMNS
            }
        )

    return result_rows


def save_csv(rows: list[dict[str, object]], output_path: Path = OUTPUT_FILE) -> None:
    """Сохраняет feature dataset М4 в CSV-файл"""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=OUTPUT_COLUMNS,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def save_parquet(
    rows: list[dict[str, object]],
    output_path: Path = PARQUET_FILE,
) -> None:
    """Сохраняет feature dataset М4 в Parquet-файл"""
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ModuleNotFoundError as error:
        raise RuntimeError("Для сохранения parquet нужен пакет pyarrow") from error

    output_path.parent.mkdir(parents=True, exist_ok=True)

    schema = pa.schema(
        [
            ("date", pa.string()),
            ("Tax_Pre_Flag", pa.int64()),
            ("Tax_Active_Flag", pa.int64()),
            ("Tax_Post_Flag", pa.int64()),
            ("Tax_Week_Flag", pa.int64()),
            ("Tax_Day_Strict", pa.int64()),
            ("is_quarter_end", pa.int64()),
            ("is_year_end", pa.int64()),
            ("is_month_end", pa.int64()),
            ("Regime_Post_ENP", pa.int64()),
            ("tax_pressure", pa.float64()),
            ("tax_pressure_smoothed", pa.float64()),
            ("tax_proximity", pa.float64()),
            ("MAD_tax_pressure", pa.float64()),
            ("MAD_tax_proximity", pa.float64()),
            ("Seasonal_Factor", pa.float64()),
            ("Seasonal_Factor_raw", pa.float64()),
        ]
    )
    ordered_rows = [
        {column: row.get(column) for column in OUTPUT_COLUMNS}
        for row in rows
    ]

    table = pa.Table.from_pylist(ordered_rows, schema=schema)
    pq.write_table(table, output_path)


def main() -> None:
    """Запускает сборку feature dataset М4 и сохраняет результат"""
    rows = build_m4_features()
    save_csv(rows)
    save_parquet(rows)
    print(f"Сохранено строк: {len(rows)}")
    print(f"Файл: {OUTPUT_FILE}")
    print(f"Файл parquet: {PARQUET_FILE}")


if __name__ == "__main__":
    main()
