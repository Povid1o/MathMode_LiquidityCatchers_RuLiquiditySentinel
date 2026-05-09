from __future__ import annotations

import csv
from datetime import date, datetime
from pathlib import Path
from statistics import median


PROJECT_ROOT = Path(__file__).resolve().parents[3]
M1_DATASET_FILE = PROJECT_ROOT / "data/processed/m1_dataset.csv"
RUONIA_FILE = PROJECT_ROOT / "data/processed/ruonia.csv"
OUTPUT_FILE = PROJECT_ROOT / "data/processed/m1_features.csv"
MAD_WINDOW = 36

OUTPUT_COLUMNS = [
    "date",
    "averaging_period_end",
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


def _read_csv(path: Path) -> list[dict[str, str]]:
    """Читает CSV-файл в список словарей"""
    with path.open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def _date_from_string(value: str) -> date:
    """Преобразует строковую дату DD-MM-YYYY в объект date"""
    return datetime.strptime(value, "%d-%m-%Y").date()


def _date_sort_key(date_text: str) -> datetime:
    """Готовит строковую дату DD-MM-YYYY для сортировки"""
    return datetime.strptime(date_text, "%d-%m-%Y")


def _to_float(value: str | None) -> float | None:
    """Преобразует строку в число с плавающей точкой"""
    if value is None or value == "":
        return None
    return float(value)


def _to_int(value: str | None) -> int | None:
    """Преобразует строку в целое число"""
    if value is None or value == "":
        return None
    return int(float(value))


def _safe_divide(numerator: float | None, denominator: float | None) -> float | None:
    """Делит числа с защитой от пустого или нулевого знаменателя"""
    if numerator is None or denominator in (None, 0):
        return None
    return numerator / denominator


def _mean(values: list[float]) -> float | None:
    """Считает среднее значение непустого списка"""
    if not values:
        return None
    return sum(values) / len(values)


def _rolling_mean(values: list[float | None], window: int) -> list[float | None]:
    """Считает скользящее среднее по полному окну"""
    result: list[float | None] = []
    for index in range(len(values)):
        current_window = values[max(0, index - window + 1) : index + 1]
        if len(current_window) < window or any(value is None for value in current_window):
            result.append(None)
            continue
        result.append(sum(value for value in current_window if value is not None) / window)
    return result


def _median_abs_deviation(values: list[float]) -> float:
    """Считает медианное абсолютное отклонение"""
    window_median = median(values)
    return median([abs(value - window_median) for value in values])


def _mad_scores(values: list[float | None], window: int) -> list[float | None]:
    """Считает rolling MAD-score по полному окну"""
    result: list[float | None] = []
    for index, value in enumerate(values):
        current_window = values[max(0, index - window + 1) : index + 1]
        if (
            value is None
            or len(current_window) < window
            or any(item is None for item in current_window)
        ):
            result.append(None)
            continue

        clean_window = [item for item in current_window if item is not None]
        window_median = median(clean_window)
        window_mad = _median_abs_deviation(clean_window)
        mad_floor = max(abs(window_median) * 0.01, 1e-6)
        safe_mad = max(window_mad, mad_floor)
        score = (value - window_median) / safe_mad
        result.append(min(max(score, -5), 5))
    return result


def _forward_fill(values: list[float | None]) -> list[float | None]:
    """Заполняет пустые значения последним известным значением"""
    result: list[float | None] = []
    last_value: float | None = None
    for value in values:
        if value is not None:
            last_value = value
        result.append(last_value)
    return result


def _build_ruonia_points(rows: list[dict[str, str]]) -> list[tuple[date, float]]:
    """Готовит отсортированные дневные значения RUONIA"""
    points: list[tuple[date, float]] = []
    for row in rows:
        ruonia_rate = _to_float(row.get("ruonia_rate"))
        if ruonia_rate is None:
            continue
        points.append((_date_from_string(row["date"]), ruonia_rate))
    return sorted(points, key=lambda item: item[0])


def _ruonia_period_avg(
    points: list[tuple[date, float]],
    start_date: date,
    end_date: date,
) -> float | None:
    """Считает среднюю RUONIA за период усреднения"""
    values = [
        value
        for point_date, value in points
        if start_date <= point_date <= end_date
    ]
    return _mean(values)


def _ruonia_start(points: list[tuple[date, float]], start_date: date) -> float | None:
    """Находит последнее значение RUONIA на начало периода"""
    result: float | None = None
    for point_date, value in points:
        if point_date > start_date:
            break
        result = value
    return result


def _calculate_signal(
    spread_mad_score: float | None,
    spread_relative_mad_score: float | None,
    spread_delta_mad_score: float | None,
    ruonia_mad_score: float | None,
    flag_end_of_period: int,
) -> tuple[float, float]:
    """Считает сигнал М1 по логике из аналитического ноутбука"""
    if ruonia_mad_score is None:
        signal = (
            0.467 * (spread_mad_score or 0)
            + 0.333 * (spread_relative_mad_score or 0)
            + 0.200 * (spread_delta_mad_score or 0)
        )
    else:
        signal = (
            0.35 * (spread_mad_score or 0)
            + 0.25 * (spread_relative_mad_score or 0)
            + 0.25 * ruonia_mad_score
            + 0.15 * (spread_delta_mad_score or 0)
        )

    signal = min(max(signal, -5), 5)
    final_signal = signal
    if flag_end_of_period == 1 and signal > 0:
        final_signal = min(max(signal * 1.15, -5), 5)
    return signal, final_signal


def build_m1_features(
    m1_dataset_path: Path = M1_DATASET_FILE,
    ruonia_path: Path = RUONIA_FILE,
) -> list[dict[str, object]]:
    """Собирает признаки М1 для аналитика и ML"""
    m1_rows = sorted(_read_csv(m1_dataset_path), key=lambda row: _date_sort_key(row["date"]))
    ruonia_points = _build_ruonia_points(_read_csv(ruonia_path))

    base_rows: list[dict[str, object]] = []
    spreads: list[float | None] = []
    spread_relatives: list[float | None] = []
    spread_deltas: list[float | None] = []
    reserve_loads: list[float | None] = []
    ruonia_period_values: list[float | None] = []

    previous_spread: float | None = None
    for row in m1_rows:
        start_date = _date_from_string(row["date"])
        end_date = _date_from_string(row["averaging_period_end"])
        actual_balances = _to_float(row["actual_balances"])
        required_reserves_avg = _to_float(row["required_reserves_avg"])
        accounting_reserves = _to_float(row["accounting_reserves"])
        spread = _to_float(row["spread"])
        period_days = _to_int(row["averaging_period_days"])

        full_reserves = None
        if required_reserves_avg is not None:
            full_reserves = required_reserves_avg + (accounting_reserves or 0)

        spread_relative = _safe_divide(spread, required_reserves_avg)
        if spread_relative is not None:
            spread_relative *= 100

        spread_delta = None
        if spread is not None and previous_spread is not None:
            spread_delta = spread - previous_spread
        if spread is not None:
            previous_spread = spread

        reserve_load = _safe_divide(full_reserves, actual_balances)
        if reserve_load is not None:
            reserve_load *= 100

        ruonia_period_avg = _ruonia_period_avg(ruonia_points, start_date, end_date)
        ruonia_value_start = _ruonia_start(ruonia_points, start_date)
        flag_end_of_period = 1 if end_date.day >= 25 else 0

        base_rows.append(
            {
                "date": row["date"],
                "averaging_period_end": row["averaging_period_end"],
                "averaging_period_days": period_days,
                "actual_balances": actual_balances,
                "required_reserves_avg": required_reserves_avg,
                "accounting_reserves": accounting_reserves,
                "full_reserves": full_reserves,
                "spread": spread,
                "spread_relative": spread_relative,
                "spread_delta": spread_delta,
                "reserve_load": reserve_load,
                "ruonia_rate": _to_float(row.get("ruonia_rate")),
                "ruonia_period_avg": ruonia_period_avg,
                "ruonia_start": ruonia_value_start,
                "flag_end_of_period": flag_end_of_period,
            }
        )

        spreads.append(spread)
        spread_relatives.append(spread_relative)
        spread_deltas.append(spread_delta if spread_delta is not None else 0)
        reserve_loads.append(reserve_load)
        ruonia_period_values.append(ruonia_period_avg)

    spread_ma3_values = _rolling_mean(spreads, 3)
    spread_mad_scores = _mad_scores(spreads, MAD_WINDOW)
    spread_relative_mad_scores = _mad_scores(spread_relatives, MAD_WINDOW)
    spread_delta_mad_scores = _mad_scores(spread_deltas, MAD_WINDOW)
    reserve_load_mad_scores = _mad_scores(reserve_loads, MAD_WINDOW)
    ruonia_mad_scores = _mad_scores(_forward_fill(ruonia_period_values), MAD_WINDOW)

    result_rows: list[dict[str, object]] = []
    for index, row in enumerate(base_rows):
        spread_mad_score = spread_mad_scores[index]
        spread_relative_mad_score = spread_relative_mad_scores[index]
        spread_delta_mad_score = spread_delta_mad_scores[index]
        ruonia_mad_score = ruonia_mad_scores[index]

        m1_signal, m1_signal_final = _calculate_signal(
            spread_mad_score,
            spread_relative_mad_score,
            spread_delta_mad_score,
            ruonia_mad_score,
            int(row["flag_end_of_period"] or 0),
        )

        row.update(
            {
                "spread_ma3": spread_ma3_values[index],
                "spread_mad_score": spread_mad_score,
                "spread_relative_mad_score": spread_relative_mad_score,
                "spread_delta_mad_score": spread_delta_mad_score,
                "reserve_load_mad_score": reserve_load_mad_scores[index],
                "ruonia_mad_score": ruonia_mad_score,
                "m1_signal": m1_signal,
                "m1_signal_final": m1_signal_final,
                "m1_reliable": 1 if spread_mad_score is not None else 0,
            }
        )
        result_rows.append(row)

    return result_rows


def save_csv(rows: list[dict[str, object]], output_path: Path = OUTPUT_FILE) -> None:
    """Сохраняет признаки М1 в CSV-файл"""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """Запускает сборку признаков М1 и сохраняет результат"""
    rows = build_m1_features()
    save_csv(rows)
    print(f"Сохранено строк: {len(rows)}")
    print(f"Файл: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
