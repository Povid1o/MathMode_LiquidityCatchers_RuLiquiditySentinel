from __future__ import annotations

import csv
from bisect import bisect_right
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
REPO_FILE = PROJECT_ROOT / "data/processed/repo.csv"
KEYRATE_FILE = PROJECT_ROOT / "data/processed/keyrate.csv"
OUTPUT_FILE = PROJECT_ROOT / "data/processed/m2_dataset.csv"

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


def _sort_term_days(value: object) -> int:
    """Готовит срок аукциона для сортировки"""
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str) and value != "":
        return int(float(value))
    return 0


def _read_csv(path: Path) -> list[dict[str, str]]:
    """Читает CSV-файл в список словарей"""
    with path.open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def _date_sort_key(date_text: str) -> datetime:
    """Готовит строковую дату DD-MM-YYYY для сортировки"""
    return datetime.strptime(date_text, "%d-%m-%Y")


def _build_keyrate_timeline(
    rows: list[dict[str, str]],
) -> list[tuple[datetime, dict[str, str]]]:
    """Создает временную шкалу ключевой ставки"""
    return sorted((_date_sort_key(row["date"]), row) for row in rows)


def _find_keyrate_for_date(
    timeline: list[tuple[datetime, dict[str, str]]],
    timeline_dates: list[datetime],
    repo_date: str,
) -> dict[str, str]:
    """Находит последнюю известную ключевую ставку на дату аукциона"""
    if not timeline:
        return {}

    index = bisect_right(timeline_dates, _date_sort_key(repo_date)) - 1
    if index < 0:
        return {}

    return timeline[index][1]


def build_m2_dataset(
    repo_path: Path = REPO_FILE,
    keyrate_path: Path = KEYRATE_FILE,
) -> list[dict[str, object]]:
    """Собирает датасет М2 из итогов репо и ключевой ставки"""
    repo_rows = _read_csv(repo_path)
    keyrate_rows = _read_csv(keyrate_path)
    keyrate_timeline = _build_keyrate_timeline(keyrate_rows)
    keyrate_dates = [item[0] for item in keyrate_timeline]

    result_rows: list[dict[str, object]] = []
    for row in repo_rows:
        keyrate_row = _find_keyrate_for_date(
            keyrate_timeline,
            keyrate_dates,
            row["date"],
        )

        result_rows.append(
            {
                "date": row["date"],
                "auction_type": row["auction_type"],
                "term_days": _to_int(row["term_days"]),
                "auction_time": row["auction_time"],
                "total_deals_volume": _to_float(row["total_deals_volume"]),
                "weighted_average_rate": _to_float(row["weighted_average_rate"]),
                "settlement_code": row["settlement_code"],
                "demand_volume": _to_float(row.get("demand_volume", "")),
                "cutoff_rate": _to_float(row.get("cutoff_rate", "")),
                "min_rate": _to_float(row.get("min_rate", "")),
                "max_rate": _to_float(row.get("max_rate", "")),
                "limit_deals_volume": _to_float(row.get("limit_deals_volume", "")),
                "weighted_average_limit_rate": _to_float(
                    row.get("weighted_average_limit_rate", "")
                ),
                "first_leg_date": row.get("first_leg_date", ""),
                "second_leg_date": row.get("second_leg_date", ""),
                "cover_ratio": _to_float(row.get("cover_ratio", "")),
                "key_rate": _to_float(keyrate_row.get("key_rate", "")),
            }
        )

    return sorted(
        result_rows,
        key=lambda item: (
            _date_sort_key(str(item["date"])),
            str(item["auction_time"]),
            _sort_term_days(item["term_days"]),
        ),
    )


def save_csv(rows: list[dict[str, object]], output_path: Path = OUTPUT_FILE) -> None:
    """Сохраняет датасет М2 в CSV-файл"""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """Запускает сборку датасета М2 и сохраняет результат"""
    rows = build_m2_dataset()
    save_csv(rows)
    print(f"Сохранено строк: {len(rows)}")
    print(f"Файл: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
