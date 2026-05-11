from __future__ import annotations

import csv
from datetime import date
from datetime import datetime
from datetime import timedelta
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
TAX_CALENDAR_FILE = PROJECT_ROOT / "data/processed/tax_calendar.csv"
OUTPUT_FILE = PROJECT_ROOT / "data/processed/m4_dataset.csv"

OUTPUT_COLUMNS = [
    "date",
    "is_tax_payment_day",
    "is_tax_reporting_day",
    "is_notification_day",
    "tax_events_count",
    "tax_payment_events_count",
    "tax_reporting_events_count",
    "notification_events_count",
    "other_events_count",
    "days_to_next_tax_payment",
    "days_since_prev_tax_payment",
    "is_month_end",
    "is_quarter_end",
    "is_year_end",
]


def _read_csv(path: Path) -> list[dict[str, str]]:
    """Читает CSV-файл в список словарей"""
    with path.open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def _parse_date(value: str) -> date:
    """Преобразует дату DD-MM-YYYY в объект date"""
    return datetime.strptime(value, "%d-%m-%Y").date()


def _format_date(value: date) -> str:
    """Преобразует объект date в DD-MM-YYYY"""
    return value.strftime("%d-%m-%Y")


def _daterange(start_date: date, end_date: date) -> list[date]:
    """Возвращает список календарных дат в заданном диапазоне"""
    days_count = (end_date - start_date).days
    return [start_date + timedelta(days=index) for index in range(days_count + 1)]


def _is_month_end(value: date) -> int:
    """Проверяет, является ли дата последним днем месяца"""
    next_day = value + timedelta(days=1)
    return int(next_day.month != value.month)


def _is_quarter_end(value: date) -> int:
    """Проверяет, является ли дата последним днем квартала"""
    return int(_is_month_end(value) and value.month in {3, 6, 9, 12})


def _is_year_end(value: date) -> int:
    """Проверяет, является ли дата последним днем года"""
    return int(value.month == 12 and value.day == 31)


def _next_payment_distance(value: date, payment_dates: list[date]) -> int | None:
    """Считает количество дней до следующей налоговой даты платежа"""
    for payment_date in payment_dates:
        if payment_date >= value:
            return (payment_date - value).days
    return None


def _previous_payment_distance(value: date, payment_dates: list[date]) -> int | None:
    """Считает количество дней после предыдущей налоговой даты платежа"""
    previous_dates = [payment_date for payment_date in payment_dates if payment_date <= value]
    if not previous_dates:
        return None
    return (value - previous_dates[-1]).days


def _empty_counts() -> dict[str, int]:
    """Создает пустые счетчики событий для одной даты"""
    return {
        "tax_events_count": 0,
        "tax_payment_events_count": 0,
        "tax_reporting_events_count": 0,
        "notification_events_count": 0,
        "other_events_count": 0,
    }


def _events_by_date(rows: list[dict[str, str]]) -> dict[date, dict[str, int]]:
    """Агрегирует события налогового календаря по датам"""
    result: dict[date, dict[str, int]] = {}

    for row in rows:
        event_date = _parse_date(row["event_date"])
        counts = result.setdefault(event_date, _empty_counts())
        event_type = row["event_type"]

        counts["tax_events_count"] += 1
        if event_type in {"payment_deadline", "mixed"}:
            counts["tax_payment_events_count"] += 1
        if event_type in {"reporting_deadline", "mixed"}:
            counts["tax_reporting_events_count"] += 1
        if event_type in {"notification_deadline", "mixed"}:
            counts["notification_events_count"] += 1
        if event_type == "other":
            counts["other_events_count"] += 1

    return result


def build_m4_dataset(
    tax_calendar_path: Path = TAX_CALENDAR_FILE,
) -> list[dict[str, object]]:
    """Собирает базовый дневной датасет М4 по налоговому календарю"""
    rows = _read_csv(tax_calendar_path)
    if not rows:
        return []

    counts_by_date = _events_by_date(rows)
    all_dates = sorted(counts_by_date)
    payment_dates = sorted(
        event_date
        for event_date, counts in counts_by_date.items()
        if counts["tax_payment_events_count"] > 0
    )

    result_rows: list[dict[str, object]] = []
    for current_date in _daterange(all_dates[0], all_dates[-1]):
        counts = counts_by_date.get(current_date, _empty_counts())
        payment_count = counts["tax_payment_events_count"]
        reporting_count = counts["tax_reporting_events_count"]
        notification_count = counts["notification_events_count"]

        result_rows.append(
            {
                "date": _format_date(current_date),
                "is_tax_payment_day": int(payment_count > 0),
                "is_tax_reporting_day": int(reporting_count > 0),
                "is_notification_day": int(notification_count > 0),
                "tax_events_count": counts["tax_events_count"],
                "tax_payment_events_count": payment_count,
                "tax_reporting_events_count": reporting_count,
                "notification_events_count": notification_count,
                "other_events_count": counts["other_events_count"],
                "days_to_next_tax_payment": _next_payment_distance(
                    current_date,
                    payment_dates,
                ),
                "days_since_prev_tax_payment": _previous_payment_distance(
                    current_date,
                    payment_dates,
                ),
                "is_month_end": _is_month_end(current_date),
                "is_quarter_end": _is_quarter_end(current_date),
                "is_year_end": _is_year_end(current_date),
            }
        )

    return result_rows


def save_csv(rows: list[dict[str, object]], output_path: Path = OUTPUT_FILE) -> None:
    """Сохраняет датасет М4 в CSV-файл"""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """Запускает сборку датасета М4 и сохраняет результат"""
    rows = build_m4_dataset()
    save_csv(rows)
    print(f"Сохранено строк: {len(rows)}")
    print(f"Файл: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
