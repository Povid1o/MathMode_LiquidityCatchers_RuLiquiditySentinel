from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
OFZ_AUCTIONS_FILE = PROJECT_ROOT / "data/processed/ofz_auctions.csv"
OUTPUT_FILE = PROJECT_ROOT / "data/processed/m3_dataset.csv"

OUTPUT_COLUMNS = [
    "auction_date",
    "published_date",
    "document_title",
    "auction_format",
    "issue",
    "security_type",
    "maturity_date",
    "days_to_maturity",
    "offered_amount",
    "demand_amount",
    "placed_amount",
    "proceeds_amount",
    "cutoff_price",
    "weighted_average_price",
    "cutoff_yield",
    "weighted_average_yield",
    "official_coefficient",
    "cover_ratio",
    "placement_ratio",
    "source_url",
    "source_file",
]


def _read_csv(path: Path) -> list[dict[str, str]]:
    """Читает CSV-файл в список словарей"""
    with path.open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


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


def _date_sort_key(value: object) -> datetime:
    """Готовит дату DD-MM-YYYY для сортировки"""
    return datetime.strptime(str(value), "%d-%m-%Y")


def build_m3_dataset(
    ofz_auctions_path: Path = OFZ_AUCTIONS_FILE,
) -> list[dict[str, object]]:
    """Собирает базовый датасет М3 по аукционам ОФЗ"""
    rows = _read_csv(ofz_auctions_path)

    result_rows: list[dict[str, object]] = []
    for row in rows:
        result_rows.append(
            {
                "auction_date": row["auction_date"],
                "published_date": row["published_date"],
                "document_title": row["document_title"],
                "auction_format": row["auction_format"],
                "issue": row["issue"],
                "security_type": row["security_type"],
                "maturity_date": row["maturity_date"],
                "days_to_maturity": _to_int(row["days_to_maturity"]),
                "offered_amount": _to_float(row["offered_amount"]),
                "demand_amount": _to_float(row["demand_amount"]),
                "placed_amount": _to_float(row["placed_amount"]),
                "proceeds_amount": _to_float(row["proceeds_amount"]),
                "cutoff_price": _to_float(row["cutoff_price"]),
                "weighted_average_price": _to_float(row["weighted_average_price"]),
                "cutoff_yield": _to_float(row["cutoff_yield"]),
                "weighted_average_yield": _to_float(row["weighted_average_yield"]),
                "official_coefficient": _to_float(row["official_coefficient"]),
                "cover_ratio": _to_float(row["cover_ratio"]),
                "placement_ratio": _to_float(row["placement_ratio"]),
                "source_url": row["source_url"],
                "source_file": row["source_file"],
            }
        )

    return sorted(
        result_rows,
        key=lambda row: (
            _date_sort_key(row["auction_date"]),
            str(row["issue"]),
            str(row["auction_format"]),
        ),
    )


def save_csv(rows: list[dict[str, object]], output_path: Path = OUTPUT_FILE) -> None:
    """Сохраняет датасет М3 в CSV-файл"""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """Запускает сборку датасета М3 и сохраняет результат"""
    rows = build_m3_dataset()
    save_csv(rows)
    print(f"Сохранено строк: {len(rows)}")
    print(f"Файл: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
