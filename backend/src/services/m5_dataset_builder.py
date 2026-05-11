from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CBR_BUDGET_FUNDS_FILE = PROJECT_ROOT / "data/processed/cbr_budget_funds.csv"
CBR_LIQUIDITY_FILE = PROJECT_ROOT / "data/processed/cbr_liquidity.csv"
ROSKAZNA_DEPOSITS_FILE = PROJECT_ROOT / "data/processed/roskazna_treasury_deposits.csv"
OUTPUT_FILE = PROJECT_ROOT / "data/processed/m5_dataset.csv"

OUTPUT_COLUMNS = [
    "date",
    "budget_funds_date",
    "budget_funds_total_mln_rub",
    "federal_budget_funds_mln_rub",
    "regional_local_budget_funds_mln_rub",
    "other_budget_funds_mln_rub",
    "extra_budgetary_funds_mln_rub",
    "roskazna_auctions_count",
    "roskazna_max_volume_mln_rub",
    "roskazna_demand_volume_mln_rub",
    "roskazna_accepted_volume_mln_rub",
    "roskazna_settled_volume_mln_rub",
    "roskazna_weighted_average_rate",
    "roskazna_cover_ratio",
    "roskazna_accepted_ratio",
    "roskazna_settled_ratio",
    "roskazna_bidders_count",
    "roskazna_accepted_bidders_count",
    "liquidity_deficit_surplus_bln_rub",
    "liquidity_deficit_surplus_without_correspondent_accounts_bln_rub",
    "cbr_claims_standard_instruments_bln_rub",
    "repo_fx_swap_auctions_bln_rub",
    "secured_loans_auctions_bln_rub",
    "repo_fx_swap_standing_bln_rub",
    "secured_loans_standing_bln_rub",
    "cbr_liabilities_standard_instruments_bln_rub",
    "deposit_auctions_bln_rub",
    "deposit_standing_bln_rub",
    "cobr_bln_rub",
    "nonstandard_refundable_operations_bln_rub",
    "correspondent_accounts_bln_rub",
    "required_reserves_avg_bln_rub",
    "source_budget_file",
    "source_liquidity_file",
]


def _read_csv(path: Path) -> list[dict[str, str]]:
    """Читает CSV-файл в список словарей"""
    with path.open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def _parse_date(value: str) -> datetime:
    """Преобразует дату DD-MM-YYYY в datetime для сортировки и сравнения"""
    return datetime.strptime(value, "%d-%m-%Y")


def _to_float(value: str | None) -> float | None:
    """Преобразует строку в число с плавающей точкой"""
    if value is None or value == "":
        return None
    return float(value)


def _latest_budget_row(
    budget_rows: list[dict[str, str]],
    row_date: str,
) -> dict[str, str] | None:
    """Находит последнюю месячную запись бюджетных средств к дневной дате"""
    current_date = _parse_date(row_date)
    selected_row: dict[str, str] | None = None

    for row in budget_rows:
        if row["currency_type"] != "total":
            continue
        if _parse_date(row["date"]) <= current_date:
            selected_row = row
        else:
            break

    return selected_row


def _sum_values(rows: list[dict[str, str]], field_name: str) -> float | None:
    """Суммирует числовое поле по строкам"""
    values = [_to_float(row.get(field_name)) for row in rows]
    values = [value for value in values if value is not None]
    if not values:
        return None
    return sum(values)


def _weighted_average_rate(rows: list[dict[str, str]]) -> float | None:
    """Считает средневзвешенную ставку Росказны по принятому объему"""
    numerator = 0.0
    denominator = 0.0

    for row in rows:
        rate = _to_float(row.get("weighted_average_accepted_rate"))
        volume = _to_float(row.get("accepted_volume_mln_rub"))
        if rate is None or volume is None:
            continue
        numerator += rate * volume
        denominator += volume

    if denominator == 0:
        return None
    return numerator / denominator


def _safe_ratio(numerator: float | None, denominator: float | None) -> float | None:
    """Считает отношение с защитой от пустых и нулевых значений"""
    if numerator is None or denominator in {None, 0}:
        return None
    return numerator / denominator


def _build_roskazna_index(rows: list[dict[str, str]]) -> dict[str, dict[str, object]]:
    """Агрегирует аукционы Росказны по дате"""
    rows_by_date: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        rows_by_date.setdefault(row["auction_date"], []).append(row)

    result: dict[str, dict[str, object]] = {}
    for auction_date, daily_rows in rows_by_date.items():
        max_volume = _sum_values(daily_rows, "max_volume_mln_rub")
        demand_volume = _sum_values(daily_rows, "demand_volume_mln_rub")
        accepted_volume = _sum_values(daily_rows, "accepted_volume_mln_rub")
        settled_volume = _sum_values(daily_rows, "settled_volume_mln_rub")

        result[auction_date] = {
            "roskazna_auctions_count": len(daily_rows),
            "roskazna_max_volume_mln_rub": max_volume,
            "roskazna_demand_volume_mln_rub": demand_volume,
            "roskazna_accepted_volume_mln_rub": accepted_volume,
            "roskazna_settled_volume_mln_rub": settled_volume,
            "roskazna_weighted_average_rate": _weighted_average_rate(daily_rows),
            "roskazna_cover_ratio": _safe_ratio(demand_volume, max_volume),
            "roskazna_accepted_ratio": _safe_ratio(accepted_volume, demand_volume),
            "roskazna_settled_ratio": _safe_ratio(settled_volume, max_volume),
            "roskazna_bidders_count": _sum_values(daily_rows, "bidders_count"),
            "roskazna_accepted_bidders_count": _sum_values(
                daily_rows,
                "accepted_bidders_count",
            ),
        }

    return result


def _empty_roskazna_row() -> dict[str, object]:
    """Возвращает пустые дневные агрегаты Росказны"""
    return {
        "roskazna_auctions_count": 0,
        "roskazna_max_volume_mln_rub": None,
        "roskazna_demand_volume_mln_rub": None,
        "roskazna_accepted_volume_mln_rub": None,
        "roskazna_settled_volume_mln_rub": None,
        "roskazna_weighted_average_rate": None,
        "roskazna_cover_ratio": None,
        "roskazna_accepted_ratio": None,
        "roskazna_settled_ratio": None,
        "roskazna_bidders_count": None,
        "roskazna_accepted_bidders_count": None,
    }


def build_m5_dataset(
    budget_funds_path: Path = CBR_BUDGET_FUNDS_FILE,
    liquidity_path: Path = CBR_LIQUIDITY_FILE,
    roskazna_deposits_path: Path = ROSKAZNA_DEPOSITS_FILE,
) -> list[dict[str, object]]:
    """Собирает базовый датасет М5 по бюджетным средствам и ликвидности"""
    budget_rows = sorted(
        _read_csv(budget_funds_path),
        key=lambda row: _parse_date(row["date"]),
    )
    liquidity_rows = sorted(
        _read_csv(liquidity_path),
        key=lambda row: _parse_date(row["date"]),
    )
    roskazna_rows = (
        _read_csv(roskazna_deposits_path)
        if roskazna_deposits_path.exists()
        else []
    )
    roskazna_index = _build_roskazna_index(roskazna_rows)

    result_rows: list[dict[str, object]] = []
    for liquidity_row in liquidity_rows:
        budget_row = _latest_budget_row(budget_rows, liquidity_row["date"])
        roskazna_row = roskazna_index.get(liquidity_row["date"], _empty_roskazna_row())

        result_rows.append(
            {
                "date": liquidity_row["date"],
                "budget_funds_date": budget_row["date"] if budget_row else "",
                "budget_funds_total_mln_rub": _to_float(
                    budget_row["budget_funds_total_mln_rub"] if budget_row else None
                ),
                "federal_budget_funds_mln_rub": _to_float(
                    budget_row["federal_budget_funds_mln_rub"] if budget_row else None
                ),
                "regional_local_budget_funds_mln_rub": _to_float(
                    budget_row["regional_local_budget_funds_mln_rub"]
                    if budget_row
                    else None
                ),
                "other_budget_funds_mln_rub": _to_float(
                    budget_row["other_budget_funds_mln_rub"] if budget_row else None
                ),
                "extra_budgetary_funds_mln_rub": _to_float(
                    budget_row["extra_budgetary_funds_mln_rub"] if budget_row else None
                ),
                **roskazna_row,
                "liquidity_deficit_surplus_bln_rub": _to_float(
                    liquidity_row["liquidity_deficit_surplus_bln_rub"]
                ),
                "liquidity_deficit_surplus_without_correspondent_accounts_bln_rub": _to_float(
                    liquidity_row[
                        "liquidity_deficit_surplus_without_correspondent_accounts_bln_rub"
                    ]
                ),
                "cbr_claims_standard_instruments_bln_rub": _to_float(
                    liquidity_row["cbr_claims_standard_instruments_bln_rub"]
                ),
                "repo_fx_swap_auctions_bln_rub": _to_float(
                    liquidity_row["repo_fx_swap_auctions_bln_rub"]
                ),
                "secured_loans_auctions_bln_rub": _to_float(
                    liquidity_row["secured_loans_auctions_bln_rub"]
                ),
                "repo_fx_swap_standing_bln_rub": _to_float(
                    liquidity_row["repo_fx_swap_standing_bln_rub"]
                ),
                "secured_loans_standing_bln_rub": _to_float(
                    liquidity_row["secured_loans_standing_bln_rub"]
                ),
                "cbr_liabilities_standard_instruments_bln_rub": _to_float(
                    liquidity_row["cbr_liabilities_standard_instruments_bln_rub"]
                ),
                "deposit_auctions_bln_rub": _to_float(
                    liquidity_row["deposit_auctions_bln_rub"]
                ),
                "deposit_standing_bln_rub": _to_float(
                    liquidity_row["deposit_standing_bln_rub"]
                ),
                "cobr_bln_rub": _to_float(liquidity_row["cobr_bln_rub"]),
                "nonstandard_refundable_operations_bln_rub": _to_float(
                    liquidity_row["nonstandard_refundable_operations_bln_rub"]
                ),
                "correspondent_accounts_bln_rub": _to_float(
                    liquidity_row["correspondent_accounts_bln_rub"]
                ),
                "required_reserves_avg_bln_rub": _to_float(
                    liquidity_row["required_reserves_avg_bln_rub"]
                ),
                "source_budget_file": budget_row["source_file"] if budget_row else "",
                "source_liquidity_file": liquidity_row["source_file"],
            }
        )

    return result_rows


def save_csv(rows: list[dict[str, object]], output_path: Path = OUTPUT_FILE) -> None:
    """Сохраняет датасет М5 в CSV-файл"""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=OUTPUT_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """Запускает сборку датасета М5 и сохраняет результат"""
    rows = build_m5_dataset()
    save_csv(rows)
    print(f"Сохранено строк: {len(rows)}")
    print(f"Файл: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
