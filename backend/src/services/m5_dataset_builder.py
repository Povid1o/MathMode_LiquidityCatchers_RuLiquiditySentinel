from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CBR_BUDGET_FUNDS_FILE = PROJECT_ROOT / "data/processed/cbr_budget_funds.csv"
CBR_LIQUIDITY_FILE = PROJECT_ROOT / "data/processed/cbr_liquidity.csv"
OUTPUT_FILE = PROJECT_ROOT / "data/processed/m5_dataset.csv"

OUTPUT_COLUMNS = [
    "date",
    "budget_funds_date",
    "budget_funds_total_mln_rub",
    "federal_budget_funds_mln_rub",
    "regional_local_budget_funds_mln_rub",
    "other_budget_funds_mln_rub",
    "extra_budgetary_funds_mln_rub",
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


def build_m5_dataset(
    budget_funds_path: Path = CBR_BUDGET_FUNDS_FILE,
    liquidity_path: Path = CBR_LIQUIDITY_FILE,
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

    result_rows: list[dict[str, object]] = []
    for liquidity_row in liquidity_rows:
        budget_row = _latest_budget_row(budget_rows, liquidity_row["date"])

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
        writer = csv.DictWriter(file, fieldnames=OUTPUT_COLUMNS)
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
