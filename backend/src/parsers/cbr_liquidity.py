from __future__ import annotations

import csv
import re
from datetime import datetime
from html.parser import HTMLParser
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
RAW_FILE = PROJECT_ROOT / "data/raw/treasury_funds/cbr_liquidity.html"
OUTPUT_FILE = PROJECT_ROOT / "data/processed/cbr_liquidity.csv"
SOURCE_URL = "https://www.cbr.ru/hd_base/bliquidity/"

OUTPUT_COLUMNS = [
    "date",
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
    "source_url",
    "source_file",
]


class _LiquidityTableParser(HTMLParser):
    """Достает строки дневной таблицы ликвидности из HTML ЦБ"""

    def __init__(self) -> None:
        super().__init__()
        self._inside_table = False
        self._inside_cell = False
        self._current_cell = ""
        self._current_row: list[str] = []
        self.rows: list[list[str]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        """Обрабатывает открывающие HTML-теги"""
        attributes = dict(attrs)
        if tag == "table" and "data" in (attributes.get("class") or ""):
            self._inside_table = True
        elif tag == "tr" and self._inside_table:
            self._current_row = []
        elif tag == "td" and self._inside_table:
            self._inside_cell = True
            self._current_cell = ""

    def handle_data(self, data: str) -> None:
        """Собирает текст внутри ячеек таблицы"""
        if self._inside_cell:
            self._current_cell += data

    def handle_endtag(self, tag: str) -> None:
        """Обрабатывает закрывающие HTML-теги"""
        if tag == "td" and self._inside_cell:
            self._inside_cell = False
            self._current_row.append(_clean_text(self._current_cell))
        elif tag == "tr" and self._inside_table and self._current_row:
            self.rows.append(self._current_row)
        elif tag == "table" and self._inside_table:
            self._inside_table = False


def _clean_text(value: str) -> str:
    """Очищает текст ячейки HTML-таблицы"""
    return re.sub(r"\s+", " ", value.replace("\xa0", " ")).strip()


def _format_date(value: str) -> str:
    """Переводит дату сайта ЦБ в формат DD-MM-YYYY"""
    return datetime.strptime(value, "%d.%m.%Y").strftime("%d-%m-%Y")


def _to_float(value: str) -> float | None:
    """Преобразует строковое число ЦБ в число с плавающей точкой"""
    text = value.replace("\xa0", "").replace(" ", "").replace(",", ".")
    if text in {"", "-", "—"}:
        return None
    return float(text)


def _to_int(value: str) -> int | None:
    """Преобразует строковое число ЦБ в целое число"""
    number = _to_float(value)
    if number is None:
        return None
    return int(number)


def parse_cbr_liquidity(input_path: Path = RAW_FILE) -> list[dict[str, object]]:
    """Парсит дневную таблицу ликвидности банковского сектора ЦБ"""
    parser = _LiquidityTableParser()
    parser.feed(input_path.read_text(encoding="utf-8"))

    parsed_rows: list[dict[str, object]] = []
    for row in parser.rows:
        if len(row) != 15:
            continue

        parsed_rows.append(
            {
                "date": _format_date(row[0]),
                "liquidity_deficit_surplus_bln_rub": _to_float(row[1]),
                "liquidity_deficit_surplus_without_correspondent_accounts_bln_rub": (
                    _to_float(row[2])
                ),
                "cbr_claims_standard_instruments_bln_rub": _to_float(row[3]),
                "repo_fx_swap_auctions_bln_rub": _to_float(row[4]),
                "secured_loans_auctions_bln_rub": _to_float(row[5]),
                "repo_fx_swap_standing_bln_rub": _to_float(row[6]),
                "secured_loans_standing_bln_rub": _to_float(row[7]),
                "cbr_liabilities_standard_instruments_bln_rub": _to_float(row[8]),
                "deposit_auctions_bln_rub": _to_float(row[9]),
                "deposit_standing_bln_rub": _to_float(row[10]),
                "cobr_bln_rub": _to_float(row[11]),
                "nonstandard_refundable_operations_bln_rub": _to_float(row[12]),
                "correspondent_accounts_bln_rub": _to_float(row[13]),
                "required_reserves_avg_bln_rub": _to_float(row[14]),
                "source_url": SOURCE_URL,
                "source_file": input_path.name,
            }
        )

    return sorted(
        parsed_rows,
        key=lambda row: datetime.strptime(str(row["date"]), "%d-%m-%Y"),
    )


def save_csv(rows: list[dict[str, object]], output_path: Path = OUTPUT_FILE) -> None:
    """Сохраняет таблицу ликвидности ЦБ в CSV-файл"""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """Запускает парсинг дневной ликвидности ЦБ и сохраняет результат"""
    rows = parse_cbr_liquidity()
    save_csv(rows)
    print(f"Сохранено строк: {len(rows)}")
    print(f"Файл: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
