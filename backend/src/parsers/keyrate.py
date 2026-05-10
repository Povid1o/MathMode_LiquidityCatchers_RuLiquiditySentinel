from __future__ import annotations

import csv
import re
from datetime import datetime
from html.parser import HTMLParser
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
RAW_FILE = PROJECT_ROOT / "data/raw/keyrate/keyrate.html"
OUTPUT_FILE = PROJECT_ROOT / "data/processed/keyrate.csv"

OUTPUT_COLUMNS = [
    "date",
    "key_rate",
]


class _TableParser(HTMLParser):
    """Читает HTML-таблицы в список строк"""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.tables: list[list[list[str]]] = []
        self._table_level = 0
        self._current_table: list[list[str]] = []
        self._current_row: list[str] = []
        self._current_cell: list[str] = []
        self._in_row = False
        self._in_cell = False

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        """Обрабатывает открывающий HTML-тег"""
        if tag == "table":
            self._table_level += 1
            if self._table_level == 1:
                self._current_table = []
        elif tag == "tr" and self._table_level > 0:
            self._in_row = True
            self._current_row = []
        elif tag in {"td", "th"} and self._in_row:
            self._in_cell = True
            self._current_cell = []

    def handle_data(self, data: str) -> None:
        """Сохраняет текст внутри ячейки"""
        if self._in_cell:
            self._current_cell.append(data)

    def handle_endtag(self, tag: str) -> None:
        """Обрабатывает закрывающий HTML-тег"""
        if tag in {"td", "th"} and self._in_cell:
            self._current_row.append(_normalize_space("".join(self._current_cell)))
            self._current_cell = []
            self._in_cell = False
        elif tag == "tr" and self._in_row:
            if any(value != "" for value in self._current_row):
                self._current_table.append(self._current_row)
            self._current_row = []
            self._in_row = False
        elif tag == "table" and self._table_level > 0:
            if self._table_level == 1 and self._current_table:
                self.tables.append(self._current_table)
            self._table_level -= 1


def _normalize_space(value: str) -> str:
    """Нормализует пробелы в строке"""
    return re.sub(r"\s+", " ", value.replace("\xa0", " ")).strip()


def _format_date(value: str) -> str | None:
    """Преобразует дату из DD.MM.YYYY в DD-MM-YYYY"""
    text = value.strip()
    if not text:
        return None

    try:
        return datetime.strptime(text, "%d.%m.%Y").strftime("%d-%m-%Y")
    except ValueError:
        return None


def _date_sort_key(date_text: object) -> datetime:
    """Готовит строковую дату DD-MM-YYYY для сортировки"""
    return datetime.strptime(str(date_text), "%d-%m-%Y")


def _to_float(value: str) -> float | None:
    """Преобразует строку в число с плавающей точкой"""
    text = value.replace("\xa0", "").replace(" ", "").replace(",", ".").strip()
    if text in {"", "-", "—"}:
        return None

    try:
        return float(text)
    except ValueError:
        return None


def _read_tables(path: Path) -> list[list[list[str]]]:
    """Читает HTML-таблицы из сырого файла ЦБ"""
    parser = _TableParser()
    parser.feed(path.read_text(encoding="utf-8-sig"))
    return parser.tables


def _find_keyrate_table(tables: list[list[list[str]]]) -> list[list[str]]:
    """Находит таблицу ключевой ставки"""
    for table in tables:
        if not table:
            continue

        headers = [value.strip().lower() for value in table[0]]
        if headers == ["дата", "ставка"]:
            return table

    raise ValueError("Не найдена таблица ключевой ставки")


def parse_keyrate(input_path: Path = RAW_FILE) -> list[dict[str, object]]:
    """Парсит ключевую ставку из HTML-файла ЦБ"""
    tables = _read_tables(input_path)
    table = _find_keyrate_table(tables)

    parsed_rows: list[dict[str, object]] = []
    for row in table[1:]:
        if len(row) < 2:
            continue

        row_date = _format_date(row[0])
        key_rate = _to_float(row[1])
        if row_date is None or key_rate is None:
            continue

        parsed_rows.append(
            {
                "date": row_date,
                "key_rate": key_rate,
            }
        )

    return sorted(parsed_rows, key=lambda item: _date_sort_key(item["date"]))


def save_csv(rows: list[dict[str, object]], output_path: Path = OUTPUT_FILE) -> None:
    """Сохраняет подготовленные строки в CSV-файл"""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """Запускает парсер и сохраняет результат в CSV"""
    rows = parse_keyrate()
    save_csv(rows)
    print(f"Сохранено строк: {len(rows)}")
    print(f"Файл: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
