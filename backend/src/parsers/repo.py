from __future__ import annotations

import csv
import re
from datetime import datetime
from html.parser import HTMLParser
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
RAW_FILE = PROJECT_ROOT / "data/raw/repo/repo.html"
DAILY_DIR = PROJECT_ROOT / "data/raw/repo/daily"
OUTPUT_FILE = PROJECT_ROOT / "data/processed/repo.csv"

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
]

HEADER_TO_FIELD = {
    "тип аукциона": "auction_type",
    "срок дни": "term_days",
    "дата": "date",
    "время аукциона": "auction_time",
    "общий объем заключенных сделок млн руб": "total_deals_volume",
    "средневзвешенная ставка годовых": "weighted_average_rate",
    "код расчета": "settlement_code",
}

DETAIL_TO_FIELD = {
    "тип аукциона": "auction_type",
    "объем спроса на операции репо млн руб": "demand_volume",
    "общий объем заключенных сделок репо млн руб": "total_deals_volume",
    "ставка отсечения годовых": "cutoff_rate",
    "средневзвешенная ставка годовых": "weighted_average_rate",
    "минимальная заявленная ставка годовых": "min_rate",
    "максимальная заявленная ставка годовых": "max_rate",
    "объем заключенных сделок репо в рамках лимита млн руб": "limit_deals_volume",
    "средневзвешенная ставка по заявкам удовлетворенным в рамках лимита годовых": (
        "weighted_average_limit_rate"
    ),
    "срок дни": "term_days",
    "дата исполнения первой части сделки": "first_leg_date",
    "дата исполнения второй части сделки": "second_leg_date",
}

DETAIL_FIELDS = [
    "demand_volume",
    "cutoff_rate",
    "min_rate",
    "max_rate",
    "limit_deals_volume",
    "weighted_average_limit_rate",
    "first_leg_date",
    "second_leg_date",
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
        elif tag == "br" and self._in_cell:
            self._current_cell.append(" ")

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


def _normalize_header(value: str) -> str:
    """Нормализует заголовок таблицы для поиска колонок"""
    text = _normalize_space(value).lower()
    text = re.sub(r"[^а-яa-z0-9\s]", " ", text, flags=re.IGNORECASE)
    return _normalize_space(text)


def _format_date(value: str) -> str | None:
    """Преобразует дату из DD.MM.YYYY в DD-MM-YYYY"""
    text = value.strip()
    if not text:
        return None

    try:
        return datetime.strptime(text, "%d.%m.%Y").strftime("%d-%m-%Y")
    except ValueError:
        return None


def _format_daily_file_date(value: str) -> str:
    """Преобразует дату из YYYY-MM-DD в DD-MM-YYYY"""
    return datetime.strptime(value, "%Y-%m-%d").strftime("%d-%m-%Y")


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


def _to_int(value: str) -> int | None:
    """Преобразует строку в целое число"""
    number = _to_float(value)
    if number is None:
        return None
    return int(number)


def _sort_term_days(value: object) -> int:
    """Готовит срок аукциона для сортировки"""
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str) and value != "":
        return int(float(value))
    return 0


def _read_tables(path: Path) -> list[list[list[str]]]:
    """Читает HTML-таблицы из сырого файла ЦБ"""
    parser = _TableParser()
    parser.feed(path.read_text(encoding="utf-8-sig"))
    return parser.tables


def _find_summary_table(tables: list[list[list[str]]]) -> list[list[str]]:
    """Находит сводную таблицу итогов репо"""
    for table in tables:
        if not table:
            continue

        normalized_headers = [_normalize_header(value) for value in table[0]]
        if "тип аукциона" in normalized_headers and "код расчета" in normalized_headers:
            return table

    raise ValueError("Не найдена сводная таблица итогов репо")


def _find_detail_tables(tables: list[list[list[str]]]) -> list[list[str]]:
    """Находит дневные таблицы с деталями аукционов репо"""
    detail_tables: list[list[str]] = []
    for table in tables:
        normalized_keys = {
            _normalize_header(row[0])
            for row in table
            if len(row) >= 2 and row[0].strip() != ""
        }
        if "тип аукциона" in normalized_keys and "срок дни" in normalized_keys:
            detail_tables.append(table)

    return detail_tables


def _find_columns(header_row: list[str]) -> dict[str, int]:
    """Находит номера нужных колонок по строке заголовков"""
    columns: dict[str, int] = {}

    for index, header in enumerate(header_row):
        normalized_header = _normalize_header(header)
        field_name = HEADER_TO_FIELD.get(normalized_header)
        if field_name is not None:
            columns[field_name] = index

    missing = sorted(set(HEADER_TO_FIELD.values()) - set(columns))
    if missing:
        raise ValueError(f"Не найдены обязательные колонки: {', '.join(missing)}")

    return columns


def _get(row: list[str], index: int) -> str:
    """Возвращает значение ячейки по номеру колонки"""
    if index >= len(row):
        return ""
    return row[index]


def _daily_path_for_date(date_text: object, daily_dir: Path = DAILY_DIR) -> Path:
    """Возвращает путь дневной страницы для даты DD-MM-YYYY"""
    repo_date = datetime.strptime(str(date_text), "%d-%m-%Y").date()
    return daily_dir / f"{repo_date.isoformat()}.html"


def _calculate_cover_ratio(row: dict[str, object]) -> float | None:
    """Считает отношение спроса к объему заключенных сделок"""
    demand_volume = row.get("demand_volume")
    total_deals_volume = row.get("total_deals_volume")
    if not isinstance(demand_volume, (int, float)):
        return None
    if not isinstance(total_deals_volume, (int, float)) or total_deals_volume == 0:
        return None

    return demand_volume / total_deals_volume


def parse_repo_summary(input_path: Path = RAW_FILE) -> list[dict[str, object]]:
    """Парсит сводную таблицу итогов аукционов репо из HTML-файла ЦБ"""
    tables = _read_tables(input_path)
    table = _find_summary_table(tables)
    columns = _find_columns(table[0])

    parsed_rows: list[dict[str, object]] = []
    for row in table[1:]:
        row_date = _format_date(_get(row, columns["date"]))
        if row_date is None:
            continue

        parsed_rows.append(
            {
                "date": row_date,
                "auction_type": _get(row, columns["auction_type"]).lower(),
                "term_days": _to_int(_get(row, columns["term_days"])),
                "auction_time": _get(row, columns["auction_time"]),
                "total_deals_volume": _to_float(
                    _get(row, columns["total_deals_volume"])
                ),
                "weighted_average_rate": _to_float(
                    _get(row, columns["weighted_average_rate"])
                ),
                "settlement_code": _get(row, columns["settlement_code"]),
            }
        )

    return sorted(
        parsed_rows,
        key=lambda item: (
            _date_sort_key(item["date"]),
            str(item["auction_time"]),
            _sort_term_days(item["term_days"]),
        ),
    )


def parse_repo_daily_detail(
    input_path: Path,
    date_text: str | None = None,
) -> list[dict[str, object]]:
    """Парсит дневную страницу ЦБ с деталями аукционов репо"""
    if date_text is None:
        date_text = _format_daily_file_date(input_path.stem)

    tables = _read_tables(input_path)
    detail_rows: list[dict[str, object]] = []
    for table in _find_detail_tables(tables):
        parsed_row: dict[str, object] = {"date": date_text}
        for row in table:
            if len(row) < 2:
                continue

            field_name = DETAIL_TO_FIELD.get(_normalize_header(row[0]))
            if field_name is None:
                continue

            value = row[1]
            if field_name in {
                "demand_volume",
                "total_deals_volume",
                "cutoff_rate",
                "weighted_average_rate",
                "min_rate",
                "max_rate",
                "limit_deals_volume",
                "weighted_average_limit_rate",
            }:
                parsed_row[field_name] = _to_float(value)
            elif field_name == "term_days":
                parsed_row[field_name] = _to_int(value)
            elif field_name in {"first_leg_date", "second_leg_date"}:
                parsed_row[field_name] = _format_date(value)
            elif field_name == "auction_type":
                parsed_row[field_name] = value.lower()
            else:
                parsed_row[field_name] = value

        if len(parsed_row) > 1:
            detail_rows.append(parsed_row)

    return detail_rows


def _detail_key(row: dict[str, object]) -> tuple[object, object, object]:
    """Создает ключ для связи сводной строки и дневной детализации"""
    return (
        row.get("date"),
        row.get("auction_type"),
        row.get("term_days"),
    )


def _build_detail_index(
    summary_rows: list[dict[str, object]],
    daily_dir: Path = DAILY_DIR,
) -> dict[tuple[object, object, object], dict[str, object]]:
    """Собирает индекс дневных деталей по дате, типу и сроку"""
    details: dict[tuple[object, object, object], dict[str, object]] = {}
    parsed_dates: set[object] = set()

    for summary_row in summary_rows:
        row_date = summary_row["date"]
        if row_date in parsed_dates:
            continue
        parsed_dates.add(row_date)

        daily_path = _daily_path_for_date(row_date, daily_dir)
        if not daily_path.exists():
            continue

        for detail_row in parse_repo_daily_detail(daily_path, str(row_date)):
            details[_detail_key(detail_row)] = detail_row

    return details


def _merge_summary_with_detail(
    summary_row: dict[str, object],
    detail_row: dict[str, object] | None,
) -> dict[str, object]:
    """Объединяет сводную строку с дневной детализацией"""
    result = {column: None for column in OUTPUT_COLUMNS}
    result.update(summary_row)

    if detail_row is not None:
        for field_name, value in detail_row.items():
            if field_name in {"date"}:
                continue
            if value is not None and value != "":
                result[field_name] = value

    result["cover_ratio"] = _calculate_cover_ratio(result)
    return result


def parse_repo(
    input_path: Path = RAW_FILE,
    daily_dir: Path = DAILY_DIR,
) -> list[dict[str, object]]:
    """Парсит итоги аукционов репо и добавляет дневную детализацию"""
    summary_rows = parse_repo_summary(input_path)
    detail_index = _build_detail_index(summary_rows, daily_dir)

    parsed_rows = [
        _merge_summary_with_detail(summary_row, detail_index.get(_detail_key(summary_row)))
        for summary_row in summary_rows
    ]

    return sorted(
        parsed_rows,
        key=lambda item: (
            _date_sort_key(item["date"]),
            str(item["auction_time"]),
            _sort_term_days(item["term_days"]),
        ),
    )


def save_csv(rows: list[dict[str, object]], output_path: Path = OUTPUT_FILE) -> None:
    """Сохраняет подготовленные строки в CSV-файл"""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """Запускает парсер и сохраняет результат в CSV"""
    rows = parse_repo()
    save_csv(rows)
    print(f"Сохранено строк: {len(rows)}")
    print(f"Файл: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
