from __future__ import annotations

import csv
import xml.etree.ElementTree as ET
from datetime import date, datetime, timedelta
from pathlib import Path
from zipfile import ZipFile


PROJECT_ROOT = Path(__file__).resolve().parents[3]
RAW_FILE = PROJECT_ROOT / "data/raw/ruonia/RC_F01_04_2010_T07_05_2026.xlsx"
OUTPUT_FILE = PROJECT_ROOT / "data/processed/ruonia.csv"
SHEET_NAME = "RC"

HEADER_ROW = 1
FIRST_DATA_ROW = 2

NS = {
    "main": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
    "office": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "rels": "http://schemas.openxmlformats.org/package/2006/relationships",
}

HEADER_TO_FIELD = {
    "DT": "date",
    "ruo": "ruonia_rate",
    "vol": "transactions_volume",
    "T": "transactions_count",
    "C": "participants_count",
}

OUTPUT_COLUMNS = [
    "date",
    "ruonia_rate",
    "transactions_volume",
    "transactions_count",
    "participants_count",
]


def _column_name(cell_ref: str) -> str:
    """Возвращает буквенное имя колонки из адреса ячейки Excel"""
    return "".join(char for char in cell_ref if char.isalpha())


def _column_index(column_name: str) -> int:
    """Переводит буквенное имя колонки Excel в номер колонки"""
    index = 0
    for char in column_name.upper():
        index = index * 26 + ord(char) - ord("A") + 1
    return index


def _format_date(value: date) -> str:
    """Форматирует дату в строку DD-MM-YYYY"""
    return value.strftime("%d-%m-%Y")


def _excel_date_to_date(value: object) -> date | None:
    """Преобразует дату Excel в объект date"""
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value

    if isinstance(value, (int, float)):
        return date(1899, 12, 30) + timedelta(days=int(value))

    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None

        for fmt in ("%d.%m.%Y", "%Y-%m-%d", "%d-%m-%Y"):
            try:
                return datetime.strptime(text[:10], fmt).date()
            except ValueError:
                pass

    return None


def _date_sort_key(date_text: object) -> str:
    """Готовит строковую дату DD-MM-YYYY для сортировки"""
    text = str(date_text)
    return text[6:10] + text[3:5] + text[:2]


def _to_float(value: object) -> float | None:
    """Преобразует значение в float"""
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        return float(value)

    text = str(value).replace("\xa0", "").replace(" ", "").replace(",", ".")
    try:
        return float(text)
    except ValueError:
        return None


def _to_int(value: object) -> int | None:
    """Преобразует значение в int"""
    number = _to_float(value)
    if number is None:
        return None
    return int(number)


def _read_shared_strings(xlsx: ZipFile) -> list[str]:
    """Читает общие строковые значения из xlsx-файла"""
    if "xl/sharedStrings.xml" not in xlsx.namelist():
        return []

    root = ET.fromstring(xlsx.read("xl/sharedStrings.xml"))
    strings: list[str] = []
    for item in root.findall("main:si", NS):
        parts = [node.text or "" for node in item.findall(".//main:t", NS)]
        strings.append("".join(parts))
    return strings


def _cell_value(cell: ET.Element, shared_strings: list[str]) -> object:
    """Достает значение из XML-ячейки Excel"""
    cell_type = cell.attrib.get("t")
    value_node = cell.find("main:v", NS)

    if cell_type == "inlineStr":
        return "".join(
            node.text or "" for node in cell.findall(".//main:is/main:t", NS)
        )

    if value_node is None or value_node.text is None:
        return None

    raw_value = value_node.text
    if cell_type == "s":
        return shared_strings[int(raw_value)]
    if cell_type == "b":
        return raw_value == "1"

    try:
        number = float(raw_value)
    except ValueError:
        return raw_value

    return int(number) if number.is_integer() else number


def _sheet_path(xlsx: ZipFile, sheet_name: str) -> str:
    """Находит путь к XML-файлу нужного листа внутри xlsx"""
    workbook = ET.fromstring(xlsx.read("xl/workbook.xml"))
    rels = ET.fromstring(xlsx.read("xl/_rels/workbook.xml.rels"))

    relation_targets = {
        rel.attrib["Id"]: rel.attrib["Target"]
        for rel in rels.findall("rels:Relationship", NS)
    }

    for sheet in workbook.findall(".//main:sheet", NS):
        if sheet.attrib.get("name") == sheet_name:
            rel_id = sheet.attrib[f"{{{NS['office']}}}id"]
            target = relation_targets[rel_id]
            target = target.lstrip("/")
            if target.startswith("xl/"):
                return target
            return "xl/" + target

    raise ValueError(f"Лист не найден: {sheet_name}")


def _read_sheet_rows(path: Path, sheet_name: str) -> list[dict[int, object]]:
    """Читает строки листа Excel в виде словарей с номерами колонок"""
    with ZipFile(path) as xlsx:
        shared_strings = _read_shared_strings(xlsx)
        worksheet = ET.fromstring(xlsx.read(_sheet_path(xlsx, sheet_name)))

        rows: list[dict[int, object]] = []
        for row in worksheet.findall(".//main:sheetData/main:row", NS):
            row_index = int(row.attrib["r"])
            values: dict[int, object] = {}
            for cell in row.findall("main:c", NS):
                cell_ref = cell.attrib["r"]
                values[_column_index(_column_name(cell_ref))] = _cell_value(
                    cell,
                    shared_strings,
                )
            while len(rows) < row_index:
                rows.append({})
            rows[row_index - 1] = values

    return rows


def _find_columns(header_row: dict[int, object]) -> dict[str, int]:
    """Находит номера нужных колонок по строке заголовков"""
    columns: dict[str, int] = {}

    for column_index, header in header_row.items():
        field_name = HEADER_TO_FIELD.get(str(header))
        if field_name is not None:
            columns[field_name] = column_index

    missing = sorted(set(HEADER_TO_FIELD.values()) - set(columns))
    if missing:
        raise ValueError(f"Не найдены обязательные колонки: {', '.join(missing)}")

    return columns


def parse_ruonia(
    input_path: Path = RAW_FILE,
    sheet_name: str = SHEET_NAME,
) -> list[dict[str, object]]:
    """Парсит дневные значения RUONIA из Excel-файла ЦБ"""
    rows = _read_sheet_rows(input_path, sheet_name)
    columns = _find_columns(rows[HEADER_ROW - 1])

    parsed_rows: list[dict[str, object]] = []
    for row in rows[FIRST_DATA_ROW - 1 :]:
        row_date = _excel_date_to_date(row.get(columns["date"]))
        ruonia_rate = _to_float(row.get(columns["ruonia_rate"]))

        if row_date is None or ruonia_rate is None:
            continue

        parsed_rows.append(
            {
                "date": _format_date(row_date),
                "ruonia_rate": ruonia_rate,
                "transactions_volume": _to_float(
                    row.get(columns["transactions_volume"])
                ),
                "transactions_count": _to_int(row.get(columns["transactions_count"])),
                "participants_count": _to_int(row.get(columns["participants_count"])),
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
    rows = parse_ruonia()
    save_csv(rows)
    print(f"Сохранено строк: {len(rows)}")
    print(f"Файл: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
