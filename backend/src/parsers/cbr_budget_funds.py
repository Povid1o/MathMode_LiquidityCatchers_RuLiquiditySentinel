from __future__ import annotations

import csv
import re
import xml.etree.ElementTree as ET
from datetime import date, datetime, timedelta
from pathlib import Path
from zipfile import ZipFile


PROJECT_ROOT = Path(__file__).resolve().parents[3]
RAW_FILE = PROJECT_ROOT / "data/raw/treasury_funds/cbr_budget_funds.xlsx"
OUTPUT_FILE = PROJECT_ROOT / "data/processed/cbr_budget_funds.csv"
SOURCE_URL = "https://www.cbr.ru/vfs/statistics/banksector/borrowings/02_29_Budget_all.xlsx"

NS = {
    "main": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
    "office": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "rels": "http://schemas.openxmlformats.org/package/2006/relationships",
}

SHEET_TO_CURRENCY_TYPE = {
    "в рублях": "rub",
    "в инвалюте": "foreign_currency_and_precious_metals",
    "итого": "total",
}

ROW_TO_FIELD = {
    "остатки бюджетных средств на счетах всего": "budget_funds_total_mln_rub",
    "средства федерального бюджета": "federal_budget_funds_mln_rub",
    "средства бюджетов субъектов российской федерации и местных бюджетов": (
        "regional_local_budget_funds_mln_rub"
    ),
    "средства прочих бюджетных средств": "other_budget_funds_mln_rub",
    "средства внебюджетных фондов": "extra_budgetary_funds_mln_rub",
}

OUTPUT_COLUMNS = [
    "date",
    "currency_type",
    "budget_funds_total_mln_rub",
    "federal_budget_funds_mln_rub",
    "regional_local_budget_funds_mln_rub",
    "other_budget_funds_mln_rub",
    "extra_budgetary_funds_mln_rub",
    "source_url",
    "source_file",
]


def _normalize_text(value: object) -> str:
    """Нормализует текст для сопоставления строк Excel"""
    text = "" if value is None else str(value)
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text.replace("\xa0", " ")).strip().lower()


def _format_date(value: date) -> str:
    """Форматирует дату в строку DD-MM-YYYY"""
    return value.strftime("%d-%m-%Y")


def _excel_date_to_string(value: object) -> str | None:
    """Преобразует дату Excel в строку DD-MM-YYYY"""
    if isinstance(value, datetime):
        return _format_date(value.date())
    if isinstance(value, date):
        return _format_date(value)
    if isinstance(value, (int, float)):
        return _format_date(date(1899, 12, 30) + timedelta(days=int(value)))
    if isinstance(value, str):
        text = value.strip()
        for fmt in ("%d.%m.%Y", "%Y-%m-%d"):
            try:
                return _format_date(datetime.strptime(text[:10], fmt).date())
            except ValueError:
                pass
    return None


def _to_float(value: object) -> float | None:
    """Преобразует значение в число с плавающей точкой"""
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        return float(value)

    text = str(value).replace("\xa0", "").replace(" ", "").replace(",", ".")
    try:
        return float(text)
    except ValueError:
        return None


def _column_name(cell_ref: str) -> str:
    """Возвращает буквенное имя колонки из адреса ячейки Excel"""
    return "".join(char for char in cell_ref if char.isalpha())


def _column_index(column_name: str) -> int:
    """Переводит буквенное имя колонки Excel в номер колонки"""
    index = 0
    for char in column_name.upper():
        index = index * 26 + ord(char) - ord("A") + 1
    return index


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

    try:
        number = float(raw_value)
    except ValueError:
        return raw_value

    return int(number) if number.is_integer() else number


def _sheet_paths(xlsx: ZipFile) -> dict[str, str]:
    """Находит пути к XML-файлам листов внутри xlsx"""
    workbook = ET.fromstring(xlsx.read("xl/workbook.xml"))
    rels = ET.fromstring(xlsx.read("xl/_rels/workbook.xml.rels"))
    relation_targets = {
        rel.attrib["Id"]: rel.attrib["Target"]
        for rel in rels.findall("rels:Relationship", NS)
    }

    result: dict[str, str] = {}
    for sheet in workbook.findall(".//main:sheet", NS):
        rel_id = sheet.attrib[f"{{{NS['office']}}}id"]
        result[sheet.attrib["name"]] = "xl/" + relation_targets[rel_id].lstrip("/")
    return result


def _read_sheet_rows(xlsx: ZipFile, sheet_path: str) -> list[dict[int, object]]:
    """Читает строки листа Excel в виде словарей с номерами колонок"""
    shared_strings = _read_shared_strings(xlsx)
    worksheet = ET.fromstring(xlsx.read(sheet_path))

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


def _find_metric_rows(rows: list[dict[int, object]]) -> dict[str, int]:
    """Находит строки с нужными показателями бюджетных средств"""
    metric_rows: dict[str, int] = {}

    for row_index, row in enumerate(rows):
        label = _normalize_text(row.get(1))
        for marker, field_name in ROW_TO_FIELD.items():
            if label == marker:
                metric_rows[field_name] = row_index

    missing = sorted(set(ROW_TO_FIELD.values()) - set(metric_rows))
    if missing:
        raise ValueError(f"Не найдены строки показателей: {', '.join(missing)}")

    return metric_rows


def parse_cbr_budget_funds(input_path: Path = RAW_FILE) -> list[dict[str, object]]:
    """Парсит бюджетные средства на счетах банков из Excel-файла ЦБ"""
    parsed_rows: list[dict[str, object]] = []

    with ZipFile(input_path) as xlsx:
        sheet_paths = _sheet_paths(xlsx)

        for sheet_name, currency_type in SHEET_TO_CURRENCY_TYPE.items():
            if sheet_name not in sheet_paths:
                raise ValueError(f"Лист не найден: {sheet_name}")

            rows = _read_sheet_rows(xlsx, sheet_paths[sheet_name])
            date_row = rows[1]
            metric_rows = _find_metric_rows(rows)

            for column_index, raw_date in date_row.items():
                row_date = _excel_date_to_string(raw_date)
                if row_date is None:
                    continue

                result_row: dict[str, object] = {
                    "date": row_date,
                    "currency_type": currency_type,
                    "source_url": SOURCE_URL,
                    "source_file": input_path.name,
                }

                for field_name, row_index in metric_rows.items():
                    result_row[field_name] = _to_float(
                        rows[row_index].get(column_index)
                    )

                parsed_rows.append(result_row)

    return sorted(
        parsed_rows,
        key=lambda row: (
            datetime.strptime(str(row["date"]), "%d-%m-%Y"),
            str(row["currency_type"]),
        ),
    )


def save_csv(rows: list[dict[str, object]], output_path: Path = OUTPUT_FILE) -> None:
    """Сохраняет данные по бюджетным средствам в CSV-файл"""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """Запускает парсинг бюджетных средств и сохраняет результат"""
    rows = parse_cbr_budget_funds()
    save_csv(rows)
    print(f"Сохранено строк: {len(rows)}")
    print(f"Файл: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
