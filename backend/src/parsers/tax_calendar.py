from __future__ import annotations

import csv
import re
import xml.etree.ElementTree as ET
from datetime import datetime
from html.parser import HTMLParser
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
RAW_DIR = PROJECT_ROOT / "data/raw/tax_calendar"
RELEASES_DIR = RAW_DIR / "releases"
OUTPUT_FILE = PROJECT_ROOT / "data/processed/tax_calendar.csv"

OUTPUT_COLUMNS = [
    "event_date",
    "day_type",
    "tax_name",
    "event_type",
    "event_text",
    "source_release_date",
    "source_file",
    "source_url",
]

MONTH_NUMBERS = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
}

SOURCE_URL_PREFIX = "https://data.nalog.ru/opendata/7707329152-kalendar/"


class _EventHtmlParser(HTMLParser):
    """Достает текстовые абзацы из HTML внутри XML-календаря"""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.paragraphs: list[str] = []
        self._in_paragraph = False
        self._parts: list[str] = []

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        """Обрабатывает открывающий HTML-тег"""
        if tag == "p":
            self._in_paragraph = True
            self._parts = []

    def handle_data(self, data: str) -> None:
        """Сохраняет текст внутри абзаца"""
        if self._in_paragraph:
            self._parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        """Обрабатывает закрывающий HTML-тег"""
        if tag != "p" or not self._in_paragraph:
            return

        text = _normalize_space(" ".join(self._parts))
        if text:
            self.paragraphs.append(text)
        self._parts = []
        self._in_paragraph = False


def _normalize_space(value: str) -> str:
    """Нормализует пробелы в строке"""
    return re.sub(r"\s+", " ", value.replace("\xa0", " ")).strip()


def _read_xml_text(path: Path) -> str:
    """Читает XML ФНС с учетом ошибок в объявленной кодировке"""
    raw = path.read_bytes()
    for encoding in ("utf-8-sig", "windows-1251"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue

    raise ValueError(f"Не удалось определить кодировку XML-файла: {path}")


def _read_xml_root(path: Path) -> ET.Element:
    """Читает корневой элемент XML-файла ФНС"""
    text = _read_xml_text(path)
    end_tag = "</calendar>"
    end_index = text.find(end_tag)
    if end_index == -1:
        raise ValueError(f"В XML-файле ФНС не найден закрывающий тег calendar: {path}")

    return ET.fromstring(text[: end_index + len(end_tag)])


def _release_date_from_name(path: Path) -> str:
    """Достает дату релиза из имени XML-файла ФНС"""
    match = re.search(r"data-(\d{8})-structure", path.name)
    if not match:
        return ""

    value = match.group(1)
    for date_format in ("%Y%m%d", "%d%m%Y"):
        try:
            return datetime.strptime(value, date_format).strftime("%d-%m-%Y")
        except ValueError:
            continue

    return ""


def _release_date_key(path: Path) -> tuple[int, int, int]:
    """Готовит дату релиза для сортировки XML-файлов"""
    release_date = _release_date_from_name(path)
    if not release_date:
        return (0, 0, 0)

    value = datetime.strptime(release_date, "%d-%m-%Y")
    return (value.year, value.month, value.day)


def _selected_release_paths(releases_dir: Path) -> list[Path]:
    """Выбирает последний XML-релиз для каждого календарного года"""
    latest_by_year: dict[str, Path] = {}

    for path in sorted(releases_dir.glob("data-*.xml")):
        root = _read_xml_root(path)
        for year in root.findall("year"):
            year_index = year.get("index", "")
            current_path = latest_by_year.get(year_index)
            if current_path is None or _release_date_key(path) > _release_date_key(
                current_path
            ):
                latest_by_year[year_index] = path

    return sorted(set(latest_by_year.values()), key=_release_date_key)


def _event_date(year: str, month_name: str, day: str) -> str:
    """Собирает дату события в формате DD-MM-YYYY"""
    month_number = MONTH_NUMBERS.get(month_name)
    if month_number is None:
        raise ValueError(f"Неизвестный месяц в XML ФНС: {month_name}")

    return datetime(int(year), month_number, int(day)).strftime("%d-%m-%Y")


def _html_to_paragraphs(value: str) -> list[str]:
    """Преобразует HTML события в список текстовых абзацев"""
    parser = _EventHtmlParser()
    parser.feed(value)
    return parser.paragraphs


def _is_tax_name(value: str) -> bool:
    """Проверяет, похож ли абзац на название налога или сбора"""
    return value.endswith(":") and not value.startswith("-") and len(value) <= 160


def _clean_event_text(value: str) -> str:
    """Очищает текст события от служебного маркера списка"""
    return _normalize_space(value.removeprefix("-").strip())


def _classify_event_type(value: str) -> str:
    """Определяет тип события по тексту ФНС"""
    text = value.lower()
    has_payment = any(
        keyword in text
        for keyword in (
            "уплачивают",
            "перечисляют",
            "вносят",
            "уплата",
            "перечисление",
        )
    )
    has_notification = any(
        keyword in text
        for keyword in (
            "уведомление",
            "уведомляют",
            "уведомляет",
        )
    )
    has_reporting = any(
        keyword in text
        for keyword in (
            "представляют",
            "представляет",
            "декларацию",
            "расчет",
            "сведения",
            "отчетность",
        )
    )

    if has_payment and (has_notification or has_reporting):
        return "mixed"
    if has_payment:
        return "payment_deadline"
    if has_notification:
        return "notification_deadline"
    if has_reporting:
        return "reporting_deadline"
    return "other"


def _parse_event_paragraphs(
    paragraphs: list[str],
    base_row: dict[str, str],
) -> list[dict[str, str]]:
    """Преобразует абзацы одного дня в строки событий"""
    rows: list[dict[str, str]] = []
    current_tax_name = ""

    for paragraph in paragraphs:
        if _is_tax_name(paragraph):
            current_tax_name = paragraph.rstrip(":")
            continue

        event_text = _clean_event_text(paragraph)
        if not event_text:
            continue

        rows.append(
            {
                **base_row,
                "tax_name": current_tax_name,
                "event_type": _classify_event_type(event_text),
                "event_text": event_text,
            }
        )

    return rows


def parse_tax_calendar(
    releases_dir: Path = RELEASES_DIR,
) -> list[dict[str, str]]:
    """Парсит XML-релизы ФНС в очищенный список налоговых событий"""
    if not releases_dir.exists():
        raise ValueError(f"Папка с XML-релизами ФНС не найдена: {releases_dir}")

    rows_by_key: dict[tuple[str, str, str], dict[str, str]] = {}
    for path in _selected_release_paths(releases_dir):
        root = _read_xml_root(path)
        release_date = _release_date_from_name(path)
        source_url = f"{SOURCE_URL_PREFIX}{path.name}"

        for year in root.findall("year"):
            year_index = year.get("index", "")
            for month in year.findall("month"):
                month_name = month.get("name", "")
                for day in month.findall("day"):
                    day_type = day.get("type", "")
                    day_text = day.text or ""
                    if day_type != "event" or not day_text.strip():
                        continue

                    base_row = {
                        "event_date": _event_date(
                            year_index,
                            month_name,
                            day.get("num", ""),
                        ),
                        "day_type": day_type,
                        "source_release_date": release_date,
                        "source_file": path.name,
                        "source_url": source_url,
                    }
                    for row in _parse_event_paragraphs(
                        _html_to_paragraphs(day_text),
                        base_row,
                    ):
                        key = (
                            row["event_date"],
                            row["tax_name"],
                            row["event_text"],
                        )
                        rows_by_key[key] = row

    return sorted(
        rows_by_key.values(),
        key=lambda row: (
            datetime.strptime(row["event_date"], "%d-%m-%Y"),
            row["tax_name"],
            row["event_text"],
        ),
    )


def save_csv(rows: list[dict[str, str]], output_path: Path = OUTPUT_FILE) -> None:
    """Сохраняет налоговый календарь в CSV-файл"""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """Запускает парсинг налогового календаря ФНС"""
    rows = parse_tax_calendar()
    save_csv(rows)
    print(f"Сохранено налоговых событий: {len(rows)}")
    print(f"Файл: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
