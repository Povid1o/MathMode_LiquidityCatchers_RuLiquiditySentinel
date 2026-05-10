from __future__ import annotations

import re
import sys
from datetime import datetime
from html.parser import HTMLParser
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.src.downloaders.common import download_file


SOURCE_URL = "https://www.nalog.gov.ru/opendata/7707329152-kalendar/"
RAW_DIR = PROJECT_ROOT / "data/raw/tax_calendar"
INDEX_FILE = RAW_DIR / "index.html"
STRUCTURE_FILE = RAW_DIR / "structure-20140228.xsd"
LATEST_FILE = RAW_DIR / "tax_calendar.xml"
RELEASES_DIR = RAW_DIR / "releases"

DATASET_URL_PATTERN = re.compile(
    r"^https://data\.nalog\.ru/opendata/7707329152-kalendar/data-.+\.xml$"
)
STRUCTURE_URL_PATTERN = re.compile(
    r"^https://data\.nalog\.ru/opendata/7707329152-kalendar/structure-.+\.xsd$"
)


def _release_date_sort_key(url: str) -> tuple[int, int, int]:
    """Готовит дату XML-релиза для сортировки"""
    match = re.search(r"data-(\d{8})-structure", url)
    if not match:
        return (0, 0, 0)

    value = match.group(1)
    for date_format in ("%Y%m%d", "%d%m%Y"):
        try:
            release_date = datetime.strptime(value, date_format)
            return (release_date.year, release_date.month, release_date.day)
        except ValueError:
            continue

    return (0, 0, 0)


class _OpenDataLinkParser(HTMLParser):
    """Ищет ссылки на XML и XSD в паспорте открытых данных ФНС"""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.data_urls: list[str] = []
        self.structure_urls: list[str] = []

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        """Обрабатывает открывающий HTML-тег"""
        if tag != "a":
            return

        attrs_dict = dict(attrs)
        href = attrs_dict.get("href")
        if href is None:
            return

        if DATASET_URL_PATTERN.match(href):
            self.data_urls.append(href)
        elif STRUCTURE_URL_PATTERN.match(href):
            self.structure_urls.append(href)


def _read_open_data_links(index_path: Path = INDEX_FILE) -> tuple[list[str], str]:
    """Читает ссылки на XML-релизы и XSD из паспорта набора данных"""
    parser = _OpenDataLinkParser()
    parser.feed(index_path.read_text(encoding="utf-8"))

    data_urls = sorted(set(parser.data_urls), key=_release_date_sort_key)
    structure_urls = sorted(set(parser.structure_urls))

    if not data_urls:
        raise ValueError("В паспорте ФНС не найдены ссылки на XML-релизы")
    if not structure_urls:
        raise ValueError("В паспорте ФНС не найдена ссылка на XSD-структуру")

    latest_url = data_urls[-1]
    historical_urls = [url for url in data_urls if url != latest_url]
    return [latest_url, *historical_urls], structure_urls[-1]


def _release_output_path(url: str, output_dir: Path = RELEASES_DIR) -> Path:
    """Возвращает путь raw-файла для XML-релиза"""
    return output_dir / Path(url).name


def download_tax_calendar(
    index_path: Path = INDEX_FILE,
    latest_path: Path = LATEST_FILE,
    structure_path: Path = STRUCTURE_FILE,
    releases_dir: Path = RELEASES_DIR,
    force: bool = False,
) -> list[Path]:
    """Скачивает налоговый календарь ФНС и предыдущие XML-релизы"""
    download_file(SOURCE_URL, index_path)
    data_urls, structure_url = _read_open_data_links(index_path)

    download_file(structure_url, structure_path)

    releases_dir.mkdir(parents=True, exist_ok=True)
    downloaded_files: list[Path] = []
    for index, url in enumerate(data_urls):
        output_path = _release_output_path(url, releases_dir)
        if force or not output_path.exists():
            download_file(url, output_path)
        downloaded_files.append(output_path)

        if index == 0:
            download_file(url, latest_path)

    return downloaded_files


def main() -> None:
    """Запускает скачивание налогового календаря ФНС"""
    downloaded_files = download_tax_calendar()
    print(f"Подготовлено XML-релизов ФНС: {len(downloaded_files)}")
    print(f"Папка: {RELEASES_DIR}")


if __name__ == "__main__":
    main()
