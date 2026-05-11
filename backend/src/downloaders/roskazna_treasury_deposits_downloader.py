from __future__ import annotations

import argparse
import ssl
import sys
from datetime import date
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen


PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.src.downloaders.common import CHUNK_SIZE, USER_AGENT, download_file


BASE_URL = "https://roskazna.gov.ru"
SOURCE_URL = (
    "https://roskazna.gov.ru/finansovye-operacii/"
    "razmeshchenie-sredstv-edinogo-kaznachejskogo-scheta/"
    "razmeshchenie-sredstv-edinogo-kaznachejskogo-scheta-na-bankovskih-depozitah"
)
RAW_DIR = PROJECT_ROOT / "data/raw/treasury_funds/roskazna_deposits"
PAGES_DIR = PROJECT_ROOT / "data/raw/treasury_funds/roskazna_pages"
LINKS_FILE = PROJECT_ROOT / "data/raw/treasury_funds/roskazna_deposit_links.txt"
DEFAULT_START_YEAR = 2021
DEFAULT_MAX_PAGES_PER_YEAR = 80
ARCHIVE_MARKER = 'id="start-files-list"'


class _XmlLinkParser(HTMLParser):
    """Достает XML-ссылки Росказны из сохраненной HTML-страницы"""

    def __init__(self) -> None:
        super().__init__()
        self.links: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        """Обрабатывает открывающие HTML-теги"""
        if tag != "a":
            return

        attributes = dict(attrs)
        href = attributes.get("href")
        if href is None or not href.lower().endswith(".xml"):
            return

        self.links.append(urljoin(BASE_URL, href))


def _xml_output_path(url: str, raw_dir: Path) -> Path:
    """Строит путь сохранения XML-файла по ссылке"""
    parsed_url = urlparse(url)
    filename = Path(parsed_url.path).name
    if not filename:
        raise ValueError(f"Не удалось определить имя XML-файла из ссылки: {url}")
    return raw_dir / filename


def _page_url(year: int, page: int) -> str:
    """Собирает ссылку на страницу архива Росказны"""
    if page == 1:
        return f"{SOURCE_URL}?filter_year={year}"
    return f"{SOURCE_URL}?filter_year={year}&page={page}"


def _page_output_path(year: int, page: int, pages_dir: Path) -> Path:
    """Строит путь сохранения HTML-страницы Росказны"""
    return pages_dir / f"{year}_page_{page:02d}.html"


def _download_roskazna_file(
    url: str,
    output_path: Path,
    allow_unverified_ssl: bool = False,
) -> None:
    """Скачивает файл Росказны с опциональным отключением проверки SSL"""
    if not allow_unverified_ssl:
        download_file(url, output_path)
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    request = Request(url, headers={"User-Agent": USER_AGENT})
    ssl_context = ssl._create_unverified_context()

    with urlopen(request, timeout=60, context=ssl_context) as response:
        with temporary_path.open("wb") as file:
            while True:
                chunk = response.read(CHUNK_SIZE)
                if not chunk:
                    break
                file.write(chunk)

    temporary_path.replace(output_path)


def _read_links_from_text(text: str) -> list[str]:
    """Читает XML-ссылки из HTML-текста"""
    parser = _XmlLinkParser()
    parser.feed(text)
    return parser.links


def _read_links_from_html(path: Path) -> list[str]:
    """Читает XML-ссылки из HTML-файла Росказны"""
    return _read_links_from_text(path.read_text(encoding="utf-8"))


def _read_archive_links_from_html(path: Path) -> list[str]:
    """Читает XML-ссылки из архивной таблицы Росказны"""
    text = path.read_text(encoding="utf-8")
    marker_position = text.find(ARCHIVE_MARKER)
    if marker_position == -1:
        return _read_links_from_text(text)
    return _read_links_from_text(text[marker_position:])


def _read_current_links_from_html(path: Path) -> list[str]:
    """Читает XML-ссылки из верхнего текущего блока Росказны"""
    text = path.read_text(encoding="utf-8")
    marker_position = text.find(ARCHIVE_MARKER)
    if marker_position == -1:
        return []
    return _read_links_from_text(text[:marker_position])


def _is_last_archive_page(path: Path) -> bool:
    """Проверяет, что страница является последней страницей архива"""
    text = path.read_text(encoding="utf-8")
    marker_position = text.find(ARCHIVE_MARKER)
    if marker_position == -1:
        return False

    archive_text = text[marker_position:]
    return (
        'class="page-item disabled" aria-disabled="true" aria-label="Вперёд &raquo;"'
        in archive_text
    )


def _read_links_from_txt(path: Path) -> list[str]:
    """Читает XML-ссылки из текстового файла"""
    if not path.exists():
        return []

    links: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        link = line.strip()
        if link and not link.startswith("#") and link.lower().endswith(".xml"):
            links.append(link)
    return links


def download_roskazna_html_pages(
    years: list[int],
    pages_dir: Path = PAGES_DIR,
    max_pages_per_year: int = DEFAULT_MAX_PAGES_PER_YEAR,
    force: bool = False,
    allow_unverified_ssl: bool = False,
) -> list[Path]:
    """Скачивает HTML-страницы архива Росказны по годам и страницам"""
    pages_dir.mkdir(parents=True, exist_ok=True)
    downloaded_pages: list[Path] = []

    for year in years:
        previous_link_sets: set[tuple[str, ...]] = set()

        for page in range(1, max_pages_per_year + 1):
            output_path = _page_output_path(year, page, pages_dir)

            if output_path.exists() and not force:
                print(f"HTML-страница Росказны уже есть: {output_path.name}")
            else:
                try:
                    _download_roskazna_file(
                        _page_url(year, page),
                        output_path,
                        allow_unverified_ssl=allow_unverified_ssl,
                    )
                    print(f"Скачана HTML-страница Росказны: {output_path.name}")
                except Exception as error:
                    print(f"Не удалось скачать HTML-страницу Росказны: {year}, page={page}")
                    print(f"Причина: {error}")
                    break

            links = tuple(_read_archive_links_from_html(output_path))
            if not links:
                print(f"Останавливаем {year}: на странице {page} нет XML-ссылок архива")
                break

            if links in previous_link_sets:
                print(f"Останавливаем {year}: страница {page} повторяет предыдущие XML архива")
                break

            previous_link_sets.add(links)
            downloaded_pages.append(output_path)

            if _is_last_archive_page(output_path):
                print(f"Останавливаем {year}: страница {page} последняя в архиве")
                break

    print(f"Готово HTML-страниц Росказны: {len(set(downloaded_pages))}")
    return sorted(set(downloaded_pages))


def collect_roskazna_xml_links(
    pages_dir: Path = PAGES_DIR,
    links_file: Path = LINKS_FILE,
) -> list[str]:
    """Собирает XML-ссылки Росказны из HTML-страниц и txt-файла"""
    links: list[str] = []

    if pages_dir.exists():
        page_paths = sorted(pages_dir.glob("*.html"))
        for path in page_paths:
            links.extend(_read_archive_links_from_html(path))

        if page_paths:
            links.extend(_read_current_links_from_html(page_paths[-1]))

    links.extend(_read_links_from_txt(links_file))

    unique_links: list[str] = []
    seen_links: set[str] = set()
    for link in links:
        if link in seen_links:
            continue
        seen_links.add(link)
        unique_links.append(link)

    return unique_links


def download_roskazna_xml_files(
    links: list[str],
    raw_dir: Path = RAW_DIR,
    force: bool = False,
    allow_unverified_ssl: bool = False,
) -> list[Path]:
    """Скачивает XML-файлы Росказны по списку ссылок"""
    raw_dir.mkdir(parents=True, exist_ok=True)

    downloaded_files: list[Path] = []
    skipped_count = 0
    failed_count = 0

    for link in links:
        output_path = _xml_output_path(link, raw_dir)
        if output_path.exists() and not force:
            skipped_count += 1
            downloaded_files.append(output_path)
            continue

        try:
            _download_roskazna_file(
                link,
                output_path,
                allow_unverified_ssl=allow_unverified_ssl,
            )
        except Exception as error:
            failed_count += 1
            print(f"Не удалось скачать XML Росказны: {link}")
            print(f"Причина: {error}")
            continue

        downloaded_files.append(output_path)

    print(f"Найдено XML-ссылок Росказны: {len(links)}")
    print(f"Пропущено уже скачанных XML: {skipped_count}")
    print(f"Не скачано XML из-за ошибок: {failed_count}")
    print(f"Готово XML-файлов Росказны: {len(downloaded_files)}")

    return sorted(set(downloaded_files))


def _local_xml_files(raw_dir: Path = RAW_DIR) -> list[Path]:
    """Возвращает локальные XML-файлы Росказны"""
    raw_dir.mkdir(parents=True, exist_ok=True)
    return sorted(
        path
        for path in raw_dir.iterdir()
        if path.is_file() and path.suffix.lower() == ".xml"
    )


def prepare_roskazna_treasury_deposits(raw_dir: Path = RAW_DIR) -> list[Path]:
    """Скачивает XML Росказны из сохраненных ссылок и проверяет raw-директорию"""
    links = collect_roskazna_xml_links()
    if links:
        download_roskazna_xml_files(links, raw_dir)

    files = _local_xml_files(raw_dir)

    if not files:
        raise FileNotFoundError(
            "Не найдены XML-файлы Росказны и XML-ссылки для скачивания"
        )

    return files


def _parse_years(value: str | None) -> list[int]:
    """Парсит список лет из аргумента командной строки"""
    if value is None:
        return list(range(DEFAULT_START_YEAR, date.today().year + 1))

    years: list[int] = []
    for part in value.split(","):
        text = part.strip()
        if not text:
            continue
        if "-" in text:
            start_text, end_text = text.split("-", 1)
            years.extend(range(int(start_text), int(end_text) + 1))
        else:
            years.append(int(text))

    return sorted(set(years))


def main() -> None:
    """Скачивает и проверяет raw-файлы Росказны"""
    argument_parser = argparse.ArgumentParser(
        description="Скачивает XML Росказны по сохраненным HTML-страницам архива"
    )
    argument_parser.add_argument(
        "--no-update-pages",
        action="store_true",
        help="Не скачивать HTML-страницы архива, использовать только локальные HTML/txt",
    )
    argument_parser.add_argument(
        "--years",
        help="Годы для скачивания HTML, например 2024 или 2021-2026",
    )
    argument_parser.add_argument(
        "--max-pages",
        type=int,
        default=DEFAULT_MAX_PAGES_PER_YEAR,
        help="Максимум страниц архива на один год",
    )
    argument_parser.add_argument(
        "--force-pages",
        action="store_true",
        help="Перекачивать HTML-страницы, даже если они уже есть",
    )
    argument_parser.add_argument(
        "--force-xml",
        action="store_true",
        help="Перекачивать XML-файлы, даже если они уже есть",
    )
    argument_parser.add_argument(
        "--allow-unverified-ssl",
        action="store_true",
        help="Отключить проверку SSL-сертификата для Росказны",
    )
    args = argument_parser.parse_args()

    if args.allow_unverified_ssl:
        print("Внимание: проверка SSL-сертификата Росказны отключена")

    if not args.no_update_pages:
        download_roskazna_html_pages(
            years=_parse_years(args.years),
            max_pages_per_year=args.max_pages,
            force=args.force_pages,
            allow_unverified_ssl=args.allow_unverified_ssl,
        )

    links = collect_roskazna_xml_links()
    if links:
        download_roskazna_xml_files(
            links,
            force=args.force_xml,
            allow_unverified_ssl=args.allow_unverified_ssl,
        )

    files = _local_xml_files()
    if not files:
        raise FileNotFoundError(
            "Не найдены XML-файлы Росказны и XML-ссылки для скачивания"
        )

    print(f"Найдено XML-файлов Росказны: {len(files)}")


if __name__ == "__main__":
    main()
