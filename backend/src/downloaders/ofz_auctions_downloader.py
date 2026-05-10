from __future__ import annotations

import sys
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urljoin
from urllib.parse import urlparse
from urllib.request import Request
from urllib.request import urlopen


PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

SOURCE_URL = (
    "https://minfin.gov.ru/ru/document/"
    "?q_4=%D0%A0%D0%B5%D0%B7%D1%83%D0%BB%D1%8C%D1%82%D0%B0%D1%82%D1%8B"
    "+%D0%BF%D1%80%D0%BE%D0%B2%D0%B5%D0%B4%D0%B5%D0%BD%D0%BD%D1%8B%D1%85"
    "+%D0%B0%D1%83%D0%BA%D1%86%D0%B8%D0%BE%D0%BD%D0%BE%D0%B2"
    "+%D0%BF%D0%BE+%D1%80%D0%B0%D0%B7%D0%BC%D0%B5%D1%89%D0%B5%D0%BD%D0%B8%D1%8E"
    "+%D0%B3%D0%BE%D1%81%D1%83%D0%B4%D0%B0%D1%80%D1%81%D1%82%D0%B2%D0%B5%D0%BD%D0%BD%D1%8B%D1%85"
    "+%D1%86%D0%B5%D0%BD%D0%BD%D1%8B%D1%85+%D0%B1%D1%83%D0%BC%D0%B0%D0%B3"
    "&input_select_search=&input_select_search=&input_select_search="
    "&P_DATE_from_4=&P_DATE_to_4=&M_DATE_from_4=&M_DATE_to_4="
    "&t_4=4706449514283688298&order_4=&dir_4=desc"
    "&by_doc_number_4=0&INF_BLOCK_ID_4=0"
)
RAW_INDEX_FILE = PROJECT_ROOT / "data/raw/ofz_auctions/index.html"
RAW_FILES_DIR = PROJECT_ROOT / "data/raw/ofz_auctions/files"
BASE_URL = "https://minfin.gov.ru"
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)
CHUNK_SIZE = 1024 * 1024


@dataclass
class OfzAuctionDocument:
    """Описывает документ Минфина с результатами аукционов ОФЗ"""

    title: str
    document_url: str
    file_url: str
    file_name: str
    published_date: str


class _DocumentIndexParser(HTMLParser):
    """Ищет карточки документов Минфина и ссылки на XLSX-файлы"""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.documents: list[OfzAuctionDocument] = []
        self._in_card = False
        self._card_depth = 0
        self._current: dict[str, str] = {}
        self._capture: str | None = None
        self._captured_text: list[str] = []

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        """Обрабатывает открывающий HTML-тег"""
        attributes = {name: value or "" for name, value in attrs}
        class_name = attributes.get("class", "")

        if tag == "div" and "document_card" in class_name:
            self._in_card = True
            self._card_depth = 1
            self._current = {}
            return

        if not self._in_card:
            return

        if tag == "div":
            self._card_depth += 1

        if tag == "a" and "document_title" in class_name:
            self._current["document_url"] = urljoin(BASE_URL, attributes.get("href", ""))
            self._capture = "title"
            self._captured_text = []
        elif tag == "a" and "file_item" in class_name:
            file_url = urljoin(BASE_URL, attributes.get("href", ""))
            self._current["file_url"] = file_url
            self._current["file_name"] = Path(urlparse(file_url).path).name
        elif tag == "span" and "date" in class_name:
            self._capture = "date"
            self._captured_text = []

    def handle_data(self, data: str) -> None:
        """Сохраняет текст внутри нужного тега"""
        if self._capture is not None:
            self._captured_text.append(data.strip())

    def handle_endtag(self, tag: str) -> None:
        """Обрабатывает закрывающий HTML-тег"""
        if not self._in_card:
            return

        if tag in {"a", "span"} and self._capture is not None:
            text = " ".join(value for value in self._captured_text if value)
            if self._capture == "title":
                self._current["title"] = text
            elif self._capture == "date" and text.startswith("Опубликовано:"):
                self._current["published_date"] = text.replace("Опубликовано:", "").strip()
            self._capture = None
            self._captured_text = []

        if tag == "div":
            self._card_depth -= 1
            if self._card_depth == 0:
                self._save_current_document()
                self._in_card = False

    def _save_current_document(self) -> None:
        """Сохраняет найденную карточку документа"""
        required_fields = {"title", "document_url", "file_url", "file_name"}
        if not required_fields.issubset(self._current):
            return
        if not self._current["file_name"].lower().endswith(".xlsx"):
            return

        self.documents.append(
            OfzAuctionDocument(
                title=self._current["title"],
                document_url=self._current["document_url"],
                file_url=self._current["file_url"],
                file_name=self._current["file_name"],
                published_date=self._current.get("published_date", ""),
            )
        )


def _download_file(url: str, output_path: Path) -> None:
    """Скачивает файл с сайта Минфина"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")

    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=60) as response:
        with temporary_path.open("wb") as file:
            while True:
                chunk = response.read(CHUNK_SIZE)
                if not chunk:
                    break
                file.write(chunk)

    temporary_path.replace(output_path)


def parse_document_index(index_path: Path = RAW_INDEX_FILE) -> list[OfzAuctionDocument]:
    """Парсит индексную страницу Минфина со списком документов"""
    parser = _DocumentIndexParser()
    parser.feed(index_path.read_text(encoding="utf-8-sig"))
    return parser.documents


def download_ofz_auctions(
    index_path: Path = RAW_INDEX_FILE,
    files_dir: Path = RAW_FILES_DIR,
) -> list[OfzAuctionDocument]:
    """Скачивает индекс Минфина и XLSX-файлы с результатами аукционов"""
    _download_file(SOURCE_URL, index_path)
    documents = parse_document_index(index_path)

    for document in documents:
        output_path = files_dir / document.file_name
        _download_file(document.file_url, output_path)

    return documents


def main() -> None:
    """Запускает скачивание документов Минфина по аукционам ОФЗ"""
    documents = download_ofz_auctions()
    print(f"Скачано XLSX-файлов: {len(documents)}")
    print(f"Папка: {RAW_FILES_DIR}")


if __name__ == "__main__":
    main()
