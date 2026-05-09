from __future__ import annotations

import sys
from datetime import date
from pathlib import Path
from urllib.parse import urlencode


PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.src.downloaders.common import download_file

SOURCE_URL = "https://www.cbr.ru/hd_base/keyrate/"
OUTPUT_FILE = PROJECT_ROOT / "data/raw/keyrate/keyrate.html"
DEFAULT_START_DATE = date(2013, 9, 17)


def _format_query_date(value: date) -> str:
    """Форматирует дату для запроса к сайту ЦБ"""
    return value.strftime("%d.%m.%Y")


def _build_keyrate_url(start_date: date, end_date: date) -> str:
    """Собирает ссылку на официальную HTML-таблицу ключевой ставки"""
    params = {
        "UniDbQuery.From": _format_query_date(start_date),
        "UniDbQuery.To": _format_query_date(end_date),
        "UniDbQuery.Posted": "True",
    }
    return f"{SOURCE_URL}?{urlencode(params)}"


def download_keyrate(
    output_path: Path = OUTPUT_FILE,
    start_date: date = DEFAULT_START_DATE,
    end_date: date | None = None,
) -> None:
    """Скачивает HTML-файл ЦБ с ключевой ставкой"""
    if end_date is None:
        end_date = date.today()

    download_file(_build_keyrate_url(start_date, end_date), output_path)


def main() -> None:
    """Запускает скачивание ключевой ставки"""
    download_keyrate()
    print(f"Файл скачан: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
