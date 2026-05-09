from __future__ import annotations

import sys
from datetime import date
from pathlib import Path
from urllib.parse import urlencode


PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.src.downloaders.common import download_file

SOURCE_URL = "https://www.cbr.ru/hd_base/repo/"
OUTPUT_FILE = PROJECT_ROOT / "data/raw/repo/repo.html"
DEFAULT_START_DATE = date(2002, 11, 21)


def _format_query_date(value: date) -> str:
    """Форматирует дату для запроса к сайту ЦБ"""
    return value.strftime("%d.%m.%Y")


def _build_repo_url(start_date: date, end_date: date) -> str:
    """Собирает ссылку на официальную HTML-таблицу итогов репо"""
    params = {
        "UniDbQuery.From": _format_query_date(start_date),
        "UniDbQuery.To": _format_query_date(end_date),
        "UniDbQuery.Posted": "True",
        "UniDbQuery.P1": "0",
    }
    return f"{SOURCE_URL}?{urlencode(params)}"


def download_repo(
    output_path: Path = OUTPUT_FILE,
    start_date: date = DEFAULT_START_DATE,
    end_date: date | None = None,
) -> None:
    """Скачивает HTML-файл ЦБ с итогами аукционов репо"""
    if end_date is None:
        end_date = date.today()

    download_file(_build_repo_url(start_date, end_date), output_path)


def main() -> None:
    """Запускает скачивание итогов аукционов репо"""
    download_repo()
    print(f"Файл скачан: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
