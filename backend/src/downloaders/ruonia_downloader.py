from __future__ import annotations

import sys
from datetime import date
from pathlib import Path
from urllib.parse import urlencode


PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.src.downloaders.common import download_file

SOURCE_URL = "https://www.cbr.ru/Queries/UniDbQuery/DownloadExcel/14315"
OUTPUT_FILE = PROJECT_ROOT / "data/raw/ruonia/ruonia.xlsx"
DEFAULT_START_DATE = date(2010, 4, 1)


def _format_query_date(value: date) -> str:
    """Форматирует дату для запроса к сайту ЦБ"""
    return value.strftime("%m/%d/%Y")


def _build_ruonia_url(start_date: date, end_date: date) -> str:
    """Собирает ссылку для скачивания Excel-файла RUONIA"""
    params = {
        "FromDate": _format_query_date(start_date),
        "ToDate": _format_query_date(end_date),
        "posted": "False",
        "backUrl": "/hd_base/ruonia/dynamics/",
    }
    return f"{SOURCE_URL}?{urlencode(params)}"


def download_ruonia(
    output_path: Path = OUTPUT_FILE,
    start_date: date = DEFAULT_START_DATE,
    end_date: date | None = None,
) -> None:
    """Скачивает Excel-файл ЦБ с динамикой RUONIA"""
    if end_date is None:
        end_date = date.today()

    download_file(_build_ruonia_url(start_date, end_date), output_path)


def main() -> None:
    """Запускает скачивание RUONIA"""
    download_ruonia()
    print(f"Файл скачан: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
