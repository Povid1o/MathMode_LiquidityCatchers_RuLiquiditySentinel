from __future__ import annotations

import sys
from datetime import date
from pathlib import Path
from urllib.parse import urlencode


PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.src.downloaders.common import download_file


BASE_URL = "https://www.cbr.ru/hd_base/bliquidity/"
DEFAULT_FROM_DATE = "01.02.2014"
OUTPUT_FILE = PROJECT_ROOT / "data/raw/treasury_funds/cbr_liquidity.html"


def _today_for_cbr() -> str:
    """Возвращает текущую дату в формате сайта ЦБ"""
    return date.today().strftime("%d.%m.%Y")


def _build_url(from_date: str = DEFAULT_FROM_DATE, to_date: str | None = None) -> str:
    """Собирает ссылку на дневную таблицу ликвидности ЦБ"""
    params = {
        "UniDbQuery.Posted": "True",
        "UniDbQuery.From": from_date,
        "UniDbQuery.To": to_date or _today_for_cbr(),
    }
    return f"{BASE_URL}?{urlencode(params)}"


def download_cbr_liquidity(
    output_path: Path = OUTPUT_FILE,
    from_date: str = DEFAULT_FROM_DATE,
    to_date: str | None = None,
) -> None:
    """Скачивает дневную HTML-таблицу ЦБ по ликвидности банковского сектора"""
    download_file(_build_url(from_date, to_date), output_path)


def main() -> None:
    """Запускает скачивание дневной таблицы ликвидности ЦБ"""
    download_cbr_liquidity()
    print(f"Файл скачан: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
