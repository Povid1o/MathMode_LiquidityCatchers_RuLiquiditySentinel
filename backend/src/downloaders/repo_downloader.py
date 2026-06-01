from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed
from datetime import date
from datetime import datetime
from datetime import timedelta
from pathlib import Path
from urllib.parse import urlencode


PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.src.downloaders.common import download_file

SOURCE_URL = "https://www.cbr.ru/hd_base/repo/"
OUTPUT_FILE = PROJECT_ROOT / "data/raw/repo/repo.html"
DAILY_DIR = PROJECT_ROOT / "data/raw/repo/daily"
DEFAULT_START_DATE = date(2002, 11, 21)
DEFAULT_END_LAG_DAYS = 7


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


def _parse_csv_date(value: str) -> date:
    """Преобразует дату из DD-MM-YYYY в объект date"""
    return datetime.strptime(value, "%d-%m-%Y").date()


def _daily_output_path(value: date, output_dir: Path = DAILY_DIR) -> Path:
    """Возвращает путь дневного raw-файла"""
    return output_dir / f"{value.isoformat()}.html"


def _unique_repo_dates(rows: list[dict[str, object]]) -> list[date]:
    """Возвращает отсортированные даты аукционов из сводной таблицы"""
    dates = {
        _parse_csv_date(str(row["date"]))
        for row in rows
        if row.get("date") not in {None, ""}
    }
    return sorted(dates)


def download_repo(
    output_path: Path = OUTPUT_FILE,
    start_date: date = DEFAULT_START_DATE,
    end_date: date | None = None,
) -> None:
    """Скачивает HTML-файл ЦБ с итогами аукционов репо"""
    if end_date is None:
        end_date = date.today() - timedelta(days=DEFAULT_END_LAG_DAYS)

    download_file(_build_repo_url(start_date, end_date), output_path)


def download_repo_daily_pages(
    rows: list[dict[str, object]],
    output_dir: Path = DAILY_DIR,
    force: bool = False,
    max_workers: int = 8,
) -> None:
    """Скачивает дневные страницы ЦБ только для дат с аукционами"""
    repo_dates = _unique_repo_dates(rows)
    output_dir.mkdir(parents=True, exist_ok=True)

    dates_to_download = []
    for repo_date in repo_dates:
        if _daily_output_path(repo_date, output_dir).exists() and not force:
            continue
        dates_to_download.append(repo_date)

    skipped_count = len(repo_dates) - len(dates_to_download)
    downloaded_count = len(dates_to_download)
    if not dates_to_download:
        print(
            "Дневные страницы репо: "
            f"скачано 0, уже было в кеше {skipped_count}"
        )
        return

    def download_one(repo_date: date) -> None:
        """Скачивает одну дневную страницу репо"""
        output_path = _daily_output_path(repo_date, output_dir)
        download_file(_build_repo_url(repo_date, repo_date), output_path)

    completed_count = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(download_one, repo_date) for repo_date in dates_to_download]
        for future in as_completed(futures):
            future.result()
            completed_count += 1
            if completed_count % 100 == 0:
                print(
                    "Скачано дневных страниц репо: "
                    f"{completed_count}/{len(dates_to_download)}"
                )

    print(
        "Дневные страницы репо: "
        f"скачано {downloaded_count}, уже было в кеше {skipped_count}"
    )


def main() -> None:
    """Запускает скачивание итогов аукционов репо"""
    download_repo()
    print(f"Файл скачан: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
