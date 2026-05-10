from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.src.downloaders.common import download_file


SOURCE_URL = "https://www.cbr.ru/vfs/statistics/banksector/borrowings/02_29_Budget_all.xlsx"
OUTPUT_FILE = PROJECT_ROOT / "data/raw/treasury_funds/cbr_budget_funds.xlsx"


def download_cbr_budget_funds(output_path: Path = OUTPUT_FILE) -> None:
    """Скачивает Excel-файл ЦБ с бюджетными средствами на счетах банков"""
    download_file(SOURCE_URL, output_path)


def main() -> None:
    """Запускает скачивание бюджетных средств на счетах банков"""
    download_cbr_budget_funds()
    print(f"Файл скачан: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
