from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.src.downloaders.common import download_file

SOURCE_URL = "https://www.cbr.ru/vfs/hd_base/RReserves/required_reserves_table.xlsx"
OUTPUT_FILE = PROJECT_ROOT / "data/raw/required_reserves/required_reserves_table.xlsx"


def download_required_reserves(output_path: Path = OUTPUT_FILE) -> None:
    """Скачивает Excel-файл ЦБ с обязательными резервами"""
    download_file(SOURCE_URL, output_path)


def main() -> None:
    """Запускает скачивание обязательных резервов"""
    download_required_reserves()
    print(f"Файл скачан: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
