from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DOWNLOADERS_PATH = PROJECT_ROOT / "backend/src/downloaders"
PARSERS_PATH = PROJECT_ROOT / "backend/src/parsers"
SERVICES_PATH = PROJECT_ROOT / "backend/src/services"

for path in (DOWNLOADERS_PATH, PARSERS_PATH, SERVICES_PATH):
    sys.path.insert(0, str(path))

from required_reserves_downloader import download_required_reserves
from ruonia_downloader import download_ruonia

from required_reserves import parse_required_reserves
from required_reserves import save_csv as save_required_reserves_csv
from ruonia import parse_ruonia
from ruonia import save_csv as save_ruonia_csv

from m1_dataset_builder import build_m1_dataset
from m1_dataset_builder import save_csv as save_m1_dataset_csv


def run_m1_pipeline() -> None:
    """Запускает полный пайплайн подготовки данных М1"""
    print("Скачиваем обязательные резервы")
    download_required_reserves()

    print("Скачиваем RUONIA")
    download_ruonia()

    print("Обрабатываем обязательные резервы")
    required_reserves_rows = parse_required_reserves()
    save_required_reserves_csv(required_reserves_rows)

    print("Обрабатываем RUONIA")
    ruonia_rows = parse_ruonia()
    save_ruonia_csv(ruonia_rows)

    print("Собираем датасет М1")
    m1_rows = build_m1_dataset()
    save_m1_dataset_csv(m1_rows)

    print(f"Готово, строк в датасете М1: {len(m1_rows)}")


def main() -> None:
    """Запускает пайплайн М1"""
    run_m1_pipeline()


if __name__ == "__main__":
    main()
