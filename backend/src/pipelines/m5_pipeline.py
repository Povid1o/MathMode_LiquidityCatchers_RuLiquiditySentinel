from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.src.downloaders.cbr_budget_funds_downloader import download_cbr_budget_funds
from backend.src.downloaders.cbr_liquidity_downloader import download_cbr_liquidity
from backend.src.downloaders.roskazna_treasury_deposits_downloader import (
    prepare_roskazna_treasury_deposits,
)

from backend.src.parsers.cbr_budget_funds import parse_cbr_budget_funds
from backend.src.parsers.cbr_budget_funds import save_csv as save_cbr_budget_funds_csv
from backend.src.parsers.cbr_liquidity import parse_cbr_liquidity
from backend.src.parsers.cbr_liquidity import save_csv as save_cbr_liquidity_csv
from backend.src.parsers.roskazna_treasury_deposits import (
    parse_roskazna_treasury_deposits,
)
from backend.src.parsers.roskazna_treasury_deposits import (
    save_csv as save_roskazna_treasury_deposits_csv,
)

from backend.src.services.m5_dataset_builder import build_m5_dataset
from backend.src.services.m5_dataset_builder import save_csv as save_m5_dataset_csv


def run_m5_pipeline() -> None:
    """Запускает полный пайплайн подготовки данных М5"""
    print("Скачиваем бюджетные средства на счетах банков с сайта ЦБ")
    download_cbr_budget_funds()

    print("Скачиваем дневную таблицу ликвидности банковского сектора ЦБ")
    download_cbr_liquidity()

    print("Проверяем XML-файлы Росказны по депозитам ЕКС")
    roskazna_files = prepare_roskazna_treasury_deposits()

    print("Обрабатываем бюджетные средства на счетах банков")
    budget_rows = parse_cbr_budget_funds()
    save_cbr_budget_funds_csv(budget_rows)

    print("Обрабатываем дневную таблицу ликвидности банковского сектора")
    liquidity_rows = parse_cbr_liquidity()
    save_cbr_liquidity_csv(liquidity_rows)

    print("Обрабатываем депозиты ЕКС Росказны")
    roskazna_rows = parse_roskazna_treasury_deposits()
    save_roskazna_treasury_deposits_csv(roskazna_rows)

    print("Собираем датасет М5")
    m5_rows = build_m5_dataset()
    save_m5_dataset_csv(m5_rows)

    print(f"Готово, строк по бюджетным средствам: {len(budget_rows)}")
    print(f"Готово, строк по ликвидности: {len(liquidity_rows)}")
    print(f"Готово, XML-файлов Росказны: {len(roskazna_files)}")
    print(f"Готово, строк по депозитам Росказны: {len(roskazna_rows)}")
    print(f"Готово, строк в датасете М5: {len(m5_rows)}")


def main() -> None:
    """Запускает пайплайн М5"""
    run_m5_pipeline()


if __name__ == "__main__":
    main()
