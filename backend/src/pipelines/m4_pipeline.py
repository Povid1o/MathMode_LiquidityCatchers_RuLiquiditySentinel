from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]

sys.path.insert(0, str(PROJECT_ROOT))

from backend.src.downloaders.tax_calendar_downloader import download_tax_calendar

from backend.src.parsers.tax_calendar import parse_tax_calendar
from backend.src.parsers.tax_calendar import save_csv as save_tax_calendar_csv

from backend.src.services.m4_dataset_builder import build_m4_dataset
from backend.src.services.m4_dataset_builder import save_csv as save_m4_dataset_csv
from backend.src.services.m4_feature_builder import build_m4_features
from backend.src.services.m4_feature_builder import save_csv as save_m4_features_csv
from backend.src.services.m4_feature_builder import save_parquet as save_m4_features_parquet


def run_m4_pipeline() -> None:
    """Запускает полный пайплайн подготовки данных М4"""
    print("Скачиваем налоговый календарь ФНС")
    downloaded_files = download_tax_calendar()
    print(f"Подготовлено XML-релизов ФНС: {len(downloaded_files)}")

    print("Обрабатываем налоговый календарь ФНС")
    tax_calendar_rows = parse_tax_calendar()
    save_tax_calendar_csv(tax_calendar_rows)

    print("Собираем датасет М4")
    m4_rows = build_m4_dataset()
    save_m4_dataset_csv(m4_rows)

    print("Собираем признаки М4")
    m4_feature_rows = build_m4_features()
    save_m4_features_csv(m4_feature_rows)
    save_m4_features_parquet(m4_feature_rows)

    print(f"Готово, строк в налоговом календаре: {len(tax_calendar_rows)}")
    print(f"Готово, строк в датасете М4: {len(m4_rows)}")
    print(f"Готово, строк в признаках М4: {len(m4_feature_rows)}")


def main() -> None:
    """Запускает пайплайн М4"""
    run_m4_pipeline()


if __name__ == "__main__":
    main()
