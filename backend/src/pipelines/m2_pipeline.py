from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]

sys.path.insert(0, str(PROJECT_ROOT))

from backend.src.downloaders.keyrate_downloader import download_keyrate
from backend.src.downloaders.repo_downloader import download_repo
from backend.src.downloaders.repo_downloader import download_repo_daily_pages

from backend.src.parsers.keyrate import parse_keyrate
from backend.src.parsers.keyrate import save_csv as save_keyrate_csv
from backend.src.parsers.repo import parse_repo
from backend.src.parsers.repo import parse_repo_summary
from backend.src.parsers.repo import save_csv as save_repo_csv

from backend.src.services.m2_dataset_builder import build_m2_dataset
from backend.src.services.m2_dataset_builder import save_csv as save_m2_dataset_csv
from backend.src.services.m2_feature_builder import build_m2_features
from backend.src.services.m2_feature_builder import save_csv as save_m2_features_csv
from backend.src.services.m2_feature_builder import save_parquet as save_m2_features_parquet


def run_m2_pipeline() -> None:
    """Запускает полный пайплайн подготовки данных М2"""
    print("Скачиваем итоги аукционов репо")
    download_repo()

    print("Скачиваем дневные детали аукционов репо")
    repo_summary_rows = parse_repo_summary()
    download_repo_daily_pages(repo_summary_rows)

    print("Скачиваем ключевую ставку")
    download_keyrate()

    print("Обрабатываем итоги аукционов репо")
    repo_rows = parse_repo()
    save_repo_csv(repo_rows)

    print("Обрабатываем ключевую ставку")
    keyrate_rows = parse_keyrate()
    save_keyrate_csv(keyrate_rows)

    print("Собираем датасет М2")
    m2_rows = build_m2_dataset()
    save_m2_dataset_csv(m2_rows)

    print("Собираем признаки М2")
    m2_feature_rows = build_m2_features()
    save_m2_features_csv(m2_feature_rows)
    save_m2_features_parquet(m2_feature_rows)

    print(f"Готово, строк в датасете М2: {len(m2_rows)}")
    print(f"Готово, строк в признаках М2: {len(m2_feature_rows)}")


def main() -> None:
    """Запускает пайплайн М2"""
    run_m2_pipeline()


if __name__ == "__main__":
    main()
