from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]

sys.path.insert(0, str(PROJECT_ROOT))

from backend.src.downloaders.ofz_auctions_downloader import download_ofz_auctions

from backend.src.parsers.ofz_auctions import parse_ofz_auctions
from backend.src.parsers.ofz_auctions import save_csv as save_ofz_auctions_csv

from backend.src.services.m3_dataset_builder import build_m3_dataset
from backend.src.services.m3_dataset_builder import save_csv as save_m3_dataset_csv


def run_m3_pipeline() -> None:
    """Запускает полный пайплайн подготовки данных М3"""
    print("Скачиваем документы Минфина по аукционам ОФЗ")
    documents = download_ofz_auctions()
    print(f"Скачано документов Минфина: {len(documents)}")

    print("Обрабатываем аукционы ОФЗ")
    ofz_rows = parse_ofz_auctions()
    save_ofz_auctions_csv(ofz_rows)

    print("Собираем датасет М3")
    m3_rows = build_m3_dataset()
    save_m3_dataset_csv(m3_rows)

    print(f"Готово, строк в датасете М3: {len(m3_rows)}")


def main() -> None:
    """Запускает пайплайн М3"""
    run_m3_pipeline()


if __name__ == "__main__":
    main()
