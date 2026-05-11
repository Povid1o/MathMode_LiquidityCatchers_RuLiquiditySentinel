from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.src.services.final_dataset_builder import build_final_ml_dataset
from backend.src.services.final_dataset_builder import save_csv
from backend.src.services.final_dataset_builder import save_parquet


def run_final_dataset_pipeline() -> None:
    """Запускает сборку финального ML dataset из готовых feature-файлов"""
    print("Собираем финальный ML dataset из признаков М1-М5")
    final_dataset = build_final_ml_dataset()
    save_csv(final_dataset)
    save_parquet(final_dataset)
    print(f"Готово, строк: {len(final_dataset)}")
    print(f"Готово, колонок: {len(final_dataset.columns)}")


def main() -> None:
    """Запускает pipeline финального ML dataset"""
    run_final_dataset_pipeline()


if __name__ == "__main__":
    main()
