from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.src.services.lsi_training_service import build_lsi_models
from backend.src.services.lsi_training_service import load_final_dataset
from backend.src.services.lsi_training_service import save_lsi_models
from backend.src.services.lsi_training_service import save_lsi_scores


def run_lsi_training_pipeline() -> None:
    """Запускает обучение глобального и локального LSI"""
    print("Загружаем финальный ML dataset")
    data = load_final_dataset()

    print("Обучаем LSI Global на всей истории")
    print("Обучаем LSI Local на последнем 365-дневном окне")
    global_artifact, local_artifact, scores = build_lsi_models(data)

    save_lsi_models(global_artifact, local_artifact)
    save_lsi_scores(scores)

    print("Готово")
    print(f"LSI Global: строк обучения {global_artifact['training_rows']}")
    print(
        "LSI Local: "
        f"{local_artifact['train_start']} — {local_artifact['train_end']}, "
        f"строк обучения {local_artifact['training_rows']}"
    )
    print(f"Строк scores: {len(scores)}")


def main() -> None:
    """Запускает pipeline обучения LSI"""
    run_lsi_training_pipeline()


if __name__ == "__main__":
    main()
