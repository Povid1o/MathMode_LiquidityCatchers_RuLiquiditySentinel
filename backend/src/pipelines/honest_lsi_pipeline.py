"""honest_lsi_pipeline — Phase B: единый прогон honest-LSI.

final_ml_dataset (+raw) → honest_ml_dataset → honest Global/Local модели + scores.
Parsing/исходные пайплайны не трогает; пишет только honest_* артефакты.

Запуск: python -m backend.src.pipelines.honest_lsi_pipeline
"""
from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.src.services.honest_feature_builder import build_honest_dataset, save_honest_dataset
from backend.src.services.honest_lsi_training import (
    build_honest_lsi_models,
    load_honest_dataset,
    save_honest_models,
    save_honest_scores,
)


def run_honest_lsi_pipeline() -> None:
    """Собирает honest-датасет и обучает honest Global/Local LSI."""
    print("Собираем honest_ml_dataset (honest-фичи M1-M5)")
    dataset = build_honest_dataset()
    save_honest_dataset(dataset)

    print("Обучаем honest Global/Local LSI (kind-aware whitelist)")
    data = load_honest_dataset()
    global_artifact, local_artifact, scores = build_honest_lsi_models(data)
    save_honest_models(global_artifact, local_artifact)
    save_honest_scores(scores)

    print(f"Готово. honest_ml_dataset: {len(dataset)} строк")
    print(f"Global: {len(global_artifact['features_list'])} фич | Local: {len(local_artifact['features_list'])} фич")


def main() -> None:
    run_honest_lsi_pipeline()


if __name__ == "__main__":
    main()
