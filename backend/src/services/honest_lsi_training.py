"""honest_lsi_training — Phase B: обучение honest Global/Local LSI (kind-aware).

Тот же пайплайн, что production (StandardScaler→PCA→IsolationForest→EMA→MinMax),
но с kind-aware whitelist: Global использует GLOBAL_WHITELIST, Local — LOCAL_WHITELIST
(+ rk_bidders, доступный на свежем окне). M4 — overlay (вне PCA).

Пишет НОВЫЕ артефакты (honest_*), production-модели не трогает.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from backend.src.services.honest_feature_builder import (
    GLOBAL_WHITELIST,
    HONEST_DATASET_PARQUET,
    LOCAL_WHITELIST,
)
from backend.src.services.lsi_training_service import (
    LOCAL_WINDOW_DAYS,
    MIN_LOCAL_ROWS,
    fit_lsi_artifact,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models"

HONEST_GLOBAL_MODEL = MODEL_DIR / "honest_lsi_global_pipeline.joblib"
HONEST_LOCAL_MODEL = MODEL_DIR / "honest_lsi_local_pipeline.joblib"
HONEST_SCORES_CSV = DATA_DIR / "honest_lsi_scores.csv"
HONEST_SCORES_PARQUET = DATA_DIR / "honest_lsi_scores.parquet"


def load_honest_dataset(path: Path = HONEST_DATASET_PARQUET) -> pd.DataFrame:
    """Грузит honest_ml_dataset, парсит дату."""
    data = pd.read_parquet(path)
    data["date"] = pd.to_datetime(data["date"])
    return data.sort_values("date").reset_index(drop=True)


def build_honest_lsi_models(
    data: pd.DataFrame,
    *,
    local_window_days: int = LOCAL_WINDOW_DAYS,
) -> tuple[dict[str, Any], dict[str, Any], pd.DataFrame]:
    """Обучает honest Global (GLOBAL_WHITELIST) и Local (LOCAL_WHITELIST)."""
    data = data.copy()
    data["date"] = pd.to_datetime(data["date"])
    data = data.sort_values("date").reset_index(drop=True)

    global_artifact, global_scores = fit_lsi_artifact(
        data, kind="global", feature_list=GLOBAL_WHITELIST
    )

    latest_date = data["date"].max()
    local_start = latest_date - pd.Timedelta(days=local_window_days)
    local_data = data[data["date"] >= local_start].reset_index(drop=True)
    if len(local_data) < MIN_LOCAL_ROWS:
        raise ValueError(
            f"Недостаточно строк для локального honest LSI: {len(local_data)}, "
            f"нужно минимум {MIN_LOCAL_ROWS}"
        )

    local_artifact, local_scores = fit_lsi_artifact(
        local_data,
        kind="local",
        window_days=local_window_days,
        feature_list=LOCAL_WHITELIST,
    )

    scores = global_scores.merge(local_scores, on="date", how="left")
    return global_artifact, local_artifact, scores


def save_honest_models(
    global_artifact: dict[str, Any],
    local_artifact: dict[str, Any],
    *,
    model_dir: Path = MODEL_DIR,
) -> None:
    """Сохраняет honest-модели (новые файлы, production не трогаем)."""
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(global_artifact, model_dir / HONEST_GLOBAL_MODEL.name)
    joblib.dump(local_artifact, model_dir / HONEST_LOCAL_MODEL.name)


def save_honest_scores(
    scores: pd.DataFrame,
    *,
    csv_path: Path = HONEST_SCORES_CSV,
    parquet_path: Path = HONEST_SCORES_PARQUET,
) -> None:
    """Сохраняет honest LSI scores."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    scores.to_csv(csv_path, index=False)
    scores.to_parquet(parquet_path, index=False)


def main() -> None:
    """Обучает и сохраняет honest Global/Local LSI."""
    data = load_honest_dataset()
    global_artifact, local_artifact, scores = build_honest_lsi_models(data)
    save_honest_models(global_artifact, local_artifact)
    save_honest_scores(scores)
    print(f"Global: {global_artifact['training_rows']} строк, {len(global_artifact['features_list'])} фич, "
          f"{global_artifact['train_start']}→{global_artifact['train_end']}")
    print(f"Local:  {local_artifact['training_rows']} строк, {len(local_artifact['features_list'])} фич, "
          f"{local_artifact['train_start']}→{local_artifact['train_end']}")
    print(f"Модели: {HONEST_GLOBAL_MODEL.name}, {HONEST_LOCAL_MODEL.name}")
    print(f"Scores: {HONEST_SCORES_CSV}")


if __name__ == "__main__":
    main()
