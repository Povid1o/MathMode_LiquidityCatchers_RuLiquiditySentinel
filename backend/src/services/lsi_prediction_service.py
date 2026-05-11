from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODEL_FILE = PROJECT_ROOT / "models/lsi_pipeline.joblib"

LSI_GREEN_MAX = 40.0
LSI_YELLOW_MAX = 70.0
EMA_ALPHA = 0.05


def get_lsi_status(lsi_value: float) -> str:
    """Возвращает статус светофора по шкале LSI 0-100"""
    if lsi_value < LSI_GREEN_MAX:
        return "ЗЕЛЕНЫЙ (Норма)"
    if lsi_value < LSI_YELLOW_MAX:
        return "ЖЕЛТЫЙ (Повышенное внимание)"
    return "КРАСНЫЙ (Стресс ликвидности)"


def _load_model_components(model_path: Path = MODEL_FILE) -> dict[str, Any]:
    """Загружает сохраненный пайплайн LSI"""
    if not model_path.exists():
        raise FileNotFoundError(f"Файл модели LSI не найден: {model_path}")
    return joblib.load(model_path)


def _prepare_feature_matrix(
    data: pd.DataFrame,
    features_list: list[str],
) -> pd.DataFrame:
    """Готовит матрицу признаков в порядке, сохраненном при обучении модели"""
    missing_columns = [column for column in features_list if column not in data.columns]
    if missing_columns:
        raise ValueError(f"В данных нет признаков для LSI: {missing_columns[:10]}")

    return data[features_list].astype(float).fillna(0)


def _top_pca_drivers(
    scaled_row: np.ndarray,
    pca_weights: np.ndarray,
    features_list: list[str],
    top_n: int = 3,
) -> list[str]:
    """Выбирает главные драйверы LSI через вклад признаков в первую компоненту PCA"""
    contributions = np.abs(scaled_row * pca_weights)
    top_indexes = np.argsort(contributions)[::-1][:top_n]
    return [features_list[index] for index in top_indexes]


def add_lsi_scores(
    data: pd.DataFrame,
    model_path: Path = MODEL_FILE,
) -> pd.DataFrame:
    """Добавляет к датафрейму LSI_Index, lsi_status и top_drivers"""
    if data.empty:
        raise ValueError("Нельзя рассчитать LSI для пустого датафрейма")

    components = _load_model_components(model_path)
    features_list = components["features_list"]
    scaler = components["scaler"]
    pca = components["pca"]
    iso_forest = components["iso_forest"]
    minmax_scaler = components["minmax_scaler"]

    result = data.copy()
    if "date" in result.columns:
        result = result.sort_values("date").reset_index(drop=True)

    feature_matrix = _prepare_feature_matrix(result, features_list)
    scaled_matrix = scaler.transform(feature_matrix)
    pca_matrix = pca.transform(scaled_matrix)

    raw_scores = -iso_forest.decision_function(pca_matrix)
    smoothed_scores = (
        pd.Series(raw_scores)
        .ewm(alpha=EMA_ALPHA, adjust=False)
        .mean()
        .to_numpy()
    )
    lsi_values = minmax_scaler.transform(smoothed_scores.reshape(-1, 1)).flatten()
    lsi_values = np.clip(lsi_values, 0, 100)

    result["LSI_Index"] = np.round(lsi_values, 2)
    result["lsi"] = result["LSI_Index"]
    result["lsi_status"] = result["LSI_Index"].map(get_lsi_status)

    pc1_weights = pca.components_[0]
    result["top_drivers"] = [
        _top_pca_drivers(scaled_row, pc1_weights, features_list)
        for scaled_row in scaled_matrix
    ]

    return result


def get_lsi_prediction(
    new_data_df: pd.DataFrame,
    model_path: Path = MODEL_FILE,
) -> dict[str, object]:
    """Формирует ответ LSI для frontend или LLM"""
    lsi_data = add_lsi_scores(new_data_df, model_path=model_path)
    latest_row = lsi_data.iloc[-1]
    today_lsi = float(latest_row["LSI_Index"])
    row_date = latest_row["date"]
    date_value = row_date.date() if hasattr(row_date, "date") else row_date

    return {
        "date": str(date_value),
        "LSI_Index": today_lsi,
        "status": get_lsi_status(today_lsi),
        "top_drivers": latest_row["top_drivers"],
    }


def main() -> None:
    """Показывает пример ответа LSI по финальному датасету"""
    data_path = PROJECT_ROOT / "data/processed/final_ml_dataset.parquet"
    data = pd.read_parquet(data_path)
    response = get_lsi_prediction(data)
    print(response)


if __name__ == "__main__":
    main()
