from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

import joblib
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.src.services.lsi_training_service import GLOBAL_MODEL_FILE
from backend.src.services.lsi_training_service import LOCAL_MODEL_FILE


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


def _load_model_components(model_path: Path) -> dict[str, Any]:
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


def _score_with_artifact(
    data: pd.DataFrame,
    components: dict[str, Any],
    *,
    prefix: str,
    apply_from_date: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Считает LSI по одному сохраненному artifact"""
    if data.empty:
        raise ValueError("Нельзя рассчитать LSI для пустого датафрейма")

    features_list = components["features_list"]
    scaler = components["scaler"]
    pca = components["pca"]
    iso_forest = components["iso_forest"]
    minmax_scaler = components["minmax_scaler"]
    ema_alpha = float(components.get("ema_alpha", EMA_ALPHA))

    result = data.copy()
    if "date" in result.columns:
        result = result.sort_values("date").reset_index(drop=True)

    if apply_from_date is not None and "date" in result.columns:
        score_mask = result["date"] >= apply_from_date
    else:
        score_mask = pd.Series(True, index=result.index)
    score_data = result.loc[score_mask].copy()
    score_index = score_data.index

    if score_data.empty:
        empty = pd.DataFrame(index=result.index)
        empty[f"LSI_{prefix.title()}"] = np.nan
        empty[f"lsi_{prefix}"] = np.nan
        empty[f"lsi_{prefix}_status"] = None
        empty[f"top_drivers_{prefix}"] = [[] for _ in range(len(result))]
        return empty

    feature_matrix = _prepare_feature_matrix(score_data, features_list)
    scaled_matrix = scaler.transform(feature_matrix)
    pca_matrix = pca.transform(scaled_matrix)

    raw_scores = -iso_forest.decision_function(pca_matrix)
    smoothed_scores = (
        pd.Series(raw_scores)
        .ewm(alpha=ema_alpha, adjust=False)
        .mean()
        .to_numpy()
    )
    lsi_values = minmax_scaler.transform(smoothed_scores.reshape(-1, 1)).flatten()
    lsi_values = np.clip(lsi_values, 0, 100)

    scored = pd.DataFrame(index=score_index)
    scored[f"LSI_{prefix.title()}"] = np.round(lsi_values, 2)
    scored[f"lsi_{prefix}"] = scored[f"LSI_{prefix.title()}"]
    scored[f"lsi_{prefix}_raw"] = raw_scores
    scored[f"lsi_{prefix}_smoothed"] = smoothed_scores
    scored[f"lsi_{prefix}_status"] = scored[f"LSI_{prefix.title()}"].map(get_lsi_status)

    pc1_weights = pca.components_[0]
    scored[f"top_drivers_{prefix}"] = [
        _top_pca_drivers(scaled_row, pc1_weights, features_list)
        for scaled_row in scaled_matrix
    ]

    output = pd.DataFrame(index=result.index)
    for column in scored.columns:
        if column.startswith("top_drivers") or column.endswith("_status"):
            output[column] = pd.Series([None] * len(result), index=result.index, dtype=object)
            output.loc[score_index, column] = pd.Series(
                scored[column].to_list(),
                index=score_index,
                dtype=object,
            )
        else:
            output[column] = np.nan
            output.loc[score_index, column] = scored[column].to_numpy()
    return output


def _merge_score_columns(result: pd.DataFrame, scores: pd.DataFrame) -> pd.DataFrame:
    """Добавляет рассчитанные LSI-колонки к результату"""
    for column in scores.columns:
        result[column] = scores[column].to_list()
    return result


def add_lsi_scores(
    data: pd.DataFrame,
    global_model_path: Path = GLOBAL_MODEL_FILE,
    local_model_path: Path = LOCAL_MODEL_FILE,
) -> pd.DataFrame:
    """Добавляет к датафрейму локальный и глобальный LSI"""
    if data.empty:
        raise ValueError("Нельзя рассчитать LSI для пустого датафрейма")

    result = data.copy()
    if "date" in result.columns:
        result["date"] = pd.to_datetime(result["date"])
        result = result.sort_values("date").reset_index(drop=True)

    has_any_model = False
    if global_model_path.exists():
        global_components = _load_model_components(global_model_path)
        global_scores = _score_with_artifact(result, global_components, prefix="global")
        result = _merge_score_columns(result, global_scores)
        has_any_model = True

    if local_model_path.exists():
        local_components = _load_model_components(local_model_path)
        train_start = local_components.get("train_start")
        apply_from_date = pd.Timestamp(train_start) if train_start else None
        local_scores = _score_with_artifact(
            result,
            local_components,
            prefix="local",
            apply_from_date=apply_from_date,
        )
        result = _merge_score_columns(result, local_scores)
        has_any_model = True

    if not has_any_model:
        raise FileNotFoundError("Файлы моделей LSI не найдены")

    if "LSI_Local" in result.columns and result["LSI_Local"].notna().any():
        result["LSI_Index"] = result["LSI_Local"].combine_first(result.get("LSI_Global"))
        result["lsi"] = result["lsi_local"].combine_first(result.get("lsi_global"))
        result["lsi_status"] = result["lsi_local_status"].combine_first(result.get("lsi_global_status"))
        result["top_drivers"] = result["top_drivers_local"].combine_first(result.get("top_drivers_global"))
    elif "LSI_Global" in result.columns:
        result["LSI_Index"] = result["LSI_Global"]
        result["lsi"] = result["lsi_global"]
        result["lsi_status"] = result["lsi_global_status"]
        result["top_drivers"] = result["top_drivers_global"]
    else:
        raise ValueError("Модели LSI не вернули ни локальный, ни глобальный индекс")

    return result


def get_lsi_prediction(
    new_data_df: pd.DataFrame,
) -> dict[str, object]:
    """Формирует ответ LSI для frontend или LLM"""
    lsi_data = add_lsi_scores(new_data_df)
    latest_row = lsi_data.iloc[-1]
    today_lsi = float(latest_row["LSI_Index"])
    row_date = latest_row["date"]
    date_value = row_date.date() if hasattr(row_date, "date") else row_date

    response: dict[str, object] = {
        "date": str(date_value),
        "LSI_Index": today_lsi,
        "status": str(latest_row["lsi_status"]),
        "top_drivers": latest_row["top_drivers"],
    }

    if "LSI_Local" in latest_row and pd.notna(latest_row["LSI_Local"]):
        response["LSI_Local"] = float(latest_row["LSI_Local"])
        response["local_status"] = str(latest_row["lsi_local_status"])
        response["local_top_drivers"] = latest_row["top_drivers_local"]
    if "LSI_Global" in latest_row and pd.notna(latest_row["LSI_Global"]):
        response["LSI_Global"] = float(latest_row["LSI_Global"])
        response["global_status"] = str(latest_row["lsi_global_status"])
        response["global_top_drivers"] = latest_row["top_drivers_global"]

    return response


def main() -> None:
    """Показывает пример ответа LSI по финальному датасету"""
    data_path = PROJECT_ROOT / "data/processed/final_ml_dataset.parquet"
    data = pd.read_parquet(data_path)
    response = get_lsi_prediction(data)
    print(response)


if __name__ == "__main__":
    main()
