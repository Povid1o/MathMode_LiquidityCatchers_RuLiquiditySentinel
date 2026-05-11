from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models"

FINAL_DATASET_FILE = DATA_DIR / "final_ml_dataset.parquet"
LSI_SCORES_CSV = DATA_DIR / "lsi_scores.csv"
LSI_SCORES_PARQUET = DATA_DIR / "lsi_scores.parquet"

GLOBAL_MODEL_FILE = MODEL_DIR / "lsi_global_pipeline.joblib"
LOCAL_MODEL_FILE = MODEL_DIR / "lsi_local_pipeline.joblib"

PCA_COMPONENTS = 10
TOP_N_PER_GROUP = 2
CONTAMINATION = 0.06
RANDOM_STATE = 42
EMA_ALPHA = 0.05
LOCAL_WINDOW_DAYS = 365
MIN_LOCAL_ROWS = 120


def load_final_dataset(path: Path = FINAL_DATASET_FILE) -> pd.DataFrame:
    """Загружает финальный ML dataset для обучения LSI"""
    if not path.exists():
        raise FileNotFoundError(f"Финальный ML dataset не найден: {path}")

    data = pd.read_parquet(path)
    if "date" not in data.columns:
        raise ValueError("В финальном ML dataset нет колонки date")

    data = data.copy()
    data["date"] = pd.to_datetime(data["date"])
    return data.sort_values("date").reset_index(drop=True)


def _feature_group(column: str) -> str:
    """Возвращает группу признака как в notebook final_chupappo"""
    parts = column.split("_")
    return "_".join(parts[:2])


def select_lsi_features(data: pd.DataFrame, top_n: int = TOP_N_PER_GROUP) -> list[str]:
    """Отбирает признаки по дисперсии внутри групп из первых двух частей имени"""
    numeric_features = list(data.select_dtypes(include=[np.number]).columns)
    if not numeric_features:
        raise ValueError("В датасете нет числовых признаков для LSI")

    feature_matrix = data[numeric_features].astype(float).fillna(0)
    groups: dict[str, list[str]] = defaultdict(list)
    for column in numeric_features:
        groups[_feature_group(column)].append(column)

    selected_features: list[str] = []
    for columns in groups.values():
        if len(columns) <= top_n:
            selected_features.extend(columns)
            continue
        variances = feature_matrix[columns].var()
        selected_features.extend(variances.nlargest(top_n).index.tolist())

    return selected_features


def _get_status(lsi_value: float) -> str:
    """Возвращает статус по шкале LSI 0-100"""
    if lsi_value < 40:
        return "ЗЕЛЕНЫЙ (Норма)"
    if lsi_value < 70:
        return "ЖЕЛТЫЙ (Повышенное внимание)"
    return "КРАСНЫЙ (Стресс ликвидности)"


def fit_lsi_artifact(
    data: pd.DataFrame,
    *,
    kind: str,
    window_days: int | None = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Обучает LSI artifact и возвращает scores на обучающем окне"""
    if data.empty:
        raise ValueError("Нельзя обучить LSI на пустом датасете")

    features_list = select_lsi_features(data)
    feature_matrix = data[features_list].astype(float).fillna(0)

    n_components = min(PCA_COMPONENTS, len(features_list), len(feature_matrix))
    if n_components < 1:
        raise ValueError("Недостаточно признаков для PCA")

    scaler = StandardScaler()
    scaled_matrix = scaler.fit_transform(feature_matrix)

    pca = PCA(n_components=n_components)
    pca_matrix = pca.fit_transform(scaled_matrix)

    iso_forest = IsolationForest(
        contamination=CONTAMINATION,
        random_state=RANDOM_STATE,
    )
    iso_forest.fit(pca_matrix)

    raw_scores = -iso_forest.decision_function(pca_matrix)
    smoothed_scores = (
        pd.Series(raw_scores)
        .ewm(alpha=EMA_ALPHA, adjust=False)
        .mean()
        .to_numpy()
    )

    minmax_scaler = MinMaxScaler(feature_range=(0, 100))
    lsi_values = minmax_scaler.fit_transform(smoothed_scores.reshape(-1, 1)).flatten()
    lsi_values = np.clip(lsi_values, 0, 100)

    artifact = {
        "version": 2,
        "kind": kind,
        "features_list": features_list,
        "scaler": scaler,
        "pca": pca,
        "iso_forest": iso_forest,
        "minmax_scaler": minmax_scaler,
        "ema_alpha": EMA_ALPHA,
        "contamination": CONTAMINATION,
        "random_state": RANDOM_STATE,
        "top_n_per_group": TOP_N_PER_GROUP,
        "pca_components": n_components,
        "window_days": window_days,
        "train_start": str(data["date"].min().date()),
        "train_end": str(data["date"].max().date()),
        "training_rows": int(len(data)),
    }

    scores = pd.DataFrame(
        {
            "date": data["date"].to_numpy(),
            f"lsi_{kind}": np.round(lsi_values, 2),
            f"lsi_{kind}_raw": raw_scores,
            f"lsi_{kind}_smoothed": smoothed_scores,
            f"lsi_{kind}_status": [_get_status(value) for value in lsi_values],
        }
    )

    return artifact, scores


def build_lsi_models(
    data: pd.DataFrame,
    *,
    local_window_days: int = LOCAL_WINDOW_DAYS,
) -> tuple[dict[str, Any], dict[str, Any], pd.DataFrame]:
    """Обучает глобальную и локальную LSI-модели"""
    data = data.copy()
    data["date"] = pd.to_datetime(data["date"])
    data = data.sort_values("date").reset_index(drop=True)

    global_artifact, global_scores = fit_lsi_artifact(data, kind="global")

    latest_date = data["date"].max()
    local_start = latest_date - pd.Timedelta(days=local_window_days)
    local_data = data[data["date"] >= local_start].reset_index(drop=True)
    if len(local_data) < MIN_LOCAL_ROWS:
        raise ValueError(
            f"Недостаточно строк для локального LSI: {len(local_data)}, нужно минимум {MIN_LOCAL_ROWS}"
        )

    local_artifact, local_scores = fit_lsi_artifact(
        local_data,
        kind="local",
        window_days=local_window_days,
    )

    scores = global_scores.merge(local_scores, on="date", how="left")
    return global_artifact, local_artifact, scores


def save_lsi_models(
    global_artifact: dict[str, Any],
    local_artifact: dict[str, Any],
    *,
    model_dir: Path = MODEL_DIR,
) -> None:
    """Сохраняет LSI-модели для backend inference"""
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(global_artifact, model_dir / GLOBAL_MODEL_FILE.name)
    joblib.dump(local_artifact, model_dir / LOCAL_MODEL_FILE.name)


def save_lsi_scores(
    scores: pd.DataFrame,
    *,
    csv_path: Path = LSI_SCORES_CSV,
    parquet_path: Path = LSI_SCORES_PARQUET,
) -> None:
    """Сохраняет рассчитанные LSI scores в CSV и parquet"""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    scores.to_csv(csv_path, index=False)
    scores.to_parquet(parquet_path, index=False)
