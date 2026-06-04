from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

# статусы берём из единого источника порогов — никаких магических 40/70 здесь
from backend.src.services.lsi_thresholds import get_lsi_status as _get_status_from_thresholds


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models"

FINAL_DATASET_FILE = DATA_DIR / "final_ml_dataset.parquet"
LSI_SCORES_CSV = DATA_DIR / "lsi_scores.csv"
LSI_SCORES_PARQUET = DATA_DIR / "lsi_scores.parquet"

GLOBAL_MODEL_FILE = MODEL_DIR / "lsi_global_pipeline.joblib"
LOCAL_MODEL_FILE = MODEL_DIR / "lsi_local_pipeline.joblib"

PCA_COMPONENTS = 10
CONTAMINATION = 0.06
RANDOM_STATE = 42
EMA_ALPHA = 0.05
LOCAL_WINDOW_DAYS = 365
MIN_LOCAL_ROWS = 120
MIN_LSI_FEATURES = 10
FEATURE_SELECTION_METHOD = "fixed_stress_whitelist"

LSI_FEATURE_CANDIDATES = [
    "m1_spread_mad_score",
    "m1_spread_relative_mad_score",
    "m1_spread_delta_mad_score",
    "m1_reserve_load_mad_score",
    "m1_ruonia_mad_score",
    "m1_flag_end_of_period",
    "m1_signal",
    "m1_signal_final",
    "m2_Flag_Demand",
    "m2_MAD_score_cover",
    "m2_MAD_score_rate_spread",
    "m2_auction_flag",
    "m3_cover_stress_score",
    "m3_yield_stress_score",
    "m3_Flag_Nedospros",
    "m3_Flag_Perespros",
    "m3_auction_flag",
    "m4_Tax_Week_Flag",
    "m4_Tax_Day_Strict",
    "m4_MAD_tax_pressure",
    "m4_MAD_tax_proximity",
    "m4_Seasonal_Factor_raw",
    "m5_cbr_liquidity_stress_mad_score",
    "m5_cbr_liquidity_drain_mad_score",
    "m5_roskazna_net_flow_stress_mad_score",
    "m5_Flag_Budget_Drain",
]


MODULES = ["m1", "m2", "m3", "m4", "m5"]


def compute_module_contributions(
    scaled_matrix: np.ndarray,
    pca: PCA,
    features_list: list[str],
) -> dict[str, np.ndarray]:
    """Вычисляет приближенный вклад каждого модуля M1-M5 в LSI-оценку.

    Метод — PCA-based weighted attribution (не SHAP, не причинная декомпозиция):
    для каждого признака j вклад = |x_scaled[j]| * sum_k(evr[k] * |components[k,j]|),
    где evr — explained_variance_ratio_, k — индекс главной компоненты.
    Вклады нормируются до 100% (по строкам) и агрегируются по префиксу модуля.

    Возвращает словарь {module_upper: ndarray вкладов в % по строкам}
    """
    evr = pca.explained_variance_ratio_            # (n_components,)
    components = pca.components_                   # (n_components, n_features)

    # "структурный вес" признака j в PCA — насколько он загружает компоненты, взвешенные по EVR
    structural_weights = np.abs(components).T @ evr   # (n_features,)

    # вклад признака j в строке i = |x_scaled[i,j]| * structural_weights[j]
    feature_contrib = np.abs(scaled_matrix) * structural_weights[np.newaxis, :]  # (n_rows, n_features)

    # нормировка до 100% по строкам
    row_sums = feature_contrib.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1.0, row_sums)
    feature_contrib_pct = feature_contrib / row_sums * 100.0

    # индексы признаков по модулям
    module_indices: dict[str, list[int]] = {}
    for j, feat_name in enumerate(features_list):
        prefix = feat_name.split("_", 1)[0].upper()
        module_indices.setdefault(prefix, []).append(j)

    result: dict[str, np.ndarray] = {}
    for module_upper in sorted(module_indices.keys()):
        indices = module_indices[module_upper]
        result[module_upper] = np.round(feature_contrib_pct[:, indices].sum(axis=1), 2)

    return result


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


def select_lsi_features(data: pd.DataFrame) -> list[str]:
    """Отбирает только согласованные стресс-признаки LSI"""
    numeric_features = set(data.select_dtypes(include=[np.number]).columns)
    selected_features = [
        column
        for column in LSI_FEATURE_CANDIDATES
        if column in numeric_features
    ]
    if len(selected_features) < MIN_LSI_FEATURES:
        missing_columns = [
            column
            for column in LSI_FEATURE_CANDIDATES
            if column not in data.columns
        ]
        raise ValueError(
            "Недостаточно стресс-признаков для LSI: "
            f"{len(selected_features)} из {len(LSI_FEATURE_CANDIDATES)}. "
            f"Отсутствуют: {missing_columns[:10]}"
        )

    return selected_features


def fit_lsi_artifact(
    data: pd.DataFrame,
    *,
    kind: str,
    window_days: int | None = None,
    feature_list: list[str] | None = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Обучает LSI artifact и возвращает scores на обучающем окне.

    feature_list — явный набор признаков (kind-aware whitelist для honest-моделей).
    Если None — используется production-whitelist через select_lsi_features (старое
    поведение, обратная совместимость).
    """
    if data.empty:
        raise ValueError("Нельзя обучить LSI на пустом датасете")

    if feature_list is None:
        features_list = select_lsi_features(data)
    else:
        numeric_features = set(data.select_dtypes(include=[np.number]).columns)
        features_list = [c for c in feature_list if c in numeric_features]
        if len(features_list) < MIN_LSI_FEATURES:
            raise ValueError(
                "Недостаточно признаков для LSI: "
                f"{len(features_list)} из {len(feature_list)}. "
                f"Отсутствуют: {[c for c in feature_list if c not in data.columns][:10]}"
            )
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
        "feature_selection_method": FEATURE_SELECTION_METHOD,
        "pca_components": n_components,
        "window_days": window_days,
        "train_start": str(data["date"].min().date()),
        "train_end": str(data["date"].max().date()),
        "training_rows": int(len(data)),
    }

    # вклады модулей M1-M5 (PCA-based approximation)
    module_contribs = compute_module_contributions(scaled_matrix, pca, features_list)

    scores_dict: dict[str, object] = {
        "date": data["date"].to_numpy(),
        f"lsi_{kind}": np.round(lsi_values, 2),
        f"lsi_{kind}_raw": raw_scores,
        f"lsi_{kind}_smoothed": smoothed_scores,
        f"lsi_{kind}_status": [_get_status_from_thresholds(value) for value in lsi_values],
    }
    for module_upper, contrib_array in module_contribs.items():
        scores_dict[f"lsi_{kind}_contrib_{module_upper.lower()}"] = contrib_array

    scores = pd.DataFrame(scores_dict)

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
