import pandas as pd
import streamlit as st
from pathlib import Path
import sys

from dashboard.config import DATASETS

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
THRESHOLD_METRICS_FILE = PROJECT_ROOT / "data" / "processed" / "lsi_threshold_metrics.csv"

from backend.src.services.lsi_thresholds import get_threshold_profile
# --- Phase B: honest LSI (новый сбалансированный индекс M1≈23/M2≈26/M3≈30/M5≈20, M4 — overlay) ---
from backend.src.services.honest_lsi_prediction import DEFAULT_HONEST_PROFILE
from backend.src.services.honest_lsi_prediction import get_honest_lsi_response
from backend.src.services.honest_lsi_prediction import honest_add_lsi_scores
from backend.src.services.honest_lsi_training import HONEST_GLOBAL_MODEL
from backend.src.services.honest_lsi_training import HONEST_LOCAL_MODEL
from backend.src.services.honest_lsi_training import load_honest_dataset

DEFAULT_THRESHOLD_PROFILE = DEFAULT_HONEST_PROFILE


def _parse_dates(df: pd.DataFrame, col: str = "date") -> pd.DataFrame:
    """Парсит дату с учетом ISO и DD-MM-YYYY форматов"""
    if col not in df.columns:
        return df
    if pd.api.types.is_datetime64_any_dtype(df[col]):
        return df
    try:
        df[col] = pd.to_datetime(df[col], format="%Y-%m-%d", errors="raise")
    except Exception:
        try:
            df[col] = pd.to_datetime(df[col], format="%d-%m-%Y", errors="raise")
        except Exception:
            df[col] = pd.to_datetime(df[col], dayfirst=True, errors="coerce")
    return df


@st.cache_data(ttl=3600, show_spinner=False)
def load_m1() -> pd.DataFrame:
    df = pd.read_parquet(DATASETS["m1"])
    df = _parse_dates(df)
    return df.sort_values("date").reset_index(drop=True)


@st.cache_data(ttl=3600, show_spinner=False)
def load_m2() -> pd.DataFrame:
    df = pd.read_parquet(DATASETS["m2"])
    df = _parse_dates(df)
    return df.sort_values("date").reset_index(drop=True)


@st.cache_data(ttl=3600, show_spinner=False)
def load_m3() -> pd.DataFrame:
    df = pd.read_parquet(DATASETS["m3"])
    df = _parse_dates(df)
    return df.sort_values("date").reset_index(drop=True)


@st.cache_data(ttl=3600, show_spinner=False)
def load_m4() -> pd.DataFrame:
    df = pd.read_parquet(DATASETS["m4"])
    df = _parse_dates(df)
    return df.sort_values("date").reset_index(drop=True)


@st.cache_data(ttl=3600, show_spinner=False)
def load_m5() -> pd.DataFrame:
    df = pd.read_parquet(DATASETS["m5"])
    df = _parse_dates(df)
    return df.sort_values("date").reset_index(drop=True)


@st.cache_data(ttl=3600, show_spinner=False)
def load_final() -> pd.DataFrame:
    df = pd.read_parquet(DATASETS["final"])
    df = _parse_dates(df)
    return df.sort_values("date").reset_index(drop=True)


@st.cache_data(ttl=3600, show_spinner=False)
def load_lsi() -> pd.DataFrame:
    """Загружает honest_ml_dataset и добавляет honest LSI (Global/Local).

    Phase B: индекс собран на сбалансированном whitelist стресс-признаков
    (M1/M2/M3/M5; M4 — налоговый overlay вне PCA). Возвращает колонки
    lsi / lsi_local / lsi_global / LSI_Index / LSI_Local / LSI_Global, как ждёт dashboard.
    """
    data = load_honest_dataset()

    if not HONEST_GLOBAL_MODEL.exists() and not HONEST_LOCAL_MODEL.exists():
        return data

    return honest_add_lsi_scores(data)


@st.cache_data(ttl=3600, show_spinner=False)
def load_lsi_response(threshold_profile: str = DEFAULT_THRESHOLD_PROFILE) -> dict[str, object]:
    """Возвращает последний honest-LSI-ответ для заданного порогового профиля.

    Числовые LSI-значения не пересчитываются — меняются только статусы (ЗЕЛЕНЫЙ/ЖЕЛТЫЙ/КРАСНЫЙ).
    Кеш учитывает threshold_profile: каждый профиль кешируется отдельно.
    """
    data = load_honest_dataset()
    return get_honest_lsi_response(data, threshold_profile=threshold_profile)


def load_threshold_profile(profile: str = DEFAULT_THRESHOLD_PROFILE) -> dict[str, object]:
    """Возвращает конфигурацию порогового профиля LSI для dashboard"""
    return get_threshold_profile(profile)


@st.cache_data(ttl=3600, show_spinner=False)
def load_threshold_metrics() -> pd.DataFrame:
    """Загружает метрики качества порогов LSI"""
    if not THRESHOLD_METRICS_FILE.exists():
        return pd.DataFrame()
    return pd.read_csv(THRESHOLD_METRICS_FILE)


def dataset_summary() -> dict:
    """Возвращает метаданные по всем датасетам для overview-страницы"""
    results = {}
    loaders = {
        "m1": load_m1,
        "m2": load_m2,
        "m3": load_m3,
        "m4": load_m4,
        "m5": load_m5,
        "final": load_final,
    }
    for key, loader in loaders.items():
        path: Path = DATASETS[key]
        try:
            df = loader()
            results[key] = {
                "ok": True,
                "rows": len(df),
                "cols": len(df.columns),
                "date_min": df["date"].min(),
                "date_max": df["date"].max(),
                "missing_pct": df.isnull().mean().max() * 100,
                "path": str(path.name),
            }
        except Exception as e:
            results[key] = {"ok": False, "error": str(e), "path": str(path.name)}
    return results
