"""lab/utils.py — reusable helpers for the RU Liquidity Sentinel research lab.

Назначение: дать ноутбукам в lab/ единый набор функций для чтения существующих
артефактов, переобучения LSI-подобной модели в процессе (без записи в production),
сравнения шкал, объяснимости и proxy-таргетов для Local.

Принципы:
- НЕ меняет production-код и не перезаписывает артефакты в models/ или data/processed/.
- Всё обучение делается in-memory; production-пайплайны не трогаются.
- Где возможно, переиспользует функции backend (compute_module_contributions,
  LSI_FEATURE_CANDIDATES), чтобы оставаться верными production-логике.

Импорт из корня проекта:
    import sys; sys.path.insert(0, ".")   # если запуск из корня
    from lab import utils
или (если CWD = lab/):
    import utils
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

# sklearn version mismatch при unpickle старых артефактов — глушим шум в лабе
warnings.filterwarnings("ignore", message=".*InconsistentVersionWarning.*")
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from scipy import stats

# ----------------------------------------------------------------------------
# Константы по умолчанию (ноутбуки могут переопределять в parameter-ячейках)
# ----------------------------------------------------------------------------
DEFAULT_PCA_COMPONENTS = 10
DEFAULT_EMA_ALPHA = 0.05
DEFAULT_CONTAMINATION = 0.06
DEFAULT_RANDOM_STATE = 42
DEFAULT_LOCAL_WINDOW_DAYS = 365

# фичи, помеченные как мёртвая / дублирующая в аудите
DEAD_FEATURES = ["m1_flag_end_of_period"]
DUPLICATE_FEATURES = ["m1_signal_final"]  # дублирует m1_signal

# стресс-эпизоды (те же окна, что в lsi_backtest_service)
STRESS_EPISODES: dict[str, tuple[str, str]] = {
    "Dec2014": ("2014-12-01", "2014-12-31"),
    "Feb-Mar2022": ("2022-02-01", "2022-03-31"),
    "Aug2023": ("2023-08-01", "2023-08-31"),
}

# дефолтные пороги production-профиля backtest_sensitive
DEFAULT_THRESHOLDS = {"green_max": 30.0, "yellow_max": 60.0}


# ----------------------------------------------------------------------------
# Пути и загрузка данных
# ----------------------------------------------------------------------------
def project_root(start: Path | None = None) -> Path:
    """Находит корень проекта, поднимаясь вверх до папки с data/processed."""
    here = (start or Path.cwd()).resolve()
    for candidate in [here, *here.parents]:
        if (candidate / "data" / "processed").is_dir() and (candidate / "backend").is_dir():
            return candidate
    # fallback: два уровня вверх от этого файла (lab/ -> root)
    return Path(__file__).resolve().parents[1]


def _ensure_backend_on_path() -> Path:
    """Добавляет корень проекта в sys.path, чтобы импортировать backend.*"""
    import sys

    root = project_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


def data_dir() -> Path:
    return project_root() / "data" / "processed"


def models_dir() -> Path:
    return project_root() / "models"


def load_final_dataset() -> pd.DataFrame:
    """Грузит финальный ML dataset (parquet), парсит дату, сортирует."""
    path = data_dir() / "final_ml_dataset.parquet"
    if not path.exists():
        path = data_dir() / "final_ml_dataset.csv"
    data = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    data["date"] = pd.to_datetime(data["date"])
    return data.sort_values("date").reset_index(drop=True)


def load_lsi_scores() -> pd.DataFrame:
    """Грузит сохранённые production LSI scores."""
    path = data_dir() / "lsi_scores.parquet"
    data = pd.read_parquet(path) if path.exists() else pd.read_csv(data_dir() / "lsi_scores.csv")
    data["date"] = pd.to_datetime(data["date"])
    return data.sort_values("date").reset_index(drop=True)


def load_backtest_scores() -> pd.DataFrame:
    """Грузит backtest scores (point-in-time expanding/rolling)."""
    path = data_dir() / "lsi_backtest_scores.csv"
    data = pd.read_csv(path)
    data["date"] = pd.to_datetime(data["date"])
    return data.sort_values("date").reset_index(drop=True)


def load_threshold_calibration() -> pd.DataFrame:
    """Грузит таблицу калибровки порогов (если есть)."""
    path = data_dir() / "lsi_threshold_calibration.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def load_model_artifact(kind: str = "global") -> dict[str, Any]:
    """Грузит сохранённый production-артефакт (joblib). Только для сравнения.

    ВНИМАНИЕ: артефакт может быть сохранён другой версией sklearn — для чистых
    экспериментов предпочитайте fit_lsi_like_model() (переобучение in-memory).
    """
    import joblib

    name = "lsi_global_pipeline.joblib" if kind == "global" else "lsi_local_pipeline.joblib"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return joblib.load(models_dir() / name)


# ----------------------------------------------------------------------------
# Whitelist и группировка по модулям
# ----------------------------------------------------------------------------
def get_lsi_whitelist() -> list[str]:
    """Возвращает production-whitelist LSI (26 фич) из backend."""
    _ensure_backend_on_path()
    from backend.src.services.lsi_training_service import LSI_FEATURE_CANDIDATES

    return list(LSI_FEATURE_CANDIDATES)


def available_whitelist(data: pd.DataFrame) -> list[str]:
    """Whitelist-фичи, фактически присутствующие в data."""
    return [c for c in get_lsi_whitelist() if c in data.columns]


def split_features_by_module(features: Iterable[str]) -> dict[str, list[str]]:
    """Группирует фичи по префиксу m1..m5."""
    out: dict[str, list[str]] = {}
    for f in features:
        prefix = f.split("_", 1)[0].lower()
        out.setdefault(prefix, []).append(f)
    return out


# ----------------------------------------------------------------------------
# Сводки по признакам
# ----------------------------------------------------------------------------
def summarize_features(data: pd.DataFrame, features: Iterable[str]) -> pd.DataFrame:
    """Таблица: dtype, n_unique, mean, std, min, max, zero_rate, null_rate."""
    rows = []
    n = len(data)
    for f in features:
        if f not in data.columns:
            rows.append({"feature": f, "present": False})
            continue
        s = pd.to_numeric(data[f], errors="coerce")
        rows.append(
            {
                "feature": f,
                "present": True,
                "module": f.split("_", 1)[0],
                "n_unique": int(s.nunique(dropna=True)),
                "mean": float(s.mean()),
                "std": float(s.std()),
                "min": float(s.min()),
                "max": float(s.max()),
                "zero_rate": float((s == 0).mean()),
                "null_rate": float(s.isna().mean()),
            }
        )
    return pd.DataFrame(rows)


def compute_zero_null_stats(data: pd.DataFrame, features: Iterable[str]) -> pd.DataFrame:
    """Узкая таблица zero_rate / null_rate / std для поиска мёртвых фич."""
    summ = summarize_features(data, features)
    cols = [c for c in ["feature", "module", "std", "n_unique", "zero_rate", "null_rate"] if c in summ.columns]
    return summ[cols].sort_values("zero_rate", ascending=False).reset_index(drop=True)


def find_dead_or_constant(data: pd.DataFrame, features: Iterable[str], *, zero_thresh: float = 0.999) -> pd.DataFrame:
    """Помечает фичи как constant (std~0) или near-dead (zero_rate высокий)."""
    summ = summarize_features(data, features)
    summ = summ[summ.get("present", True)].copy()
    summ["is_constant"] = summ["std"].fillna(0).abs() < 1e-12
    summ["is_near_dead"] = summ["zero_rate"] >= zero_thresh
    return summ[["feature", "std", "zero_rate", "n_unique", "is_constant", "is_near_dead"]]


# ----------------------------------------------------------------------------
# Корреляции (nan-safe)
# ----------------------------------------------------------------------------
def spearman(a, b) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 5:
        return float("nan")
    return float(stats.spearmanr(a[m], b[m]).correlation)


def pearson(a, b) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 5:
        return float("nan")
    return float(stats.pearsonr(a[m], b[m])[0])


# ----------------------------------------------------------------------------
# EMA и LSI-подобная модель (переобучение in-memory, faithful к production)
# ----------------------------------------------------------------------------
def compute_ema(values, alpha: float = DEFAULT_EMA_ALPHA) -> np.ndarray:
    """EMA как в production: ewm(alpha, adjust=False).mean()."""
    return pd.Series(np.asarray(values, dtype=float)).ewm(alpha=alpha, adjust=False).mean().to_numpy()


def fit_lsi_like_model(
    data: pd.DataFrame,
    features: list[str],
    *,
    use_pca: bool = True,
    n_pca: int = DEFAULT_PCA_COMPONENTS,
    ema_alpha: float = DEFAULT_EMA_ALPHA,
    contamination: float = DEFAULT_CONTAMINATION,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> dict[str, Any]:
    """Обучает LSI-подобную модель тем же пайплайном, что production, с ручками.

    Пайплайн: StandardScaler -> [PCA] -> IsolationForest -> -decision_function
              -> EMA(alpha) -> MinMaxScaler(0,100) -> clip(0,100).

    Возвращает dict-артефакт со scaler/pca/iso/minmax и массивами lsi/raw/smoothed,
    а также scaled_matrix (для объяснимости). Не пишет ничего на диск.
    """
    feats = [f for f in features if f in data.columns]
    X = data[feats].astype(float).fillna(0).to_numpy()

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    pca = None
    if use_pca:
        nc = min(n_pca, len(feats), len(data))
        pca = PCA(n_components=nc)
        mat = pca.fit_transform(Xs)
    else:
        mat = Xs

    iso = IsolationForest(contamination=contamination, random_state=random_state)
    iso.fit(mat)
    raw = -iso.decision_function(mat)
    smoothed = compute_ema(raw, ema_alpha)

    minmax = MinMaxScaler(feature_range=(0, 100))
    lsi = minmax.fit_transform(smoothed.reshape(-1, 1)).flatten()
    lsi = np.clip(lsi, 0, 100)

    return {
        "features": feats,
        "scaler": scaler,
        "pca": pca,
        "iso": iso,
        "minmax": minmax,
        "use_pca": use_pca,
        "ema_alpha": ema_alpha,
        "scaled_matrix": Xs,
        "lsi": lsi,
        "raw": raw,
        "smoothed": smoothed,
        "date": data["date"].to_numpy() if "date" in data.columns else np.arange(len(data)),
    }


def score_lsi_like_model(data: pd.DataFrame, artifact: dict[str, Any]) -> np.ndarray:
    """Применяет уже обученный lab-артефакт к новым данным (transform-only)."""
    feats = artifact["features"]
    X = data[feats].astype(float).fillna(0).to_numpy()
    Xs = artifact["scaler"].transform(X)
    mat = artifact["pca"].transform(Xs) if artifact["use_pca"] and artifact["pca"] is not None else Xs
    raw = -artifact["iso"].decision_function(mat)
    smoothed = compute_ema(raw, artifact["ema_alpha"])
    lsi = artifact["minmax"].transform(smoothed.reshape(-1, 1)).flatten()
    return np.clip(lsi, 0, 100)


# ----------------------------------------------------------------------------
# Сравнение шкал и эпизоды
# ----------------------------------------------------------------------------
def compare_scores(a, b) -> dict[str, float]:
    """Spearman, Pearson, max|diff|, mean diff между двумя сериями LSI."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    m = np.isfinite(a) & np.isfinite(b)
    return {
        "spearman": spearman(a, b),
        "pearson": pearson(a, b),
        "max_abs_diff": float(np.nanmax(np.abs(a[m] - b[m]))) if m.any() else float("nan"),
        "mean_diff": float(np.nanmean(a[m] - b[m])) if m.any() else float("nan"),
        "n": int(m.sum()),
    }


def compute_episode_summary(
    dates,
    lsi,
    episodes: dict[str, tuple[str, str]] = STRESS_EPISODES,
    thresholds: dict[str, float] = DEFAULT_THRESHOLDS,
) -> pd.DataFrame:
    """Таблица по стресс-эпизодам: n, max, mean, доля выше YELLOW/RED."""
    s = pd.DataFrame({"date": pd.to_datetime(pd.Series(dates)), "lsi": np.asarray(lsi, dtype=float)})
    green = thresholds.get("green_max", 30.0)
    yellow = thresholds.get("yellow_max", 60.0)
    rows = []
    for name, (a, b) in episodes.items():
        seg = s[(s["date"] >= pd.Timestamp(a)) & (s["date"] <= pd.Timestamp(b))]["lsi"]
        if seg.empty:
            rows.append({"episode": name, "n": 0})
            continue
        rows.append(
            {
                "episode": name,
                "n": int(len(seg)),
                "max": round(float(seg.max()), 2),
                "mean": round(float(seg.mean()), 2),
                "n_yellow": int(((seg >= green) & (seg < yellow)).sum()),
                "n_red": int((seg >= yellow).sum()),
                "pct_red": round(float((seg >= yellow).mean() * 100), 1),
            }
        )
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Объяснимость
# ----------------------------------------------------------------------------
def module_contributions(scaled_matrix: np.ndarray, pca: PCA, features: list[str]) -> pd.DataFrame:
    """EVR-взвешенные вклады модулей M1-M5 (та же формула, что в backend)."""
    _ensure_backend_on_path()
    from backend.src.services.lsi_training_service import compute_module_contributions as _mc

    contribs = _mc(scaled_matrix, pca, features)
    return pd.DataFrame(contribs)


def structural_weights(pca: PCA) -> np.ndarray:
    """structural_weights[j] = sum_k evr[k]*|components[k,j]| (как в backend)."""
    return np.abs(pca.components_).T @ pca.explained_variance_ratio_


def pc1_drivers(scaled_row: np.ndarray, pca: PCA, features: list[str], top_n: int = 3) -> list[str]:
    """Top-N драйверов по ПЕРВОЙ компоненте PCA (как production top_drivers)."""
    contrib = np.abs(scaled_row * pca.components_[0])
    idx = np.argsort(contrib)[::-1][:top_n]
    return [features[i] for i in idx]


def evr_weighted_drivers(scaled_row: np.ndarray, pca: PCA, features: list[str], top_n: int = 3) -> list[str]:
    """Top-N драйверов по EVR-взвешенным structural_weights (как attribution)."""
    contrib = np.abs(scaled_row) * structural_weights(pca)
    idx = np.argsort(contrib)[::-1][:top_n]
    return [features[i] for i in idx]


def dominant_module(contrib_row: pd.Series) -> str:
    """Имя модуля с максимальным вкладом в строке attribution."""
    return str(contrib_row.idxmax()).lower()


def driver_agreement_rate(artifact: dict[str, Any]) -> dict[str, float]:
    """Доля строк, где доминирующий модуль по PC1 совпадает с EVR-attribution."""
    pca = artifact["pca"]
    feats = artifact["features"]
    Xs = artifact["scaled_matrix"]
    if pca is None:
        return {"agreement": float("nan"), "note": "no PCA"}
    mc = module_contributions(Xs, pca, feats)
    evr_dom = mc.idxmax(axis=1).str.lower().to_numpy()
    pc1 = pca.components_[0]
    pc1_dom = np.array([feats[int(np.argmax(np.abs(r * pc1)))].split("_", 1)[0] for r in Xs])
    return {
        "agreement": float((evr_dom == pc1_dom).mean()),
        "pc1_variance_share": float(pca.explained_variance_ratio_[0]),
        "n": int(len(Xs)),
    }


# ----------------------------------------------------------------------------
# Local: RUONIA/keyrate proxy и proxy-таргеты
# ----------------------------------------------------------------------------
def build_ruonia_keyrate_proxy() -> pd.DataFrame:
    """Строит ДНЕВНОЙ proxy стоимости фондирования: spread = ruonia - key_rate.

    Источники: data/processed/ruonia.csv (дневной), data/processed/keyrate.csv.
    Обе ставки forward-fill на дневной календарь. Возвращает df с колонками
    date, ruonia_rate, key_rate, spread.
    """
    dd = data_dir()
    r = pd.read_csv(dd / "ruonia.csv")
    r["date"] = pd.to_datetime(r["date"], dayfirst=True, format="mixed")
    r = r[["date", "ruonia_rate"]].sort_values("date")

    k = pd.read_csv(dd / "keyrate.csv")
    date_col = k.columns[0]
    rate_col = [c for c in k.columns if c != date_col][0]
    k["date"] = pd.to_datetime(k[date_col], dayfirst=True, format="mixed")
    k = k[["date", rate_col]].rename(columns={rate_col: "key_rate"}).sort_values("date")

    cal = pd.DataFrame({"date": pd.date_range(r["date"].min(), r["date"].max(), freq="D")})
    m = cal.merge(r, on="date", how="left").merge(k, on="date", how="left")
    m["ruonia_rate"] = m["ruonia_rate"].ffill()
    m["key_rate"] = m["key_rate"].ffill()
    m["spread"] = m["ruonia_rate"] - m["key_rate"]
    return m


def add_forward_targets(
    proxy: pd.DataFrame, *, horizons: Iterable[int] = (1, 7), spread_col: str = "spread"
) -> pd.DataFrame:
    """Добавляет forward level и forward change по заданным горизонтам."""
    out = proxy.copy()
    for h in horizons:
        out[f"fwd_level_{h}"] = out[spread_col].shift(-h)
        out[f"fwd_change_{h}"] = out[spread_col].shift(-h) - out[spread_col]
    return out


def attach_feature_to_proxy(proxy: pd.DataFrame, data: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Прикрепляет (ffill) выбранные фичи из final_ml_dataset к дневному proxy."""
    cols = ["date"] + [f for f in features if f in data.columns]
    merged = proxy.merge(data[cols], on="date", how="left")
    for f in features:
        if f in merged.columns:
            merged[f] = merged[f].ffill()
    return merged


# ----------------------------------------------------------------------------
# Графики (matplotlib; seaborn опционально)
# ----------------------------------------------------------------------------
def _mpl():
    import matplotlib.pyplot as plt

    return plt


def plot_lsi_series(
    dates,
    series: dict[str, np.ndarray],
    *,
    episodes: dict[str, tuple[str, str]] | None = None,
    thresholds: dict[str, float] | None = None,
    title: str = "LSI series",
    figsize=(13, 4.5),
):
    """Рисует одну или несколько LSI-серий с порогами и подсветкой эпизодов."""
    plt = _mpl()
    dates = pd.to_datetime(pd.Series(dates))
    fig, ax = plt.subplots(figsize=figsize)
    for name, vals in series.items():
        ax.plot(dates, np.asarray(vals, dtype=float), lw=1.1, label=name)
    if thresholds:
        if "green_max" in thresholds:
            ax.axhline(thresholds["green_max"], color="green", ls="--", lw=0.8, alpha=0.6)
        if "yellow_max" in thresholds:
            ax.axhline(thresholds["yellow_max"], color="red", ls="--", lw=0.8, alpha=0.6)
    if episodes:
        for nm, (a, b) in episodes.items():
            ax.axvspan(pd.Timestamp(a), pd.Timestamp(b), color="orange", alpha=0.12)
    ax.set_title(title)
    ax.set_ylabel("LSI 0-100")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    return fig, ax


FEATURE_DESCRIPTIONS: dict[str, str] = {
    "m1_spread_mad_score": (
        "Абсолютный избыток резервов (MAD-score). Измеряет отклонение спреда резервов "
        "(фактические остатки минус норматив ЦБ, в рублях) от скользящей медианы за 3 года. "
        "Рост в плюс означает аномальное накопление («хординг») денег банками (сигнал стресса)."
    ),
    "m1_spread_relative_mad_score": (
        "Относительный избыток резервов (MAD-score). Оценивает отклонение спреда резервов, "
        "выраженного в процентах от требуемых резервов ЦБ. "
        "Учитывает изменение масштаба банковского сектора (обеспечивает сопоставимость периодов 2014 и 2024 гг.)."
    ),
    "m1_spread_delta_mad_score": (
        "Скорость изменения резервов (MAD-score). Оценивает периодные изменения спреда "
        "(spread_t - spread_t-1). Резкие положительные скачки сигнализируют о "
        "внезапной панике и импульсном изъятии банками ликвидности с рынка."
    ),
    "m1_reserve_load_mad_score": (
        "Коэффициент резервной нагрузки (MAD-score). Доля полного норматива обязательных "
        "и учетных резервов относительно фактических остатков банков на корсчетах. "
        "Высокое значение указывает на сильную «связанность» средств и уязвимость расчетов к шокам."
    ),
    # --- M2: REPO-аукционы ЦБ (sparse, event-driven) ---
    "m2_auction_flag": (
        "Флаг дня REPO-аукциона ЦБ (бинарный). 1 = в этот день проходил аукцион РЕПО. "
        "Большинство дней = 0; это маркер события, а не уровня стресса."
    ),
    "m2_Flag_Demand": (
        "Флаг повышенного спроса на REPO-аукционе (бинарный, sparse). "
        "Отмечает дни, когда спрос банков на рефинансирование был аномально высок."
    ),
    "m2_MAD_score_cover": (
        "MAD-score коэффициента покрытия REPO-аукциона (bid-to-cover). Ненулевой только в дни "
        "аукционов. Высокое значение = ажиотажный спрос на ликвидность ЦБ (сигнал стресса)."
    ),
    "m2_MAD_score_rate_spread": (
        "MAD-score спреда ставки отсечения над минимальной на REPO-аукционе. Ненулевой в дни "
        "аукционов. Рост = банки готовы платить дороже за рефинансирование (стресс)."
    ),
    # --- M3: ОФЗ-аукционы Минфина (sparse, event-driven) ---
    "m3_auction_flag": (
        "Флаг дня аукциона ОФЗ (бинарный). 1 = проходил аукцион размещения ОФЗ. "
        "Маркер события; нули = отсутствие аукциона, а не отсутствие стресса."
    ),
    "m3_cover_stress_score": (
        "Стресс по спросу на ОФЗ — ИНВЕРСИЯ cover MAD (= -m3_MAD_score_cover). Низкий спрос на "
        "размещении трактуется как стресс, поэтому знак перевёрнут. Высокое значение = слабый спрос."
    ),
    "m3_yield_stress_score": (
        "Стресс по доходности на аукционе ОФЗ (MAD-score). Высокое значение = Минфин вынужден давать "
        "премию к доходности, чтобы разместить выпуск (признак напряжения на рынке госдолга)."
    ),
    "m3_Flag_Nedospros": (
        "Флаг «недоспроса» на аукционе ОФЗ (бинарный, sparse). Спрос ниже предложения — слабый рынок."
    ),
    "m3_Flag_Perespros": (
        "Флаг «переспроса» на аукционе ОФЗ (бинарный, очень sparse). Спрос значительно выше "
        "предложения. В коротком окне может быть почти мёртвым (мало единиц)."
    ),
    # --- M4: налоговый календарь (детерминированный, не рыночный) ---
    "m4_Tax_Week_Flag": (
        "Флаг налоговой недели (бинарный). 1 = идёт период налоговых выплат. Детерминированный "
        "календарный контекст, НЕ объём изъятия ликвидности. Может быть очень частым."
    ),
    "m4_Tax_Day_Strict": (
        "Строгий флаг налогового дня (бинарный). 1 = день (±1) ключевого налогового платежа. "
        "Уже, чем Tax_Week_Flag; календарный, а не рыночный сигнал."
    ),
    "m4_MAD_tax_pressure": (
        "MAD-score налогового давления. Сглаженная интенсивность налоговых событий "
        "(квартальный/годовой налог на прибыль усилен весами). Календарная конструкция."
    ),
    "m4_MAD_tax_proximity": (
        "MAD-score близости к налоговому дню. Экспоненциальная близость к ближайшему платежу "
        "(до/после). Непрерывный календарный контекст, повторяется каждый месяц."
    ),
    "m4_Seasonal_Factor_raw": (
        "Сырой сезонный фактор [1.0..1.4]. Мультипликатор по налоговой неделе / концу квартала / "
        "концу года. Детерминированный, ограниченный сверху; почти не имеет хвостов."
    ),
    # --- M5: ликвидность ЦБ и Казначейство (daily structural + Roskazna post-2021) ---
    "m5_cbr_liquidity_stress_mad_score": (
        "MAD-score стресса ликвидности банковского сектора (ЦБ). Низкая структурная ликвидность "
        "трактуется как стресс (stress_when_lower=True). Ключевой дневной структурный признак."
    ),
    "m5_cbr_liquidity_drain_mad_score": (
        "MAD-score оттока (дренажа) ликвидности из сектора. Высокое значение = быстрое сокращение "
        "ликвидности (изъятие средств), потенциальный стресс."
    ),
    "m5_roskazna_net_flow_stress_mad_score": (
        "MAD-score стресса чистого потока Казначейства (Roskazna). Данные доступны в основном "
        "ПОСЛЕ 2021 г.; до 2021 ряд преимущественно нулевой/пустой — учитывать при интерпретации."
    ),
    "m5_Flag_Budget_Drain": (
        "Флаг бюджетного изъятия ликвидности (бинарный, sparse). 1 = крупный чистый отток средств "
        "по бюджетному каналу (порог -300 млрд руб)."
    ),
}


def plot_feature_distribution(data: pd.DataFrame, feature: str, *, bins: int = 60, figsize=(11, 3.5)):
    """Гистограмма + временной ряд одного признака."""
    plt = _mpl()
    s = pd.to_numeric(data[feature], errors="coerce")
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    axes[0].hist(s.dropna(), bins=bins, color="steelblue", alpha=0.8)
    axes[0].set_title(f"{feature} — hist")
    axes[0].grid(alpha=0.2)
    if "date" in data.columns:
        axes[1].plot(pd.to_datetime(data["date"]), s, lw=0.8, color="darkorange")
    else:
        axes[1].plot(s.values, lw=0.8, color="darkorange")
    axes[1].set_title(f"{feature} — over time")
    axes[1].grid(alpha=0.2)
    
    desc = FEATURE_DESCRIPTIONS.get(feature)
    if desc:
        fig.tight_layout()
        plt.figtext(
            0.5, -0.15, desc,
            wrap=True,
            horizontalalignment="center",
            fontsize=9,
            style="italic",
            color="dimgray"
        )
        plt.subplots_adjust(bottom=0.25)
    else:
        fig.tight_layout()
        
    return fig, axes


def correlation_heatmap(data: pd.DataFrame, features: list[str], *, method: str = "spearman", figsize=(11, 9)):
    """Корреляционная матрица выбранных фич (seaborn если есть, иначе matplotlib)."""
    plt = _mpl()
    feats = [f for f in features if f in data.columns]
    corr = data[feats].astype(float).corr(method=method)
    fig, ax = plt.subplots(figsize=figsize)
    try:
        import seaborn as sns

        sns.heatmap(corr, ax=ax, cmap="coolwarm", center=0, square=False,
                    cbar_kws={"shrink": 0.6}, xticklabels=True, yticklabels=True)
    except Exception:
        im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
        ax.set_xticks(range(len(feats)))
        ax.set_xticklabels(feats, rotation=90, fontsize=6)
        ax.set_yticks(range(len(feats)))
        ax.set_yticklabels(feats, fontsize=6)
        fig.colorbar(im, ax=ax, shrink=0.6)
    ax.set_title(f"{method.title()} correlation — LSI whitelist")
    fig.tight_layout()
    return fig, ax, corr


def savefig(fig, name: str) -> Path:
    """Сохраняет график в lab/outputs/ (не в git по умолчанию)."""
    out = project_root() / "lab" / "outputs"
    out.mkdir(parents=True, exist_ok=True)
    path = out / name
    fig.savefig(path, dpi=110, bbox_inches="tight")
    return path


# ----------------------------------------------------------------------------
# Поблочный разбор распределений (используется в 01_feature_distributions)
# ----------------------------------------------------------------------------
def summarize_features_extended(data: pd.DataFrame, features: Iterable[str]) -> pd.DataFrame:
    """Расширенная сводка с перцентилями (p01/p05/p50/p95/p99) по списку фич.

    Колонки: module, feature, null_rate, zero_rate, n_unique, mean, std,
    min, p01, p05, p50, p95, p99, max. Удобно для поблочного просмотра модуля.
    """
    rows = []
    for f in features:
        if f not in data.columns:
            rows.append({"feature": f, "present": False})
            continue
        s = pd.to_numeric(data[f], errors="coerce")
        clean = s.dropna()
        q = clean.quantile([0.01, 0.05, 0.50, 0.95, 0.99]) if len(clean) else pd.Series()
        rows.append(
            {
                "module": f.split("_", 1)[0],
                "feature": f,
                "null_rate": round(float(s.isna().mean()), 4),
                "zero_rate": round(float((s == 0).mean()), 4),
                "n_unique": int(s.nunique(dropna=True)),
                "mean": round(float(s.mean()), 4),
                "std": round(float(s.std()), 4),
                "min": round(float(s.min()), 4),
                "p01": round(float(q.get(0.01, np.nan)), 4),
                "p05": round(float(q.get(0.05, np.nan)), 4),
                "p50": round(float(q.get(0.50, np.nan)), 4),
                "p95": round(float(q.get(0.95, np.nan)), 4),
                "p99": round(float(q.get(0.99, np.nan)), 4),
                "max": round(float(s.max()), 4),
            }
        )
    return pd.DataFrame(rows)


def classify_feature_type(data: pd.DataFrame, features: Iterable[str], *, sparse_zero: float = 0.8) -> pd.DataFrame:
    """Эвристически помечает тип фичи: continuous / binary / sparse_event /
    calendar / dead_or_constant. Для ручного просмотра, не строгая типизация.

    Правила (по порядку):
    - dead_or_constant: std ~ 0 или n_unique <= 1
    - binary: значения подмножество {0,1}
    - calendar: имя содержит flag/tax/seasonal/day/week (детерминированный календарь)
    - sparse_event: zero_rate >= sparse_zero (преимущественно нули = событие)
    - continuous: иначе
    """
    calendar_tokens = ("flag", "tax", "seasonal", "_day", "_week", "auction")
    rows = []
    for f in features:
        if f not in data.columns:
            rows.append({"feature": f, "feature_type": "absent"})
            continue
        s = pd.to_numeric(data[f], errors="coerce")
        clean = s.dropna()
        std = float(s.std()) if len(clean) else 0.0
        nuniq = int(s.nunique(dropna=True))
        zero_rate = float((s == 0).mean())
        vals = set(np.unique(clean.values)) if len(clean) else set()
        is_binary = vals.issubset({0.0, 1.0}) and nuniq <= 2
        name = f.lower()
        if std < 1e-12 or nuniq <= 1:
            ftype = "dead_or_constant"
        elif is_binary:
            ftype = "binary"
        elif any(tok in name for tok in calendar_tokens):
            ftype = "calendar"
        elif zero_rate >= sparse_zero:
            ftype = "sparse_event"
        else:
            ftype = "continuous"
        rows.append(
            {
                "module": f.split("_", 1)[0],
                "feature": f,
                "feature_type": ftype,
                "zero_rate": round(zero_rate, 4),
                "n_unique": nuniq,
                "std": round(std, 4),
            }
        )
    return pd.DataFrame(rows)


def plot_module_small_multiples(
    data: pd.DataFrame,
    features: list[str],
    *,
    mask: pd.Series | None = None,
    bins: int = 40,
    ncols: int = 3,
    title: str | None = None,
):
    """Small-multiples гистограммы для фич одного модуля (одна фигура на модуль).

    mask — опциональный булев фильтр строк (например, только auction days).
    """
    plt = _mpl()
    feats = [f for f in features if f in data.columns]
    if not feats:
        return None, None
    sub = data[mask] if mask is not None else data
    nrows = int(np.ceil(len(feats) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 2.8 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax, f in zip(axes, feats):
        s = pd.to_numeric(sub[f], errors="coerce").dropna()
        ax.hist(s, bins=bins, color="steelblue", alpha=0.85)
        ax.set_title(f, fontsize=9)
        ax.grid(alpha=0.2)
    for ax in axes[len(feats):]:
        ax.axis("off")
    if title:
        fig.suptitle(title, fontsize=12, y=1.02)
    fig.tight_layout()
    return fig, axes


def plot_nonzero_timeline(
    data: pd.DataFrame,
    features: list[str],
    *,
    episodes: dict[str, tuple[str, str]] | None = None,
    title: str = "Non-zero signal timeline",
    figsize=(13, 0.7),
):
    """Таймлайн ненулевых наблюдений по нескольким фичам (для sparse/event модулей).

    Каждая фича — отдельная строка-«дорожка»; точка ставится там, где значение != 0.
    Полезно увидеть, концентрируются ли события вокруг стресс-периодов.
    """
    plt = _mpl()
    feats = [f for f in features if f in data.columns]
    if not feats or "date" not in data.columns:
        return None, None
    dates = pd.to_datetime(data["date"])
    fig, ax = plt.subplots(figsize=(figsize[0], max(2.0, figsize[1] * len(feats) + 1)))
    for i, f in enumerate(feats):
        s = pd.to_numeric(data[f], errors="coerce").fillna(0)
        nz = s != 0
        ax.scatter(dates[nz], np.full(int(nz.sum()), i), s=6, marker="|", color="darkred", alpha=0.6)
    if episodes:
        for nm, (a, b) in episodes.items():
            ax.axvspan(pd.Timestamp(a), pd.Timestamp(b), color="orange", alpha=0.12)
    ax.set_yticks(range(len(feats)))
    ax.set_yticklabels(feats, fontsize=8)
    ax.set_title(title)
    ax.grid(alpha=0.15, axis="x")
    fig.tight_layout()
    return fig, ax


def bar_metric_by_module(
    summary: pd.DataFrame,
    metric: str,
    *,
    title: str | None = None,
    figsize=(13, 5),
):
    """Bar chart значения метрики (например zero_rate или std) по фичам,
    цвет — по модулю. Ожидает df с колонками feature, module, <metric>.
    """
    plt = _mpl()
    df = summary.dropna(subset=[metric]).copy()
    df = df.sort_values(["module", metric])
    modules = sorted(df["module"].unique())
    cmap = plt.get_cmap("tab10")
    color_map = {m: cmap(i % 10) for i, m in enumerate(modules)}
    colors = [color_map[m] for m in df["module"]]
    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(range(len(df)), df[metric].values, color=colors)
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(df["feature"], rotation=90, fontsize=7)
    ax.set_ylabel(metric)
    ax.set_title(title or f"{metric} by feature (color = module)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=color_map[m]) for m in modules]
    ax.legend(handles, modules, title="module", fontsize=8, ncol=len(modules))
    ax.grid(alpha=0.2, axis="y")
    fig.tight_layout()
    return fig, ax


# ----------------------------------------------------------------------------
# Сырые данные по модулям (используется в 00_data_inventory)
# ----------------------------------------------------------------------------
# Карта «модуль -> сырой/исходный датасет в data/processed». Это входные данные
# модулей M1-M5 ДО сборки фич — удобно глазами посмотреть исходные ряды.
RAW_MODULE_SOURCES: dict[str, str] = {
    "m1": "m1_dataset.csv",   # резервы банков + RUONIA (по периодам усреднения)
    "m2": "m2_dataset.csv",   # аукционы РЕПО ЦБ (event-driven, все срочности)
    "m3": "m3_dataset.csv",   # аукционы ОФЗ Минфина (event-driven)
    "m4": "m4_dataset.csv",   # налоговый календарь (дневной)
    "m5": "m5_dataset.csv",   # ликвидность ЦБ + Казначейство (дневной + Roskazna)
}


def load_raw_csv(
    filename: str,
    *,
    date_col: str | None = None,
    parse_dates: bool = True,
) -> tuple[pd.DataFrame, str]:
    """Загружает сырой CSV из data/processed и парсит дату (dayfirst).

    Возвращает (df, date_col). Если date_col не указан — берётся первая колонка,
    содержащая 'date' (учитывает auction_date / event_date), иначе первая колонка.
    """
    path = data_dir() / filename
    df = pd.read_csv(path)
    if date_col is None:
        date_col = next((c for c in df.columns if "date" in c.lower()), df.columns[0])
    if parse_dates and date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col], dayfirst=True, format="mixed", errors="coerce")
        df = df.sort_values(date_col).reset_index(drop=True)
    return df, date_col


def plot_raw_timeseries(
    df: pd.DataFrame,
    columns: Iterable[str],
    *,
    date_col: str = "date",
    ncols: int = 2,
    kind: str = "line",
    title: str | None = None,
    figsize_per: tuple[float, float] = (6.6, 2.6),
):
    """Small-multiples временных рядов сырых колонок (одна фигура на модуль).

    kind: 'line' (по умолчанию) или 'scatter' — для разреженных event-driven рядов
    (аукционы) часто нагляднее scatter. Нечисловые/отсутствующие колонки пропускаются.
    """
    plt = _mpl()
    cols = [c for c in columns if c in df.columns]
    if not cols:
        return None, None
    x = pd.to_datetime(df[date_col], errors="coerce")
    nrows = int(np.ceil(len(cols) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(figsize_per[0] * ncols, figsize_per[1] * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax, c in zip(axes, cols):
        y = pd.to_numeric(df[c], errors="coerce")
        if kind == "scatter":
            ax.scatter(x, y, s=7, alpha=0.5, color="steelblue")
        else:
            ax.plot(x, y, lw=0.8, color="steelblue")
        ax.set_title(c, fontsize=9)
        ax.grid(alpha=0.2)
        ax.tick_params(axis="x", labelsize=7)
    for ax in axes[len(cols):]:
        ax.axis("off")
    if title:
        fig.suptitle(title, fontsize=12, y=1.02)
    fig.tight_layout()
    return fig, axes


# ----------------------------------------------------------------------------
# M2 term-aware артефакты (Phase A) — для 08_m2_term_structure
# ----------------------------------------------------------------------------
def load_m2_features() -> pd.DataFrame:
    """Per-auction M2 features (с tier и tier-MAD). Дата -> datetime."""
    p = data_dir() / "m2_features.parquet"
    df = pd.read_parquet(p) if p.exists() else pd.read_csv(data_dir() / "m2_features.csv")
    df["date"] = pd.to_datetime(df["date"], dayfirst=True, format="mixed", errors="coerce")
    return df.sort_values("date").reset_index(drop=True)


def load_m2_daily_profile() -> pd.DataFrame:
    """Дневной term-профиль M2 (m2_daily_profile). Дата -> datetime."""
    p = data_dir() / "m2_daily_profile.parquet"
    df = pd.read_parquet(p) if p.exists() else pd.read_csv(data_dir() / "m2_daily_profile.csv")
    df["date"] = pd.to_datetime(df["date"], dayfirst=True, format="mixed", errors="coerce")
    return df.sort_values("date").reset_index(drop=True)
