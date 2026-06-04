"""honest_lsi_backtest — Phase B: честный point-in-time backtest honest-LSI.

Для каждой даты `t` модель обучается ТОЛЬКО на данных до `t` (Global — expanding,
Local — rolling 365д), затем LSI считается на строке `t`. Это исключает look-ahead
(в отличие от in-sample full-history fit). Сверяем с in-sample honest-скорами.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from backend.src.services.honest_feature_builder import GLOBAL_WHITELIST, LOCAL_WHITELIST
from backend.src.services.honest_lsi_training import load_honest_dataset
from backend.src.services.lsi_training_service import (
    LOCAL_WINDOW_DAYS,
    MIN_LOCAL_ROWS,
    fit_lsi_artifact,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
HONEST_BACKTEST_CSV = DATA_DIR / "honest_lsi_backtest_scores.csv"

MIN_GLOBAL_ROWS = 120
BACKTEST_EPISODES = {
    "Декабрь 2014": ("2014-12-01", "2014-12-31"),
    "Февраль-март 2022": ("2022-02-01", "2022-03-31"),
    "Август 2023": ("2023-08-01", "2023-08-31"),
}


def _score_last(scoring_data: pd.DataFrame, artifact: dict[str, Any]) -> float:
    """LSI для последней строки с EMA на прошлой истории (point-in-time)."""
    feats = artifact["features_list"]
    X = scoring_data[feats].astype(float).fillna(0).to_numpy()
    scaled = artifact["scaler"].transform(X)
    pca_mat = artifact["pca"].transform(scaled)
    raw = -artifact["iso_forest"].decision_function(pca_mat)
    smoothed = pd.Series(raw).ewm(alpha=artifact["ema_alpha"], adjust=False).mean().to_numpy()
    lsi = artifact["minmax_scaler"].transform(smoothed.reshape(-1, 1)).flatten()
    return float(np.clip(lsi, 0, 100)[-1])


def _fit_and_score(train: pd.DataFrame, current: pd.DataFrame, *, kind: str,
                   feature_list: list[str], window_days: int | None = None) -> float:
    artifact, _ = fit_lsi_artifact(train.reset_index(drop=True), kind=kind,
                                   window_days=window_days, feature_list=feature_list)
    scoring = pd.concat([train, current], ignore_index=True)
    return _score_last(scoring, artifact)


def run_honest_backtest(data: pd.DataFrame | None = None,
                        episodes: dict[str, tuple[str, str]] = BACKTEST_EPISODES) -> pd.DataFrame:
    """Честный point-in-time backtest honest Global/Local по стресс-эпизодам."""
    if data is None:
        data = load_honest_dataset()
    data = data.copy()
    data["date"] = pd.to_datetime(data["date"])
    data = data.sort_values("date").reset_index(drop=True)

    rows: list[dict[str, object]] = []
    for episode, (start, end) in episodes.items():
        seg = data.loc[(data["date"] >= start) & (data["date"] <= end), "date"]
        for score_date in seg:
            current = data[data["date"] == score_date].copy()
            train_g = data[data["date"] < score_date].copy()
            if len(train_g) < MIN_GLOBAL_ROWS:
                continue
            g = _fit_and_score(train_g, current, kind="global", feature_list=GLOBAL_WHITELIST)

            local_start = score_date - pd.Timedelta(days=LOCAL_WINDOW_DAYS)
            train_l = data[(data["date"] < score_date) & (data["date"] >= local_start)].copy()
            local = None
            if len(train_l) >= MIN_LOCAL_ROWS:
                local = _fit_and_score(train_l, current, kind="local",
                                       feature_list=LOCAL_WHITELIST, window_days=LOCAL_WINDOW_DAYS)
            rows.append({"date": score_date.date().isoformat(), "episode": episode,
                         "lsi_global_pit": round(g, 2),
                         "lsi_local_pit": round(local, 2) if local is not None else None,
                         "global_train_rows": len(train_g), "local_train_rows": len(train_l)})
    return pd.DataFrame(rows)


def save_backtest(scores: pd.DataFrame, path: Path = HONEST_BACKTEST_CSV) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    scores.to_csv(path, index=False)


def main() -> None:
    scores = run_honest_backtest()
    save_backtest(scores)
    insample = pd.read_csv(DATA_DIR / "honest_lsi_scores.csv")
    insample["date"] = pd.to_datetime(insample["date"])
    print("Честный point-in-time backtest honest-LSI (Global):")
    for episode in BACKTEST_EPISODES:
        seg = scores[scores["episode"] == episode]
        if seg.empty:
            print(f"  {episode}: нет строк"); continue
        gmax = seg["lsi_global_pit"].max()
        s, e = BACKTEST_EPISODES[episode]
        ism = insample[(insample["date"] >= s) & (insample["date"] <= e)]["lsi_global"].max()
        print(f"  {episode:20s} PIT max={gmax:5.1f}  | in-sample max={ism:5.1f}")
    print(f"Сохранено: {HONEST_BACKTEST_CSV}")


if __name__ == "__main__":
    main()
