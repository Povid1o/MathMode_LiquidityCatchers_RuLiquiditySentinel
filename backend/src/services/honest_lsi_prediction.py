"""honest_lsi_prediction — Phase B: скоринг и объяснимость honest LSI.

- Скорит honest Global/Local артефактами (тот же пайплайн, что обучение).
- LSI_Index = Local.combine_first(Global) (как в production).
- Объяснимость: EVR-attribution драйверы (единый метод вместо PC1-only) + вклады
  модулей + декомпозиция по компонентам PCA.
- M4 — overlay: налоговый контекст отдаётся отдельным полем (НЕ влияет на PCA/LSI).
- Порог-профиль по умолчанию — 'honest'.

Порт валидированных лаб-функций (explain_lsi_point / explain_components_point).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from backend.src.services.honest_lsi_training import (
    HONEST_GLOBAL_MODEL,
    HONEST_LOCAL_MODEL,
    load_honest_dataset,
)
from backend.src.services.lsi_thresholds import get_lsi_status

DEFAULT_HONEST_PROFILE = "honest"
_MOD_NAMES = {"M1": "резервы/RUONIA", "M2": "РЕПО-аукционы", "M3": "ОФЗ-аукционы", "M5": "ликвидность ЦБ/ЕКС"}
M4_CONTEXT_COLUMNS = ["m4_Tax_Week_Flag", "m4_Tax_Day_Strict", "m4_MAD_tax_pressure"]


def load_honest_models() -> tuple[dict[str, Any], dict[str, Any]]:
    """Грузит honest Global/Local артефакты."""
    g = joblib.load(HONEST_GLOBAL_MODEL)
    lo = joblib.load(HONEST_LOCAL_MODEL)
    return g, lo


def _structural_weights(pca) -> np.ndarray:
    return np.abs(pca.components_).T @ pca.explained_variance_ratio_


def score_honest(data: pd.DataFrame, artifact: dict[str, Any]) -> dict[str, Any]:
    """Скорит данные honest-артефактом. Возвращает lsi, scaled_matrix и пр."""
    feats = artifact["features_list"]
    X = data[feats].astype(float).fillna(0).to_numpy()
    scaled = artifact["scaler"].transform(X)
    pca_mat = artifact["pca"].transform(scaled)
    raw = -artifact["iso_forest"].decision_function(pca_mat)
    smoothed = pd.Series(raw).ewm(alpha=artifact["ema_alpha"], adjust=False).mean().to_numpy()
    lsi = artifact["minmax_scaler"].transform(smoothed.reshape(-1, 1)).flatten()
    lsi = np.clip(lsi, 0, 100)
    return {"lsi": lsi, "scaled_matrix": scaled, "features": feats, "pca": artifact["pca"]}


def _row_index(data: pd.DataFrame, date) -> int:
    return int(np.argmin(np.abs(pd.to_datetime(data["date"]) - pd.Timestamp(date))))


def honest_drivers(scored: dict[str, Any], idx: int, top_n: int = 8) -> list[dict[str, Any]]:
    """Топ-драйверы (EVR-attribution) с z и направлением."""
    feats = scored["features"]
    scaled = scored["scaled_matrix"][idx]
    sw = _structural_weights(scored["pca"])
    contrib = np.abs(scaled) * sw
    pct = contrib / contrib.sum() * 100
    order = np.argsort(pct)[::-1][:top_n]
    return [
        {"feature": feats[j], "module": feats[j][:2].upper(), "contrib_pct": round(float(pct[j]), 1),
         "z_scaled": round(float(scaled[j]), 2),
         "direction": "выше нормы" if scaled[j] > 0 else "ниже нормы"}
        for j in order
    ]


def honest_module_contributions(scored: dict[str, Any], idx: int) -> dict[str, float]:
    """Вклад модулей M1/M2/M3/M5 в точке (M4 — overlay, не участвует)."""
    feats = scored["features"]
    scaled = scored["scaled_matrix"][idx]
    sw = _structural_weights(scored["pca"])
    contrib = np.abs(scaled) * sw
    pct = contrib / contrib.sum() * 100
    return {m: round(float(sum(pct[j] for j, f in enumerate(feats) if f[:2].upper() == m)), 1)
            for m in ["M1", "M2", "M3", "M5"]}


def honest_components(scored: dict[str, Any], idx: int, top_load: int = 4) -> list[dict[str, Any]]:
    """Декомпозиция по компонентам PCA: активный фактор + топ-loadings."""
    pca = scored["pca"]; feats = scored["features"]
    all_scores = pca.transform(scored["scaled_matrix"])
    row = all_scores[idx]
    out = []
    for k in range(min(3, len(row))):
        z = (row[k] - all_scores[:, k].mean()) / (all_scores[:, k].std() + 1e-9)
        load = pd.Series(pca.components_[k], index=feats)
        top = load.reindex(load.abs().sort_values(ascending=False).index).head(top_load)
        dom = pd.Series([f[:2].upper() for f in top.index]).mode().iloc[0]
        out.append({"component": f"PC{k+1}", "evr_pct": round(float(pca.explained_variance_ratio_[k] * 100), 1),
                    "activation_z": round(float(z), 2), "factor": _MOD_NAMES.get(dom, dom),
                    "top_loadings": [f"{f}({v:+.2f})" for f, v in top.items()]})
    return out


def tax_overlay(row: pd.Series) -> dict[str, Any]:
    """M4 overlay: налоговый контекст (метаданные, НЕ влияет на LSI)."""
    return {c.replace("m4_", ""): (float(row[c]) if c in row.index and pd.notna(row[c]) else None)
            for c in M4_CONTEXT_COLUMNS}


def get_honest_lsi_prediction(
    data: pd.DataFrame | None = None,
    *,
    date=None,
    profile: str = DEFAULT_HONEST_PROFILE,
) -> dict[str, Any]:
    """Полный honest-ответ для даты (по умолчанию — последняя)."""
    if data is None:
        data = load_honest_dataset()
    g_art, l_art = load_honest_models()
    g = score_honest(data, g_art)
    lo = score_honest(data, l_art)

    idx = len(data) - 1 if date is None else _row_index(data, date)
    row = data.iloc[idx]
    # Local доступен только на своём окне (train_start..); иначе берём Global
    local_ok = pd.Timestamp(l_art["train_start"]) <= pd.to_datetime(row["date"])
    if local_ok:
        lsi_index = float(lo["lsi"][idx]); kind, sc = "local", lo
    else:
        lsi_index = float(g["lsi"][idx]); kind, sc = "global", g

    return {
        "date": str(pd.to_datetime(row["date"]).date()),
        "LSI_Index": round(lsi_index, 2),
        "kind": kind,
        "status": get_lsi_status(lsi_index, profile=profile),
        "LSI_Global": round(float(g["lsi"][idx]), 2),
        "LSI_Local": round(float(lo["lsi"][idx]), 2) if local_ok else None,
        "module_contributions": honest_module_contributions(sc, idx),
        "top_drivers": honest_drivers(sc, idx),
        "components": honest_components(sc, idx),
        "tax_overlay": tax_overlay(row),
        "threshold_profile": profile,
    }


def main() -> None:
    """Пример honest-ответа на последнюю дату."""
    import json
    resp = get_honest_lsi_prediction()
    print(json.dumps(resp, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
