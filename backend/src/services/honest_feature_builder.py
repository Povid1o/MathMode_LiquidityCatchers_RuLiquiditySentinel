"""honest_feature_builder — Phase B: сборка honest-фич M1–M5 для LSI.

Точный порт валидированного лаб-эталона lab/utils.build_honest_features:
из текущего final_ml_dataset + raw-источников считает все honest-фичи (m*x_) и
формирует honest_ml_dataset. M4 — overlay (вне PCA, в whitelist не входит).

Единый source of truth: эта логика обязана давать те же фичи, что лаб-эталон
(проверяется в тестах сверкой колонок).

Состав (зафиксирован в Phase A):
- M1: spread_mad, spread_relative_mad, reserve_load_mad, ruonia_mad, spread_vol(=|spread_delta_mad|)
- M2: auction_flag, Flag_Demand, base_cover_mad, cutoff_spread(+available), short_active30, days_since_short
- M3: auction_flag, Flag_Nedospros, ea_cover, ea_placement, ea_yield_to_key, age, available, days_since, failed
- M4: overlay (вне PCA)
- M5 (Global): claims, liabilities, repo_standing, secured_standing; (+Local) rk_bidders
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data" / "processed"

FINAL_DATASET_FILE = DATA_DIR / "final_ml_dataset.parquet"
HONEST_DATASET_CSV = DATA_DIR / "honest_ml_dataset.csv"
HONEST_DATASET_PARQUET = DATA_DIR / "honest_ml_dataset.parquet"

ROLLING_WINDOW_DAYS = 365 * 3
MAD_ROLLING_WIN = 756            # ~3 года торговых дней (как в лаб-эталоне)
MAD_MIN_PERIODS = 120
MAD_FLOOR = 0.05
MAD_CLIP = 5.0
CUTOFF_SPREAD_CAP_DAYS = 7
M3_DAYS_SINCE_CAP = 250
M3_AGE_CAP = 90
M3_AVAILABLE_AGE = 10
M2_SHORT_ACTIVE_DAYS = 30
M2_DAYS_SINCE_SHORT_CAP = 90

# --- whitelist'ы (kind-aware): M4 не входит (overlay) ---
M1_FEATURES = ["m1_spread_mad_score", "m1_spread_relative_mad_score",
               "m1_reserve_load_mad_score", "m1_ruonia_mad_score", "m1_spread_vol"]
M2_FEATURES = ["m2_auction_flag", "m2_Flag_Demand", "m2_base_cover_mad",
               "m2_cutoff_spread", "m2_cutoff_spread_available",
               "m2_short_active30", "m2_days_since_short"]
M3_FEATURES = ["m3_auction_flag", "m3_Flag_Nedospros", "m3x_cover", "m3x_placement",
               "m3x_yield_to_key", "m3x_age", "m3x_available", "m3x_days_since", "m3x_failed"]
M5_GLOBAL_FEATURES = ["m5x_claims", "m5x_liab", "m5x_repostd", "m5x_secured"]
M5_LOCAL_ONLY = ["m5x_rk_bidders"]

GLOBAL_WHITELIST = M1_FEATURES + M2_FEATURES + M3_FEATURES + M5_GLOBAL_FEATURES
LOCAL_WHITELIST = GLOBAL_WHITELIST + M5_LOCAL_ONLY


def _read_dated(path: Path, date_col: str = "date") -> pd.DataFrame:
    """Читает CSV с датой DD-MM-YYYY (или ISO), сортирует по дате."""
    df = pd.read_csv(path)
    df[date_col] = pd.to_datetime(df[date_col], dayfirst=True, format="mixed", errors="coerce")
    return df.sort_values(date_col).reset_index(drop=True)


def _mad_rolling(s: pd.Series, win: int = MAD_ROLLING_WIN) -> pd.Series:
    """MAD-score по скользящему окну (идентично лаб-эталону)."""
    s = pd.to_numeric(s, errors="coerce")
    med = s.rolling(win, min_periods=MAD_MIN_PERIODS).median()
    m = (s - med).abs().rolling(win, min_periods=MAD_MIN_PERIODS).median()
    return ((s - med) / m.clip(lower=MAD_FLOOR)).clip(-MAD_CLIP, MAD_CLIP)


def build_honest_dataset(final_dataset_path: Path = FINAL_DATASET_FILE) -> pd.DataFrame:
    """Собирает honest_ml_dataset: текущий final + honest-фичи M1/M2/M3/M5."""
    d = pd.read_parquet(final_dataset_path)
    d["date"] = pd.to_datetime(d["date"])
    d = d.sort_values("date").reset_index(drop=True)
    cal = d[["date"]].copy()

    # ---- M1: волатильность резервов = |MAD(spread_delta)| ----
    d["m1_spread_vol"] = pd.to_numeric(d["m1_spread_delta_mad_score"], errors="coerce").abs()

    # ---- M2: base_cover + cutoff_spread + short-события ----
    prof = pd.read_parquet(DATA_DIR / "m2_daily_profile.parquet") \
        if (DATA_DIR / "m2_daily_profile.parquet").exists() else _read_dated(DATA_DIR / "m2_daily_profile.csv")
    prof["date"] = pd.to_datetime(prof["date"], dayfirst=True, format="mixed", errors="coerce")
    d = d.merge(prof[["date", "m2_base_cover_mad", "m2_short_age_days"]], on="date", how="left")
    d["m2_short_active30"] = (d["m2_short_age_days"] <= M2_SHORT_ACTIVE_DAYS).astype(int)
    d["m2_days_since_short"] = np.minimum(d["m2_short_age_days"].fillna(365), M2_DAYS_SINCE_SHORT_CAP)

    f2 = _read_dated(DATA_DIR / "m2_features.csv")
    r = _read_dated(DATA_DIR / "ruonia.csv")
    fb = f2[f2["tier"] == "base"].copy()
    fb["cutoff_rate"] = pd.to_numeric(fb["cutoff_rate"], errors="coerce")
    fb = fb.dropna(subset=["cutoff_rate"]).merge(r[["date", "ruonia_rate"]], on="date", how="left")
    fb["cs"] = fb["cutoff_rate"] - fb["ruonia_rate"]
    fb = fb.dropna(subset=["cs"]).sort_values("date")[["date", "cs"]]
    cs = pd.merge_asof(cal.sort_values("date"), fb, on="date", direction="backward",
                       tolerance=pd.Timedelta(days=CUTOFF_SPREAD_CAP_DAYS))
    d["m2_cutoff_spread"] = cs["cs"].values
    d["m2_cutoff_spread_available"] = cs["cs"].notna().astype(int).values

    # ---- M3: event-aware cover/placement/yield_to_key + age/available/days_since/failed ----
    raw = _read_dated(DATA_DIR / "ofz_auctions.csv", date_col="auction_date").rename(columns={"auction_date": "date"})
    for c in ["offered_amount", "demand_amount", "placed_amount", "cutoff_yield"]:
        raw[c] = pd.to_numeric(raw[c], errors="coerce")

    def _agg(x: pd.DataFrame) -> pd.Series:
        off, dem, pla = x.offered_amount.sum(), x.demand_amount.sum(), x.placed_amount.sum()
        wy = (x.cutoff_yield * x.placed_amount).sum() / pla if pla > 0 else np.nan
        return pd.Series({"offered": off, "demand": dem, "placed": pla, "cutoff_y": wy})

    g = raw.groupby("date").apply(_agg).reset_index().sort_values("date").reset_index(drop=True)
    g["cover"] = g.demand / g.offered
    g["placement"] = g.placed / g.offered
    g["failed"] = (g.placed == 0).astype(int)
    k = _read_dated(DATA_DIR / "keyrate.csv")
    g = pd.merge_asof(g, k, on="date", direction="backward")
    g["yield_to_key"] = g.cutoff_y - g.key_rate

    def _mad_series(col: str) -> pd.DataFrame:
        s = g.dropna(subset=[col]).copy().sort_values("date")
        vals = s[col].values
        ds = s["date"].values
        out = []
        window = np.timedelta64(ROLLING_WINDOW_DAYS, "D")
        for i in range(len(s)):
            w = vals[(ds > ds[i] - window) & (ds <= ds[i])]
            med = np.median(w)
            mad = max(np.median(np.abs(w - med)), MAD_FLOOR)
            out.append((vals[i] - med) / mad)
        s["mad"] = out
        return s[["date", "mad"]]

    for name, col, sign in [("m3x_cover", "cover", -1), ("m3x_placement", "placement", -1),
                            ("m3x_yield_to_key", "yield_to_key", 1)]:
        ms = _mad_series(col)
        ms["mad"] = ms["mad"] * sign
        d[name] = pd.merge_asof(cal.sort_values("date"), ms.sort_values("date"),
                                on="date", direction="backward")["mad"].values

    af = d["m3_auction_flag"].fillna(0).values
    age = np.empty(len(d))
    last = -10 ** 9
    for i, v in enumerate(af):
        if v == 1:
            last = i
        age[i] = i - last if last > -10 ** 8 else 9999
    age = pd.Series(age)
    first = int(np.argmax(af == 1))
    dss = age.copy()
    dss[:first] = 0
    dss = dss.clip(0, M3_DAYS_SINCE_CAP)
    d["m3x_age"] = np.minimum(age.clip(lower=0), M3_AGE_CAP)
    d["m3x_available"] = (age.between(0, M3_AVAILABLE_AGE)).astype(int)
    d["m3x_days_since"] = dss.values
    fdays = set(g[g.failed == 1]["date"])
    d["m3x_failed"] = pd.to_datetime(d["date"]).isin(fdays).astype(int)

    # ---- M5: claims/liabilities/repo_standing/secured_standing (+rk_bidders) ----
    liq = _read_dated(DATA_DIR / "cbr_liquidity.csv")
    for c in liq.columns:
        if c != "date":
            liq[c] = pd.to_numeric(liq[c], errors="coerce")

    def _dly(src: pd.DataFrame, col: str) -> np.ndarray:
        t = src[["date", col]].copy()
        t["m"] = _mad_rolling(t[col])
        return pd.merge_asof(cal.sort_values("date"), t[["date", "m"]],
                             on="date", direction="backward")["m"].values

    d["m5x_claims"] = _dly(liq, "cbr_claims_standard_instruments_bln_rub")
    d["m5x_liab"] = _dly(liq, "cbr_liabilities_standard_instruments_bln_rub")
    d["m5x_repostd"] = _dly(liq, "repo_fx_swap_standing_bln_rub")
    d["m5x_secured"] = _dly(liq, "secured_loans_standing_bln_rub")
    rk = _read_dated(DATA_DIR / "roskazna_treasury_deposits.csv", date_col="auction_date").rename(columns={"auction_date": "date"})
    gb = rk.groupby("date")["bidders_count"].sum().reset_index()
    d["m5x_rk_bidders"] = _dly(gb, "bidders_count")

    new_cols = [c for c in d.columns if c.startswith(
        ("m1_spread_vol", "m2_base_cover", "m2_cutoff", "m2_short", "m2_days_since", "m3x_", "m5x_"))]
    d[new_cols] = d[new_cols].fillna(0)
    return d


def save_honest_dataset(d: pd.DataFrame, *, csv_path: Path = HONEST_DATASET_CSV,
                        parquet_path: Path = HONEST_DATASET_PARQUET) -> None:
    """Сохраняет honest_ml_dataset в CSV и parquet."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    out = d.copy()
    out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")
    out.to_csv(csv_path, index=False)
    out.to_parquet(parquet_path, index=False)


def main() -> None:
    """Собирает и сохраняет honest_ml_dataset."""
    d = build_honest_dataset()
    save_honest_dataset(d)
    missing = [c for c in LOCAL_WHITELIST if c not in d.columns]
    print(f"honest_ml_dataset: {len(d)} строк, {d.shape[1]} колонок")
    print(f"Global whitelist: {len(GLOBAL_WHITELIST)} фич | Local: {len(LOCAL_WHITELIST)} фич")
    print(f"Отсутствующих whitelist-колонок: {missing if missing else 'нет'}")
    print(f"Файлы: {HONEST_DATASET_CSV}, {HONEST_DATASET_PARQUET}")


if __name__ == "__main__":
    main()
