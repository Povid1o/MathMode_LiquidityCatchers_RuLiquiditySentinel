from pathlib import Path

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data" / "processed"

DATASETS = {
    "m1": DATA_DIR / "m1_features.parquet",
    "m2": DATA_DIR / "m2_features.parquet",
    "m3": DATA_DIR / "m3_features.parquet",
    "m4": DATA_DIR / "m4_features.parquet",
    "m5": DATA_DIR / "m5_features.parquet",
    "final": DATA_DIR / "final_ml_dataset.parquet",
}

MODULE_LABELS = {
    "m1": "M1 — Резервы",
    "m2": "M2 — Репо ЦБ",
    "m3": "M3 — ОФЗ",
    "m4": "M4 — Налоги",
    "m5": "M5 — Ликвидность",
}

COLORS = {
    "primary": "#1f77b4",
    "secondary": "#ff7f0e",
    "danger": "#d62728",
    "success": "#2ca02c",
    "neutral": "#7f7f7f",
    "warn": "#bcbd22",
    "bg_dark": "#0e1117",
    "stress_high": "rgba(214,39,40,0.15)",
    "stress_low": "rgba(44,160,44,0.15)",
}

PLOTLY_TEMPLATE = "plotly_dark"

MAD_STRESS_THRESHOLD = 2.0
