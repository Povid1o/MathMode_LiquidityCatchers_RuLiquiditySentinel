"""warehouse — встроенный DuckDB как serving-слой (source of truth) для дашборда.

Архитектура (point 4):
- Pipelines (m1..m5 / final / honest) по-прежнему пишут processed parquet/csv —
  их контракт не ломаем (валидированный honest feature-build не трогаем).
- `sync_processed_to_warehouse()` загружает все processed-выходы в один DuckDB-файл
  `data/warehouse.duckdb` + ведёт таблицу свежести `_manifest`.
- Дашборд читает через `read_table()` (с fallback на parquet, если таблицы ещё нет).

Почему DuckDB, а не SQLite: колоночный аналитический движок, нативно читает/пишет
pandas и parquet, тянет широкий final_ml_dataset (100+ колонок) и временные ряды
без ORM. Один встроенный файл — нулевая инфраструктура.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import duckdb
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
WAREHOUSE_PATH = PROJECT_ROOT / "data" / "warehouse.duckdb"
MANIFEST_TABLE = "_manifest"

# Канонические таблицы warehouse: имя -> исходный processed-файл.
# parquet в приоритете; для источников без parquet берём csv.
PROCESSED_TABLES: dict[str, Path] = {
    # признаки модулей
    "m1_features": DATA_DIR / "m1_features.parquet",
    "m2_features": DATA_DIR / "m2_features.parquet",
    "m2_daily_profile": DATA_DIR / "m2_daily_profile.parquet",
    "m3_features": DATA_DIR / "m3_features.parquet",
    "m4_features": DATA_DIR / "m4_features.parquet",
    "m5_features": DATA_DIR / "m5_features.parquet",
    # сводные датасеты
    "final_ml_dataset": DATA_DIR / "final_ml_dataset.parquet",
    "honest_ml_dataset": DATA_DIR / "honest_ml_dataset.parquet",
    # скоринг и метрики
    "honest_lsi_scores": DATA_DIR / "honest_lsi_scores.parquet",
    "lsi_threshold_metrics": DATA_DIR / "lsi_threshold_metrics.csv",
    # распарсенные сырьевые источники (для будущего инкрементального апдейта)
    "ruonia": DATA_DIR / "ruonia.csv",
    "keyrate": DATA_DIR / "keyrate.csv",
    "ofz_auctions": DATA_DIR / "ofz_auctions.csv",
    "repo": DATA_DIR / "repo.csv",
    "cbr_liquidity": DATA_DIR / "cbr_liquidity.csv",
    "required_reserves": DATA_DIR / "required_reserves.csv",
    "roskazna_treasury_deposits": DATA_DIR / "roskazna_treasury_deposits.csv",
    "cbr_budget_funds": DATA_DIR / "cbr_budget_funds.csv",
}

# Возможные имена колонки-даты в разных источниках (для свежести).
_DATE_CANDIDATES = ("date", "dt", "auction_date", "published_date", "Date")


def connect(read_only: bool = False) -> duckdb.DuckDBPyConnection:
    """Открывает короткоживущее соединение к warehouse (вызывающий закрывает)."""
    WAREHOUSE_PATH.parent.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(str(WAREHOUSE_PATH), read_only=read_only)


def _detect_date_column(df: pd.DataFrame) -> str | None:
    for cand in _DATE_CANDIDATES:
        if cand in df.columns:
            return cand
    return None


def _to_datetime(series: pd.Series) -> pd.Series:
    """Канонический парсер дат (как dashboard.data.loader._parse_dates):
    сперва ISO %Y-%m-%d, затем %d-%m-%Y, иначе dayfirst-coerce.
    Нужен, т.к. processed-источники хранят дату строкой в разных форматах."""
    if pd.api.types.is_datetime64_any_dtype(series):
        return series
    for fmt in ("%Y-%m-%d", "%d-%m-%Y"):
        try:
            return pd.to_datetime(series, format=fmt, errors="raise")
        except Exception:
            continue
    return pd.to_datetime(series, dayfirst=True, errors="coerce")


def _read_source_file(path: Path) -> pd.DataFrame:
    """Читает processed-файл (parquet или csv)."""
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def has_table(name: str, conn: duckdb.DuckDBPyConnection | None = None) -> bool:
    """Есть ли таблица в warehouse."""
    own = conn is None
    conn = conn or connect(read_only=WAREHOUSE_PATH.exists())
    try:
        rows = conn.execute(
            "SELECT 1 FROM information_schema.tables WHERE table_name = ?", [name]
        ).fetchone()
        return rows is not None
    finally:
        if own:
            conn.close()


def list_tables() -> list[str]:
    """Список пользовательских таблиц warehouse (без служебных)."""
    if not WAREHOUSE_PATH.exists():
        return []
    conn = connect(read_only=True)
    try:
        names = [r[0] for r in conn.execute("SHOW TABLES").fetchall()]
    finally:
        conn.close()
    return [n for n in names if not n.startswith("_")]


def write_table(
    name: str,
    df: pd.DataFrame,
    *,
    source: str | None = None,
    conn: duckdb.DuckDBPyConnection | None = None,
) -> None:
    """Перезаписывает таблицу warehouse из DataFrame и обновляет manifest."""
    own = conn is None
    conn = conn or connect(read_only=False)
    try:
        conn.register("_incoming_df", df)
        conn.execute(f'CREATE OR REPLACE TABLE "{name}" AS SELECT * FROM _incoming_df')
        conn.unregister("_incoming_df")
        _update_manifest(conn, name, df, source)
    finally:
        if own:
            conn.close()


def read_table(name: str) -> pd.DataFrame:
    """Читает таблицу warehouse. С fallback на processed-файл, если таблицы нет."""
    if WAREHOUSE_PATH.exists() and has_table(name):
        conn = connect(read_only=True)
        try:
            return conn.execute(f'SELECT * FROM "{name}"').df()
        finally:
            conn.close()
    # fallback: warehouse ещё не наполнен — читаем исходный файл
    path = PROCESSED_TABLES.get(name)
    if path is None or not path.exists():
        raise KeyError(f"Таблица '{name}' отсутствует в warehouse и нет processed-файла")
    return _read_source_file(path)


def _ensure_manifest(conn: duckdb.DuckDBPyConnection) -> None:
    conn.execute(
        f"""CREATE TABLE IF NOT EXISTS {MANIFEST_TABLE} (
                table_name VARCHAR PRIMARY KEY,
                row_count  BIGINT,
                col_count  BIGINT,
                date_min   VARCHAR,
                date_max   VARCHAR,
                source     VARCHAR,
                updated_at TIMESTAMP
            )"""
    )


def _update_manifest(
    conn: duckdb.DuckDBPyConnection,
    name: str,
    df: pd.DataFrame,
    source: str | None,
) -> None:
    _ensure_manifest(conn)
    date_col = _detect_date_column(df)
    date_min = date_max = None
    if date_col is not None:
        dates = _to_datetime(df[date_col])
        if dates.notna().any():
            date_min = str(dates.min().date())
            date_max = str(dates.max().date())
    conn.execute(f"DELETE FROM {MANIFEST_TABLE} WHERE table_name = ?", [name])
    conn.execute(
        f"INSERT INTO {MANIFEST_TABLE} VALUES (?, ?, ?, ?, ?, ?, now())",
        [name, len(df), df.shape[1], date_min, date_max, source],
    )


def manifest() -> pd.DataFrame:
    """Таблица свежести warehouse: строки, даты, время обновления по каждой таблице."""
    if not WAREHOUSE_PATH.exists():
        return pd.DataFrame(
            columns=["table_name", "row_count", "col_count", "date_min", "date_max", "source", "updated_at"]
        )
    conn = connect(read_only=True)
    try:
        if not has_table(MANIFEST_TABLE, conn):
            return pd.DataFrame(
                columns=["table_name", "row_count", "col_count", "date_min", "date_max", "source", "updated_at"]
            )
        return conn.execute(
            f"SELECT * FROM {MANIFEST_TABLE} ORDER BY table_name"
        ).df()
    finally:
        conn.close()


def sync_processed_to_warehouse(
    tables: Iterable[str] | None = None,
    *,
    verbose: bool = True,
) -> pd.DataFrame:
    """Загружает processed-файлы в warehouse (одной транзакцией соединения).

    Возвращает manifest после синхронизации. Отсутствующие файлы пропускает.
    """
    names = list(tables) if tables is not None else list(PROCESSED_TABLES.keys())
    conn = connect(read_only=False)
    loaded: list[str] = []
    try:
        _ensure_manifest(conn)
        for name in names:
            path = PROCESSED_TABLES.get(name)
            if path is None or not path.exists():
                if verbose:
                    print(f"  · пропуск {name}: файл не найден ({path})")
                continue
            df = _read_source_file(path)
            write_table(name, df, source=path.name, conn=conn)
            loaded.append(name)
            if verbose:
                print(f"  ✓ {name}: {len(df)} строк, {df.shape[1]} колонок ← {path.name}")
    finally:
        conn.close()
    if verbose:
        print(f"Синхронизировано таблиц: {len(loaded)} → {WAREHOUSE_PATH.name}")
    return manifest()


def main() -> None:
    """Наполняет warehouse из текущих processed-файлов."""
    print(f"Инициализация warehouse: {WAREHOUSE_PATH}")
    mani = sync_processed_to_warehouse()
    print("\nManifest свежести:")
    if not mani.empty:
        print(mani.to_string(index=False))


if __name__ == "__main__":
    main()
