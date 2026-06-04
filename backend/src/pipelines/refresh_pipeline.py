"""refresh_pipeline — оркестратор полного обновления данных (point 2).

Запускает весь конвейер от загрузки сырья до пересчёта honest-LSI и наполнения
DuckDB warehouse, поэтому кнопка «Данные ⚙️» в дашборде может одним нажатием
дотянуть данные до последней доступной даты и обновить все графики.

Порядок шагов:
  m1 → m2 → m3 → m4 → m5   (download → parse → features; модули независимы)
  → final_ml_dataset       (сборка из feature-файлов)
  → honest LSI             (honest_ml_dataset + переобучение Global/Local)
  → warehouse sync         (загрузка processed-выходов в DuckDB)

Каждый шаг изолирован: падение одного источника (например, сетевой сбой ОФЗ)
не блокирует остальные — шаг помечается ошибкой, конвейер продолжается на том,
что уже лежит в processed.
"""
from __future__ import annotations

import io
import sys
import time
import traceback
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator


PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.src.db import warehouse as wh
from backend.src.pipelines.final_dataset_pipeline import run_final_dataset_pipeline
from backend.src.pipelines.honest_lsi_pipeline import run_honest_lsi_pipeline
from backend.src.pipelines.m1_pipeline import run_m1_pipeline
from backend.src.pipelines.m2_pipeline import run_m2_pipeline
from backend.src.pipelines.m3_pipeline import run_m3_pipeline
from backend.src.pipelines.m4_pipeline import run_m4_pipeline
from backend.src.pipelines.m5_pipeline import run_m5_pipeline


@dataclass
class StepResult:
    key: str
    label: str
    status: str = "pending"   # pending | running | ok | error
    seconds: float = 0.0
    log_tail: str = ""
    error: str = ""


@dataclass
class RefreshStep:
    key: str
    label: str
    func: Callable[[], None]


def _warehouse_sync() -> None:
    wh.sync_processed_to_warehouse(verbose=True)


def build_steps() -> list[RefreshStep]:
    """Шаги полного обновления в порядке выполнения."""
    return [
        RefreshStep("m1", "M1 — Резервы / RUONIA", run_m1_pipeline),
        RefreshStep("m2", "M2 — РЕПО-аукционы", run_m2_pipeline),
        RefreshStep("m3", "M3 — ОФЗ-аукционы", run_m3_pipeline),
        RefreshStep("m4", "M4 — Налоговый календарь", run_m4_pipeline),
        RefreshStep("m5", "M5 — Ликвидность ЦБ / ЕКС", run_m5_pipeline),
        RefreshStep("final", "Сборка final_ml_dataset", run_final_dataset_pipeline),
        RefreshStep("honest", "Honest LSI (датасет + модели)", run_honest_lsi_pipeline),
        RefreshStep("warehouse", "Синхронизация DuckDB warehouse", _warehouse_sync),
    ]


def _run_step(step: RefreshStep, *, log_lines: int = 40) -> StepResult:
    """Выполняет один шаг, перехватывая stdout/stderr и ошибки."""
    res = StepResult(key=step.key, label=step.label, status="running")
    buf = io.StringIO()
    t0 = time.monotonic()
    try:
        with redirect_stdout(buf), redirect_stderr(buf):
            step.func()
        res.status = "ok"
    except Exception as exc:  # noqa: BLE001 — изолируем падение источника
        res.status = "error"
        res.error = f"{type(exc).__name__}: {exc}"
        buf.write("\n" + traceback.format_exc())
    res.seconds = round(time.monotonic() - t0, 1)
    tail = buf.getvalue().strip().splitlines()
    res.log_tail = "\n".join(tail[-log_lines:])
    return res


def iter_full_refresh(steps: list[RefreshStep] | None = None) -> Iterator[StepResult]:
    """Генератор: выполняет шаги по очереди, отдавая результат каждого.

    Дашборд итерирует и обновляет UI между шагами. warehouse-sync выполняется
    всегда (последним), даже если выше были ошибки, чтобы подтянуть в БД то,
    что успешно пересчиталось.
    """
    steps = steps or build_steps()
    for step in steps:
        yield _run_step(step)


def run_full_refresh(steps: list[RefreshStep] | None = None) -> list[StepResult]:
    """Синхронный полный прогон (для CLI/скриптов). Возвращает все результаты."""
    results: list[StepResult] = []
    for res in iter_full_refresh(steps):
        mark = {"ok": "✓", "error": "✗"}.get(res.status, "·")
        print(f"  {mark} {res.label}: {res.status} ({res.seconds}s)"
              + (f" — {res.error}" if res.error else ""))
        results.append(res)
    return results


def main() -> None:
    print("Полное обновление данных RU Liquidity Sentinel")
    results = run_full_refresh()
    ok = sum(r.status == "ok" for r in results)
    print(f"\nИтог: {ok}/{len(results)} шагов успешно.")


if __name__ == "__main__":
    main()
