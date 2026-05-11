from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.src.services.lsi_backtest_service import build_backtest_report
from backend.src.services.lsi_backtest_service import run_backtest
from backend.src.services.lsi_backtest_service import save_backtest_outputs


def run_lsi_backtest_pipeline() -> None:
    """Запускает point-in-time backtest LSI по стресс-эпизодам"""
    print("Считаем rolling/expanding backtest LSI")
    scores, sensitivity = run_backtest()
    save_backtest_outputs(scores, sensitivity)
    build_backtest_report(scores, sensitivity)
    print(f"Готово, строк backtest: {len(scores)}")
    print(f"Готово, строк sensitivity: {len(sensitivity)}")


def main() -> None:
    """Запускает pipeline backtest LSI"""
    run_lsi_backtest_pipeline()


if __name__ == "__main__":
    main()
