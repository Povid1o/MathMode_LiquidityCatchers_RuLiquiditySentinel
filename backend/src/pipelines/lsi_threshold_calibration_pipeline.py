from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.src.services.lsi_threshold_calibration_service import build_calibration_report
from backend.src.services.lsi_threshold_calibration_service import build_threshold_metrics
from backend.src.services.lsi_threshold_calibration_service import run_threshold_calibration
from backend.src.services.lsi_threshold_calibration_service import save_calibration_outputs


def run_lsi_threshold_calibration_pipeline() -> None:
    """Калибрует пороги светофора LSI и сохраняет результаты"""
    print("Запускаем калибровку порогов LSI")
    calibration = run_threshold_calibration()
    metrics = build_threshold_metrics()
    save_calibration_outputs(calibration, metrics)
    rec_green, rec_red, rec_reason = build_calibration_report(calibration, metrics)
    print(f"Готово, строк калибровки: {len(calibration)}")
    print(f"Готово, строк метрик: {len(metrics)}")
    print(f"Рекомендованные пороги: зелёный < {rec_green}, красный >= {rec_red}")
    print(f"Причина: {rec_reason}")


def main() -> None:
    """Запускает pipeline калибровки порогов LSI"""
    run_lsi_threshold_calibration_pipeline()


if __name__ == "__main__":
    main()
