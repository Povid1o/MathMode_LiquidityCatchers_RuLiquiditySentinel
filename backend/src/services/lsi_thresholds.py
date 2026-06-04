"""Конфигурация пороговых профилей светофора LSI.

Единое место для всех порогов: backend inference, training, backtest, dashboard.
Никаких магических констант в других модулях — только импорт отсюда.

Два профиля:
- backtest_sensitive (30 / 60): production default.
  Выбран по бизнес-логике раннего предупреждения: ошибка пропуска стрессового
  эпизода дороже, чем ложная тревога. Event Recall Red = 100% по Global.
  Высокий false alert rate (~10.5% global, ~47% local) является осознанной ценой
  чувствительности и требует ручной фильтрации аналитиком.

- conservative (40 / 70): альтернативный профиль с меньшим числом ложных тревог.
  FP rate ~3.95% global. Декабрь 2014 не попадает в красную зону, но даёт
  19 жёлтых дней. Применять при необходимости снизить нагрузку на аналитика.
"""

from __future__ import annotations

# Профиль по умолчанию для production inference и dashboard.
# Выбор подтверждён куратором: приоритет раннего предупреждения.
DEFAULT_THRESHOLD_PROFILE = "backtest_sensitive"

# FP-rates взяты из data/processed/lsi_threshold_calibration.csv
# (non_stress_fp строки, global_fp_rate_pct / local_fp_rate_pct)
LSI_THRESHOLD_PROFILES: dict[str, dict[str, object]] = {
    "backtest_sensitive": {
        "green_max": 30.0,
        "yellow_max": 60.0,
        "description": (
            "Production default (выбор куратора). Зелёный < 30, жёлтый 30–60, красный ≥ 60. "
            "Event Recall Red = 100% по Global backtest: все размеченные стресс-эпизоды "
            "детектируются красным, включая Декабрь 2014 (1 красный день из 23). "
            "Global FP rate ≈ 10.5%, Local FP rate ≈ 47.1% на вне-стрессовой истории. "
            "Высокий false alert rate — осознанная цена чувствительности; "
            "требует ручного подтверждения аналитиком."
        ),
        "global_fp_rate_pct": 10.50,
        "local_fp_rate_pct": 47.13,
        "dec2014_red_days": 1,
        "dec2014_yellow_days": 23,
        "feb2022_red_days": 41,
    },
    "honest": {
        "green_max": 40.0,
        "yellow_max": 60.0,
        "description": (
            "Phase B honest LSI (percentile-anchored на honest Global). "
            "Зелёный < 40 (≈p80), жёлтый 40–60, красный ≥ 60 (≈p95). "
            "Распределение времени: GREEN ~80%, YELLOW ~15%, RED ~5%. "
            "Эпизоды на honest-шкале: Фев-мар 2022 — RED (острый кризис: пауза ОФЗ + "
            "рост кредитования ЦБ); Дек 2014 и Авг 2023 — YELLOW (многоканальный индекс "
            "оценивает их умереннее, т.к. в Дек 2014 нет данных ОФЗ). "
            "Пороги перекалиброваны под новый сбалансированный индекс (M1≈23, M2≈26, "
            "M3≈30, M5≈20, M4 — overlay вне PCA)."
        ),
        "global_pctl_yellow": 80,
        "global_pctl_red": 95,
    },
    "conservative": {
        "green_max": 40.0,
        "yellow_max": 70.0,
        "description": (
            "Альтернативный профиль с меньшим числом ложных тревог. "
            "Зелёный < 40, жёлтый 40–70, красный ≥ 70. "
            "Global FP rate ≈ 3.95% на вне-стрессовой истории 2014–2026. "
            "Декабрь 2014 не попадает в красную зону (max backtest LSI ≈ 63), "
            "но детектируется 19 жёлтых дней из 23. "
            "Применять, если требуется снизить нагрузку на аналитика."
        ),
        "global_fp_rate_pct": 3.95,
        "local_fp_rate_pct": 35.63,
        "dec2014_red_days": 0,
        "dec2014_yellow_days": 19,
        "feb2022_red_days": 41,
    },
}


def get_threshold_profile(profile: str = DEFAULT_THRESHOLD_PROFILE) -> dict[str, object]:
    """Возвращает конфигурацию порогового профиля по имени"""
    if profile not in LSI_THRESHOLD_PROFILES:
        available = ", ".join(sorted(LSI_THRESHOLD_PROFILES.keys()))
        raise ValueError(
            f"Пороговый профиль '{profile}' не найден. "
            f"Доступные профили: {available}"
        )
    return LSI_THRESHOLD_PROFILES[profile]


def get_lsi_status(lsi_value: float, profile: str = DEFAULT_THRESHOLD_PROFILE) -> str:
    """Возвращает статус светофора по шкале LSI 0-100 для заданного профиля"""
    config = get_threshold_profile(profile)
    green_max = float(config["green_max"])
    yellow_max = float(config["yellow_max"])
    if lsi_value < green_max:
        return "ЗЕЛЕНЫЙ (Норма)"
    if lsi_value < yellow_max:
        return "ЖЕЛТЫЙ (Повышенное внимание)"
    return "КРАСНЫЙ (Стресс ликвидности)"
