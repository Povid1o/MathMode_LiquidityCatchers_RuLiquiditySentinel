"""Оценка свежести таблиц витрины с поправкой на график публикации источника.

Разница «сегодня минус последняя дата» сама по себе ничего не значит: месячный
ряд обязательных резервов с отставанием 50 дней абсолютно здоров, а дневная
ликвидность ЦБ с отставанием 5 дней — уже нет. Пока метрика была одна для всех,
месячные ряды постоянно горели красным и заглушали настоящие сбои: именно так
двухмесячный застой депозитов Росказны потерялся среди «нормальных» 50–70 дней.

Здесь для каждой таблицы задан ожидаемый лаг публикации, и статус считается
относительно него.

Ожидаемые значения выведены из наблюдённого поведения источников:
- ЦБ, дневная ликвидность (bliquidity) — публикуется на следующий рабочий день;
- ЦБ, обязательные резервы — месячный ряд, значение появляется на 16-й рабочий
  день месяца, следующего за отчётным (сноска 4 в исходном xlsx);
- ЦБ, бюджетные средства на счетах банков — месячный ряд с задержкой ~2 месяца;
- Минфин, ОФЗ-аукционы — один накопительный файл на год, обновляется
  нерегулярно (наблюдались версии 07.05, 28.05, 16.07);
- РЕПО ЦБ — аукционы примерно раз в неделю;
- Росказна, депозиты ЕКС — аукционы почти ежедневно.
"""
from __future__ import annotations

from dataclasses import dataclass


# Запас поверх ожидаемого лага: в пределах него считаем, что источник просто
# задержал публикацию на цикл, и показываем «внимание», а не «застой».
GRACE_DAYS = 7

STATUS_OK = "ok"
STATUS_WARN = "warn"
STATUS_STALE = "stale"
STATUS_UNKNOWN = "unknown"


@dataclass(frozen=True)
class FreshnessRule:
    """Ожидаемый лаг публикации и человекочитаемая причина."""

    expected_lag_days: int
    cadence: str
    forward_looking: bool = False


# Таблицы без даты (справочники, метрики порогов) в проверку не входят.
DATELESS_TABLES = frozenset({"lsi_threshold_metrics"})

RULES: dict[str, FreshnessRule] = {
    # --- дневные ряды ---
    "cbr_liquidity": FreshnessRule(3, "дневной ряд ЦБ, следующий рабочий день"),
    "ruonia": FreshnessRule(3, "дневная ставка, публикуется на следующий рабочий день"),
    "keyrate": FreshnessRule(7, "меняется только на заседаниях ЦБ"),
    "m5_features": FreshnessRule(3, "производная от дневной ликвидности ЦБ"),
    "final_ml_dataset": FreshnessRule(3, "сводный дневной датасет"),
    "honest_ml_dataset": FreshnessRule(3, "сводный дневной датасет"),
    "honest_lsi_scores": FreshnessRule(3, "дневные скоры LSI"),

    # --- аукционные ряды ---
    "roskazna_treasury_deposits": FreshnessRule(5, "аукционы депозитов ЕКС, почти ежедневно"),
    "repo": FreshnessRule(10, "аукционы РЕПО ЦБ, примерно раз в неделю"),
    "m2_features": FreshnessRule(10, "производная от аукционов РЕПО"),
    "m2_daily_profile": FreshnessRule(10, "производная от аукционов РЕПО"),
    "ofz_auctions": FreshnessRule(35, "накопительный файл Минфина, обновляется нерегулярно"),
    "m3_features": FreshnessRule(35, "производная от файла Минфина по ОФЗ"),

    # --- месячные ряды ЦБ ---
    "required_reserves": FreshnessRule(60, "месячный ряд, 16-й рабочий день следующего месяца"),
    "m1_features": FreshnessRule(60, "производная от месячных обязательных резервов"),
    "cbr_budget_funds": FreshnessRule(70, "месячный ряд, задержка публикации ~2 месяца"),

    # --- календарь вперёд ---
    "m4_features": FreshnessRule(
        0,
        "налоговый календарь строится вперёд, отрицательный лаг — норма",
        forward_looking=True,
    ),
}


def rule_for(table_name: str) -> FreshnessRule | None:
    """Возвращает правило свежести для таблицы витрины"""
    return RULES.get(table_name)


def classify(table_name: str, lag_days: float | None) -> str:
    """Определяет статус свежести таблицы с учётом графика публикации источника"""
    if table_name in DATELESS_TABLES or lag_days is None:
        return STATUS_UNKNOWN

    rule = rule_for(table_name)
    if rule is None:
        return STATUS_UNKNOWN

    lag = int(lag_days)

    # Календарь вперёд: данные впереди сегодняшней даты — так и задумано.
    if rule.forward_looking and lag <= 0:
        return STATUS_OK

    if lag <= rule.expected_lag_days:
        return STATUS_OK
    if lag <= rule.expected_lag_days + GRACE_DAYS:
        return STATUS_WARN
    return STATUS_STALE


def expected_lag(table_name: str) -> int | None:
    """Ожидаемый лаг публикации в днях, если он задан для таблицы"""
    rule = rule_for(table_name)
    return None if rule is None else rule.expected_lag_days


def cadence(table_name: str) -> str:
    """Человекочитаемое описание графика публикации источника"""
    rule = rule_for(table_name)
    return "" if rule is None else rule.cadence


def overdue_days(table_name: str, lag_days: float | None) -> int | None:
    """На сколько дней таблица просрочена сверх ожидаемого лага (иначе 0)"""
    if lag_days is None:
        return None
    rule = rule_for(table_name)
    if rule is None:
        return None
    if rule.forward_looking and int(lag_days) <= 0:
        return 0
    return max(0, int(lag_days) - rule.expected_lag_days)
