"""feature_catalog — единый словарь человекочитаемых названий признаков.

Зачем отдельный модуль в backend, а не словарь в дашборде: потребителей двое —
страницы дашборда и AI-аналитик. Пока подписи лежали в dashboard/components и
дублировались строками в заголовках графиков, источник правды успел размножиться,
а покрытие остановилось на honest-whitelist.

Каждая запись описывает не только «как назвать», но и «как читать»:

- label        — короткая подпись для оси, легенды, заголовка;
- description  — одна фраза о смысле, идёт в тултип и в ответ ассистента;
- unit         — единица измерения, чтобы «млн ₽» и «п.п.» не терялись;
- higher_means — что означает рост значения: stress / calm / neutral;
- status       — verified / derived / needs_review.

Про `higher_means`. Из имени признака направление не выводится. Живой пример:
m3x_cover в honest_feature_builder умножается на -1, поэтому ВЫСОКОЕ покрытие
аукциона даёт ОТРИЦАТЕЛЬНОЕ значение признака, а положительное значение — это
недоспрос, то есть стресс. Без этого поля и человек, и модель прочитают знак
наоборот.

Про `status`. Неверная русская подпись хуже технического имени: техническое имя
заставляет специалиста пойти и проверить, а уверенная подпись — не заставляет.
Поэтому там, где смысл не выводится из кода однозначно, стоит needs_review, и эта
метка видна и в дашборде, и в ответах ассистента, пока предметный специалист не
подтвердит формулировку.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Literal


HigherMeans = Literal["stress", "calm", "neutral"]
SpecStatus = Literal["verified", "derived", "needs_review"]


@dataclass(frozen=True)
class FeatureSpec:
    """Описание одного признака."""

    label: str
    description: str
    module: str
    unit: str = ""
    higher_means: HigherMeans = "neutral"
    status: SpecStatus = "derived"

    @property
    def needs_review(self) -> bool:
        return self.status == "needs_review"


# Пояснение направления для подсказок в интерфейсе и промпте ассистента.
HIGHER_MEANS_HINT: dict[str, str] = {
    "stress": "рост значения — в сторону стресса",
    "calm": "рост значения — в сторону спокойного рынка",
    "neutral": "направление не однозначно: отклонение в обе стороны считается аномалией",
}


CATALOG: dict[str, FeatureSpec] = {
    # ------------------------------------------------------------------
    # M1 — обязательные резервы и RUONIA
    # ------------------------------------------------------------------
    "m1_spread": FeatureSpec(
        label="Спред резервов",
        description=(
            "Фактические среднедневные остатки на корсчетах минус обязательные резервы, "
            "подлежащие усреднению. Положительное значение — банки держат сверх норматива, "
            "то есть у них есть запас свободных средств."
        ),
        module="M1", unit="млн ₽", higher_means="calm", status="verified",
    ),
    "m1_spread_mad_score": FeatureSpec(
        label="Аномальность спреда резервов (MAD)",
        description=(
            "Насколько спред резервов отклонился от своей нормы за 3 года (в единицах MAD). "
            "Отклонение в обе стороны считается аномалией."
        ),
        module="M1", unit="MAD", higher_means="neutral", status="needs_review",
    ),
    "m1_spread_relative_mad_score": FeatureSpec(
        label="Аномальность относительного спреда (MAD)",
        description=(
            "То же отклонение, но спред берётся в процентах от обязательных резервов — "
            "сравнимо между периодами с разным объёмом резервов."
        ),
        module="M1", unit="MAD", higher_means="neutral", status="needs_review",
    ),
    "m1_reserve_load_mad_score": FeatureSpec(
        label="Аномальность нагрузки резервов (MAD)",
        description=(
            "Отклонение показателя нагрузки: какая доля остатка на корсчетах связана "
            "резервными обязательствами (резервы к усреднению плюс резервы на спецсчетах). "
            "Чем выше нагрузка, тем меньше у банков свободных средств."
        ),
        module="M1", unit="MAD", higher_means="neutral", status="needs_review",
    ),
    "m1_ruonia_mad_score": FeatureSpec(
        label="Аномальность ставки RUONIA (MAD)",
        description=(
            "Отклонение ставки RUONIA от собственной нормы за 3 года. "
            "Рост ставки относительно нормы — признак удорожания рублёвой ликвидности."
        ),
        module="M1", unit="MAD", higher_means="stress", status="derived",
    ),
    "m1_ruonia_rate": FeatureSpec(
        label="Ставка RUONIA",
        description="Ставка межбанковского рублёвого рынка овернайт.",
        module="M1", unit="%", higher_means="neutral", status="verified",
    ),
    "m1_spread_vol": FeatureSpec(
        label="Волатильность подушки резервов",
        description=(
            "Модуль дневного изменения подушки в единицах MAD. Растёт, когда остатки "
            "на корсчетах начинают колебаться сильнее обычного."
        ),
        module="M1", unit="MAD", higher_means="stress", status="verified",
    ),
    "m1_signal_final": FeatureSpec(
        label="Итоговый сигнал M1",
        description=(
            "Взвешенная сводка MAD-скоров модуля, ограниченная диапазоном ±5. "
            "В последний день периода усреднения положительный сигнал усиливается "
            "на 15%: в этот момент нехватка резервов особенно болезненна."
        ),
        module="M1", unit="", higher_means="stress", status="verified",
    ),

    # ------------------------------------------------------------------
    # M2 — аукционы РЕПО ЦБ
    # ------------------------------------------------------------------
    "m2_auction_flag": FeatureSpec(
        label="Был аукцион РЕПО",
        description="Признак того, что в этот день проводился аукцион РЕПО ЦБ.",
        module="M2", unit="0/1", higher_means="neutral", status="verified",
    ),
    "m2_Flag_Demand": FeatureSpec(
        label="Флаг повышенного спроса на РЕПО",
        description=(
            "Спрос банков на аукционе РЕПО выше порога относительно предложения — "
            "банки активнее обычного привлекают ликвидность у регулятора."
        ),
        module="M2", unit="0/1", higher_means="stress", status="derived",
    ),
    "m2_MAD_score_cover": FeatureSpec(
        label="Аномальность переподписки РЕПО (MAD)",
        description=(
            "Отклонение отношения спроса к объёму сделок на аукционе РЕПО от нормы "
            "за 3 года. Считается по всем срочностям вместе."
        ),
        module="M2", unit="MAD", higher_means="stress", status="derived",
    ),
    "m2_base_cover_mad": FeatureSpec(
        label="Аномальность переподписки РЕПО, основная срочность (MAD)",
        description=(
            "То же отклонение переподписки, но только внутри основной срочности "
            "(недельные аукционы). Окно однородно по срочности, поэтому показатель "
            "чище, чем общий по всем срокам."
        ),
        module="M2", unit="MAD", higher_means="stress", status="derived",
    ),
    "m2_cutoff_spread": FeatureSpec(
        label="Спред ставки отсечения к RUONIA",
        description=(
            "Ставка отсечения аукциона РЕПО минус ставка RUONIA. Внимание: спред "
            "считается именно к RUONIA, а не к ключевой ставке. Рост означает, что "
            "банки готовы платить за ликвидность дороже межбанковского рынка."
        ),
        module="M2", unit="п.п.", higher_means="stress", status="verified",
    ),
    "m2_cutoff_spread_available": FeatureSpec(
        label="Служебный: спред отсечения доступен",
        description=(
            "Служебный флаг: был ли аукцион с известной ставкой отсечения в пределах "
            "7 дней. Показывает наличие данных, а не состояние рынка."
        ),
        module="M2", unit="0/1", higher_means="neutral", status="verified",
    ),
    "m2_short_active30": FeatureSpec(
        label="Короткое РЕПО активно (30 дней)",
        description=(
            "С последнего аукциона короткого РЕПО прошло не больше 30 дней. "
            "ЦБ проводит их не постоянно, а при потребности в тонкой настройке."
        ),
        module="M2", unit="0/1", higher_means="stress", status="derived",
    ),
    "m2_days_since_short": FeatureSpec(
        label="Дней с последнего короткого РЕПО",
        description=(
            "Сколько дней прошло с последнего аукциона короткого РЕПО, "
            "ограничено 90 днями. Большое значение — регулятор давно не вмешивался."
        ),
        module="M2", unit="дней", higher_means="calm", status="derived",
    ),

    # ------------------------------------------------------------------
    # M3 — аукционы ОФЗ Минфина
    # ------------------------------------------------------------------
    "m3_auction_flag": FeatureSpec(
        label="Был аукцион ОФЗ",
        description="Признак того, что в этот день проводился аукцион ОФЗ.",
        module="M3", unit="0/1", higher_means="neutral", status="verified",
    ),
    "m3_Flag_Nedospros": FeatureSpec(
        label="Флаг недоспроса на ОФЗ",
        description=(
            "Спрос на аукционе оказался ниже порога относительно предложения: "
            "рынок забрал меньше, чем предлагал Минфин."
        ),
        module="M3", unit="0/1", higher_means="stress", status="verified",
    ),
    "m3_Flag_Perespros": FeatureSpec(
        label="Флаг переспроса на ОФЗ",
        description=(
            "Спрос на аукционе выше порога относительно предложения — "
            "признак сильного аппетита к госбумагам."
        ),
        module="M3", unit="0/1", higher_means="calm", status="verified",
    ),
    "m3_cover_stress_score": FeatureSpec(
        label="Стресс-скор покрытия ОФЗ",
        description=(
            "Оценка стресса по покрытию аукциона: слабый спрос относительно истории "
            "даёт положительное значение."
        ),
        module="M3", unit="", higher_means="stress", status="needs_review",
    ),
    "m3x_cover": FeatureSpec(
        label="Стресс покрытия ОФЗ (event-aware)",
        description=(
            "MAD-отклонение покрытия аукциона (спрос к предложению) с ОБРАТНЫМ знаком: "
            "положительное значение — покрытие ниже нормы, то есть недоспрос и стресс; "
            "отрицательное — спрос выше обычного. Между аукционами значение "
            "протягивается до следующего."
        ),
        module="M3", unit="MAD", higher_means="stress", status="verified",
    ),
    "m3x_placement": FeatureSpec(
        label="Стресс размещения ОФЗ (event-aware)",
        description=(
            "MAD-отклонение доли размещённого объёма от предложенного, знак обратный: "
            "положительное значение — разместили меньше обычного. Несостоявшееся "
            "размещение даёт заметный положительный вклад."
        ),
        module="M3", unit="MAD", higher_means="stress", status="verified",
    ),
    "m3x_yield_to_key": FeatureSpec(
        label="Премия доходности ОФЗ к ключевой ставке",
        description=(
            "MAD-отклонение разницы между доходностью отсечения и ключевой ставкой. "
            "Считается только по ОФЗ-ПД: флоатеры и инфляционные бумаги исказили бы "
            "картину. Рост премии — рынок требует больше за длинный риск."
        ),
        module="M3", unit="MAD", higher_means="stress", status="verified",
    ),
    "m3x_failed": FeatureSpec(
        label="Аукцион ОФЗ не состоялся",
        description="В этот день по аукциону не было размещено ни одной бумаги.",
        module="M3", unit="0/1", higher_means="stress", status="verified",
    ),
    "m3x_age": FeatureSpec(
        label="Дней с последнего аукциона ОФЗ",
        description=(
            "Возраст последних аукционных данных, ограничен 90 днями. "
            "Показывает, насколько свежи значения показателей ОФЗ."
        ),
        module="M3", unit="дней", higher_means="neutral", status="verified",
    ),
    "m3x_days_since": FeatureSpec(
        label="Дней с последнего аукциона ОФЗ (расширенный)",
        description=(
            "То же расстояние до последнего аукциона, но с потолком 250 дней и нулём "
            "до первого аукциона в истории. Отличается от m3x_age только границами."
        ),
        module="M3", unit="дней", higher_means="neutral", status="verified",
    ),
    "m3x_available": FeatureSpec(
        label="Служебный: данные аукциона ОФЗ свежие",
        description=(
            "Служебный флаг: последнему аукциону не больше 10 дней. "
            "Отражает наличие данных, а не состояние рынка."
        ),
        module="M3", unit="0/1", higher_means="neutral", status="verified",
    ),

    # ------------------------------------------------------------------
    # M4 — налоговый календарь (overlay, вне PCA)
    # ------------------------------------------------------------------
    "m4_Tax_Day_Strict": FeatureSpec(
        label="Строгий налоговый день",
        description=(
            "День пиковой налоговой выплаты по календарю ФНС — момент максимального "
            "оттока рублёвой ликвидности из банковской системы."
        ),
        module="M4", unit="0/1", higher_means="stress", status="verified",
    ),
    "m4_MAD_tax_pressure": FeatureSpec(
        label="Аномальность налогового давления (MAD)",
        description=(
            "Отклонение расчётного налогового давления от собственной нормы. "
            "M4 — контекстный overlay: он объясняет календарный отток, но в PCA "
            "индекса не входит."
        ),
        module="M4", unit="MAD", higher_means="stress", status="needs_review",
    ),

    # ------------------------------------------------------------------
    # M5 — ликвидность ЦБ и ЕКС Казначейства
    # ------------------------------------------------------------------
    "m5x_claims": FeatureSpec(
        label="Требования ЦБ к банкам",
        description=(
            "MAD-отклонение объёма требований ЦБ к банкам по стандартным инструментам. "
            "Рост означает, что банки больше занимают у регулятора."
        ),
        module="M5", unit="MAD", higher_means="stress", status="derived",
    ),
    "m5x_liab": FeatureSpec(
        label="Обязательства ЦБ перед банками",
        description=(
            "MAD-отклонение объёма обязательств ЦБ перед банками по стандартным "
            "инструментам: депозиты и облигации ЦБ. Рост — у банков избыток средств, "
            "которые они размещают в регуляторе."
        ),
        module="M5", unit="MAD", higher_means="calm", status="derived",
    ),
    "m5x_repostd": FeatureSpec(
        label="Постоянное РЕПО и валютный своп",
        description=(
            "MAD-отклонение задолженности по операциям постоянного действия. "
            "Это дорогой инструмент: банки идут в него, когда аукционов не хватило."
        ),
        module="M5", unit="MAD", higher_means="stress", status="derived",
    ),
    "m5x_secured": FeatureSpec(
        label="Обеспеченные кредиты ЦБ (постоянные)",
        description=(
            "MAD-отклонение задолженности по обеспеченным кредитам постоянного действия. "
            "Как и постоянное РЕПО, инструмент последней очереди."
        ),
        module="M5", unit="MAD", higher_means="stress", status="derived",
    ),
    "m5x_rk_bidders": FeatureSpec(
        label="Число банков на аукционах ЕКС",
        description=(
            "MAD-отклонение количества банков, подавших заявки на депозитные аукционы "
            "Казначейства. Входит только в модель Local. Внимание: при короткой истории "
            "источника показатель может быть нулевым на всём периоде — это отсутствие "
            "данных, а не отсутствие спроса."
        ),
        module="M5", unit="MAD", higher_means="neutral", status="needs_review",
    ),
}


# Алиасы: страницы модулей M4 и M5 читают нативные датасеты, где те же признаки
# лежат без префикса модуля.
_ALIAS_PREFIXES = ("m1_", "m2_", "m3_", "m3x_", "m4_", "m5_", "m5x_")


def _humanize(column: str) -> str:
    """Автоподпись для признака без записи в каталоге.

    Намеренно выглядит машинно: автоподпись не должна выдавать себя за выверенную.
    """
    match = re.match(r"^(m[1-5]x?)_(.+)$", column)
    if match:
        module, rest = match.group(1).upper(), match.group(2)
    else:
        module, rest = "", column
    readable = rest.replace("_", " ")
    return f"{module} · {readable}" if module else readable


def lookup(column: str) -> FeatureSpec | None:
    """Ищет признак в каталоге, учитывая префиксные алиасы."""
    if column in CATALOG:
        return CATALOG[column]

    # Нативное имя без префикса: m4-страница отдаёт MAD_tax_pressure
    for prefix in _ALIAS_PREFIXES:
        candidate = f"{prefix}{column}"
        if candidate in CATALOG:
            return CATALOG[candidate]

    # Обратный случай: пришло с префиксом, а в каталоге запись без него
    stripped = re.sub(r"^m[1-5]x?_", "", column)
    if stripped != column and stripped in CATALOG:
        return CATALOG[stripped]

    return None


def label(column: str) -> str:
    """Человекочитаемая подпись признака (или автоподпись, если записи нет)."""
    spec = lookup(column)
    return spec.label if spec else _humanize(column)


def label_with_flag(column: str) -> str:
    """Подпись с пометкой о неподтверждённой формулировке — для интерфейса."""
    spec = lookup(column)
    if spec is None:
        return f"{_humanize(column)} ⚠︎"
    return f"{spec.label} ⚠︎" if spec.needs_review else spec.label


def description(column: str) -> str:
    """Пояснение смысла признака, пустая строка если записи нет."""
    spec = lookup(column)
    return spec.description if spec else ""


def describe(column: str) -> dict[str, object]:
    """Полное описание признака для инструментов ассистента."""
    spec = lookup(column)
    if spec is None:
        return {
            "column": column,
            "label": _humanize(column),
            "description": "",
            "module": "",
            "unit": "",
            "higher_means": "neutral",
            "status": "missing",
            "warning": (
                "Признака нет в каталоге: название сгенерировано автоматически. "
                "Не выдавай его за проверенное описание."
            ),
        }

    payload: dict[str, object] = {
        "column": column,
        "label": spec.label,
        "description": spec.description,
        "module": spec.module,
        "unit": spec.unit,
        "higher_means": spec.higher_means,
        "higher_means_hint": HIGHER_MEANS_HINT[spec.higher_means],
        "status": spec.status,
    }
    if spec.needs_review:
        payload["warning"] = (
            "Формулировка ещё не подтверждена предметным специалистом — "
            "сообщи об этом, если строишь на ней вывод."
        )
    return payload


def missing_labels(columns: Iterable[str]) -> list[str]:
    """Возвращает признаки без записи в каталоге.

    Нужна, чтобы пробелы были видимы: новая фича попадает в отчёт на странице
    «Качество данных», а не молча откатывается к техническому имени.
    """
    return sorted({column for column in columns if lookup(column) is None})


def review_queue() -> list[str]:
    """Признаки с неподтверждённой формулировкой — очередь на верификацию."""
    return sorted(column for column, spec in CATALOG.items() if spec.needs_review)


def coverage(columns: Iterable[str]) -> dict[str, object]:
    """Сводка покрытия каталога по переданному набору колонок."""
    unique = sorted({c for c in columns if c != "date"})
    missing = missing_labels(unique)
    covered = len(unique) - len(missing)
    return {
        "total": len(unique),
        "covered": covered,
        "missing_count": len(missing),
        "missing": missing,
        "coverage_pct": round(100 * covered / len(unique), 1) if unique else 0.0,
        "needs_review": review_queue(),
    }
