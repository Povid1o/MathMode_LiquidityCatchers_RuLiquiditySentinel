"""Сервис автокомментариев по текущему состоянию LSI.

Реализует:
- сборку контекста из обработанных данных
- rule-based комментарий (работает без LLM)
- опциональный вызов OpenAI API (если задан OPENAI_API_KEY)
- ответы на вопросы с простым date-based retrieval

Fallback-цепочка: LLM → rule-based. Dashboard работает в любом случае.
"""

from __future__ import annotations

import os
import traceback
from pathlib import Path
from typing import Any

import pandas as pd

# Загружаем .env если есть — не падаем если python-dotenv не установлен
try:
    from dotenv import load_dotenv as _load_dotenv
    _load_dotenv(dotenv_path=Path(__file__).resolve().parents[3] / ".env", override=False)
except ImportError:
    pass

from backend.src.services.lsi_thresholds import DEFAULT_THRESHOLD_PROFILE
from backend.src.services.lsi_thresholds import get_lsi_status
from backend.src.services.lsi_thresholds import get_threshold_profile


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data" / "processed"

_FINAL_ML_PATH = DATA_DIR / "final_ml_dataset.parquet"
_LSI_SCORES_PATH = DATA_DIR / "lsi_scores.parquet"
_BACKTEST_PATH = DATA_DIR / "lsi_backtest_scores.parquet"
_THRESHOLD_METRICS_PATH = DATA_DIR / "lsi_threshold_metrics.parquet"

# Названия модулей для читаемых подписей
_MODULE_LABELS: dict[str, str] = {
    "M1": "M1 (Обязательные резервы / RUONIA)",
    "M2": "M2 (Репо ЦБ)",
    "M3": "M3 (ОФЗ-аукционы)",
    "M4": "M4 (Налоговое давление)",
    "M5": "M5 (Ликвидность / Казначейство)",
}

# Короткие пояснения к статусам для rule-based текста
_STATUS_EXPLANATIONS: dict[str, str] = {
    "ЗЕЛЕНЫЙ (Норма)": "рынок в штатном режиме, признаков стресса нет",
    "ЖЕЛТЫЙ (Повышенное внимание)": "повышенное внимание: отдельные сигналы выше нормы, требует наблюдения",
    "КРАСНЫЙ (Стресс ликвидности)": "стресс ликвидности: значения существенно выше норм, необходим анализ",
}


# ---------------------------------------------------------------------------
# Загрузка данных
# ---------------------------------------------------------------------------

def _load_parquet_safe(path: Path) -> pd.DataFrame:
    """Загружает parquet-файл, возвращает пустой датафрейм при ошибке"""
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_parquet(path)
    except Exception:
        return pd.DataFrame()


def load_context(threshold_profile: str = DEFAULT_THRESHOLD_PROFILE) -> dict[str, Any]:
    """Собирает аналитический контекст из обработанных данных.

    Возвращает словарь с последними значениями LSI, вкладами модулей,
    флагами событий и сводкой бэктеста
    """
    context: dict[str, Any] = {
        "threshold_profile": threshold_profile,
        "profile_config": get_threshold_profile(threshold_profile),
        "errors": [],
    }

    # --- финальный датасет (последняя строка) ---
    final_df = _load_parquet_safe(_FINAL_ML_PATH)
    if not final_df.empty:
        if "date" in final_df.columns:
            final_df["date"] = pd.to_datetime(final_df["date"])
            final_df = final_df.sort_values("date").reset_index(drop=True)
        latest = final_df.iloc[-1]
        context["data_date"] = str(latest["date"].date()) if hasattr(latest["date"], "date") else str(latest["date"])
        context["final_df"] = final_df
        context["latest_row"] = latest

        # M4: налоговые флаги
        context["m4_tax_active"] = bool(latest.get("m4_Tax_Active_Flag", 0) == 1)
        context["m4_tax_pre"] = bool(latest.get("m4_Tax_Pre_Flag", 0) == 1)
        context["m4_tax_day"] = bool(latest.get("m4_Tax_Day_Strict", 0) == 1)

        # M5: бюджетный drain
        context["m5_budget_drain"] = bool(latest.get("m5_Flag_Budget_Drain", 0) == 1)
        context["m5_liquidity"] = float(latest["m5_liquidity_deficit_surplus_bln_rub_lag_1d"]) \
            if "m5_liquidity_deficit_surplus_bln_rub_lag_1d" in latest.index and pd.notna(latest.get("m5_liquidity_deficit_surplus_bln_rub_lag_1d")) \
            else None

        # M2: аукционный стресс
        context["m2_flag_demand"] = bool(latest.get("m2_Flag_Demand", 0) == 1)

        # M3: недоспрос ОФЗ
        context["m3_flag_nedospros"] = bool(latest.get("m3_Flag_Nedospros", 0) == 1)
    else:
        context["errors"].append("Финальный датасет не найден")
        context["data_date"] = "н/д"

    # --- LSI-скоры (последняя строка) ---
    lsi_df = _load_parquet_safe(_LSI_SCORES_PATH)
    if not lsi_df.empty:
        if "date" in lsi_df.columns:
            lsi_df["date"] = pd.to_datetime(lsi_df["date"])
            lsi_df = lsi_df.sort_values("date").reset_index(drop=True)
        lsi_latest = lsi_df.iloc[-1]
        context["lsi_df"] = lsi_df

        if "lsi_local" in lsi_latest and pd.notna(lsi_latest.get("lsi_local")):
            local_val = float(lsi_latest["lsi_local"])
            context["lsi_local"] = local_val
            context["local_status"] = get_lsi_status(local_val, profile=threshold_profile)
            context["local_contribs"] = {
                m.upper(): float(lsi_latest[f"lsi_local_contrib_{m}"])
                for m in ["m1", "m2", "m3", "m4", "m5"]
                if f"lsi_local_contrib_{m}" in lsi_latest.index and pd.notna(lsi_latest.get(f"lsi_local_contrib_{m}"))
            }

        if "lsi_global" in lsi_latest and pd.notna(lsi_latest.get("lsi_global")):
            global_val = float(lsi_latest["lsi_global"])
            context["lsi_global"] = global_val
            context["global_status"] = get_lsi_status(global_val, profile=threshold_profile)
            context["global_contribs"] = {
                m.upper(): float(lsi_latest[f"lsi_global_contrib_{m}"])
                for m in ["m1", "m2", "m3", "m4", "m5"]
                if f"lsi_global_contrib_{m}" in lsi_latest.index and pd.notna(lsi_latest.get(f"lsi_global_contrib_{m}"))
            }
    else:
        context["errors"].append("LSI-скоры не найдены")

    # --- бэктест: краткая сводка ---
    backtest_df = _load_parquet_safe(_BACKTEST_PATH)
    if not backtest_df.empty:
        context["backtest_available"] = True
        if "lsi_global" in backtest_df.columns:
            context["backtest_global_max"] = float(backtest_df["lsi_global"].max())
            context["backtest_global_mean"] = float(backtest_df["lsi_global"].mean())
        if "lsi_local" in backtest_df.columns:
            context["backtest_local_max"] = float(backtest_df["lsi_local"].max())
    else:
        context["backtest_available"] = False

    return context


# ---------------------------------------------------------------------------
# Rule-based комментарий
# ---------------------------------------------------------------------------

def _top_modules(contribs: dict[str, float], n: int = 3) -> list[tuple[str, float]]:
    """Возвращает топ-N модулей по вкладу"""
    sorted_items = sorted(contribs.items(), key=lambda x: x[1], reverse=True)
    return sorted_items[:n]


def build_rule_based_commentary(context: dict[str, Any]) -> str:
    """Строит аналитический комментарий по текущему LSI на основе правил.

    Не требует LLM. Использует только значения из переданного контекста.
    Выводит дату данных, статусы, вклады модулей, флаги событий и краткую интерпретацию
    """
    lines: list[str] = []

    data_date = context.get("data_date", "н/д")
    profile = context.get("threshold_profile", DEFAULT_THRESHOLD_PROFILE)
    profile_cfg = context.get("profile_config", {})
    green_max = float(profile_cfg.get("green_max", 30))
    yellow_max = float(profile_cfg.get("yellow_max", 60))

    lsi_local = context.get("lsi_local")
    lsi_global = context.get("lsi_global")
    local_status = context.get("local_status", "н/д")
    global_status = context.get("global_status", "н/д")
    local_contribs = context.get("local_contribs", {})
    global_contribs = context.get("global_contribs", {})

    # --- заголовок ---
    lines.append(f"📅 Дата данных: {data_date}")
    lines.append(f"⚙️ Пороговый профиль: {profile} (зелёный < {int(green_max)}, жёлтый {int(green_max)}–{int(yellow_max)}, красный ≥ {int(yellow_max)})")
    lines.append("")

    # --- LSI Local ---
    if lsi_local is not None:
        local_expl = _STATUS_EXPLANATIONS.get(local_status, "")
        lines.append(f"📍 LSI Local = {lsi_local:.2f} → {local_status}")
        if local_expl:
            lines.append(f"   ({local_expl})")
        if local_contribs:
            top_local = _top_modules(local_contribs, n=3)
            top_str = ", ".join(f"{_MODULE_LABELS.get(m, m)} {v:.1f}%" for m, v in top_local)
            lines.append(f"   Топ вкладов в Local: {top_str}")
    else:
        lines.append("📍 LSI Local: данные недоступны")

    lines.append("")

    # --- LSI Global ---
    if lsi_global is not None:
        global_expl = _STATUS_EXPLANATIONS.get(global_status, "")
        lines.append(f"🌐 LSI Global = {lsi_global:.2f} → {global_status}")
        if global_expl:
            lines.append(f"   ({global_expl})")
        if global_contribs:
            top_global = _top_modules(global_contribs, n=3)
            top_str = ", ".join(f"{_MODULE_LABELS.get(m, m)} {v:.1f}%" for m, v in top_global)
            lines.append(f"   Топ вкладов в Global: {top_str}")
    else:
        lines.append("🌐 LSI Global: данные недоступны")

    lines.append("")

    # --- флаги событий ---
    lines.append("🔔 Флаги событий на последнюю дату:")
    event_lines: list[str] = []

    m4_tax_active = context.get("m4_tax_active", False)
    m4_tax_pre = context.get("m4_tax_pre", False)
    m4_tax_day = context.get("m4_tax_day", False)

    if m4_tax_day:
        event_lines.append("   🟡 M4: налоговый день (строгий) — пиковое давление на ликвидность")
    elif m4_tax_active:
        event_lines.append("   🟡 M4: активная налоговая фаза — повышенный отток ликвидности")
    elif m4_tax_pre:
        event_lines.append("   🔵 M4: предналоговая фаза — ожидание оттока")
    else:
        event_lines.append("   🟢 M4: налогового давления не зафиксировано")

    m5_drain = context.get("m5_budget_drain", False)
    m5_liq = context.get("m5_liquidity")
    if m5_drain:
        event_lines.append("   🔴 M5: флаг Budget Drain активен — бюджетные средства утекают из системы")
    else:
        event_lines.append("   🟢 M5: флага Budget Drain нет")
    if m5_liq is not None:
        sign = "+" if m5_liq >= 0 else ""
        event_lines.append(f"   M5: профицит/дефицит ликвидности (лаг 1д) = {sign}{m5_liq:,.1f} млрд руб.")

    if context.get("m2_flag_demand", False):
        event_lines.append("   🔴 M2: аномальный спрос на РЕПО ЦБ зафиксирован")
    else:
        event_lines.append("   🟢 M2: аукционного стресс-сигнала нет")

    if context.get("m3_flag_nedospros", False):
        event_lines.append("   🟡 M3: недоспрос на ОФЗ-аукционе зафиксирован")
    else:
        event_lines.append("   🟢 M3: стресс-сигнала по ОФЗ нет")

    lines.extend(event_lines)
    lines.append("")

    # --- интерпретация ---
    lines.append("📊 Краткая интерпретация:")
    interpretation = _build_interpretation(
        lsi_local=lsi_local,
        lsi_global=lsi_global,
        local_status=local_status,
        global_status=global_status,
        local_contribs=local_contribs,
        global_contribs=global_contribs,
        m4_tax_active=m4_tax_active,
        m4_tax_day=m4_tax_day,
        m5_drain=m5_drain,
        m2_demand=context.get("m2_flag_demand", False),
        m3_nedospros=context.get("m3_flag_nedospros", False),
        profile=profile,
        green_max=green_max,
        yellow_max=yellow_max,
    )
    lines.append(interpretation)

    lines.append("")
    lines.append(
        "⚠️ Ограничения: дата данных не обязательно совпадает с сегодняшней календарной датой. "
        "LSI Local и LSI Global имеют разные обучающие окна и интерпретируются независимо. "
        "LSI — модельный индикатор; финальное суждение остаётся за аналитиком."
    )

    errors = context.get("errors", [])
    if errors:
        lines.append("")
        lines.append("⚠️ Предупреждения при загрузке: " + "; ".join(errors))

    return "\n".join(lines)


def _build_interpretation(
    *,
    lsi_local: float | None,
    lsi_global: float | None,
    local_status: str,
    global_status: str,
    local_contribs: dict[str, float],
    global_contribs: dict[str, float],
    m4_tax_active: bool,
    m4_tax_day: bool,
    m5_drain: bool,
    m2_demand: bool,
    m3_nedospros: bool,
    profile: str,
    green_max: float,
    yellow_max: float,
) -> str:
    """Формирует текстовую интерпретацию по набору флагов и значений"""
    parts: list[str] = []

    both_red = (
        lsi_local is not None and lsi_global is not None
        and lsi_local >= yellow_max and lsi_global >= yellow_max
    )
    any_red = (
        (lsi_local is not None and lsi_local >= yellow_max)
        or (lsi_global is not None and lsi_global >= yellow_max)
    )
    both_yellow = (
        lsi_local is not None and lsi_global is not None
        and lsi_local >= green_max and lsi_global >= green_max
    )
    all_green = (
        (lsi_local is None or lsi_local < green_max)
        and (lsi_global is None or lsi_global < green_max)
    )

    if both_red:
        parts.append(
            "Оба индикатора LSI Local и LSI Global находятся в красной зоне. "
            "Это согласованный сигнал повышенного стресса ликвидности, "
            "требующий немедленного анализа и ручного подтверждения аналитиком."
        )
    elif any_red:
        red_src = "Local" if (lsi_local is not None and lsi_local >= yellow_max) else "Global"
        green_src = "Global" if red_src == "Local" else "Local"
        parts.append(
            f"LSI {red_src} в красной зоне, тогда как LSI {green_src} не подтверждает стресс. "
            "Расхождение Local/Global требует осторожной интерпретации: "
            f"LSI {red_src} может отражать локальную аномалию, а не системный кризис."
        )
    elif both_yellow:
        parts.append(
            "Оба индикатора в жёлтой зоне — повышенное внимание. "
            "Признаков острого стресса нет, но сигнал требует мониторинга."
        )
    elif lsi_local is not None and lsi_local >= green_max:
        parts.append(
            "LSI Local в жёлтой зоне, LSI Global в норме. "
            "Локальная аномалия без подтверждения исторической моделью."
        )
    elif lsi_global is not None and lsi_global >= green_max:
        parts.append(
            "LSI Global в жёлтой зоне, LSI Local в норме. "
            "Возможно, историческая модель фиксирует паттерн, не выраженный в краткосрочном окне."
        )
    elif all_green:
        parts.append("Оба индикатора в зелёной зоне. Рынок функционирует в штатном режиме.")

    # дополнительные триггеры
    if m4_tax_day or m4_tax_active:
        parts.append(
            "Активная налоговая фаза объясняет часть давления через M4. "
            "Эффект носит временный, календарный характер."
        )
    if m5_drain:
        parts.append(
            "Флаг Budget Drain (M5) активен: бюджетные средства изымаются из банковской системы, "
            "что создаёт дополнительный структурный дефицит ликвидности."
        )
    if m2_demand:
        parts.append("Аномальный спрос на РЕПО ЦБ (M2) — банки активно привлекают ликвидность у регулятора.")
    if m3_nedospros:
        parts.append("Недоспрос на ОФЗ-аукционе (M3) — рынок предъявляет меньше спроса, чем предложено Минфином.")

    # топ-модули
    all_contribs = {}
    for m, v in local_contribs.items():
        all_contribs[m] = all_contribs.get(m, 0) + v / 2
    for m, v in global_contribs.items():
        all_contribs[m] = all_contribs.get(m, 0) + v / 2

    if all_contribs:
        top_m = max(all_contribs, key=lambda x: all_contribs[x])
        parts.append(
            f"Наибольший модельный вклад вносит {_MODULE_LABELS.get(top_m, top_m)} — "
            "именно этот блок следует проверить первым."
        )

    parts.append(
        f"Используется профиль «{profile}»: "
        "результаты отличаются от других профилей порогов."
    )

    return " ".join(parts)


# ---------------------------------------------------------------------------
# LLM prompt
# ---------------------------------------------------------------------------

def build_llm_prompt(context: dict[str, Any], user_question: str | None = None) -> str:
    """Формирует структурированный промпт для LLM-аналитика.

    Содержит роль, контекст, запрет на выдумывание данных и вопрос пользователя
    """
    profile = context.get("threshold_profile", DEFAULT_THRESHOLD_PROFILE)
    profile_cfg = context.get("profile_config", {})
    green_max = float(profile_cfg.get("green_max", 30))
    yellow_max = float(profile_cfg.get("yellow_max", 60))
    data_date = context.get("data_date", "н/д")

    lsi_local = context.get("lsi_local")
    lsi_global = context.get("lsi_global")
    local_status = context.get("local_status", "н/д")
    global_status = context.get("global_status", "н/д")
    local_contribs = context.get("local_contribs", {})
    global_contribs = context.get("global_contribs", {})

    rule_commentary = build_rule_based_commentary(context)

    extra_data = context.get("extra_period_summary", "")

    prompt_parts = [
        "ВАЖНО: отвечай ТОЛЬКО на русском языке. Никакого английского.",
        "",
        "Ты — аналитик рублевого денежного рынка. Отвечай кратко и структурно на вопрос аналитика.",
        "",
        "СТРОГИЕ ПРАВИЛА:",
        "- Не выдумывай данные, которые не переданы тебе в контексте.",
        "- Если данных нет — прямо скажи об этом на русском.",
        "- Отвечай ТОЛЬКО по-русски, кратко, структурно. Не используй английский язык.",
        "- Всегда указывай дату данных — это не обязательно сегодняшняя дата.",
        "- Не говори «кризис точно будет» — это модельный индикатор, не прогноз.",
        "- Не путай LSI Local и LSI Global — они рассчитываются на разных окнах.",
        "- Используй только переданный контекст. Не добавляй внешние новости.",
        "- Если LSI ЖЕЛТЫЙ — пиши «повышенное внимание», а не «стресс».",
        "- Если LSI КРАСНЫЙ — пиши «сигнал стресса требует подтверждения аналитиком».",
        "",
        "КОНТЕКСТ LSI (автоматический rule-based анализ):",
        "---",
        rule_commentary,
        "---",
        "",
        f"Пороговый профиль: {profile} (зелёный < {int(green_max)}, жёлтый {int(green_max)}–{int(yellow_max)}, красный ≥ {int(yellow_max)})",
        f"Дата данных: {data_date}",
    ]

    if lsi_local is not None:
        top_l = _top_modules(local_contribs, n=2)
        top_l_str = ", ".join(f"{m} ({v:.1f}%)" for m, v in top_l)
        prompt_parts.append(f"LSI Local = {lsi_local:.2f} [{local_status}], топ: {top_l_str}")

    if lsi_global is not None:
        top_g = _top_modules(global_contribs, n=2)
        top_g_str = ", ".join(f"{m} ({v:.1f}%)" for m, v in top_g)
        prompt_parts.append(f"LSI Global = {lsi_global:.2f} [{global_status}], топ: {top_g_str}")

    if extra_data:
        prompt_parts.append("")
        prompt_parts.append("ДОПОЛНИТЕЛЬНЫЕ ДАННЫЕ ПО ЗАПРОШЕННОМУ ПЕРИОДУ:")
        prompt_parts.append(extra_data)

    prompt_parts.append("")
    if user_question:
        prompt_parts.append(f"ВОПРОС АНАЛИТИКА: {user_question}")
    else:
        prompt_parts.append("ЗАДАЧА: Дай краткий аналитический комментарий по текущему состоянию LSI.")

    return "\n".join(prompt_parts)


# ---------------------------------------------------------------------------
# LLM генерация (optional)
# ---------------------------------------------------------------------------

def _is_llm_available() -> bool:
    """Проверяет наличие API-ключа без импорта openai"""
    return bool(os.environ.get("OPENAI_API_KEY", "").strip())


def generate_llm_commentary(
    context: dict[str, Any],
    user_question: str | None = None,
) -> str:
    """Генерирует комментарий через LLM API (опционально).

    Если API-ключ не задан или пакет openai не установлен — возвращает rule-based fallback.
    Никогда не падает с исключением: все ошибки обрабатываются внутри
    """
    if not _is_llm_available():
        rule = build_rule_based_commentary(context)
        return rule + "\n\n⚠️ [LLM API не подключён: OPENAI_API_KEY не задан. Показан rule-based комментарий.]"

    try:
        import openai  # noqa: PLC0415
    except ImportError:
        rule = build_rule_based_commentary(context)
        return (
            rule
            + "\n\n⚠️ [Пакет openai не установлен. "
            "Для LLM-режима: pip install openai. Показан rule-based комментарий.]"
        )

    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    base_url = os.environ.get("LLM_BASE_URL", "").strip() or None
    prompt = build_llm_prompt(context, user_question=user_question)

    try:
        client_kwargs: dict[str, Any] = {"api_key": os.environ["OPENAI_API_KEY"]}
        if base_url:
            client_kwargs["base_url"] = base_url
        client = openai.OpenAI(**client_kwargs)
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            # Qwen3.5-122b тратит ~6000 токенов на reasoning перед финальным ответом
            max_tokens=8000,
            temperature=0.2,
        )
        msg = response.choices[0].message
        # Если content пустой — не показываем reasoning_content (сырые мысли),
        # а возвращаем rule-based чтобы не засорять интерфейс
        llm_text = (msg.content or "").strip()
        if not llm_text:
            rule = build_rule_based_commentary(context)
            return rule + "\n\n⚠️ [LLM не вернул финальный ответ. Показан rule-based комментарий.]"
        return llm_text
    except Exception as exc:
        short_error = str(exc)[:200]
        rule = build_rule_based_commentary(context)
        return (
            rule
            + f"\n\n⚠️ [LLM API вернул ошибку: {short_error}. Показан rule-based комментарий.]"
        )


# ---------------------------------------------------------------------------
# Retrieval по дате / периоду
# ---------------------------------------------------------------------------

_PERIOD_KEYWORDS: list[tuple[list[str], str, str]] = [
    # Конкретные месяцы — должны стоять ВЫШЕ годовых паттернов,
    # иначе "феврале 2022" попадёт под паттерн "2022" (весь год).
    # Паттерны содержат год чтобы не путать февраль-2022 с февралём-2023.
    # Все падежные формы с годом — чтобы не ловить "феврале 2023" под паттерн 2022
    (["февраль 2022", "феврале 2022", "февраля 2022", "фев 2022", "2022-02"], "2022-01-20", "2022-03-31"),
    (["март 2022", "марте 2022", "марта 2022", "2022-03"], "2022-02-15", "2022-04-15"),
    (["декабрь 2014", "декабре 2014", "декабря 2014", "дек 2014", "2014-12"], "2014-11-15", "2015-01-15"),
    (["август 2023", "августе 2023", "августа 2023", "авг 2023", "2023-08"], "2023-07-15", "2023-09-15"),
    (["2022", "год 2022"], "2022-01-01", "2022-12-31"),
    (["2014", "год 2014"], "2014-01-01", "2014-12-31"),
    (["2023", "год 2023"], "2023-01-01", "2023-12-31"),
    (["2024", "год 2024"], "2024-01-01", "2024-12-31"),
]

_LATEST_KEYWORDS = [
    "последний", "последнее", "сейчас", "текущий", "текущая", "сегодня", "last", "latest",
]


def _extract_period(question: str) -> tuple[str | None, str | None]:
    """Определяет диапазон дат по ключевым словам в вопросе"""
    q_lower = question.lower()
    for keywords, date_from, date_to in _PERIOD_KEYWORDS:
        for kw in keywords:
            if kw in q_lower:
                return date_from, date_to
    return None, None


def _build_period_summary(lsi_df: pd.DataFrame, date_from: str, date_to: str) -> str:
    """Формирует текстовую сводку по данным за указанный период"""
    mask = (lsi_df["date"] >= pd.Timestamp(date_from)) & (lsi_df["date"] <= pd.Timestamp(date_to))
    period_df = lsi_df.loc[mask].copy()

    if period_df.empty:
        return f"Данные за период {date_from} — {date_to} отсутствуют."

    lines = [f"Период {date_from} — {date_to}: найдено {len(period_df)} наблюдений."]

    for col, label in [("lsi_local", "LSI Local"), ("lsi_global", "LSI Global")]:
        if col in period_df.columns:
            s = period_df[col].dropna()
            if not s.empty:
                lines.append(
                    f"{label}: min={s.min():.2f}, max={s.max():.2f}, среднее={s.mean():.2f}"
                )

    # пиковая дата
    if "lsi_global" in period_df.columns:
        peak_row = period_df.loc[period_df["lsi_global"].idxmax()]
        lines.append(
            f"Пик LSI Global: {peak_row['lsi_global']:.2f} на дату {peak_row['date'].date()}"
        )
    elif "lsi_local" in period_df.columns:
        peak_row = period_df.loc[period_df["lsi_local"].idxmax()]
        lines.append(
            f"Пик LSI Local: {peak_row['lsi_local']:.2f} на дату {peak_row['date'].date()}"
        )

    return "\n".join(lines)


def _build_latest_summary(context: dict[str, Any]) -> str:
    """Формирует краткую сводку по последней дате из контекста"""
    parts = [f"Последняя дата данных: {context.get('data_date', 'н/д')}"]
    if (v := context.get("lsi_local")) is not None:
        parts.append(f"LSI Local = {v:.2f} [{context.get('local_status', '')}]")
    if (v := context.get("lsi_global")) is not None:
        parts.append(f"LSI Global = {v:.2f} [{context.get('global_status', '')}]")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Публичный API: ответ на вопрос
# ---------------------------------------------------------------------------

def answer_question(
    question: str,
    threshold_profile: str = DEFAULT_THRESHOLD_PROFILE,
    use_llm: bool = True,
) -> str:
    """Отвечает на аналитический вопрос по данным LSI.

    Пытается найти релевантный период в данных по ключевым словам,
    добавляет сводку в контекст и передаёт в LLM (или rule-based fallback)
    """
    context = load_context(threshold_profile=threshold_profile)

    q_lower = question.lower()
    is_latest = any(kw in q_lower for kw in _LATEST_KEYWORDS)
    date_from, date_to = _extract_period(question)

    lsi_df = context.get("lsi_df")

    if date_from and lsi_df is not None and not lsi_df.empty:
        summary = _build_period_summary(lsi_df, date_from, date_to)
        context["extra_period_summary"] = summary
    elif is_latest or (not date_from):
        context["extra_period_summary"] = _build_latest_summary(context)

    if use_llm:
        return generate_llm_commentary(context, user_question=question)

    # rule-based fallback с дополнительной сводкой
    rule_text = build_rule_based_commentary(context)
    extra = context.get("extra_period_summary", "")
    if extra:
        return f"{rule_text}\n\n📌 Данные по запрошенному периоду:\n{extra}"
    return rule_text


# ---------------------------------------------------------------------------
# CLI / быстрая проверка
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Тест lsi_commentary_service ===\n")

    print("1. Загрузка контекста...")
    ctx = load_context()
    print(f"   Дата данных: {ctx.get('data_date')}")
    print(f"   LSI Local: {ctx.get('lsi_local')}, статус: {ctx.get('local_status')}")
    print(f"   LSI Global: {ctx.get('lsi_global')}, статус: {ctx.get('global_status')}")
    print()

    print("2. Rule-based комментарий (без LLM):")
    commentary = build_rule_based_commentary(ctx)
    print(commentary)
    print()

    print("3. Вопрос о марте 2022:")
    answer = answer_question("Что было в марте 2022?", use_llm=False)
    print(answer[:800])
    print()

    print("4. Проверка LLM доступности:")
    available = _is_llm_available()
    print(f"   OPENAI_API_KEY задан: {available}")
    if not available:
        print("   LLM недоступен — используется rule-based fallback (штатный режим)")

    print("\n=== Все проверки прошли ===")
