# 📋 M2 — Аукционы РЕПО Банка России

## 1. Описание модуля

**Экономический смысл.** Через аукционы РЕПО ЦБ предоставляет банкам рублёвую ликвидность под обеспечение. Спрос банков на это фондирование — прямой барометр дефицита: когда своих денег не хватает, банки **активно занимают у регулятора и готовы платить дороже**. Два ключевых сигнала:

- **Переподписка (cover ratio)** — спрос превышает предложенный лимит → нехватка ликвидности;
- **Премия по отсечению (cutoff spread)** — ставка отсечения аукциона уходит выше рыночной RUONIA: банки соглашаются на *премию*, лишь бы получить деньги. Это классический признак напряжённости.

ЦБ проводит **основные** (base, 1-нед) и **тонкой настройки / короткие** (short, 1–6 дн) аукционы. Активизация коротких аукционов — сигнал, что регулятор тушит локальный дефицит.

**За чем следим:** аномальная переподписка основных аукционов, премия отсечения к RUONIA, активность коротких РЕПО. Данные **разрежены** (только в дни аукционов).

---

## 2. Полный пул фичей

**`m2_features`** (уровень аукциона):

| Группа | Колонки |
|--------|---------|
| Параметры | `auction_type`, `term_days`, `tier`, `auction_time`, `settlement_code`, `first_leg_date`, `second_leg_date` |
| Объёмы/ставки | `total_deals_volume`, `demand_volume`, `cutoff_rate`, `min_rate`, `max_rate`, `weighted_average_rate`, `limit_deals_volume`, `weighted_average_limit_rate` |
| Производные | `cover_ratio`, `key_rate`, `rate_for_spread`, `rate_spread` |
| Сигналы | `Flag_Demand`, `MAD_score_cover`, `MAD_score_rate_spread`, `MAD_score_cover_tier`, `MAD_score_rate_spread_tier` |

**`m2_daily_profile`** (дневной профиль по срочности):

| Колонки | Смысл |
|---------|-------|
| `m2_short_active`, `m2_short_age_days`, `m2_short_available`, `m2_short_cover_mad`, `m2_short_ratespread_mad` | состояние коротких аукционов |
| `m2_base_active`, `m2_base_age_days`, `m2_base_available`, `m2_base_cover_mad`, `m2_base_ratespread_mad` | состояние основных аукционов |
| `m2_long_active`, `m2_long_age_days`, `m2_long_available` | длинные аукционы |
| `m2_short_share_w252/w63`, `m2_term_slope_w252/w63`, `m2_term_slope_available_*` | доля коротких и наклон срочности (скользящие окна) |

---

## 3. Whitelist (что отобрано в LSI)

```python
M2_FEATURES = ["m2_auction_flag", "m2_Flag_Demand", "m2_base_cover_mad",
               "m2_cutoff_spread", "m2_cutoff_spread_available",
               "m2_short_active30", "m2_days_since_short"]
```

| Признак | Почему в whitelist |
|---------|--------------------|
| `m2_base_cover_mad` | аномальная переподписка **основного** аукциона — ядро сигнала спроса на ликвидность |
| `m2_cutoff_spread` | **премия по отсечению**: `cutoff_rate − RUONIA`. Банки платят выше рынка → дефицит |
| `m2_cutoff_spread_available` | флаг свежести премии (был ли недавно base-аукцион) — отличает «0 = норма» от «нет данных» |
| `m2_short_active30` | активны ли короткие РЕПО за 30 дн. — регулятор тушит локальный дефицит |
| `m2_days_since_short` | сколько дней с последнего короткого аукциона (рецентность стресса) |
| `m2_auction_flag`, `m2_Flag_Demand` | контекст: был ли аукцион и флаг повышенного спроса |

**Почему НЕ вошли:**
- `MAD_score_rate_spread`, `*_tier`-варианты — дублируют сигнал ставки/переподписки в менее робастной форме; `cutoff_spread` к RUONIA экономически чище.
- `m2_term_slope_*`, `m2_short_share_*` — наклон срочности и доли информативны для исследования, но шумны и слабо коррелируют со стрессом → контекст, не вход.
- Сырые объёмы/ставки — нестационарны (зависят от лимитов ЦБ), не несут самостоятельной аномалии.

---

## 4. Методика расчёта

### `m2_base_cover_mad` / `m2_short_age_days`
Берутся из `m2_daily_profile`: `base_cover_mad` — MAD-аномалия переподписки основного аукциона; `short_age_days` — возраст последнего короткого аукциона.

### `m2_cutoff_spread` (премия по отсечению) — as-of join с допуском
```python
base = m2_features[tier == "base"]                     # только основные аукционы
cs   = base.cutoff_rate − ruonia_rate                  # премия к рыночной ставке
# протягиваем последнюю премию вперёд, но не дольше 7 дней (свежесть аукциона)
m2_cutoff_spread = merge_asof(calendar, cs, direction="backward",
                              tolerance=7 дней)
m2_cutoff_spread_available = (премия определена) ? 1 : 0
```
`tolerance=7д` (`CUTOFF_SPREAD_CAP_DAYS`) не даёт «тянуть» устаревшую премию дольше недельного цикла РЕПО.

### События коротких РЕПО
```python
m2_short_active30   = (m2_short_age_days ≤ 30) ? 1 : 0        # активность за месяц
m2_days_since_short = min(m2_short_age_days.fillna(365), 90)  # рецентность, кап 90 дн.
```

### MAD-score
Все `*_mad` считаются робастной скользящей z-оценкой (медиана + MAD), см. методику в [M1](M1_Reserves.md#4-методика-расчёта).

> Код расчёта фич — `backend/src/services/m2_feature_builder.py` и `honest_feature_builder.py`; исходные ряды — [`../data/README_DATA.md`](../data/README_DATA.md).
