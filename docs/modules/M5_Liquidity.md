# 💧 M5 — Ликвидность ЦБ и Казначейство (ЕКС)

## 1. Описание модуля

**Экономический смысл.** M5 смотрит на **баланс операций ЦБ с банковским сектором** — самый прямой «термометр» структурной ликвидности. Логика балансовая:

- **Требования ЦБ к банкам** (`claims`) растут, когда банки **активно занимают у регулятора** — классический признак **структурного дефицита** ликвидности;
- **Обязательства ЦБ перед банками** (`liab`) растут, когда у банков **избыток** и они паркуют деньги на депозитах/в КОБР (профицит);
- использование **постоянных механизмов** (`repo_standing`, `secured_standing`) — дорогих «окон последней инстанции» — всплеск означает, что банкам не хватило аукционного лимита и они идут на премиальные постоянные операции (острый стресс);
- активность **аукционов Росказна** (размещение средств ЕКС на депозитах банков) и число заявителей — индикатор того, как бюджет управляет временно свободными средствами.

**За чем следим:** аномальный рост требований ЦБ, всплеск постоянных механизмов рефинансирования, сжатие обязательств, активность заявителей Росказна. M5 — стабильный вклад ≈20%.

---

## 2. Полный пул фичей

**`m5_features`** (инжиниринг старого набора):

| Группа | Колонки |
|--------|---------|
| Структурная ликвидность | `liquidity_deficit_surplus_bln_rub_lag_1d`, `..._change_1d`, `..._change_5d`, `cbr_liquidity_stress_mad_score`, `cbr_liquidity_drain_mad_score` |
| Бюджетные средства | `budget_funds_total_mln_rub_lagged` (+ `_change`/`_pct_change`), `budget_funds_rub_mln_rub_lagged`, `budget_funds_rub_share_lagged` |
| Росказна (потоки) | `roskazna_demand_volume_*`, `roskazna_cover_ratio_*`, `roskazna_bidders_count_*`, `roskazna_first/second_leg_*`, `roskazna_net_flow_by_legs`, `..._rolling_7/14/30d`, `roskazna_net_flow_stress_mad_score` |
| Флаги | `roskazna_auction_day_flag_lag_1d`, `Flag_Budget_Drain`, `days_since_last_roskazna_auction` |

**Honest-признаки `m5x_*`** (строятся в `honest_feature_builder` из сырого `cbr_liquidity` и `roskazna_treasury_deposits`): `m5x_claims`, `m5x_liab`, `m5x_repostd`, `m5x_secured`, `m5x_rk_bidders`.

---

## 3. Whitelist (что отобрано в LSI)

```python
M5_GLOBAL_FEATURES = ["m5x_claims", "m5x_liab", "m5x_repostd", "m5x_secured"]   # Global
M5_LOCAL_ONLY      = ["m5x_rk_bidders"]                                         # только Local
```

| Признак | Источник | Почему в whitelist |
|---------|----------|--------------------|
| `m5x_claims` | `cbr_claims_standard_instruments` | требования ЦБ к банкам — **прямой сигнал дефицита** (банки занимают) |
| `m5x_liab` | `cbr_liabilities_standard_instruments` | обязательства ЦБ — зеркальный сигнал профицита (банки паркуют) |
| `m5x_repostd` | `repo_fx_swap_standing` | постоянное РЕПО/своп — премиальное «окно», всплеск = острый стресс |
| `m5x_secured` | `secured_loans_standing` | обеспеченные кредиты standing — аналогично, дорогое рефинансирование |
| `m5x_rk_bidders` | `roskazna … bidders_count` | число заявителей Росказна — **только Local** (см. ниже) |

**Почему `rk_bidders` только в Local.** Детальные данные по заявителям Росказна доступны и наполнены лишь на **свежем окне**; на полной истории ряд разрежён/отсутствует и зашумил бы Global. Поэтому kind-aware whitelist: `LOCAL_WHITELIST = GLOBAL_WHITELIST + m5x_rk_bidders` — единственное отличие Local от Global.

**Почему НЕ вошли (старый набор `m5_features`):**
- лаговые `roskazna_*_rolling_*`, `budget_funds_*`, `Flag_Budget_Drain` — это производные второго порядка (скользящие потоки, флаги-композиты) на разрежённых данных Росказны; они шумны и частично дублируют друг друга. Honest-подход берёт **первичные балансовые статьи ЦБ** (`claims/liab/standing`) — чище, стационарнее, прямее экономически.
- сырые уровни дефицита/профицита — нестационарны; их аномалию ловит MAD на балансовых статьях.

---

## 4. Методика расчёта

### `m5x_*` — MAD на балансовых статьях ЦБ
Каждый honest-признак M5 — это робастная скользящая MAD-аномалия (окно ~3 года, 756 торговых дней), протянутая на дневной календарь:

```python
def _dly(src, col):
    t["m"] = _mad_rolling(src[col])        # медиана+MAD, окно 756, пол 0.05, клип ±5
    return merge_asof(calendar, t, direction="backward")   # на дневную шкалу

m5x_claims  = _dly(cbr_liquidity, "cbr_claims_standard_instruments_bln_rub")
m5x_liab    = _dly(cbr_liquidity, "cbr_liabilities_standard_instruments_bln_rub")
m5x_repostd = _dly(cbr_liquidity, "repo_fx_swap_standing_bln_rub")
m5x_secured = _dly(cbr_liquidity, "secured_loans_standing_bln_rub")
```

### `m5x_rk_bidders`
```python
gb = roskazna.groupby("date")["bidders_count"].sum()   # заявители за день
m5x_rk_bidders = _dly(gb, "bidders_count")              # та же MAD-нормализация
```

### MAD-score
`_mad_rolling`: медиана и median absolute deviation по окну 756 дней (`min_periods=120`), пол MAD `0.05`, клип `±5` — устойчиво к кризисным выбросам. Подробнее — [M1](M1_Reserves.md#4-методика-расчёта).

> Код расчёта фич — `backend/src/services/m5_feature_builder.py` и `honest_feature_builder.py`; исходные ряды — [`../data/README_DATA.md`](../data/README_DATA.md).
