# Финальный ML dataset

## Статус

Добавлен builder финального дневного ML dataset из готовых feature-файлов М1-М5

Файлы результата:

```text
data/processed/final_ml_dataset.csv
data/processed/final_ml_dataset.parquet
```

Код:

```text
backend/src/services/final_dataset_builder.py
backend/src/pipelines/final_dataset_pipeline.py
ml/notebooks/final.ipynb
```

## Запуск

```bash
.venv/bin/python backend/src/pipelines/final_dataset_pipeline.py
```

## Правила гранулярности

Базовая дневная ось берется из `m5_features.csv`, то есть из дат таблицы ликвидности банковского сектора ЦБ

М1 присоединяется через `merge_asof` по `averaging_period_end`, а не по началу периода усреднения. Это снижает риск использования `ruonia_period_avg` и других периодных признаков до окончания периода

М2 присоединяется только по точной дате аукциона. В дни без 7-дневного аукциона аукционные stress-поля заполняются нулем, но сырые поля `cover_ratio`, `rate_spread` и ставки остаются пустыми

М3 присоединяется только по точной дате аукциона. Для ОФЗ добавлен отдельный стресс-признак:

```text
m3_cover_stress_score = -m3_MAD_score_cover
```

Это нужно потому, что для ОФЗ низкий cover ratio является стрессом, а высокий cover ratio обычно означает устойчивый спрос

М4 присоединяется по точной календарной дате. В финальный dataset включен `m4_Seasonal_Factor_raw`, но не включены центрированные сглаживания `tax_pressure_smoothed` и `Seasonal_Factor`, потому что они используют будущие дни в окне

М5 присоединяется по точной дате, потому что `m5_features.csv` уже собран на дневной сетке ЦБ и содержит лагированные budget/auction-demand признаки

В М5 добавлены нормализованные stress-признаки для LSI:

```text
m5_cbr_liquidity_stress_mad_score
m5_cbr_liquidity_drain_mad_score
m5_roskazna_net_flow_stress_mad_score
m5_Flag_Budget_Drain
```

Положительные значения MAD-score означают усиление стресса. `m5_Flag_Budget_Drain` включается при 14-дневном чистом оттоке средств Росказны из банков больше 300 млрд руб

## Проверки после сборки

На текущих данных:

```text
строк: 3077
колонок: 106
даты: 2014-02-03 — 2026-05-08
дубли дат: 0
```

CSV и parquet имеют одинаковые колонки и значения

## Ограничения

Финальный dataset сам по себе не обучает LSI. Обучение вынесено в отдельный pipeline:

```bash
.venv/bin/python backend/src/pipelines/lsi_training_pipeline.py
```

Результаты обучения:

```text
models/lsi_global_pipeline.joblib
models/lsi_local_pipeline.joblib
data/processed/lsi_scores.csv
data/processed/lsi_scores.parquet
```

LSI обучается только на согласованном whitelist стресс-признаков: MAD-score, флаги модулей, налоговые календарные признаки и M5 stress-признаки. Raw-level признаки вроде уровней резервов, объемов спроса и балансов бюджетных средств в LSI не подаются

`lsi_global_pipeline.joblib` обучается на всей истории и показывает режим относительно полного периода наблюдений

`lsi_local_pipeline.joblib` обучается на последнем 365-дневном окне и показывает локальную структуру текущего рынка

Важно: локальный LSI предназначен для текущего мониторинга. Историческая линия `lsi_local` внутри последнего окна пересчитывается моделью, обученной на всем этом окне, поэтому ее нельзя использовать как строгий point-in-time backtest без отдельной rolling-валидации

Для point-in-time проверки добавлен отдельный backtest pipeline:

```bash
.venv/bin/python backend/src/pipelines/lsi_backtest_pipeline.py
```

Результаты backtest:

```text
data/processed/lsi_backtest_scores.csv
data/processed/lsi_backtest_scores.parquet
data/processed/lsi_backtest_sensitivity.csv
data/processed/lsi_backtest_sensitivity.parquet
docs/backend/lsi_backtest_report.md
```

Backtest считается по стресс-эпизодам из ТЗ. Для даты `t` Global обучается только на данных до `t`, Local обучается на rolling-окне 365 дней до `t`

M5 flow-признаки по `first_leg_date` и `second_leg_date` оставлены как в `m5_features.csv`. Перед финальным backtest нужно согласовать cutoff прогноза: начало дня, конец дня или прогноз на следующий день

Для строгого point-in-time backtest желательно добавить календарь публикаций ЦБ и Минфина. Текущий М1 использует `averaging_period_end` как консервативную дату доступности, но это не заменяет фактическую дату публикации

M2 и M3 остаются sparse event-модулями. Нули в MAD-полях означают отсутствие события в этот день, а не фактическое значение источника

## LSI: whitelist стресс-признаков

LSI обучается только на следующих 26 согласованных стресс-признаках:

```text
# M1 — межбанковский рынок
m1_spread_mad_score, m1_spread_relative_mad_score, m1_spread_delta_mad_score,
m1_reserve_load_mad_score, m1_ruonia_mad_score, m1_flag_end_of_period,
m1_signal, m1_signal_final

# M2 — аукционы РЕПО
m2_Flag_Demand, m2_MAD_score_cover, m2_MAD_score_rate_spread, m2_auction_flag

# M3 — аукционы ОФЗ
m3_cover_stress_score, m3_yield_stress_score,
m3_Flag_Nedospros, m3_Flag_Perespros, m3_auction_flag

# M4 — налоговый календарь
m4_Tax_Week_Flag, m4_Tax_Day_Strict, m4_MAD_tax_pressure,
m4_MAD_tax_proximity, m4_Seasonal_Factor_raw

# M5 — ликвидность ЦБ и Росказна
m5_cbr_liquidity_stress_mad_score, m5_cbr_liquidity_drain_mad_score,
m5_roskazna_net_flow_stress_mad_score, m5_Flag_Budget_Drain
```

Raw-level признаки (уровни резервов, объемы спроса, балансы) в модель не подаются.
Метод отбора: `fixed_stress_whitelist` — фиксированный набор, согласованный с методологией ТЗ.

## LSI: вклад модулей M1-M5

LSI pipeline рассчитывает вклад каждого модуля M1-M5 методом **PCA-based weighted attribution**.

Метод (не SHAP, не причинная декомпозиция):

```text
structural_weight[j] = Σₖ  evr[k] · |components[k, j]|
feature_contrib[i, j] = |x_scaled[i, j]| · structural_weight[j]
contrib_pct[i, j] = feature_contrib[i, j] / Σⱼ feature_contrib[i, j] · 100%
module_contrib[i, m] = Σⱼ∈m contrib_pct[i, j]
```

Где: `evr` — explained_variance_ratio_, `components` — матрица PCA, `x_scaled` — StandardScaler output.

Интерпретация: метрика показывает, насколько активно признаки модуля m «нагружают» первые главные
компоненты в данной строке. Вклады нормированы до 100% по каждой строке.

Ограничение: вклад зависит от структуры PCA на текущем обучающем окне, а не от причинного влияния
модуля на итоговую LSI-оценку. Высокий вклад M4 в спокойный налоговый день означает, что признаки M4
активны относительно текущих PCA-весов, а не что налоговый фактор вызывает стресс.

Колонки в `lsi_scores.csv`:

```text
lsi_global_contrib_m1, lsi_global_contrib_m2, lsi_global_contrib_m3,
lsi_global_contrib_m4, lsi_global_contrib_m5
lsi_local_contrib_m1, lsi_local_contrib_m2, lsi_local_contrib_m3,
lsi_local_contrib_m4, lsi_local_contrib_m5
```

Сумма по M1-M5 для каждой строки и каждой модели = 100%.

## LSI: пороговые профили светофора

Пороги светофора хранятся в едином модуле конфигурации:

```text
backend/src/services/lsi_thresholds.py
```

Два профиля:

| Профиль | Зелёный | Красный | Global FP % | Декабрь 2014 | Event Recall Red (Global) |
|---------|---------|---------|-------------|--------------|--------------------------|
| **backtest_sensitive** (**production default**) | < 30 | ≥ 60 | ~10.5% | 23 жёлтых, 1 красный | **100%** |
| conservative (альтернатива) | < 40 | ≥ 70 | ~3.95% | 19 жёлтых, 0 красных | 33.3% |

**Production default** = `backtest_sensitive` (30 / 60). Выбор подтверждён куратором.

Бизнес-позиция: для задач раннего предупреждения стресса ликвидности **ошибка
пропуска кризисного эпизода дороже, чем ложная тревога**. Красный сигнал требует
ручного подтверждения аналитиком; пропущенный кризис не поддаётся коррекции.

Цена чувствительности (осознанная):
- Global FP rate ≈ 10.5% (~25–26 ложных красных дней в год вне кризисов)
- Local FP rate ≈ 47.1% на последнем 365-дневном окне (Local обучена в спокойный
  период — её шкала 0–100 не охватывает исторические кризисные уровни)

`conservative` (40 / 70) доступен как альтернатива с меньшим числом ложных тревог
(FP ~3.95%), но Event Recall Red = 33.3% — Декабрь 2014 и Август 2023 пропускаются.

Все сервисы (training, prediction, backtest) импортируют `get_lsi_status()` только из
`lsi_thresholds.py` — никаких дублирующих констант в коде.

## LSI: калибровка порогов светофора

Пороги откалиброваны на трёх исторических стресс-эпизодах методом
point-in-time rolling/expanding backtest.

```bash
.venv/bin/python backend/src/pipelines/lsi_threshold_calibration_pipeline.py
```

Результаты:

```text
data/processed/lsi_threshold_calibration.csv
data/processed/lsi_threshold_calibration.parquet
data/processed/lsi_threshold_metrics.csv
data/processed/lsi_threshold_metrics.parquet
docs/backend/lsi_threshold_calibration.md
```

Метод: для каждой пары порогов (зелёный_макс, красный_мин) подсчитывается:
- число дней с global/local LSI ≥ red_threshold в каждом стресс-эпизоде (detection rate)
- доля дней вне стресс-эпизодов с global LSI ≥ red_threshold (false positive rate)
- Event Recall Yellow / Red
- Lead Time Yellow / Red в 30-дневном окне до начала стресс-эпизода
- False Red Alerts/year вне размеченных стресс-окон

Рекомендованная пара выбирается алгоритмически:
1. Обязательное условие: оба главных эпизода (Декабрь 2014, Февраль-март 2022) дают
   хотя бы один красный день по Global LSI в backtest
2. Среди прошедших — минимальный global FP rate
3. При равном FP rate — более низкий red_threshold (чувствительнее)

Важная оговорка: backtest-пороги и production-пороги могут различаться. Backtest обучает
модели point-in-time, поэтому MinMaxScaler нормирует оценки относительно данных, доступных
на дату t. Production-модель нормирует относительно всей истории с 2014 по 2026. В результате
один и тот же кризисный период получает разные абсолютные значения LSI в backtest и production:
Декабрь 2014 в production-модели имеет max global LSI ≈ 34 (хуже эпизоды есть — 2022),
но в backtest-модели того же периода — max ≈ 63 (самый жёсткий кризис на виденных данных).

Это ограничение метода, а не ошибка. Подробнее см. `docs/backend/lsi_threshold_calibration.md`.
