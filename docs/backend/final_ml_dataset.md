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

## Проверки после сборки

На текущих данных:

```text
строк: 3077
колонок: 102
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

`lsi_global_pipeline.joblib` обучается на всей истории и показывает режим относительно полного периода наблюдений

`lsi_local_pipeline.joblib` обучается на последнем 365-дневном окне и показывает локальную структуру текущего рынка

Важно: локальный LSI предназначен для текущего мониторинга. Историческая линия `lsi_local` внутри последнего окна пересчитывается моделью, обученной на всем этом окне, поэтому ее нельзя использовать как строгий point-in-time backtest без отдельной rolling-валидации

M5 flow-признаки по `first_leg_date` и `second_leg_date` оставлены как в `m5_features.csv`. Перед финальным backtest нужно согласовать cutoff прогноза: начало дня, конец дня или прогноз на следующий день

Для строгого point-in-time backtest желательно добавить календарь публикаций ЦБ и Минфина. Текущий М1 использует `averaging_period_end` как консервативную дату доступности, но это не заменяет фактическую дату публикации

M2 и M3 остаются sparse event-модулями. Нули в MAD-полях означают отсутствие события в этот день, а не фактическое значение источника
