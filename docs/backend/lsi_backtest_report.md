# Бэктест LSI

Бэктест считается point-in-time: для даты `t` модель обучается только на данных до `t`, после чего LSI считается на строке `t`. Global использует expanding window, Local — rolling window 365 дней.

## Итоги по стресс-эпизодам

### Декабрь 2014

- Global: максимум 63.08 на 2014-12-10, статус `КРАСНЫЙ (Стресс ликвидности)`, драйверы: m1_signal, m1_signal_final, m4_Tax_Day_Strict
- Local: максимум 63.08 на 2014-12-10, статус `КРАСНЫЙ (Стресс ликвидности)`, драйверы: m1_signal, m1_signal_final, m4_Tax_Day_Strict

### Февраль-март 2022

- Global: максимум 100.00 на 2022-02-09, статус `КРАСНЫЙ (Стресс ликвидности)`, драйверы: m1_spread_delta_mad_score, m1_spread_mad_score, m4_Tax_Week_Flag
- Local: максимум 100.00 на 2022-03-02, статус `КРАСНЫЙ (Стресс ликвидности)`, драйверы: m5_cbr_liquidity_stress_mad_score, m1_ruonia_mad_score, m1_spread_relative_mad_score

### Август 2023

- Global: максимум 64.98 на 2023-08-03, статус `КРАСНЫЙ (Стресс ликвидности)`, драйверы: m1_reserve_load_mad_score, m4_Tax_Week_Flag, m4_Seasonal_Factor_raw
- Local: максимум 100.00 на 2023-08-01, статус `КРАСНЫЙ (Стресс ликвидности)`, драйверы: m1_reserve_load_mad_score, m1_spread_relative_mad_score, m1_signal_final

## Анализ чувствительности ±20%

Анализ чувствительности считается на дате максимального LSI внутри каждого стресс-эпизода. Для каждого модуля признаки этого модуля умножаются на 0.8 и 1.2, затем пересчитывается LSI тем же point-in-time artifact.

### 2014-12-10
- global M1 x0.8: 63.08 (delta +0.00)
- global M1 x1.2: 63.25 (delta +0.17)
- global M2 x0.8: 63.08 (delta +0.00)
- global M2 x1.2: 63.08 (delta +0.00)
- global M3 x0.8: 63.08 (delta +0.00)
- global M3 x1.2: 63.08 (delta +0.00)
- global M4 x0.8: 63.38 (delta +0.30)
- global M4 x1.2: 65.23 (delta +2.15)
- global M5 x0.8: 62.86 (delta -0.22)
- global M5 x1.2: 63.50 (delta +0.42)
- local M1 x0.8: 63.08 (delta +0.00)
- local M1 x1.2: 63.25 (delta +0.17)
- local M2 x0.8: 63.08 (delta +0.00)
- local M2 x1.2: 63.08 (delta +0.00)
- local M3 x0.8: 63.08 (delta +0.00)
- local M3 x1.2: 63.08 (delta +0.00)
- local M4 x0.8: 63.38 (delta +0.30)
- local M4 x1.2: 65.23 (delta +2.15)
- local M5 x0.8: 62.86 (delta -0.22)
- local M5 x1.2: 63.50 (delta +0.42)

### 2022-02-09
- global M1 x0.8: 100.00 (delta +0.00)
- global M1 x1.2: 100.00 (delta +0.00)
- global M2 x0.8: 100.00 (delta +0.00)
- global M2 x1.2: 100.00 (delta +0.00)
- global M3 x0.8: 99.61 (delta -0.39)
- global M3 x1.2: 100.00 (delta +0.00)
- global M4 x0.8: 100.00 (delta +0.00)
- global M4 x1.2: 100.00 (delta +0.00)
- global M5 x0.8: 100.00 (delta +0.00)
- global M5 x1.2: 100.00 (delta +0.00)

### 2022-03-02
- local M1 x0.8: 99.50 (delta -0.50)
- local M1 x1.2: 100.00 (delta +0.00)
- local M2 x0.8: 100.00 (delta +0.00)
- local M2 x1.2: 100.00 (delta +0.00)
- local M3 x0.8: 100.00 (delta +0.00)
- local M3 x1.2: 100.00 (delta +0.00)
- local M4 x0.8: 100.00 (delta +0.00)
- local M4 x1.2: 100.00 (delta +0.00)
- local M5 x0.8: 99.01 (delta -0.99)
- local M5 x1.2: 100.00 (delta +0.00)

### 2023-08-01
- local M1 x0.8: 100.00 (delta +0.00)
- local M1 x1.2: 100.00 (delta +0.00)
- local M2 x0.8: 100.00 (delta +0.00)
- local M2 x1.2: 100.00 (delta +0.00)
- local M3 x0.8: 100.00 (delta +0.00)
- local M3 x1.2: 100.00 (delta +0.00)
- local M4 x0.8: 100.00 (delta +0.00)
- local M4 x1.2: 100.00 (delta +0.00)
- local M5 x0.8: 100.00 (delta +0.00)
- local M5 x1.2: 100.00 (delta +0.00)

### 2023-08-03
- global M1 x0.8: 64.81 (delta -0.17)
- global M1 x1.2: 65.11 (delta +0.13)
- global M2 x0.8: 64.98 (delta +0.00)
- global M2 x1.2: 64.98 (delta +0.00)
- global M3 x0.8: 64.98 (delta +0.00)
- global M3 x1.2: 64.98 (delta +0.00)
- global M4 x0.8: 66.76 (delta +1.78)
- global M4 x1.2: 63.87 (delta -1.11)
- global M5 x0.8: 64.52 (delta -0.46)
- global M5 x1.2: 65.30 (delta +0.32)
