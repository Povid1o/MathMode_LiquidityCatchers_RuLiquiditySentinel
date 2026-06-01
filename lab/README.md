# RU Liquidity Sentinel — Research Lab (`lab/`)

Рабочая лаборатория для **ручной** проверки гипотез по моделям Global и Local LSI.
Это **не** production-код и **не** финальный отчёт: ноутбуки читают существующие
артефакты, переобучают LSI-подобные модели **в памяти** и дают тебе крутилки для
экспериментов. Production-пайплайны (`backend/src/...`, `models/`, `data/processed/`)
**не меняются** — лаборатория только читает их.

## Как запускать

1. Запускай из **корня проекта** (где лежат `data/`, `backend/`, `models/`):
   ```bash
   cd <project_root>
   jupyter lab        # или jupyter notebook / VS Code
   ```
2. Открой любой ноутбук из `lab/`. Первая ячейка (`bootstrap`) сама находит корень
   проекта, делает `os.chdir` в него и импортирует `lab.utils as u`.
3. Все общие функции — в `lab/utils.py`. Меняй параметры в ячейке **Parameters**
   в начале каждого ноутбука и перезапускай.

### Зависимости
`numpy, pandas, matplotlib, seaborn, scikit-learn, scipy, joblib, nbformat`.
`plotly` **не обязателен** (графики на matplotlib/seaborn). Всё уже есть в окружении,
которым считается production.

## Порядок прохождения

Ноутбуки независимы, но логически идут так:

| # | Notebook | О чём |
|---|---|---|
| 00 | `00_data_inventory.ipynb` | Какие данные есть: файлы, даты, колонки, whitelist, missingness, частота обновления |
| 01 | `01_feature_distributions.ipynb` | Распределения 26 whitelist-фич, поиск dead/dup/const, корреляции |
| 02 | `02_current_lsi_reproduction.ipynb` | Воспроизведение текущей Global/Local через backend, сверка со scores, эпизоды |
| 03 | `03_global_ablation_lab.ipynb` | Ablation Global: без dead/dup, без PCA, разное число PCA, без M4/M5, моно-модули |
| 04 | `04_scale_and_thresholds_lab.ipynb` | production vs backtest шкала, дрейф по cutoff, ретроактивный/перцентильный scaling и пороги |
| 05 | `05_explainability_lab.ipynb` | PC1-drivers vs EVR-attribution рассогласование, loadings, драйверы на ключевых датах |
| 06 | `06_local_signal_lab.ipynb` | Local как tactical signal, RUONIA-keyrate proxy, circularity, baselines |
| 07 | `07_candidate_models_lab.ipynb` | Песочница кандидатов Global/Local + сводная таблица и заглушка supervised proxy |

Рекомендуемый первый проход: **00 → 02 → 03 → 04 → 06**, затем 01/05/07 по интересу.

## Структура

```
lab/
├── README.md            # этот файл
├── utils.py             # все reusable-функции (loaders, fit/score, explainability, proxy, plots)
├── 00_data_inventory.ipynb
├── 01_feature_distributions.ipynb
├── 02_current_lsi_reproduction.ipynb
├── 03_global_ablation_lab.ipynb
├── 04_scale_and_thresholds_lab.ipynb
├── 05_explainability_lab.ipynb
├── 06_local_signal_lab.ipynb
├── 07_candidate_models_lab.ipynb
├── src/                 # место для дополнительных модулей, если utils.py разрастётся
└── outputs/             # сюда сохраняются графики/таблицы (в git не коммитятся, кроме .gitkeep)
```

## Что лаборатория помогает проверить

- **Global** полезен как структурный anomaly detector, но: production-шкала ≠ backtest-шкала
  (Dec2014 на production = YELLOW, не RED); MinMaxScaler делает шкалу нестабильной при
  переобучении; `m1_flag_end_of_period` мёртвая; `m1_signal_final` дублирует `m1_signal`;
  объяснимость рассогласована (top_drivers по PC1 vs module_contributions по всем PC); PCA
  стоит проверить ablation'ом.
- **Local** в текущем виде — IsolationForest на последнем 365-дневном окне; **не доказан**
  как tactical directional signal. Прежняя идея `m1_ruonia_mad_score` как сильного forward-baseline
  оказалась **circular** (корреляция с текущим `|spread|`, а не с forward-направлением). Local
  исследуем через proxy-targets и baselines, а не внедряем сразу.

## Важные оговорки

- Все выводы внутри ноутбуков — **предварительные** (lab, не отчёт). Не делай категоричных
  заключений из одной ячейки.
- Артефакты в `models/*.joblib` могли быть сохранены другой версией sklearn — для чистых
  экспериментов ноутбуки **переобучают** модель через `u.fit_lsi_like_model(...)`, а не
  доверяют pickled-объектам.
- Тяжёлые выводы (графики/CSV) пишутся в `lab/outputs/` и **не** коммитятся.
- Параметры вынесены в отдельные ячейки: число PCA, EMA alpha, исключения фич, окна эпизодов,
  пороги, rolling-окна, горизонты T+1/T+7.
