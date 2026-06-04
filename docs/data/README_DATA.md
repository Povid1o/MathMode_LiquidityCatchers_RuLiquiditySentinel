# 🗄️ Data — сырые данные и скрейперы

Документация слоя данных: исходные таблицы и их колонки, источники, скрипты загрузки/парсинга и «костыли» (quirks), связанные с тем, как сайты отдают данные.

> Поток: `downloaders/` → `data/raw/<source>/` (html · xml · xlsx) → `parsers/` → `data/processed/*.csv` → feature builders → витрина DuckDB.
> Все источники — **публичные**: ЦБ РФ, Минфин, Росказна, ФНС. (CBonds не используется — аукционы ОФЗ берутся напрямую с сайта Минфина.)

---

## 1. Карта источников

| Таблица (processed) | Источник | Формат | Downloader | Parser |
|---------------------|----------|--------|-----------|--------|
| `required_reserves` | **ЦБ РФ** — обязательные резервы | XLSX | `required_reserves_downloader` | `parsers/required_reserves` |
| `ruonia` | **ЦБ РФ** — ставка RUONIA | XLSX | `ruonia_downloader` | `parsers/ruonia` |
| `keyrate` | **ЦБ РФ** — ключевая ставка | HTML | `keyrate_downloader` | `parsers/keyrate` |
| `repo` | **ЦБ РФ** — аукционы РЕПО | HTML (по дням) | `repo_downloader` | `parsers/repo` |
| `ofz_auctions` | **Минфин** — аукционы ОФЗ | HTML-индекс → XLSX | `ofz_auctions_downloader` | `parsers/ofz_auctions` |
| `cbr_liquidity` | **ЦБ РФ** — ликвидность банк. сектора | HTML | `cbr_liquidity_downloader` | `parsers/cbr_liquidity` |
| `cbr_budget_funds` | **ЦБ РФ** — бюджетные средства в банках | XLSX | `cbr_budget_funds_downloader` | `parsers/cbr_budget_funds` |
| `roskazna_treasury_deposits` | **Росказна** — депозиты ЕКС | HTML-архив → XML | `roskazna_treasury_deposits_downloader` | `parsers/roskazna_treasury_deposits` |
| *(календарь)* → `m4_features` | **ФНС** — налоговый календарь | opendata XML | `tax_calendar_downloader` | `parsers/tax_calendar` |

Точные URL-ы — в константах `SOURCE_URL`/`BASE_URL` каждого `*_downloader.py`.

---

## 2. Словарь колонок (raw / processed)

### `required_reserves` — обязательные резервы (ЦБ РФ)
Публикуется на **конец периода усреднения** (≈ месяц), не ежедневно.

| Колонка | Смысл |
|---------|-------|
| `date` | дата записи (начало периода) |
| `averaging_period_end` | дата окончания периода усреднения (дата доступности значения) |
| `actual_balances` | фактические остатки на корсчетах, млн руб. |
| `required_reserves_avg` | требуемый усреднённый объём резервов, млн руб. |
| `accounting_reserves` | резервы на спецсчетах |
| `averaging_period_days` | длина периода усреднения, дней |
| `spread` | `actual_balances − required_reserves_avg` (ключевой сигнал M1) |

### `ruonia` — ставка RUONIA (ЦБ РФ)
| Колонка | Смысл |
|---------|-------|
| `date` | дата |
| `ruonia_rate` | ставка RUONIA, % годовых |
| `transactions_volume` | объём сделок, млрд руб. |
| `transactions_count` | число сделок |
| `participants_count` | число участников |

### `keyrate` — ключевая ставка (ЦБ РФ)
| Колонка | Смысл |
|---------|-------|
| `date` | дата вступления ставки в силу |
| `key_rate` | ключевая ставка ЦБ, % годовых |

### `repo` — аукционы РЕПО (ЦБ РФ)
| Колонка | Смысл |
|---------|-------|
| `date` | дата аукциона |
| `auction_type` | тип (основной/тонкой настройки и т.п.) |
| `term_days` | срок РЕПО, дней |
| `auction_time` | время проведения |
| `total_deals_volume` | объём заключённых сделок |
| `weighted_average_rate` | средневзвешенная ставка |
| `settlement_code` | код расчётов |
| `demand_volume` | объём спроса (заявки) |
| `cutoff_rate` / `min_rate` / `max_rate` | ставка отсечения / мин / макс |
| `limit_deals_volume` | лимит/объём по лимиту |
| `weighted_average_limit_rate` | средневзв. ставка по лимиту |
| `first_leg_date` / `second_leg_date` | даты первой/второй ноги |
| `cover_ratio` | `demand_volume / предложение` — переподписка |

### `ofz_auctions` — аукционы ОФЗ (Минфин)
| Колонка | Смысл |
|---------|-------|
| `auction_date` / `published_date` | дата аукциона / публикации |
| `document_title` | заголовок документа Минфина |
| `auction_format` | формат аукциона |
| `issue` / `security_type` | выпуск / тип бумаги (ОФЗ-ПД/ПК/ИН) |
| `maturity_date` / `days_to_maturity` | погашение / дней до погашения |
| `offered_amount` | предложенный объём |
| `demand_amount` | объём спроса |
| `placed_amount` | размещённый объём |
| `proceeds_amount` | выручка |
| `cutoff_price` / `weighted_average_price` | цена отсечения / средневзв. |
| `cutoff_yield` / `weighted_average_yield` | доходность отсечения / средневзв. |
| `official_coefficient` | официальный коэффициент Минфина |
| `cover_ratio` | `demand / offered` — спрос/предложение |
| `placement_ratio` | `placed / offered` — доля размещения |
| `source_url` / `source_file` | трассировка: исходный документ и файл |

### `cbr_liquidity` — ликвидность банковского сектора (ЦБ РФ)
Дневной баланс операций ЦБ — основной источник honest-фич M5 (`m5x_*`).

| Колонка | Смысл |
|---------|-------|
| `date` | дата |
| `liquidity_deficit_surplus_bln_rub` | структурный дефицит(−)/профицит(+), млрд руб. |
| `liquidity_deficit_surplus_without_correspondent_accounts_bln_rub` | то же без корсчетов |
| `cbr_claims_standard_instruments_bln_rub` | **требования ЦБ к банкам** (стандартные инструменты) → `m5x_claims` |
| `repo_fx_swap_auctions_bln_rub` / `secured_loans_auctions_bln_rub` | аукционные РЕПО/валютный своп / обеспеченные кредиты |
| `repo_fx_swap_standing_bln_rub` | РЕПО/своп по постоянным механизмам → `m5x_repostd` |
| `secured_loans_standing_bln_rub` | обеспеченные кредиты standing → `m5x_secured` |
| `cbr_liabilities_standard_instruments_bln_rub` | **обязательства ЦБ перед банками** → `m5x_liab` |
| `deposit_auctions_bln_rub` / `deposit_standing_bln_rub` | депозитные аукционы / постоянные |
| `cobr_bln_rub` | КОБР (облигации ЦБ) |
| `nonstandard_refundable_operations_bln_rub` | нестандартные возвратные операции |
| `correspondent_accounts_bln_rub` | корсчета |
| `required_reserves_avg_bln_rub` | усреднённые резервы |
| `source_url` / `source_file` | трассировка |

### `cbr_budget_funds` — бюджетные средства в банках (ЦБ РФ)
| Колонка | Смысл |
|---------|-------|
| `date` | дата |
| `currency_type` | валюта (рубли / всего) |
| `budget_funds_total_mln_rub` | всего бюджетных средств в банках, млн руб. |
| `federal_budget_funds_mln_rub` | федеральный бюджет |
| `regional_local_budget_funds_mln_rub` | региональные/местные |
| `other_budget_funds_mln_rub` | прочие |
| `extra_budgetary_funds_mln_rub` | внебюджетные фонды |
| `source_url` / `source_file` | трассировка |

### `roskazna_treasury_deposits` — депозиты ЕКС (Росказна)
Один XML = один аукцион операционного дня. 42 колонки; ключевые ниже.

| Колонка | Смысл |
|---------|-------|
| `auction_date` / `auction_id` | дата / id аукциона |
| `currency` | валюта |
| `max_volume_mln_rub` | максимальный объём размещения |
| `term_days` | срок депозита |
| `first_leg_date` / `second_leg_date` | размещение / возврат |
| `rate_type` / `base_floating_rate` / `min_spread` | тип ставки / база плавающей / мин. спред |
| `cutoff_rate` / `min_bid_rate` / `max_bid_rate` | ставка отсечения / мин / макс заявок |
| `demand_volume_mln_rub` | спрос |
| `accepted_volume_mln_rub` / `settled_volume_mln_rub` | принято / расчётный объём |
| `weighted_average_accepted_rate` | средневзв. принятая ставка |
| `bidders_count` / `accepted_bidders_count` | число заявителей / удовлетворённых → `m5x_rk_bidders` |
| `cover_ratio` / `accepted_ratio` / `settled_ratio` | переподписка / доля приёма / расчётов |
| *(+ тайминги аукциона, нетто, комментарии)* | служебные поля операционного дня |
| `source_url` / `source_file` | трассировка |

> Инжиниринговые таблицы `m1_features … m5_features`, `m2_daily_profile` документированы помодульно в [`../modules/`](../modules/) (M1–M5). Здесь — только исходные/распарсенные источники.

---

## 3. Логика скрейперов и quirks

### ЦБ РФ — XLSX (`required_reserves`, `cbr_budget_funds`)
Прямая ссылка на `.xlsx` (`download_file`), без авторизации. `cbr_budget_funds` берётся из публикации статистики `…/02_29_Budget_all.xlsx`.

### ЦБ РФ — RUONIA (`ruonia`) ⚠️ нестандартный парсинг XLSX
- Скачивается через endpoint выгрузки: `…/Queries/UniDbQuery/DownloadExcel/14315` с date-параметрами.
- **Quirk:** парсер `parsers/ruonia.py` читает XLSX **вручную** через `zipfile.ZipFile` + `xml.etree.ElementTree` (а не pandas/openpyxl): разбирает `sharedStrings`, ссылки на ячейки (`A1`-нотация) и конвертирует **Excel-серийные даты** (`_excel_date_to_date`). Так сделано ради контроля над формами выгрузки ЦБ.

### ЦБ РФ — HTML-таблицы (`keyrate`, `repo`, `cbr_liquidity`)
Парсятся **самописными `HTMLParser`-классами** (стандартная библиотека), без BeautifulSoup:
- `keyrate`: класс `_TableParser` обходит `<table>/<tr>/<td>`, считает уровень вложенности таблиц и берёт **таблицу первого уровня** (на странице ЦБ есть вложенные таблицы-обёртки).
- `cbr_liquidity`: `_LiquidityTableParser` ищет именно `<table class="data">` и аккумулирует текст ячеек.
- `repo`: ⚠️ данные отдаются **по дням** — `repo_downloader` качает отдельные HTML-страницы за каждую дату в `data/raw/repo/daily/<date>.html` (этот каталог в `.gitignore`), парсер агрегирует их в один CSV.

### Минфин — ОФЗ (`ofz_auctions`) ⚠️ двухступенчатый разбор
1. Скачивается **HTML-индекс** документов аукционов (`SOURCE_URL` на minfin.gov.ru) → `data/raw/ofz_auctions/index.html`.
2. `parse_document_index` достаёт из индекса ссылки на **`.xlsx`-файлы** аукционов и качает их в `data/raw/ofz_auctions/files/` (с `User-Agent`).
3. `parsers/ofz_auctions.py` (openpyxl, `data_only=True, read_only=True`) разбирает каждый XLSX.
- **Quirks:** заголовок таблицы **не на первой строке** — его ищет `_find_header_row`; имена колонок нормализуются (`_normalize_header`) и матчатся по словарю → устойчивость к разнице формулировок в файлах разных лет. Один документ = один аукцион; трассировка через `source_url`/`source_file`.

### Росказна — депозиты ЕКС (`roskazna_treasury_deposits`) ⚠️ SSL + пагинация архива
- Архив на `roskazna.gov.ru` постраничный: `…?filter_year=<год>&page=<n>`. `download_roskazna_html_pages` обходит годы/страницы и останавливается по маркеру последней страницы.
- Маркер `id="start-files-list"` (`ARCHIVE_MARKER`) делит страницу на **текущий блок** и **архивную таблицу** — ссылки собираются из обоих, затем дедуплицируются.
- Данные — **XML операционного дня** (`storage/operation-day-files/*.XML`), один файл = один аукцион.
- ⚠️ **Главный костыль — SSL.** Сертификат сайта не проходит проверку в Python (`CERTIFICATE_VERIFY_FAILED: unable to get local issuer certificate`). `prepare_roskazna_treasury_deposits` сначала пробует обычную загрузку, и при провале **автоматически повторяет с отключённой проверкой SSL** (`allow_unverified_ssl`). Текущий год перекачивается принудительно (`force=True`), чтобы видеть свежие депозиты.
- ⚠️ Историческая ловушка: раньше `prepare_*` **не скачивал** HTML-архив и полагался на кеш — на чистой машине падал с `FileNotFoundError`. Теперь скачивает сам (см. CLI-флаги ниже).

### ФНС — налоговый календарь (`tax_calendar` → M4) ⚠️ opendata + XSD
- Точка входа — opendata-страница `nalog.gov.ru/opendata/7707329152-kalendar/`.
- Парсер по **regex** находит ссылки на `data-*.xml` (данные) и `structure-*.xsd` (схема) на `data.nalog.ru`, качает последнюю версию в `tax_calendar.xml` и складывает релизы в `releases/`.
- M4 — детерминированный **overlay**: из календаря строятся флаги налоговых дней/недель/предналоговых периодов (в индекс не входит).

---

## 4. Как взаимодействовать со скрейперами

**Через пайплайн модуля** (рекомендуется — download → parse → features):

```bash
python -m backend.src.pipelines.m1_pipeline   # резервы + RUONIA
python -m backend.src.pipelines.m5_pipeline   # ликвидность ЦБ + бюджет + Росказна
```

**Полное обновление всего** (как кнопка «Данные ⚙️»):

```bash
python -m backend.src.pipelines.refresh_pipeline
```

**Точечный запуск загрузчика** (например, Росказна с явным диапазоном лет и SSL-флагом):

```bash
python -m backend.src.downloaders.roskazna_treasury_deposits_downloader \
        --years 2021-2026 --allow-unverified-ssl
```

**Правила:**
- сырьё пишется в `data/raw/<source>/`, результат парсинга — в `data/processed/<table>.csv`;
- крупные/регенерируемые дампы (`data/raw/repo/daily/`, `roskazna_deposits/`, `roskazna_pages/`) — в `.gitignore`;
- после ручного обновления processed-файлов синхронизируйте витрину: `python -m backend.src.db.warehouse`;
- идемпотентность: загрузчики пропускают уже скачанные файлы (`force=False`), кроме принудительно обновляемого текущего года Росказны.

> Свежесть каждой таблицы видна на странице **Данные ⚙️** дашборда (manifest витрины: строки, диапазон дат, отставание от сегодня).
