from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.src.downloaders.cbr_budget_funds_downloader import download_cbr_budget_funds
from backend.src.downloaders.cbr_liquidity_downloader import download_cbr_liquidity
from backend.src.downloaders.roskazna_treasury_deposits_downloader import (
    prepare_roskazna_treasury_deposits,
)

from backend.src.parsers.cbr_budget_funds import parse_cbr_budget_funds
from backend.src.parsers.cbr_budget_funds import save_csv as save_cbr_budget_funds_csv
from backend.src.parsers.cbr_liquidity import parse_cbr_liquidity
from backend.src.parsers.cbr_liquidity import save_csv as save_cbr_liquidity_csv
from backend.src.parsers.roskazna_treasury_deposits import (
    parse_roskazna_treasury_deposits,
)
from backend.src.parsers.roskazna_treasury_deposits import (
    save_csv as save_roskazna_treasury_deposits_csv,
)

from backend.src.services.m5_dataset_builder import build_m5_dataset
from backend.src.services.m5_dataset_builder import save_csv as save_m5_dataset_csv
from backend.src.services.m5_feature_builder import build_m5_features
from backend.src.services.m5_feature_builder import save_csv as save_m5_features_csv
from backend.src.services.m5_feature_builder import save_parquet as save_m5_features_parquet


def run_m5_pipeline() -> None:
    """Запускает полный пайплайн подготовки данных М5.

    Сбой Росказны не отменяет обновление данных ЦБ: сначала доводим до конца всё,
    что скачалось, и только потом поднимаем ошибку источника. Так свежая дневная
    ликвидность попадает в processed, а шаг всё равно становится красным —
    вместо прежнего «ok» на устаревшем кеше Росказны.
    """
    print("Скачиваем бюджетные средства на счетах банков с сайта ЦБ")
    download_cbr_budget_funds()

    print("Скачиваем дневную таблицу ликвидности банковского сектора ЦБ")
    download_cbr_liquidity()

    print("Проверяем XML-файлы Росказны по депозитам ЕКС")
    roskazna_files: list = []
    roskazna_error: Exception | None = None
    try:
        roskazna_files = prepare_roskazna_treasury_deposits()
    except Exception as error:  # noqa: BLE001 — источник изолируем, но не глотаем
        roskazna_error = error
        print(f"ОШИБКА источника Росказны: {error}")
        print("Продолжаем на ранее сохранённых XML; шаг будет помечен ошибкой в конце")

    print("Обрабатываем бюджетные средства на счетах банков")
    budget_rows = parse_cbr_budget_funds()
    save_cbr_budget_funds_csv(budget_rows)

    print("Обрабатываем дневную таблицу ликвидности банковского сектора")
    liquidity_rows = parse_cbr_liquidity()
    save_cbr_liquidity_csv(liquidity_rows)

    print("Обрабатываем депозиты ЕКС Росказны")
    roskazna_rows = parse_roskazna_treasury_deposits()
    save_roskazna_treasury_deposits_csv(roskazna_rows)

    print("Собираем датасет М5")
    m5_rows = build_m5_dataset()
    save_m5_dataset_csv(m5_rows)

    print("Собираем признаки М5")
    m5_feature_rows = build_m5_features()
    save_m5_features_csv(m5_feature_rows)
    save_m5_features_parquet(m5_feature_rows)

    print(f"Готово, строк по бюджетным средствам: {len(budget_rows)}")
    print(f"Готово, строк по ликвидности: {len(liquidity_rows)}")
    print(f"Готово, XML-файлов Росказны: {len(roskazna_files)}")
    print(f"Готово, строк по депозитам Росказны: {len(roskazna_rows)}")
    print(f"Готово, строк в датасете М5: {len(m5_rows)}")
    print(f"Готово, строк в признаках М5: {len(m5_feature_rows)}")

    if roskazna_error is not None:
        raise RuntimeError(
            "Данные ЦБ обновлены, но депозиты ЕКС Росказны остались на прежней дате: "
            f"{roskazna_error}"
        ) from roskazna_error


def main() -> None:
    """Запускает пайплайн М5"""
    run_m5_pipeline()


if __name__ == "__main__":
    main()
