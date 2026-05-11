from __future__ import annotations

import csv
import re
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
RAW_DIR = PROJECT_ROOT / "data/raw/treasury_funds/roskazna_deposits"
OUTPUT_FILE = PROJECT_ROOT / "data/processed/roskazna_treasury_deposits.csv"
SOURCE_URL = (
    "https://roskazna.gov.ru/finansovye-operacii/"
    "razmeshchenie-sredstv-edinogo-kaznachejskogo-scheta/"
    "razmeshchenie-sredstv-edinogo-kaznachejskogo-scheta-na-bankovskih-depozitah"
)

OUTPUT_COLUMNS = [
    "auction_date",
    "auction_id",
    "currency",
    "funds_placed",
    "max_volume_mln_rub",
    "term_days",
    "first_leg_date",
    "second_leg_date",
    "rate_type",
    "min_rate",
    "base_floating_rate",
    "min_spread",
    "deposit_type",
    "min_order_mln_rub",
    "max_orders_per_bank",
    "order_form",
    "place",
    "bidding_time",
    "prediction_time",
    "conclusion_time",
    "registry_time",
    "cutoff_time",
    "pay_time",
    "cutoff_rate",
    "demand_volume_mln_rub",
    "accepted_volume_mln_rub",
    "settled_volume_mln_rub",
    "weighted_average_accepted_rate",
    "min_bid_rate",
    "max_bid_rate",
    "bidders_count",
    "accepted_bidders_count",
    "random_time_seconds",
    "end_of_extension_period",
    "netting",
    "deposit_contracts_time",
    "comment",
    "cover_ratio",
    "accepted_ratio",
    "settled_ratio",
    "source_url",
    "source_file",
]


def _format_date(value: str | None) -> str:
    """Преобразует дату Росказны в формат DD-MM-YYYY"""
    text = "" if value is None else value.strip()
    if not text:
        return ""
    return datetime.strptime(text, "%d.%m.%Y").strftime("%d-%m-%Y")


def _to_float(value: str | None) -> float | None:
    """Преобразует строковое значение Росказны в число с плавающей точкой"""
    text = "" if value is None else value.strip()
    text = text.replace("\xa0", "").replace(" ", "").replace(",", ".")
    if text in {"", "-", "—"}:
        return None
    return float(text)


def _to_int(value: str | None) -> int | None:
    """Преобразует строковое значение Росказны в целое число"""
    number = _to_float(value)
    if number is None:
        return None
    return int(number)


def _clean_text(value: str | None) -> str:
    """Очищает текстовое значение Росказны"""
    text = "" if value is None else value
    return re.sub(r"\s+", " ", text.replace("\xa0", " ")).strip()


def _safe_ratio(numerator: float | None, denominator: float | None) -> float | None:
    """Считает отношение с защитой от пустых и нулевых значений"""
    if numerator is None or denominator in {None, 0}:
        return None
    return numerator / denominator


def _child_text(element: ET.Element, name: str) -> str:
    """Возвращает текст дочернего XML-элемента"""
    child = element.find(name)
    if child is None or child.text is None:
        return ""
    return child.text


def _parse_auction(element: ET.Element, source_file: str) -> dict[str, object]:
    """Парсит один аукцион Росказны из XML-элемента"""
    max_volume = _to_float(_child_text(element, "maxvol"))
    demand_volume = _to_float(_child_text(element, "totalbid"))
    accepted_volume = _to_float(_child_text(element, "totalaccept"))
    settled_volume = _to_float(_child_text(element, "totalsettle"))

    return {
        "auction_date": _format_date(_child_text(element, "aucdate")),
        "auction_id": _clean_text(_child_text(element, "id")),
        "currency": _clean_text(_child_text(element, "cur")),
        "funds_placed": _clean_text(_child_text(element, "FundsPlaced")),
        "max_volume_mln_rub": max_volume,
        "term_days": _to_int(_child_text(element, "term")),
        "first_leg_date": _format_date(_child_text(element, "firstdate")),
        "second_leg_date": _format_date(_child_text(element, "seconddate")),
        "rate_type": _clean_text(_child_text(element, "ratetype")),
        "min_rate": _to_float(_child_text(element, "minrate")),
        "base_floating_rate": _clean_text(_child_text(element, "baseflrate")),
        "min_spread": _to_float(_child_text(element, "minspread")),
        "deposit_type": _clean_text(_child_text(element, "depotype")),
        "min_order_mln_rub": _to_float(_child_text(element, "minorder")),
        "max_orders_per_bank": _to_int(_child_text(element, "maxcrorder")),
        "order_form": _clean_text(_child_text(element, "orderform")),
        "place": _clean_text(_child_text(element, "place")),
        "bidding_time": _clean_text(_child_text(element, "biddingtime")),
        "prediction_time": _clean_text(_child_text(element, "predtime")),
        "conclusion_time": _clean_text(_child_text(element, "conctime")),
        "registry_time": _clean_text(_child_text(element, "registrytime")),
        "cutoff_time": _clean_text(_child_text(element, "cutofftime")),
        "pay_time": _clean_text(_child_text(element, "paytime")),
        "cutoff_rate": _to_float(_child_text(element, "cutoffrate")),
        "demand_volume_mln_rub": demand_volume,
        "accepted_volume_mln_rub": accepted_volume,
        "settled_volume_mln_rub": settled_volume,
        "weighted_average_accepted_rate": _to_float(
            _child_text(element, "waacceptrate")
        ),
        "min_bid_rate": _to_float(_child_text(element, "minbidrate")),
        "max_bid_rate": _to_float(_child_text(element, "maxbidrate")),
        "bidders_count": _to_int(_child_text(element, "crbidders")),
        "accepted_bidders_count": _to_int(_child_text(element, "acceptcrbidders")),
        "random_time_seconds": _to_int(_child_text(element, "randtime")),
        "end_of_extension_period": _clean_text(
            _child_text(element, "endofextensionperiod")
        ),
        "netting": _clean_text(_child_text(element, "netting")),
        "deposit_contracts_time": _clean_text(
            _child_text(element, "depositcontractstime")
        ),
        "comment": _clean_text(_child_text(element, "Comment")),
        "cover_ratio": _safe_ratio(demand_volume, max_volume),
        "accepted_ratio": _safe_ratio(accepted_volume, demand_volume),
        "settled_ratio": _safe_ratio(settled_volume, max_volume),
        "source_url": SOURCE_URL,
        "source_file": source_file,
    }


def parse_roskazna_treasury_deposits(raw_dir: Path = RAW_DIR) -> list[dict[str, object]]:
    """Парсит XML-файлы Росказны по депозитам ЕКС"""
    rows: list[dict[str, object]] = []

    for path in sorted(raw_dir.iterdir()):
        if not path.is_file() or path.suffix.lower() != ".xml":
            continue

        root = ET.parse(path).getroot()
        for auction in list(root):
            rows.append(_parse_auction(auction, path.name))

    return sorted(
        rows,
        key=lambda row: (
            datetime.strptime(str(row["auction_date"]), "%d-%m-%Y"),
            int(str(row["auction_id"])),
        ),
    )


def save_csv(rows: list[dict[str, object]], output_path: Path = OUTPUT_FILE) -> None:
    """Сохраняет аукционы депозитов Росказны в CSV-файл"""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=OUTPUT_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """Запускает парсинг XML-файлов Росказны и сохраняет результат"""
    rows = parse_roskazna_treasury_deposits()
    save_csv(rows)
    print(f"Сохранено строк: {len(rows)}")
    print(f"Файл: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
