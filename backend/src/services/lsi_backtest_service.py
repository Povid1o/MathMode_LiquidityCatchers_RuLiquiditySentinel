from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from backend.src.services.lsi_prediction_service import _score_with_artifact
from backend.src.services.lsi_training_service import LOCAL_WINDOW_DAYS
from backend.src.services.lsi_training_service import MIN_LOCAL_ROWS
from backend.src.services.lsi_training_service import fit_lsi_artifact
from backend.src.services.lsi_training_service import load_final_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
DOCS_DIR = PROJECT_ROOT / "docs" / "backend"

BACKTEST_SCORES_CSV = DATA_DIR / "lsi_backtest_scores.csv"
BACKTEST_SCORES_PARQUET = DATA_DIR / "lsi_backtest_scores.parquet"
BACKTEST_SENSITIVITY_CSV = DATA_DIR / "lsi_backtest_sensitivity.csv"
BACKTEST_SENSITIVITY_PARQUET = DATA_DIR / "lsi_backtest_sensitivity.parquet"
BACKTEST_REPORT = DOCS_DIR / "lsi_backtest_report.md"

MIN_GLOBAL_ROWS = 120
SENSITIVITY_MULTIPLIERS = (0.8, 1.2)

BACKTEST_EPISODES = {
    "Декабрь 2014": ("2014-12-01", "2014-12-31"),
    "Февраль-март 2022": ("2022-02-01", "2022-03-31"),
    "Август 2023": ("2023-08-01", "2023-08-31"),
}


def _drivers_to_text(value: object) -> str:
    """Преобразует список драйверов в строку для CSV"""
    if isinstance(value, list):
        return ", ".join(str(item) for item in value)
    if value is None:
        return ""
    return str(value)


def _module_name(feature_name: str) -> str:
    """Возвращает имя модуля по префиксу признака"""
    return feature_name.split("_", 1)[0]


def _score_last_row(
    scoring_data: pd.DataFrame,
    artifact: dict[str, Any],
    *,
    prefix: str,
) -> dict[str, object]:
    """Считает LSI для последней строки с учетом EMA на прошлой истории"""
    scores = _score_with_artifact(scoring_data, artifact, prefix=prefix)
    scored_row = scores.iloc[-1]
    return {
        "value": float(scored_row[f"lsi_{prefix}"]),
        "status": str(scored_row[f"lsi_{prefix}_status"]),
        "drivers": _drivers_to_text(scored_row[f"top_drivers_{prefix}"]),
    }


def _fit_and_score(
    train_data: pd.DataFrame,
    current_row: pd.DataFrame,
    *,
    kind: str,
    window_days: int | None = None,
) -> tuple[dict[str, Any], dict[str, object]]:
    """Обучает point-in-time artifact и считает текущий LSI"""
    artifact, _ = fit_lsi_artifact(
        train_data.reset_index(drop=True),
        kind=kind,
        window_days=window_days,
    )
    scoring_data = pd.concat([train_data, current_row], ignore_index=True)
    score = _score_last_row(scoring_data, artifact, prefix=kind)
    return artifact, score


def _sensitivity_for_score_date(
    train_data: pd.DataFrame,
    current_row: pd.DataFrame,
    artifact: dict[str, Any],
    *,
    kind: str,
    base_value: float,
) -> list[dict[str, object]]:
    """Считает sensitivity ±20% по группам M1-M5 для одной даты"""
    features = artifact["features_list"]
    modules = sorted({_module_name(feature) for feature in features})
    result: list[dict[str, object]] = []

    for module in modules:
        module_features = [
            feature
            for feature in features
            if _module_name(feature) == module
        ]
        for multiplier in SENSITIVITY_MULTIPLIERS:
            perturbed_row = current_row.copy()
            for feature in module_features:
                perturbed_row[feature] = perturbed_row[feature].astype(float) * multiplier

            scoring_data = pd.concat([train_data, perturbed_row], ignore_index=True)
            score = _score_last_row(scoring_data, artifact, prefix=kind)
            result.append(
                {
                    "date": current_row["date"].iloc[0].date().isoformat(),
                    "model": kind,
                    "module": module.upper(),
                    "multiplier": multiplier,
                    "base_lsi": round(base_value, 4),
                    "perturbed_lsi": round(float(score["value"]), 4),
                    "delta_lsi": round(float(score["value"]) - base_value, 4),
                }
            )

    return result


def run_backtest(
    data: pd.DataFrame | None = None,
    *,
    episodes: dict[str, tuple[str, str]] = BACKTEST_EPISODES,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Считает честный rolling/expanding backtest по стресс-эпизодам"""
    if data is None:
        data = load_final_dataset()

    data = data.copy()
    data["date"] = pd.to_datetime(data["date"])
    data = data.sort_values("date").reset_index(drop=True)

    score_rows: list[dict[str, object]] = []
    sensitivity_rows: list[dict[str, object]] = []
    artifacts_for_sensitivity: dict[tuple[str, str], tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], float]] = {}

    for episode_name, (start_text, end_text) in episodes.items():
        start_date = pd.Timestamp(start_text)
        end_date = pd.Timestamp(end_text)
        episode_dates = data.loc[
            (data["date"] >= start_date) & (data["date"] <= end_date),
            "date",
        ]

        for score_date in episode_dates:
            current_row = data[data["date"] == score_date].copy()
            train_global = data[data["date"] < score_date].copy()
            if len(train_global) < MIN_GLOBAL_ROWS:
                continue

            global_artifact, global_score = _fit_and_score(
                train_global,
                current_row,
                kind="global",
            )

            local_start = score_date - pd.Timedelta(days=LOCAL_WINDOW_DAYS)
            train_local = data[
                (data["date"] < score_date) & (data["date"] >= local_start)
            ].copy()

            local_score: dict[str, object] | None = None
            local_artifact: dict[str, Any] | None = None
            if len(train_local) >= MIN_LOCAL_ROWS:
                local_artifact, local_score = _fit_and_score(
                    train_local,
                    current_row,
                    kind="local",
                    window_days=LOCAL_WINDOW_DAYS,
                )

            score_row = {
                "date": score_date.date().isoformat(),
                "episode": episode_name,
                "lsi_global_backtest": round(float(global_score["value"]), 2),
                "global_status": global_score["status"],
                "global_drivers": global_score["drivers"],
                "global_train_rows": len(train_global),
                "lsi_local_backtest": None,
                "local_status": None,
                "local_drivers": "",
                "local_train_rows": len(train_local),
            }

            if local_score is not None:
                score_row["lsi_local_backtest"] = round(float(local_score["value"]), 2)
                score_row["local_status"] = local_score["status"]
                score_row["local_drivers"] = local_score["drivers"]

            score_rows.append(score_row)

            current_key = (episode_name, "global")
            previous = artifacts_for_sensitivity.get(current_key)
            if previous is None or float(global_score["value"]) > previous[3]:
                artifacts_for_sensitivity[current_key] = (
                    train_global,
                    current_row,
                    global_artifact,
                    float(global_score["value"]),
                )

            if local_artifact is not None and local_score is not None:
                current_key = (episode_name, "local")
                previous = artifacts_for_sensitivity.get(current_key)
                if previous is None or float(local_score["value"]) > previous[3]:
                    artifacts_for_sensitivity[current_key] = (
                        train_local,
                        current_row,
                        local_artifact,
                        float(local_score["value"]),
                    )

    for (_episode_name, kind), payload in artifacts_for_sensitivity.items():
        train_data, current_row, artifact, base_value = payload
        sensitivity_rows.extend(
            _sensitivity_for_score_date(
                train_data,
                current_row,
                artifact,
                kind=kind,
                base_value=base_value,
            )
        )

    scores = pd.DataFrame(score_rows)
    sensitivity = pd.DataFrame(sensitivity_rows)
    return scores, sensitivity


def save_backtest_outputs(
    scores: pd.DataFrame,
    sensitivity: pd.DataFrame,
    *,
    scores_csv_path: Path = BACKTEST_SCORES_CSV,
    scores_parquet_path: Path = BACKTEST_SCORES_PARQUET,
    sensitivity_csv_path: Path = BACKTEST_SENSITIVITY_CSV,
    sensitivity_parquet_path: Path = BACKTEST_SENSITIVITY_PARQUET,
) -> None:
    """Сохраняет backtest и sensitivity в CSV и parquet"""
    scores_csv_path.parent.mkdir(parents=True, exist_ok=True)
    scores.to_csv(scores_csv_path, index=False)
    scores.to_parquet(scores_parquet_path, index=False)
    sensitivity.to_csv(sensitivity_csv_path, index=False)
    sensitivity.to_parquet(sensitivity_parquet_path, index=False)


def build_backtest_report(
    scores: pd.DataFrame,
    sensitivity: pd.DataFrame,
    *,
    output_path: Path = BACKTEST_REPORT,
) -> None:
    """Сохраняет краткий markdown-отчет по backtest"""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Бэктест LSI",
        "",
        "Бэктест считается point-in-time: для даты `t` модель обучается только на данных до `t`, "
        "после чего LSI считается на строке `t`. Global использует expanding window, Local — "
        "rolling window 365 дней.",
        "",
        "## Итоги по стресс-эпизодам",
        "",
    ]

    for episode_name in BACKTEST_EPISODES:
        episode_scores = scores[scores["episode"] == episode_name].copy()
        if episode_scores.empty:
            lines.extend([f"### {episode_name}", "", "Нет рассчитанных строк.", ""])
            continue

        lines.extend([f"### {episode_name}", ""])
        for model_name, value_column, status_column, drivers_column in [
            ("Global", "lsi_global_backtest", "global_status", "global_drivers"),
            ("Local", "lsi_local_backtest", "local_status", "local_drivers"),
        ]:
            model_scores = episode_scores.dropna(subset=[value_column])
            if model_scores.empty:
                lines.append(f"- {model_name}: нет значений")
                continue

            max_row = model_scores.loc[model_scores[value_column].idxmax()]
            lines.append(
                f"- {model_name}: максимум {max_row[value_column]:.2f} "
                f"на {max_row['date']}, статус `{max_row[status_column]}`, "
                f"драйверы: {max_row[drivers_column]}"
            )
        lines.append("")

    lines.extend(
        [
            "## Анализ чувствительности ±20%",
            "",
            "Анализ чувствительности считается на дате максимального LSI внутри каждого стресс-эпизода. "
            "Для каждого модуля признаки этого модуля умножаются на 0.8 и 1.2, затем "
            "пересчитывается LSI тем же point-in-time artifact.",
            "",
        ]
    )

    if sensitivity.empty:
        lines.append("Sensitivity не рассчитан.")
    else:
        for date_value, date_rows in sensitivity.groupby("date"):
            lines.append(f"### {date_value}")
            for _, row in date_rows.sort_values(["model", "module", "multiplier"]).iterrows():
                lines.append(
                    f"- {row['model']} {row['module']} x{row['multiplier']}: "
                    f"{row['perturbed_lsi']:.2f} "
                    f"(delta {row['delta_lsi']:+.2f})"
                )
            lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    """Запускает backtest LSI и сохраняет результаты"""
    scores, sensitivity = run_backtest()
    save_backtest_outputs(scores, sensitivity)
    build_backtest_report(scores, sensitivity)
    print(f"Сохранено строк backtest: {len(scores)}")
    print(f"Сохранено строк sensitivity: {len(sensitivity)}")
    print(f"Файл: {BACKTEST_SCORES_CSV}")
    print(f"Файл parquet: {BACKTEST_SCORES_PARQUET}")
    print(f"Отчет: {BACKTEST_REPORT}")


if __name__ == "__main__":
    main()
