"""Рисует две верифицированные диаграммы LSI в PNG (matplotlib).

Diagram 1 — Модель расчёта LSI (вертикальный поток сверху вниз).
Diagram 2 — Архитектура RU Liquidity Sentinel (свободный layout, источники вынесены).

Все факты сверены с исходниками проекта (см. сопроводительный ответ).
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
from matplotlib.lines import Line2D

plt.rcParams["font.family"] = "DejaVu Sans"

# --- палитра (C4-стиль) ---
BLUE = "#2477B8"          # контейнеры
BLUE_DARK = "#0D3B66"     # person / акценты
GREEN = "#2E8B57"         # скоринг / индекс
ORANGE = "#C77A2B"        # explainability / overlay
GREY_BORDER = "#9aa7b3"
EXT = "#3A6EA5"           # внешние источники
TXT = "#FFFFFF"
EDGE_TXT = "#33414d"


def box(ax, x, y, w, h, title, subtitle="", desc="", fc=BLUE, tc=TXT,
        title_fs=15, sub_fs=9.5, desc_fs=10):
    """Скруглённый блок с заголовком/[Container]/описанием. (x,y)=левый-нижний угол."""
    p = FancyBboxPatch((x, y), w, h,
                       boxstyle="round,pad=0.0,rounding_size=0.7",
                       linewidth=1.2, edgecolor="#16334d", facecolor=fc, zorder=3)
    ax.add_patch(p)
    cx = x + w / 2
    cur = y + h - 1.15
    ax.text(cx, cur, title, ha="center", va="top", color=tc,
            fontsize=title_fs, fontweight="bold", zorder=4)
    cur -= 1.7
    if subtitle:
        ax.text(cx, cur, subtitle, ha="center", va="top", color=tc,
                fontsize=sub_fs, style="italic", alpha=0.92, zorder=4)
        cur -= 1.5
    if desc:
        ax.text(cx, cur, desc, ha="center", va="top", color=tc,
                fontsize=desc_fs, zorder=4, linespacing=1.25)
    return (cx, y + h, cx, y, x, y + h / 2, x + w, y + h / 2)  # top,bottom,left,right anchors


def person(ax, cx, cy, w, h, title, desc):
    """Узел-актор (Person)."""
    head_r = h * 0.16
    head_cy = cy + h / 2 + head_r * 0.6
    ax.add_patch(Circle((cx, head_cy), head_r, facecolor=BLUE_DARK,
                        edgecolor="#16334d", linewidth=1.2, zorder=3))
    p = FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                       boxstyle="round,pad=0.0,rounding_size=0.9",
                       linewidth=1.2, edgecolor="#16334d", facecolor=BLUE_DARK, zorder=3)
    ax.add_patch(p)
    ax.text(cx, cy + h * 0.16, title, ha="center", va="center", color=TXT,
            fontsize=14, fontweight="bold", zorder=4)
    ax.text(cx, cy - h * 0.05, "[Person]", ha="center", va="center", color="#cdd9e5",
            fontsize=8.5, style="italic", zorder=4)
    ax.text(cx, cy - h * 0.26, desc, ha="center", va="center", color="#e6edf3",
            fontsize=9.5, zorder=4)


def arrow(ax, p1, p2, label="", dashed=False, color=EDGE_TXT, rad=0.0,
          fs=9.5, lx=None, ly=None, lcolor=EDGE_TXT, connector="arc3"):
    style = (0, (5, 4)) if dashed else "solid"
    a = FancyArrowPatch(p1, p2, connectionstyle=f"{connector},rad={rad}",
                        arrowstyle="-|>", mutation_scale=16, linewidth=1.6,
                        linestyle=style, color=color, zorder=2)
    ax.add_patch(a)
    if label:
        mx = lx if lx is not None else (p1[0] + p2[0]) / 2
        my = ly if ly is not None else (p1[1] + p2[1]) / 2
        ax.text(mx, my, label, ha="center", va="center", fontsize=fs,
                color=lcolor, zorder=5,
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.9))


# ============================================================ DIAGRAM 1
def diagram1(path):
    fig, ax = plt.subplots(figsize=(13.33, 7.5), dpi=200)
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")

    # системная граница
    ax.add_patch(FancyBboxPatch((2, 4), 96, 88, boxstyle="round,pad=0,rounding_size=1.0",
                 linewidth=1.5, edgecolor=BLUE, facecolor="#f4f8fc", zorder=1))
    ax.text(4, 90.5, "Модель расчёта Liquidity Stress Index", color=BLUE,
            fontsize=13, fontweight="bold", va="top")
    ax.text(4, 87.6, "[Software System]", color=BLUE, fontsize=8.5, style="italic", va="top")

    # узлы
    final = box(ax, 34, 80, 32, 9, "final_ml_dataset", "[Parquet]",
                "Итоговый датасет\nпризнаков M1–M5", fc=BLUE)
    hfb = box(ax, 30, 64.5, 40, 11, "honest_feature_builder", "[Python]",
              "Строит honest_ml_dataset\nи два whitelist-а признаков", fc=BLUE)
    glob = box(ax, 7, 44, 36, 14, "LSI Global", "[Python · scikit-learn]",
               "Обучение на ВСЕЙ истории.\nScaler→PCA(10)→IsolationForest\n→EMA→MinMax(0–100)",
               fc=BLUE, desc_fs=9.5)
    loc = box(ax, 57, 44, 36, 14, "LSI Local", "[Python · scikit-learn]",
              "Обучение на окне 365 дней.\nТот же pipeline\n(Scaler→PCA→IF→EMA→MinMax)",
              fc=BLUE, desc_fs=9.5)
    score = box(ax, 31, 28, 38, 11.5, "Скоринг LSI", "[Python]",
                "LSI_Index = Local где доступен,\nиначе Global (combine_first)", fc=GREEN)
    expl = box(ax, 64, 11, 33, 12.5, "Explainability", "[Python]",
               "EVR-attribution:\nвклады модулей, топ-драйверы,\nкомпоненты PCA", fc=ORANGE,
               title_fs=14, desc_fs=9)
    m4 = box(ax, 4, 12.5, 22, 10, "M4 overlay", "[Python]",
             "Налоговый контекст\n(вне PCA)", fc=ORANGE, title_fs=13, desc_fs=9)
    dash = box(ax, 30, 7, 30, 11, "Выход в дашборд", "[Streamlit]",
               "LSI, статусы-светофор,\noverlay M4, метрики", fc=BLUE_DARK)

    # аналитик (вне основного потока, справа сверху)
    person(ax, 84, 80, 24, 12, "Аналитик", "Использует результат LSI")

    # связи (box = [topx,topy, botx,boty, leftx,lefty, rightx,righty])
    arrow(ax, (50, final[3]), (50, hfb[1]), "входные признаки", fs=9)
    # honest -> Global / Local
    arrow(ax, (38, hfb[3]), (25, glob[1]), "GLOBAL_WHITELIST\n25 фич", rad=0.18, fs=8.8,
          lx=20, ly=62)
    arrow(ax, (62, hfb[3]), (75, loc[1]), "LOCAL_WHITELIST\n26 фич", rad=-0.18, fs=8.8,
          lx=80, ly=62)
    # Global/Local -> Скоринг
    arrow(ax, (25, glob[3]), (40, score[1]), "", rad=0.12)
    arrow(ax, (75, loc[3]), (60, score[1]), "", rad=-0.12)
    # Скоринг -> Explainability
    arrow(ax, (score[6], score[5]), (expl[0], expl[1]),
          "scaled values + PCA", rad=-0.18, fs=8.8, lx=84, ly=30)
    # Скоринг -> Дашборд
    arrow(ax, (45, score[3]), (45, dash[1]), "LSI_Global / LSI_Local /\nLSI_Index",
          fs=8.6, lx=45, ly=23)
    # Explainability -> Дашборд
    arrow(ax, (expl[4], expl[5]), (dash[6], dash[5]),
          "вклады, топ-признаки", rad=0.12, fs=8.6, lx=62.5, ly=21)
    # M4 overlay -> Дашборд (пунктир, минуя PCA)
    arrow(ax, (m4[6], m4[5]), (dash[4], dash[5]), "overlay,\nминуя PCA", dashed=True,
          fs=8.4, lx=23.5, ly=16.5, color=ORANGE, lcolor=ORANGE)
    # Дашборд -> Аналитик (пунктир, в обход блоков справа)
    arrow(ax, (dash[6], 13.5), (78, 73), "анализирует", dashed=True, rad=-0.45,
          fs=9, lx=88, ly=44)

    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ============================================================ DIAGRAM 2
def diagram2(path):
    fig, ax = plt.subplots(figsize=(13.33, 7.5), dpi=200)
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")

    # системная граница
    ax.add_patch(FancyBboxPatch((20, 30), 60, 64, boxstyle="round,pad=0,rounding_size=1.0",
                 linewidth=1.6, edgecolor=BLUE, facecolor="#f4f8fc", zorder=1))
    ax.text(22, 92, "RU Liquidity Sentinel", color=BLUE, fontsize=13,
            fontweight="bold", va="top")
    ax.text(22, 89, "[Software System]", color=BLUE, fontsize=8.5, style="italic", va="top")

    # полоса внешних источников
    ax.text(50, 25.5, "Внешние источники данных", color="#6b7a88", fontsize=10,
            ha="center", style="italic")

    # внутренние контейнеры
    dash = box(ax, 24, 70, 24, 13, "Дашборд", "[Streamlit]",
               "Визуализация\nи анализ", fc=BLUE)
    vitr = box(ax, 53, 70, 24, 13, "Витрина", "[DuckDB]",
               "warehouse.duckdb —\nsource of truth (UI)", fc=BLUE, desc_fs=9)
    lsi = box(ax, 53, 48, 24, 14, "LSI-модуль", "[Python · scikit-learn]",
              "final dataset, honest LSI,\nскоринг, объяснимость", fc=GREEN, desc_fs=9)
    etl = box(ax, 24, 46, 24, 14, "ETL-пайплайны", "[Python]",
              "Загрузка, парсинг,\nпризнаки M1–M5", fc=BLUE)

    # аналитик
    person(ax, 9, 76, 17, 12, "Аналитик", "Пользователь")

    # внешние источники
    cbr = box(ax, 5, 6, 20, 13.5, "Банк России", "[Software System]",
              "M1, M2, M5", fc=EXT, title_fs=12.5, sub_fs=8.5, desc_fs=9.5)
    mf = box(ax, 28.5, 6, 18, 13.5, "Минфин", "[Software System]",
             "M3 (ОФЗ)", fc=EXT, title_fs=12.5, sub_fs=8.5, desc_fs=9.5)
    fns = box(ax, 50, 6, 18, 13.5, "ФНС", "[Software System]",
              "M4 (налоги)", fc=EXT, title_fs=12.5, sub_fs=8.5, desc_fs=9.5)
    rk = box(ax, 71, 6, 22, 13.5, "Росказна", "[Software System]",
             "M5 (депозиты ЕКС)", fc=EXT, title_fs=12.5, sub_fs=8.5, desc_fs=9.5)

    # связи внутри (box = [topx,topy, botx,boty, leftx,lefty, rightx,righty])
    arrow(ax, (17.5, 76), (dash[4], 76), "Использует", fs=9.5)
    arrow(ax, (dash[6], 73), (vitr[4], 78), "Читает", fs=9.5, lx=50.5, ly=81)
    arrow(ax, (lsi[0], lsi[1]), (vitr[2], vitr[3]), "Публикует\nрезультаты",
          fs=9, lx=65.5, ly=66)
    arrow(ax, (etl[6], 53), (lsi[4], 53), "Передаёт признаки", fs=9, lx=50.5, ly=55.5)
    arrow(ax, (dash[6], 71), (lsi[2], 62), "Запрашивает LSI", dashed=True, fs=8.8,
          lx=43, ly=65, rad=-0.12)

    # ETL -> внешние источники (Скачивает), по одной стрелке к каждому
    arrow(ax, (30, etl[3]), (cbr[0], cbr[1]), "Скачивает", dashed=True, fs=9,
          lx=19, ly=33, rad=0.08)
    arrow(ax, (33, etl[3]), (mf[0], mf[1]), "", dashed=True, rad=0.05)
    arrow(ax, (37, etl[3]), (fns[0], fns[1]), "", dashed=True, rad=-0.04)
    arrow(ax, (41, etl[3]), (rk[0], rk[1]), "", dashed=True, rad=-0.08)

    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    out = "docs/diagrams"
    diagram1(f"{out}/diagram1_lsi_model.png")
    diagram2(f"{out}/diagram2_architecture.png")
    print("saved:",
          f"{out}/diagram1_lsi_model.png",
          f"{out}/diagram2_architecture.png")
