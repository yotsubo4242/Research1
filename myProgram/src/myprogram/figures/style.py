"""論文用の図と表を出力するための共通設定.

図は PDF, 表は LaTeX の tabular として figures/ に書き出す.
ラベルはすべて英語.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # use("Agg") の後に import する必要がある

from myprogram.paths import FIGURES_DIR

# 本文と並べたときに読める大きさ. 単段組を想定した幅 (inch).
COLUMN_WIDTH = 3.4
FULL_WIDTH = 6.9

# 提案手法と比較対象で色を固定する.
PROPOSED = "#2a78d6"
BASELINE = "#eb6834"
NEUTRAL = "#6b7076"
GRID = "#d8d7d3"

DOMAIN_LABELS = {
    "physics": "Physics",
    "geometry": "Geometry",
    "precalculus": "Precalculus",
    "data_mining": "Data mining",
}


def setup() -> None:
    """全図で共通の描画設定."""
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            "pdf.fonttype": 42,  # TrueType で埋め込む. TeX 側で文字が崩れない
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "axes.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.4,
            "legend.fontsize": 7.5,
            "legend.frameon": False,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "lines.linewidth": 1.4,
            "lines.markersize": 4,
        }
    )


def save(fig, name: str) -> Path:
    """図を figures/<name>.pdf に保存する."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / f"{name}.pdf"
    fig.savefig(path)
    plt.close(fig)
    print(f"  {path}")
    return path


def save_table(lines: list[str], name: str, caption: str, label: str) -> Path:
    """LaTeX の table 環境として figures/<name>.tex に保存する."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / f"{name}.tex"
    body = "\n".join(lines)
    path.write_text(
        f"% 自動生成: script/figures.py\n"
        f"\\begin{{table}}[t]\n"
        f"  \\centering\n"
        f"  \\caption{{{caption}}}\n"
        f"  \\label{{{label}}}\n"
        f"{body}\n"
        f"\\end{{table}}\n",
        encoding="utf-8",
    )
    print(f"  {path}")
    return path
