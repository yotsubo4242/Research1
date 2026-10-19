"""先修関係を推定する手法の一覧.

新しい手法を足すときは, ここに 1 行加えるだけで評価にも hSBM にも乗る.
各手法は CorpusView を受け取り DirectedGraph を返す関数.
"""

from collections.abc import Callable

from myprogram.methods import baselines, gordon, ours
from myprogram.methods.base import (
    CorpusView,
    DirectedGraph,
    latest_run,
    load_view,
    pair_score,
)

Builder = Callable[[CorpusView], DirectedGraph]

# 表示名も一緒に持つ. 表の列順はこの並びに従う.
REGISTRY: dict[str, tuple[str, Builder]] = {
    "ours": ("提案: 順序の一致度", ours.order_agreement),
    "gordon_ce": ("Gordon: クロスエントロピー", gordon.cross_entropy),
    "gordon_if": ("Gordon: 情報流", gordon.information_flow),
    "freq": ("基準: 出現頻度差", baselines.frequency_gap),
    "books": ("基準: 登場冊数差", baselines.book_gap),
    "relatedness": ("基準: 共起の総量", baselines.relatedness),
    "random": ("基準: 乱数", baselines.random_direction),
}

__all__ = [
    "REGISTRY",
    "Builder",
    "CorpusView",
    "DirectedGraph",
    "latest_run",
    "load_view",
    "pair_score",
]
