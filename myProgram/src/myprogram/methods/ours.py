"""提案手法: 複数の教科書の導入順の一致度で向きを決める.

グラフの構築そのものは subgraph.build が持っている. ここは他の手法と
同じ型で取り出せるようにする薄い包みで, 比較の土俵を揃えるためにある.
"""

from myprogram.methods.base import CorpusView, DirectedGraph
from myprogram.subgraph import build


def order_agreement(view: CorpusView) -> DirectedGraph:
    """辺 (u, v) の重みは「u を先に導入した教科書の数」.

    重みが投票数なので, ペアのスコアは比 forward / (forward + backward),
    つまり著者の一致率になる.
    """
    _, votes = build(view.occurrences, set(view.books))
    return DirectedGraph(
        method="ours",
        domain=view.domain,
        concepts=view.concepts,
        weights={pair: float(count) for pair, count in votes.items()},
        normalize="ratio",
    )
