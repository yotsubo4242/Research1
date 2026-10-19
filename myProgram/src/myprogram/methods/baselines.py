"""向きの決め方として素朴な基準. 提案手法が何を上回っているかを示す.

いずれも辺の候補は共起したペアに限る. 候補集合を全手法で揃えておかないと,
AUC や AP の差が「どのペアを見たか」の差と混ざってしまう.
"""

import random
from collections import defaultdict

import numpy as np

from myprogram.methods.base import CorpusView, DirectedGraph


def _directed(
    view: CorpusView, method: str, value: np.ndarray, normalize: str = "difference"
) -> DirectedGraph:
    """概念ごとのスカラー value の大小で向きを決める.

    value が大きい方を先修とする. 一般的な概念ほど先に導入される, という
    発想の基準をまとめて作るための補助.
    """
    weights: dict[tuple[str, str], float] = {}
    for i, j in view.cooccurring_pairs():
        gap = float(value[i] - value[j])
        # 互角のときも重み 0 の辺として残す. 落とすと, この基準が判定できない
        # ペアが共通ペア集合から消えて, 他の手法の評価まで狭くなる.
        if gap >= 0:
            weights[(view.concepts[i], view.concepts[j])] = gap
        if gap <= 0:
            weights[(view.concepts[j], view.concepts[i])] = -gap
    return DirectedGraph(
        method=method,
        domain=view.domain,
        concepts=view.concepts,
        weights=weights,
        normalize=normalize,
    )


def frequency_gap(view: CorpusView) -> DirectedGraph:
    """出現回数が多い方を先修とする."""
    return _directed(view, "freq", np.log1p(view.counts.sum(axis=1)))


def book_gap(view: CorpusView) -> DirectedGraph:
    """登場した教科書が多い方を先修とする."""
    books_by_term: dict[str, set[str]] = defaultdict(set)
    for o in view.occurrences:
        books_by_term[o.term_id].add(o.book_id)
    value = np.array([len(books_by_term[c]) for c in view.concepts], dtype=np.float64)
    return _directed(view, "books", value)


def relatedness(view: CorpusView) -> DirectedGraph:
    """共起の総量. 対称なので向きの情報を持たない.

    AUC が 0.5 付近に落ちることで,「関連度だけでは先修関係は決まらない」
    ことを示すための基準.
    """
    present = (view.counts > 0).astype(np.int64)
    cooccurrence = present @ present.T

    weights: dict[tuple[str, str], float] = {}
    for i, j in view.cooccurring_pairs():
        count = float(cooccurrence[i, j])
        weights[(view.concepts[i], view.concepts[j])] = count
        weights[(view.concepts[j], view.concepts[i])] = count
    return DirectedGraph(
        method="relatedness",
        domain=view.domain,
        concepts=view.concepts,
        weights=weights,
        normalize="sum",
    )


def random_direction(view: CorpusView, seed: int = 0) -> DirectedGraph:
    """乱数で向きを決める. 下限の確認用."""
    rng = random.Random(seed)
    weights: dict[tuple[str, str], float] = {}
    for i, j in sorted(view.cooccurring_pairs()):
        if rng.random() < 0.5:
            weights[(view.concepts[i], view.concepts[j])] = rng.random()
        else:
            weights[(view.concepts[j], view.concepts[i])] = rng.random()
    return DirectedGraph(
        method="random",
        domain=view.domain,
        concepts=view.concepts,
        weights=weights,
        normalize="difference",
    )
