"""比較する手法に共通の入出力.

各手法は CorpusView を受け取り DirectedGraph を返す. 辺 (u, v) の重みは
「u が v の先修である証拠の強さ」で, 値域は手法ごとに違ってよい.
この向きの約束だけ揃えておけば, 評価も hSBM も同じ道具で回せる.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from myprogram.subgraph import load_occurrences
from myprogram.types import Occurrence

# hSBM に渡す辺の重みの段階数. 全手法で共通にする.
WEIGHT_LEVELS = 10


@dataclass(frozen=True)
class CorpusView:
    """手法に渡すコーパスの共通ビュー.

    どの手法も「概念がどの節に何回出たか」を起点にするので, そこまでを
    ここで作っておく. 初出の順序など行列に落ちない情報が要る手法のために,
    生の Occurrence も持たせる.
    """

    domain: str
    # 頂点. 以降の添字はすべてこの並び.
    concepts: list[str]
    books: list[str]
    # 節の識別子 (教科書 ID, 本の先頭からの通し番号). counts の列に対応.
    documents: list[tuple[str, int]]
    # (概念数, 節数) の出現回数
    counts: np.ndarray
    occurrences: list[Occurrence]

    def concept_index(self) -> dict[str, int]:
        return {concept: i for i, concept in enumerate(self.concepts)}

    def cooccurring_pairs(self) -> set[tuple[int, int]]:
        """同じ節に現れた概念の組を, 添字の (小, 大) で返す.

        辺の候補をここに限る. どの手法も同じ候補集合を見ることになるので,
        手法間の差が「候補の広さ」ではなく「向きと強さの決め方」に絞られる.
        """
        present = self.counts > 0
        pairs: set[tuple[int, int]] = set()
        for column in range(present.shape[1]):
            rows = np.flatnonzero(present[:, column])
            for position, i in enumerate(rows):
                for j in rows[position + 1 :]:
                    pairs.add((int(i), int(j)))
        return pairs


@dataclass(frozen=True)
class DirectedGraph:
    """手法が出力する有向重み付きグラフ.

    normalize は, ペアのスコアを 2 方向の重みからどう作るかの指定.
      "ratio"      forward / (forward + backward). 重みが投票数のとき向き
      "difference" forward - backward. 重みがすでに非対称な量のとき向き
      "sum"        forward + backward. 向きを持たない基準用
    """

    method: str
    domain: str
    concepts: list[str]
    weights: dict[tuple[str, str], float]
    normalize: str = "difference"

    def to_payload(self, books: list[str], levels: int = WEIGHT_LEVELS) -> dict:
        """run_sbm_repeat.py が読む形に変換する.

        hSBM の eweight は辺の多重度, つまり整数を期待する. 手法によって
        重みの単位が違う (投票数, 確率, エントロピー差) ので, そのままでは
        比べられない. そこで重みの順位を levels 段階に量子化して渡し,
        生の値は score に残す.

        量子化は「重みの絶対値が違うせいで階層が変わる」ことを防ぐためで,
        どの辺を張るかは手法ごとのまま. 密度の違いは残る.
        """
        edges = [(pair, w) for pair, w in sorted(self.weights.items()) if w > 0]
        # 同じ重みの辺は同じ段階に入れる. 順位で機械的に割ると, 投票数が
        # 同じ辺が並び順だけの理由で違う段階に散ってしまう.
        ordered = sorted({w for _, w in edges})
        below = 0
        level_of: dict[float, int] = {}
        for value in ordered:
            count = sum(1 for _, w in edges if w == value)
            level_of[value] = 1 + below * levels // max(len(edges), 1)
            below += count

        return {
            "domain": self.domain,
            "group": self.method,
            "books": books,
            "nodes": [{"term_id": c} for c in self.concepts],
            "edges": [
                {
                    "source": source,
                    "target": target,
                    "weight": level_of[weight],
                    "score": weight,
                }
                for (source, target), weight in edges
            ],
        }


def pair_score(graph: DirectedGraph, concept: str, prerequisite: str) -> float | None:
    """「prerequisite が concept の先修である」スコア. 判定できなければ None.

    重み 0 の辺と, 辺が無いことは区別する. 両者を混ぜると「互角と判断した」
    ペアが評価から落ち, 手法ごとに見ているペアが変わってしまう."""
    forward = graph.weights.get((prerequisite, concept))
    backward = graph.weights.get((concept, prerequisite))
    if forward is None and backward is None:
        return None
    forward = forward or 0.0
    backward = backward or 0.0
    if forward == 0.0 and backward == 0.0:
        return 0.0
    if graph.normalize == "ratio":
        return forward / (forward + backward)
    if graph.normalize == "sum":
        return forward + backward
    return forward - backward


def load_view(run_dir: Path, domain: str) -> CorpusView:
    """実行結果の occurrences.json から CorpusView を組む."""
    occurrences = load_occurrences(run_dir)

    concepts = sorted({o.term_id for o in occurrences})
    books = sorted({o.book_id for o in occurrences})
    documents = sorted({(o.book_id, o.doc_order) for o in occurrences})

    concept_index = {c: i for i, c in enumerate(concepts)}
    document_index = {d: j for j, d in enumerate(documents)}

    counts = np.zeros((len(concepts), len(documents)), dtype=np.int64)
    for o in occurrences:
        counts[concept_index[o.term_id], document_index[(o.book_id, o.doc_order)]] += 1

    return CorpusView(
        domain=domain,
        concepts=concepts,
        books=books,
        documents=documents,
        counts=counts,
        occurrences=occurrences,
    )


def latest_run(runs_dir: Path, domain: str) -> Path | None:
    """その分野で occurrences.json を持つ最新の実行ディレクトリ."""
    candidates = [
        path
        for path in runs_dir.glob(f"*_{domain}")
        if (path / "occurrences.json").exists()
    ]
    return max(candidates, key=lambda p: p.name) if candidates else None
