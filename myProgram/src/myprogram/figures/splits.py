"""教科書を 2 群に分けたときの hSBM 結果を比較する.

hSBM の推論は試行ごとに揺れるので, 1 回の分割を見ても一致しているか
判断できない. そこで各群で K 回試行し,

  共起行列 : 概念 i と j が同じブロックに入った試行の割合
  ブロック数の分布 : 各レベルで何ブロックになったかの頻度

の 2 つを取る. 共起行列は試行の揺れを平均化して構造を比べられる.

比較の基準として「同じグラフでも試行が違えばどれだけずれるか」(ノイズ床)
を測る. 群 A の 50 試行を前半 25 / 後半 25 に割り, 互いの共起行列を
比べたものがそれにあたる. 群 A 対 群 B の一致がノイズ床と区別できて
初めて「教科書が違うから構造が違う」と言える.
"""

import json
from dataclasses import dataclass

import numpy as np

from myprogram.paths import RUNS_DIR

# 分割のシードごとに別ディレクトリに置いてある.
SPLIT_ROOTS = {
    0: RUNS_DIR / "splits",
    1: RUNS_DIR / "splits_seed1",
    2: RUNS_DIR / "splits_seed2",
}
SPLITS_DIR = SPLIT_ROOTS[0]


@dataclass(frozen=True)
class Trials:
    domain: str
    group: str
    books: list[str]
    term_ids: list[str]
    # 試行 × 頂点 の所属. レベルごと.
    assignments: dict[str, np.ndarray]
    block_counts: dict[str, list[int]]


def load(domain: str, group: str, seed: int = 0, n_runs: int = 50) -> Trials | None:
    path = SPLIT_ROOTS[seed] / domain / f"{group}.sbm{n_runs}.json"
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    levels = sorted(payload["runs"][0]["levels"])
    return Trials(
        domain=domain,
        group=group,
        books=payload["books"],
        term_ids=payload["term_ids"],
        assignments={
            lv: np.array([r["levels"][lv] for r in payload["runs"]]) for lv in levels
        },
        block_counts={
            lv: [len(set(r["levels"][lv])) for r in payload["runs"]] for lv in levels
        },
    )


def coassignment(assignments: np.ndarray, columns: np.ndarray) -> np.ndarray:
    """指定した頂点について, 同じブロックに入った試行の割合を返す.

    assignments は (試行数, 頂点数). columns は使う頂点の添字.
    """
    selected = assignments[:, columns]
    same = selected[:, :, None] == selected[:, None, :]
    return same.mean(axis=0)


def shared_terms(left: Trials, right: Trials) -> list[str]:
    """両群に現れた概念. 順序は左に合わせる."""
    other = set(right.term_ids)
    return [t for t in left.term_ids if t in other]


def _upper(matrix: np.ndarray) -> np.ndarray:
    return matrix[np.triu_indices_from(matrix, k=1)]


def similarity(left: np.ndarray, right: np.ndarray) -> float:
    """2 つの共起行列の一致度. 上三角のピアソン相関."""
    a, b = _upper(left), _upper(right)
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def noise_floor(trials: Trials, level: str, columns: np.ndarray) -> float:
    """同じグラフの試行を前半 / 後半に割って測った一致度.

    これより上でなければ, 群の違いは hSBM の揺れと区別できない.
    """
    runs = trials.assignments[level]
    half = len(runs) // 2
    return similarity(
        coassignment(runs[:half], columns), coassignment(runs[half:], columns)
    )


def compare(domain: str, level: str, seed: int = 0, n_runs: int = 50) -> dict | None:
    """群 A と群 B の共起行列を比べ, ノイズ床と並べて返す."""
    a, b = load(domain, "A", seed, n_runs), load(domain, "B", seed, n_runs)
    if a is None or b is None:
        return None

    terms = shared_terms(a, b)
    if len(terms) < 10:
        return None

    index_a = np.array([a.term_ids.index(t) for t in terms])
    index_b = np.array([b.term_ids.index(t) for t in terms])
    matrix_a = coassignment(a.assignments[level], index_a)
    matrix_b = coassignment(b.assignments[level], index_b)

    return {
        "domain": domain,
        "seed": seed,
        "level": level,
        "only_a": len(set(a.term_ids) - set(b.term_ids)),
        "only_b": len(set(b.term_ids) - set(a.term_ids)),
        "n_runs": n_runs,
        "n_terms": len(terms),
        "terms": terms,
        "matrix_a": matrix_a,
        "matrix_b": matrix_b,
        "between": similarity(matrix_a, matrix_b),
        "floor_a": noise_floor(a, level, index_a),
        "floor_b": noise_floor(b, level, index_b),
        "books_a": len(a.books),
        "books_b": len(b.books),
    }


def order_by_consensus(matrix: np.ndarray) -> np.ndarray:
    """共起行列の構造が見えるように概念を並べ替える添字を返す."""
    from scipy.cluster.hierarchy import leaves_list, linkage
    from scipy.spatial.distance import squareform

    distance = 1.0 - matrix
    np.fill_diagonal(distance, 0.0)
    # 数値誤差で非対称になることがあるので対称化してから渡す
    distance = (distance + distance.T) / 2
    return leaves_list(linkage(squareform(distance, checks=False), method="average"))


def consensus_blocks(
    matrix: np.ndarray,
    term_ids: list[str],
    threshold: float = 0.5,
    min_size: int = 3,
) -> list[list[str]]:
    """共起行列から概念の塊を取り出す.

    平均連結法で 1 - 共起率 を距離としてまとめ, 塊の中で中心的な
    (他の成員との共起率が高い) 概念から順に並べて返す. 塊自体は
    大きい順.
    """
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform

    distance = 1.0 - matrix
    np.fill_diagonal(distance, 0.0)
    distance = (distance + distance.T) / 2
    labels = fcluster(
        linkage(squareform(distance, checks=False), method="average"),
        t=1.0 - threshold,
        criterion="distance",
    )

    blocks = []
    for label in set(labels):
        members = np.where(labels == label)[0]
        if len(members) < min_size:
            continue
        # 塊の中心から順に並べる
        central = sorted(members, key=lambda i: -matrix[i, members].mean())
        blocks.append([term_ids[i] for i in central])
    return sorted(blocks, key=len, reverse=True)
