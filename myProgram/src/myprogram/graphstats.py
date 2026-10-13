"""共起グラフの構造的な指標.

辺は「source を target より先に導入した教科書の数」を重みに持つ有向辺.
割れたペアには両向きの辺が立つ.
"""

from collections.abc import Iterable

Edges = dict[tuple[str, str], int]


def majority_edges(weights: Edges) -> set[tuple[str, str]]:
    """ペアごとに多数派の向きだけを残した辺集合. 同数のペアは落とす."""
    kept: set[tuple[str, str]] = set()
    for a, b in {tuple(sorted(pair)) for pair in weights}:
        forward = weights.get((a, b), 0)
        reverse = weights.get((b, a), 0)
        if forward > reverse:
            kept.add((a, b))
        elif reverse > forward:
            kept.add((b, a))
    return kept


def reciprocity(edges: Iterable[tuple[str, str]]) -> float:
    """逆向きの辺も存在する辺の割合.

    この値は「両向きを張る」表現でのみ意味を持つ. 多数派だけを残した
    グラフでは定義上 0 になる.
    """
    edge_set = set(edges)
    if not edge_set:
        return 0.0
    return sum(1 for a, b in edge_set if (b, a) in edge_set) / len(edge_set)


def _reachable(start: str, adjacency: dict[str, set[str]]) -> set[str]:
    seen = {start}
    stack = [start]
    while stack:
        current = stack.pop()
        for nxt in adjacency[current]:
            if nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return seen


def strongly_connected(
    nodes: list[str], edges: Iterable[tuple[str, str]]
) -> tuple[int, int]:
    """強連結成分の数と最大サイズ.

    向きに従って互いに行き来できる頂点のまとまり. すべてサイズ 1 なら
    そのグラフは閉路を持たない (DAG) ことを意味する. この研究では
    最大サイズが「先修関係として循環している概念の塊」にあたる.
    """
    forward: dict[str, set[str]] = {n: set() for n in nodes}
    backward: dict[str, set[str]] = {n: set() for n in nodes}
    for a, b in edges:
        forward[a].add(b)
        backward[b].add(a)

    remaining = set(nodes)
    sizes = []
    while remaining:
        seed = next(iter(remaining))
        component = _reachable(seed, forward) & _reachable(seed, backward) & remaining
        sizes.append(len(component))
        remaining -= component
    return len(sizes), max(sizes, default=0)


def weakly_connected(
    nodes: list[str], edges: Iterable[tuple[str, str]]
) -> tuple[int, int]:
    """弱連結成分の数と最大サイズ. 向きを無視して繋がりを数える."""
    neighbours: dict[str, set[str]] = {n: set() for n in nodes}
    for a, b in edges:
        neighbours[a].add(b)
        neighbours[b].add(a)

    remaining = set(nodes)
    sizes = []
    while remaining:
        seed = next(iter(remaining))
        component = _reachable(seed, neighbours) & remaining
        sizes.append(len(component))
        remaining -= component
    return len(sizes), max(sizes, default=0)
