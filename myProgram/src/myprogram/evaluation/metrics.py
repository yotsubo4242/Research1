"""分割どうしを比較する指標とベースライン."""

import random
from collections.abc import Sequence

from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
)


def compare(a: Sequence, b: Sequence) -> dict[str, float]:
    """2 つの分割の一致度を返す."""
    return {
        "nmi": float(normalized_mutual_info_score(a, b)),
        "ami": float(adjusted_mutual_info_score(a, b)),
        "ari": float(adjusted_rand_score(a, b)),
    }


def order_baseline(positions: Sequence[tuple], k: int) -> list[int]:
    """初出順に並べて k 個に等分した分割.

    順序の情報しか使わないので, hSBM がこれを上回らなければ
    順序以上のことを何も見つけていないことになる.
    """
    order = sorted(range(len(positions)), key=lambda i: positions[i])
    labels = [0] * len(positions)
    for rank, index in enumerate(order):
        labels[index] = rank * k // len(positions)
    return labels


def shuffled_baseline(labels: Sequence, seed: int = 0) -> list:
    """同じグループサイズ分布を保ったままランダムに割り当て直した分割."""
    shuffled = list(labels)
    random.Random(seed).shuffle(shuffled)
    return shuffled
