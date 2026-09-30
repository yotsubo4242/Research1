"""先修関係の予測を評価する指標."""

import random
from collections.abc import Sequence

from sklearn.metrics import average_precision_score, roc_auc_score


def evaluate(labels: Sequence[bool], scores: Sequence[float]) -> dict[str, float]:
    """予測スコアを正解ラベルと突き合わせる.

    ap は先行研究が AUPRC として報告している値に対応する.
    accuracy は「スコアが中央値より上なら正」とした場合の正答率.
    """
    ordered = sorted(scores)
    threshold = ordered[len(ordered) // 2]
    correct = sum((score > threshold) == label for score, label in zip(scores, labels))
    return {
        "auc": float(roc_auc_score(labels, scores)),
        "ap": float(average_precision_score(labels, scores)),
        "accuracy": correct / len(labels),
    }


def random_scores(n: int, seed: int = 0) -> list[float]:
    """下限を確認するための乱数スコア."""
    rng = random.Random(seed)
    return [rng.random() for _ in range(n)]
