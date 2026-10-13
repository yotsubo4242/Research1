"""保存済みの出現情報から, 教科書の部分集合だけで共起グラフを組み直す.

acquire / extract を回し直さずに「この N 冊だけで作ったグラフ」を得るための道具.
Gr.A / Gr.B の比較や leave-one-out に使う.
"""

import json
from collections import Counter
from pathlib import Path

from myprogram.stages.extract import (
    _cooccurring_pairs,
    _count_votes,
    first_occurrences,
)
from myprogram.types import Occurrence


def load_occurrences(run_dir: Path) -> list[Occurrence]:
    """occurrences.json を Occurrence に戻す."""
    raw = json.loads((run_dir / "occurrences.json").read_text(encoding="utf-8"))
    return [
        Occurrence(
            term_id=o["term_id"],
            book_id=o["book_id"],
            doc_order=o["doc_order"],
            toc_path=tuple(o["toc_path"]),
            paragraph=o["paragraph"],
            char_offset=o["char_offset"],
        )
        for o in raw
    ]


def books_of(occurrences: list[Occurrence]) -> list[str]:
    """出現の多い順に教科書 ID を返す. 分割を再現可能にするため順序を固定する."""
    counts = Counter(o.book_id for o in occurrences)
    return [book for book, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))]


def build(
    occurrences: list[Occurrence], books: set[str]
) -> tuple[list[str], dict[tuple[str, str], int]]:
    """指定した教科書だけで共起グラフを作り, (頂点, 重み付き有向辺) を返す.

    extract.run と同じ規則で組む. 辺を張るかは共起で決め, 向きの投票は
    両方の概念を扱う教科書すべてから取る.
    """
    selected = [o for o in occurrences if o.book_id in books]
    if not selected:
        return [], {}

    cooccurring: set[tuple[str, str]] = set()
    first_by_book: dict[str, dict[str, Occurrence]] = {}
    for book in books:
        book_occurrences = [o for o in selected if o.book_id == book]
        if not book_occurrences:
            continue
        cooccurring |= _cooccurring_pairs(book_occurrences)
        first_by_book[book] = first_occurrences(book_occurrences)

    weights = _count_votes(cooccurring, first_by_book)
    nodes = sorted({o.term_id for o in selected})
    return nodes, dict(weights)
