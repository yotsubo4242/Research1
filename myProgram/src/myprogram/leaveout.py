"""N-1 冊の合意が, 残り 1 冊の導入順をどれだけ言い当てるか.

各教科書を 1 冊ずつ伏せ, 残りの教科書の多数決で決めた順序と
伏せた教科書の実際の順序を突き合わせる. 「著者の合意」が
新しい教科書に対して予測力を持つかを測る.

比較のため, 概念の配置をランダムにした帰無モデルでも同じ測定を行う.
"""

import random
from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations

from myprogram.types import Occurrence

Position = tuple[int, int, int]


def _position(occurrence: Occurrence) -> Position:
    return (occurrence.doc_order, occurrence.paragraph, occurrence.char_offset)


def first_positions(occurrences: list[Occurrence]) -> dict[str, Position]:
    """概念ごとの初出位置."""
    first: dict[str, Position] = {}
    for occurrence in sorted(occurrences, key=_position):
        first.setdefault(occurrence.term_id, _position(occurrence))
    return first


def by_book(occurrences: list[Occurrence]) -> dict[str, list[Occurrence]]:
    grouped: dict[str, list[Occurrence]] = defaultdict(list)
    for occurrence in occurrences:
        grouped[occurrence.book_id].append(occurrence)
    return dict(grouped)


def cooccurring(occurrences: list[Occurrence]) -> set[tuple[str, str]]:
    """同じ節に現れた概念のペア. 常に (小, 大) の順."""
    by_section: dict[int, set[str]] = defaultdict(set)
    for occurrence in occurrences:
        by_section[occurrence.doc_order].add(occurrence.term_id)
    pairs: set[tuple[str, str]] = set()
    for terms in by_section.values():
        pairs.update(combinations(sorted(terms), 2))
    return pairs


@dataclass(frozen=True)
class Result:
    """1 冊を伏せたときの一致の内訳."""

    book_id: str
    # 判定できたペア数と, 合意と一致した数
    total: int
    matched: int
    # 合意の強さで分けた内訳. しきい値 -> (判定数, 一致数)
    by_strength: dict[str, tuple[int, int]]

    @property
    def accuracy(self) -> float:
        return self.matched / self.total if self.total else float("nan")


STRENGTH_BINS = (("0.5-0.7", 0.5, 0.7), ("0.7-0.9", 0.7, 0.9), ("0.9-1.0", 0.9, 1.01))


def leave_one_out(books: dict[str, list[Occurrence]]) -> list[Result]:
    """各冊を 1 回ずつ伏せて, 残りの合意との一致を測る."""
    firsts = {book: first_positions(occ) for book, occ in books.items()}
    pairs_of = {book: cooccurring(occ) for book, occ in books.items()}

    results = []
    for held_out in books:
        others = [b for b in books if b != held_out]
        held_first = firsts[held_out]

        # 残りの教科書で共起したペアを候補にする
        candidates: set[tuple[str, str]] = set()
        for book in others:
            candidates |= pairs_of[book]

        total = matched = 0
        buckets = {name: [0, 0] for name, _, _ in STRENGTH_BINS}
        for a, b in candidates:
            if a not in held_first or b not in held_first:
                continue
            forward = sum(
                1
                for book in others
                if a in firsts[book]
                and b in firsts[book]
                and firsts[book][a] < firsts[book][b]
            )
            reverse = sum(
                1
                for book in others
                if a in firsts[book]
                and b in firsts[book]
                and firsts[book][a] > firsts[book][b]
            )
            if forward == reverse:  # 合意が割れたペアは予測しない
                continue

            consensus_a_first = forward > reverse
            strength = max(forward, reverse) / (forward + reverse)
            agrees = consensus_a_first == (held_first[a] < held_first[b])

            total += 1
            matched += agrees
            for name, low, high in STRENGTH_BINS:
                if low <= strength < high:
                    buckets[name][0] += 1
                    buckets[name][1] += agrees

        results.append(
            Result(
                book_id=held_out,
                total=total,
                matched=matched,
                by_strength={k: (v[0], v[1]) for k, v in buckets.items()},
            )
        )
    return results


def shuffle_book(occurrences: list[Occurrence], rng: random.Random) -> list[Occurrence]:
    """1 冊の中で概念の配置をランダムにする.

    節の数, 節あたりの概念数, その本の語彙は保ったまま, どの概念が
    どの節に現れるかと節内の順序だけを入れ替える. 共起も導入順も
    壊れるので, 「教科書の構造に意味がある」ことの帰無モデルになる.
    """
    vocabulary = sorted({o.term_id for o in occurrences})
    sizes: dict[int, int] = {}
    template: dict[int, Occurrence] = {}
    for occurrence in occurrences:
        template.setdefault(occurrence.doc_order, occurrence)
    for doc_order, terms in by_section_terms(occurrences).items():
        sizes[doc_order] = len(terms)

    shuffled: list[Occurrence] = []
    for doc_order, size in sizes.items():
        sample = rng.sample(vocabulary, min(size, len(vocabulary)))
        rng.shuffle(sample)
        reference = template[doc_order]
        for offset, term in enumerate(sample):
            shuffled.append(
                Occurrence(
                    term_id=term,
                    book_id=reference.book_id,
                    doc_order=doc_order,
                    toc_path=reference.toc_path,
                    paragraph=0,
                    char_offset=offset,
                )
            )
    return shuffled


def by_section_terms(occurrences: list[Occurrence]) -> dict[int, set[str]]:
    sections: dict[int, set[str]] = defaultdict(set)
    for occurrence in occurrences:
        sections[occurrence.doc_order].add(occurrence.term_id)
    return dict(sections)
