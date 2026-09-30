"""AL-CPL データセットの読み込み.

概念は Wikipedia の記事名 (Position_(vector) など) で表される.
"""

import re

from myprogram.paths import DATA_DIR

ALCPL_DIR = DATA_DIR / "alcpl"
PAREN_PATTERN = re.compile(r"\s*\([^)]*\)")


def _read_pairs(path) -> list[tuple[str, str]]:
    pairs = []
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) == 2 and all(fields):
            pairs.append((fields[0], fields[1]))
    return pairs


def load_pairs(domain: str) -> list[tuple[str, str, bool]]:
    """AL-CPL のラベル付きペアを返す.

    AL-CPL の (A, B) は「A を学ぶには先に B が必要」を意味する.
    """
    pairs = _read_pairs(ALCPL_DIR / f"{domain}.pairs")
    positives = set(_read_pairs(ALCPL_DIR / f"{domain}.preqs"))
    return [(a, b, (a, b) in positives) for a, b in pairs]


def load_concepts(domain: str) -> list[str]:
    """その分野に現れる概念名を返す."""
    pairs = _read_pairs(ALCPL_DIR / f"{domain}.pairs")
    return sorted({concept for pair in pairs for concept in pair})


def surface_form(concept: str) -> str:
    """概念名を本文照合用の文字列に変換する.

    Position_(vector) -> position
    括弧は Wikipedia の曖昧性解消なので落とすが, その分だけ語義の
    取り違えが起きうる (Work_(physics) が動詞の work に一致する等).
    """
    text = concept.replace("_", " ")
    text = PAREN_PATTERN.sub("", text)
    return text.strip()
