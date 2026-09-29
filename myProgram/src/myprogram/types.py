from dataclasses import dataclass


@dataclass(frozen=True)
class Term:
    # 用語の表層形
    surface: str
    # Key Terms 上の章. 診断用であり正解ラベルではない
    source_chapter: str


@dataclass(frozen=True)
class Document:
    # URL から生成した安定した ID
    doc_id: str
    # ページのタイトル
    title: str
    # 本文セクションの HTML
    html: str
    # 正解ラベルとなるパス
    toc_path: tuple[str, ...]
    # 本の先頭からの通し番号
    order: int


@dataclass(frozen=True)
class Corpus:
    book_id: str
    book_title: str
    documents: list[Document]
    terms: list[Term]


@dataclass(frozen=True)
class WordGraph:
    """
    A graph representation of words in a corpus.
    """


@dataclass
class SBMResult:
    """
    The result of a Stochastic Block Model (SBM) analysis on a word graph.
    """
