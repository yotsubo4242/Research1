from dataclasses import dataclass


@dataclass(frozen=True)
class Occurrence:
    # 用語の ID
    term_id: str
    # どの教科書か
    book_id: str
    # 本の先頭から何番目の節か
    doc_order: int
    # 正解ラベルとなるパス
    toc_path: tuple[str, ...]
    # 節内の段落番号
    paragraph: int
    # 段落内の文字位置
    char_offset: int


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
class BookSpec:
    """棚に並んでいる1冊の識別情報."""

    book_id: str
    title: str
    # URL の接頭辞. 末尾はスラッシュ.
    prefix: str


@dataclass(frozen=True)
class Corpus:
    book_id: str
    book_title: str
    documents: list[Document]


@dataclass(frozen=True)
class TermNode:
    """グラフの頂点. 教科書に依存しない属性だけを持つ."""

    term_id: str
    # 何冊の教科書に現れたか
    n_books: int
    # 総出現回数
    n_occurrences: int


@dataclass(frozen=True)
class Edge:
    """有向辺. source が先に導入される用語."""

    source: str
    target: str
    # 共起した (教科書, 節) の数
    weight: int


@dataclass(frozen=True)
class WordGraph:
    """用語の共起グラフ."""

    nodes: list[TermNode]
    edges: list[Edge]
    occurrences: list[Occurrence]


@dataclass(frozen=True)
class PrereqPair:
    # 学びたい概念
    concept: str
    # その前提となる概念(の候補)
    prerequisite: str
    # 実際に先修関係が成立するか
    is_prerequisite: bool
