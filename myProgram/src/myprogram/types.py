from dataclasses import dataclass


@dataclass(frozen=True)
class Term:
    # グラフ上の同一性を決める ID. 正規化前は表層形そのもの,
    # normalize 導入後は Wikipedia の正規タイトルが入る
    term_id: str
    # 教科書に現れた表層形
    surface: str
    # Key Terms 上の章. 診断用
    source_chapter: str

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
class Corpus:
    book_id: str
    book_title: str
    documents: list[Document]
    terms: list[Term]


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



@dataclass
class SBMResult:
    """
    The result of a Stochastic Block Model (SBM) analysis on a word graph.
    """
