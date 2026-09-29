from dataclasses import dataclass


@dataclass(frozen=True)
class Document:
    # LibreTexts上のページID
    doc_id: str
    # ページのタイトル
    title: str
    # LibreTextsから取得したHTMLの文字列
    html: str
    # 正解ラベルとなるパス
    toc_path: tuple[str, ...]


@dataclass(frozen=True)
class Corpus:
    book_id: str
    book_title: str
    documents: list[Document]


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
