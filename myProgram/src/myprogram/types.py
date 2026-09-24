from dataclasses import dataclass


@dataclass
class Corpus:
    documents: list[str]


@dataclass
class WordGraph:
    """
    A graph representation of words in a corpus.
    """


@dataclass
class SBMResult:
    """
    The result of a Stochastic Block Model (SBM) analysis on a word graph.
    """
