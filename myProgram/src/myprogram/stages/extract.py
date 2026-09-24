from myprogram.context import RunContext
from myprogram.types import Corpus, WordGraph


def run(corpus: Corpus, ctx: RunContext) -> WordGraph:
    """
    Extract a word graph from the given corpus.
    """
    ctx.log("extract.start", n_docs=len(corpus.documents))
    word_graph = WordGraph()
    ctx.log("extract.done")
    return word_graph
