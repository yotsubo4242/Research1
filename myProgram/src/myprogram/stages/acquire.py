from myprogram.context import RunContext
from myprogram.types import Corpus


def run(ctx: RunContext) -> Corpus:
    """
    Acquire a corpus of documents.
    """
    ctx.log("acquire.start")
    corpus = Corpus(
        documents=["doc1", "doc2", "doc3"]
    )  # Placeholder for actual acquisition logic
    ctx.log("acquire.done", n_docs=len(corpus.documents))

    return corpus
