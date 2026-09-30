from pathlib import Path

from myprogram.artifacts import save_word_graph
from myprogram.context import RunContext
from myprogram.stages import acquire, extract


def run_all(name: str) -> Path:
    ctx = RunContext.create(name)
    ctx.log("pipeline.start", name=name)

    corpora = acquire.run(ctx)
    word_graph = extract.run(corpora, ctx)
    save_word_graph(word_graph, ctx.run_dir)

    ctx.log("pipeline.done")
    return ctx.run_dir
