from pathlib import Path

from myprogram.context import RunContext
from myprogram.stages import acquire, extract, sbm


def run_all(name: str) -> Path:
    ctx = RunContext.create(name)
    ctx.log("pipeline.start", name=name)
    corpus = acquire.run(ctx)
    word_graph = extract.run(corpus, ctx)
    sbm_result = sbm.run(word_graph, ctx)
    ctx.log("pipeline.done")
    return ctx.run_dir
