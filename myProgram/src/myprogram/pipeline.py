from pathlib import Path

from myprogram.context import RunContext
from myprogram.stages import acquire, extract, sbm


def run_all(name: str) -> Path:
    ctx = RunContext.create(name)
    ctx.log("pipeline.start", name=name)
    corpora = acquire.run(ctx)
    word_graph = extract.run(corpora, ctx)
    sbm_result = sbm.run(word_graph, ctx)
    ctx.log("pipeline.done")
    return ctx.run_dir
