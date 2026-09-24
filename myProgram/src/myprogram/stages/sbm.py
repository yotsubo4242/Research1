from myprogram.context import RunContext
from myprogram.types import SBMResult, WordGraph


def run(word_graph: WordGraph, ctx: RunContext) -> SBMResult:
    """
    Run Stochastic Block Model (SBM) analysis on the given word graph.
    """
    ctx.log("sbm.start")
    sbm_result = SBMResult()
    ctx.log("sbm.done")
    return sbm_result
