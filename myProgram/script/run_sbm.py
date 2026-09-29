"""共起グラフに階層的 SBM を適用する.

    ~/.local/bin/micromamba run -n gt python script/run_sbm.py runs/<run_id>
"""

import json
import sys
from pathlib import Path

import graph_tool.all as gt
import numpy as np

N_INIT = 20        # 異なる出発点から探索する回数
DEG_CORR = True    # 次数補正
MIN_WEIGHT = 1     # この重み未満の辺を捨てる
N_SWEEPS = 300 # 各出発点からのMCMC掃引


def build_graph(nodes, edges):
    """JSON のノード・辺から有向重み付きグラフを作る."""
    g = gt.Graph(directed=True)
    g.add_vertex(len(nodes))
    index = {node["term_id"]: i for i, node in enumerate(nodes)}

    weight = g.new_edge_property("int")
    for edge in edges:
        if edge["weight"] < MIN_WEIGHT:
            continue
        e = g.add_edge(index[edge["source"]], index[edge["target"]])
        weight[e] = edge["weight"]
    return g, weight


def infer(g, weight):
    """複数の出発点から探索し, MCMC で精錬して最良を返す."""
    args = dict(deg_corr=DEG_CORR, eweight=weight)
    best_state, best_entropy, history = None, None, []

    for seed in range(N_INIT):
        gt.seed_rng(seed)
        state = gt.minimize_nested_blockmodel_dl(g, base_state_args=args)
        for _ in range(N_SWEEPS):
            state.multiflip_mcmc_sweep(beta=np.inf, niter=10)   # ← 追加

        entropy = float(state.entropy())
        history.append(entropy)
        print(f"  seed={seed} entropy={entropy:.2f}", flush=True)
        if best_entropy is None or entropy < best_entropy:
            best_state, best_entropy = state, entropy

    return best_state, best_entropy, history


def partitions_by_level(state, n_vertices):
    """各レベルの 頂点 -> ブロック を取り出す."""
    levels = []
    for i in range(len(state.get_levels())):
        blocks = state.project_level(i).get_blocks()
        assignment = np.asarray(blocks.a).tolist()
        levels.append(assignment)
        if len(set(assignment)) == 1:
            break
    return levels


def main() -> None:
    gt.openmp_set_num_threads(1)

    run_dir = Path(sys.argv[1])
    graph = json.loads((run_dir / "graph.json").read_text(encoding="utf-8"))
    nodes, edges = graph["nodes"], graph["edges"]

    g, weight = build_graph(nodes, edges)
    print(f"頂点 {g.num_vertices()} / 辺 {g.num_edges()}", flush=True)

    state, entropy, history = infer(g, weight)
    levels = partitions_by_level(state, g.num_vertices())

    result = {
        # 配列の並び順の参照表
        "term_ids": [node["term_id"] for node in nodes],
        "levels": levels,
        "n_blocks": [len(set(level)) for level in levels],
        "entropy": entropy,
        "entropy_history": history,
        "config": {
            "n_init": N_INIT,
            "deg_corr": DEG_CORR,
            "min_weight": MIN_WEIGHT,
            "graph_tool_version": gt.__version__,
        },
    }
    (run_dir / "sbm.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print(f"階層: {result['n_blocks']}  記述長: {entropy:.2f}", flush=True)


if __name__ == "__main__":
    main()
