"""1 つの共起グラフに階層的 SBM を K 回かけ, 各試行の分割を保存する.

    ~/.local/bin/micromamba run -n gt python script/run_sbm_repeat.py runs/splits [試行数]

推論は局所最適に落ちるため 1 回では分割が安定しない. 何度も試行して
分布として扱えるように, 試行ごとの所属をそのまま書き出す.
共起行列や一致度は後段 (uv 環境) で計算する.

入力  runs/splits/<domain>/{all,A,B}.json
出力  runs/splits/<domain>/{all,A,B}.sbm<試行数>.json

試行数を変えた結果を並べて比べられるよう, 出力名に試行数を含める.
"""

import json
import sys
import time
from pathlib import Path

import graph_tool.all as gt
import numpy as np

DEFAULT_RUNS = 50
N_SWEEPS = 50
DEG_CORR = True
LEVELS = 2  # 所属を保存する階層の深さ


def build_graph(payload: dict):
    nodes = [n["term_id"] for n in payload["nodes"]]
    index = {t: i for i, t in enumerate(nodes)}

    graph = gt.Graph(directed=True)
    graph.add_vertex(len(nodes))
    weight = graph.new_edge_property("int")
    for edge in payload["edges"]:
        e = graph.add_edge(index[edge["source"]], index[edge["target"]])
        weight[e] = edge["weight"]
    return nodes, graph, weight


def repeat(graph, weight, n_runs: int) -> list[dict]:
    args = {"deg_corr": DEG_CORR, "eweight": weight}
    runs = []
    for seed in range(n_runs):
        gt.seed_rng(seed)
        state = gt.minimize_nested_blockmodel_dl(graph, base_state_args=args)
        for _ in range(N_SWEEPS):
            state.multiflip_mcmc_sweep(beta=np.inf, niter=10)

        levels = {}
        for depth in range(LEVELS):
            # project_level が返す一時オブジェクトを変数に受けること.
            # 直接 .a を辿ると解放済みメモリを指して壊れた値になる.
            blocks = state.project_level(depth).get_blocks()
            levels[str(depth)] = [int(x) for x in np.asarray(blocks.a)]

        runs.append(
            {
                "seed": seed,
                "entropy": float(state.entropy()),
                "n_blocks": [lv.get_B() for lv in state.get_levels()],
                "levels": levels,
            }
        )
    return runs


def main() -> None:
    root = Path(sys.argv[1])
    n_runs = int(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_RUNS
    gt.openmp_set_num_threads(1)  # 並列のままだとシードを固定しても再現しない

    targets = sorted(p for p in root.glob("*/*.json") if ".sbm" not in p.name)
    print(f"対象 {len(targets)} グラフ / 各 {n_runs} 試行", flush=True)

    for path in targets:
        output = path.with_name(f"{path.stem}.sbm{n_runs}.json")
        if output.exists():
            print(f"{path.parent.name}/{path.stem}: 済み", flush=True)
            continue

        payload = json.loads(path.read_text(encoding="utf-8"))
        nodes, graph, weight = build_graph(payload)
        if graph.num_vertices() < 10 or graph.num_edges() < 20:
            print(f"{path.parent.name}/{path.stem}: 小さすぎるため省略", flush=True)
            continue

        started = time.time()
        runs = repeat(graph, weight, n_runs)
        output.write_text(
            json.dumps(
                {
                    "domain": payload["domain"],
                    "group": payload["group"],
                    "books": payload["books"],
                    "term_ids": nodes,
                    "n_runs": len(runs),
                    "n_sweeps": N_SWEEPS,
                    "runs": runs,
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        entropies = [r["entropy"] for r in runs]
        blocks = [r["n_blocks"][1] if len(r["n_blocks"]) > 1 else 1 for r in runs]
        print(
            f"{path.parent.name}/{path.stem}: 頂点 {graph.num_vertices()} "
            f"辺 {graph.num_edges()} / 記述長 {min(entropies):.0f}-{max(entropies):.0f} "
            f"/ level1 ブロック数 {min(blocks)}-{max(blocks)} "
            f"({time.time() - started:.0f}秒)",
            flush=True,
        )


if __name__ == "__main__":
    main()
