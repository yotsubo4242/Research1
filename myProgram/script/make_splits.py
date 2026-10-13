"""教科書を 2 群に分け, 各群の共起グラフを書き出す.

    uv run python script/make_splits.py         # 既定のシード 0
    uv run python script/make_splits.py 1 2     # 別の分割も作る

分割の仕方で結果が変わりうるので, シードを変えて複数の分割を作れる.
hSBM は別環境で動かすので, ここでは graph を JSON にするところまで行う.
出力は runs/splits_seed<N>/<domain>/{A,B,all}.json.
"""

import json
import random
import sys
from pathlib import Path

from myprogram.domain import DOMAINS
from myprogram.paths import RUNS_DIR
from myprogram.subgraph import books_of, build, load_occurrences


def _latest(domain: str):
    runs = [
        p for p in RUNS_DIR.glob(f"*_{domain}") if (p / "occurrences.json").exists()
    ]
    if not runs:
        raise SystemExit(f"{domain}: occurrences.json を持つ実行がありません")
    return max(runs, key=lambda p: p.name)


def _dump(
    output: Path, name: str, domain: str, books: list[str], nodes, weights
) -> None:
    path = output / domain / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "domain": domain,
                "group": name,
                "books": sorted(books),
                "nodes": [{"term_id": t} for t in nodes],
                "edges": [
                    {"source": a, "target": b, "weight": w}
                    for (a, b), w in sorted(weights.items())
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(
        f"  {name:4} 教科書 {len(books):2} / 頂点 {len(nodes):4} / 辺 {len(weights):6}"
    )


def main() -> None:
    seeds = [int(s) for s in sys.argv[1:]] or [0]
    for seed in seeds:
        output = RUNS_DIR / f"splits_seed{seed}"
        print(f"########## seed {seed} -> {output.name} ##########")
        _build_all(seed, output)


def _build_all(seed: int, output: Path) -> None:
    for domain in DOMAINS:
        run = _latest(domain)
        occurrences = load_occurrences(run)
        books = books_of(occurrences)

        # 出現数の多い順に交互に配ることで, 群の規模を揃える
        rng = random.Random(seed)
        shuffled = list(books)
        rng.shuffle(shuffled)
        group_a = set(shuffled[::2])
        group_b = set(shuffled[1::2])

        print(f"{domain}  ({run.name}, {len(books)} 冊)")
        for name, selected in [("all", set(books)), ("A", group_a), ("B", group_b)]:
            nodes, weights = build(occurrences, selected)
            _dump(output, name, domain, sorted(selected), nodes, weights)
        print()


if __name__ == "__main__":
    main()
