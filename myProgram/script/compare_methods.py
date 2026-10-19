"""複数の手法を共通のペア集合で比べる.

    uv run python script/compare_methods.py            # 全分野
    uv run python script/compare_methods.py physics

手法ごとに判定できるペアは違うが, それぞれの得意なペアだけで測ると
比較にならない. そこで「全手法が判定できたペア」に揃えて評価する.

評価と同時に, 各手法のグラフを runs/methods/<分野>/<手法>.json に書き出す.
これは hSBM にかけるための入力で, 形式は make_splits.py の出力と同じ:

    ~/.local/bin/micromamba run -n gt python script/run_sbm_repeat.py runs/methods 50
"""

import json
import sys

from myprogram.alcpl import load_pairs
from myprogram.domain import DOMAINS
from myprogram.evaluation.metrics import evaluate
from myprogram.methods import REGISTRY, latest_run, load_view, pair_score
from myprogram.paths import RUNS_DIR

OUTPUT_DIR = RUNS_DIR / "methods"


def _scores_by_method(graphs: dict, domain: str) -> tuple[list[bool], dict[str, list]]:
    """全手法が判定できたペアだけを残し, ラベルとスコアを返す."""
    labels: list[bool] = []
    scores: dict[str, list[float]] = {name: [] for name in graphs}

    for concept, prerequisite, is_prerequisite in load_pairs(domain):
        row = {
            name: pair_score(graph, concept, prerequisite)
            for name, graph in graphs.items()
        }
        if any(value is None for value in row.values()):
            continue
        labels.append(is_prerequisite)
        for name, value in row.items():
            scores[name].append(value)

    return labels, scores


def _run_domain(domain: str) -> dict | None:
    run_dir = latest_run(RUNS_DIR, domain)
    if run_dir is None:
        print(f"{domain}: occurrences.json を持つ実行が見つかりません")
        return None

    view = load_view(run_dir, domain)
    graphs = {name: build(view) for name, (_, build) in REGISTRY.items()}

    output_dir = OUTPUT_DIR / domain
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, graph in graphs.items():
        (output_dir / f"{name}.json").write_text(
            json.dumps(graph.to_payload(view.books), ensure_ascii=False),
            encoding="utf-8",
        )

    labels, scores = _scores_by_method(graphs, domain)
    if not (0 < sum(labels) < len(labels)):
        print(f"{domain}: 共通ペアが足りません (n={len(labels)})")
        return None

    rows = [
        {"method": name, "label": REGISTRY[name][0], **evaluate(labels, scores[name])}
        for name in graphs
    ]
    return {
        "domain": domain,
        "run": run_dir.name,
        "n_concepts": len(view.concepts),
        "n_books": len(view.books),
        "n_pairs": len(labels),
        "n_positive": sum(labels),
        "n_edges": {name: len(g.weights) for name, g in graphs.items()},
        "rows": rows,
    }


def _print_domain(result: dict) -> None:
    print("#" * 70)
    print(f"# {result['domain']}   {result['run']}")
    print(
        f"# 概念 {result['n_concepts']} / 教科書 {result['n_books']}冊"
        f" / 共通ペア {result['n_pairs']} (正例 {result['n_positive']}"
        f", 正例率 {result['n_positive'] / result['n_pairs']:.3f})"
    )
    print("#" * 70)
    print(f"{'手法':30} {'AUC':>7} {'AP':>7} {'正答率':>8} {'辺数':>9}")
    for row in result["rows"]:
        print(
            f"{row['label']:30} {row['auc']:7.3f} {row['ap']:7.3f}"
            f" {row['accuracy']:8.3f} {result['n_edges'][row['method']]:9d}"
        )
    print()


def main() -> None:
    domains = sys.argv[1:] or list(DOMAINS)

    results = [r for r in (_run_domain(d) for d in domains) if r is not None]
    for result in results:
        _print_domain(result)

    if len(results) > 1:
        print("=" * 70)
        print("分野横断: AUC")
        print("=" * 70)
        names = [row["method"] for row in results[0]["rows"]]
        print(f"{'分野':14}{'n':>6}  " + "".join(f"{n[:13]:>15}" for n in names))
        for result in results:
            found = {row["method"]: row["auc"] for row in result["rows"]}
            cells = "".join(f"{found[n]:>15.3f}" for n in names)
            print(f"{result['domain']:14}{result['n_pairs']:>6}  " + cells)
        print()

    output = OUTPUT_DIR / "summary.json"
    output.write_text(
        json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print(f"保存しました: {output}")
    print(f"グラフ: {OUTPUT_DIR}/<分野>/<手法>.json")


if __name__ == "__main__":
    main()
