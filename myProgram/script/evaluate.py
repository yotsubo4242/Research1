"""各分野の最新の実行結果をまとめて評価する.

    uv run python script/evaluate.py                # 全分野
    uv run python script/evaluate.py physics        # 分野を指定

分野ごとの表と, 分野横断の要約を標準出力に出し,
各 run_dir の metrics.json と runs/summary.json に保存する.
"""

import json
import sys
from pathlib import Path

from myprogram.domain import DOMAINS
from myprogram.evaluation.report import (
    build_candidates,
    evaluate_candidates,
    format_table,
)
from myprogram.paths import RUNS_DIR


def _latest_run(domain: str) -> Path | None:
    """その分野の最新の実行ディレクトリを返す."""
    candidates = [
        path for path in RUNS_DIR.glob(f"*_{domain}") if (path / "graph.json").exists()
    ]
    return max(candidates, key=lambda p: p.name) if candidates else None


def main() -> None:
    domains = sys.argv[1:] or list(DOMAINS)

    summary = []
    for domain in domains:
        run_dir = _latest_run(domain)
        print("#" * 64)
        if run_dir is None:
            print(f"# {domain}: graph.json を持つ実行が見つかりません")
            print()
            continue

        graph = json.loads((run_dir / "graph.json").read_text(encoding="utf-8"))
        candidates = build_candidates(graph, domain)
        rows = evaluate_candidates(candidates)
        positives = sum(c.is_prerequisite for c in candidates)

        print(f"# {domain}   {run_dir.name}")
        print(
            f"# 頂点 {len(graph['nodes'])} / 辺 {len(graph['edges'])}"
            f" / 評価対象 {len(candidates)} ペア / 正例 {positives}"
        )
        print("#" * 64)
        print(format_table(rows))

        (run_dir / "metrics.json").write_text(
            json.dumps({"domain": domain, "rows": rows}, ensure_ascii=False, indent=1),
            encoding="utf-8",
        )
        summary.append(
            {
                "domain": domain,
                "run": run_dir.name,
                "n_nodes": len(graph["nodes"]),
                "n_edges": len(graph["edges"]),
                "n_pairs": len(candidates),
                "n_positive": positives,
                "rows": rows,
            }
        )

    if not summary:
        return

    print("=" * 64)
    print("分野横断の要約")
    print("=" * 64)
    _print_summary(summary, "all", "全ペア")
    _print_summary(summary, "small_frequency_gap", "頻度差が小さいペアのみ")

    output = RUNS_DIR / "summary.json"
    output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print(f"保存しました: {output}")


def _print_summary(summary: list[dict], subset: str, label: str) -> None:
    """分野を行, 手法を列にした AUC の表を出す."""
    methods = [row["method"] for row in summary[0]["rows"] if row["subset"] == subset]
    if not methods:
        return

    print()
    print(f"--- {label}: AUC ---")
    print(f"{'分野':14}{'n':>6}  " + "".join(f"{m[:14]:>16}" for m in methods))
    for entry in summary:
        found = {row["method"]: row for row in entry["rows"] if row["subset"] == subset}
        if not found:
            continue
        n = next(iter(found.values()))["n"]
        cells = "".join(
            f"{found[m]['auc']:>16.3f}" if m in found else f"{'-':>16}" for m in methods
        )
        print(f"{entry['domain']:14}{n:>6}  " + cells)


if __name__ == "__main__":
    main()
