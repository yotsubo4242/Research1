import json
from dataclasses import asdict
from pathlib import Path

from myprogram.types import Edge, Occurrence, TermNode, WordGraph


def save_word_graph(word_graph: WordGraph, run_dir: Path) -> None:
    """WordGraph を JSON で保存する.

    sbm が必要とするグラフと, 評価が必要とする出現情報を別ファイルに分ける.
    """
    graph = {
        "nodes": [asdict(node) for node in word_graph.nodes],
        "edges": [asdict(edge) for edge in word_graph.edges],
    }
    (run_dir / "graph.json").write_text(
        json.dumps(graph, ensure_ascii=False), encoding="utf-8"
    )
    (run_dir / "occurrences.json").write_text(
        json.dumps([asdict(o) for o in word_graph.occurrences], ensure_ascii=False),
        encoding="utf-8",
    )


def load_word_graph(run_dir: Path) -> WordGraph:
    """保存した WordGraph を読み戻す."""
    graph = json.loads((run_dir / "graph.json").read_text(encoding="utf-8"))
    occurrences = json.loads((run_dir / "occurrences.json").read_text(encoding="utf-8"))
    return WordGraph(
        nodes=[TermNode(**node) for node in graph["nodes"]],
        edges=[Edge(**edge) for edge in graph["edges"]],
        # JSON にタプルは無いので復元する
        occurrences=[
            Occurrence(**{**o, "toc_path": tuple(o["toc_path"])}) for o in occurrences
        ],
    )
