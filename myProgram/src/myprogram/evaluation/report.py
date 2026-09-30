"""共起グラフが AL-CPL の先修関係を予測できるかを評価する.

    uv run python -m myprogram.evaluation.report runs/<run_id> [分野名]

AL-CPL の (a, b) が正例とは「a を学ぶには先に b が必要」の意味.
b が先に導入されるはずなので, 辺 b -> a が a -> b より重ければ正と予測する.
"""

import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

from myprogram.alcpl import load_pairs
from myprogram.domain import ACTIVE
from myprogram.evaluation.metrics import evaluate, random_scores

# 表示順を固定するため, 名前とスコアの取り出し方をここで定義する.
METHODS = (
    ("提案: 順序の一致度", "agreement"),
    ("基準: 出現頻度差", "frequency_gap"),
    ("基準: 登場冊数差", "book_gap"),
    ("基準: 共起の総量", "relatedness"),
)


@dataclass(frozen=True)
class Candidate:
    """評価対象の 1 ペアと, そこから計算した各種スコア."""

    is_prerequisite: bool
    # 提案手法: 先修候補を先に導入した教科書の割合
    agreement: float
    # 基準: 出現頻度の差 (一般的な概念ほど先修になりやすい)
    frequency_gap: float
    # 基準: 登場した教科書数の差
    book_gap: float
    # 基準: 共起の総量 (関連度のみ. 向きの情報を含まない)
    relatedness: float


def build_candidates(graph: dict, domain: str) -> list[Candidate]:
    """グラフと AL-CPL のラベルから評価対象を組み立てる."""
    weights = {(e["source"], e["target"]): e["weight"] for e in graph["edges"]}
    occurrences = {n["term_id"]: n["n_occurrences"] for n in graph["nodes"]}
    books = {n["term_id"]: n["n_books"] for n in graph["nodes"]}

    candidates = []
    for concept, prerequisite, is_prerequisite in load_pairs(domain):
        if concept not in occurrences or prerequisite not in occurrences:
            continue
        # 先修候補が先に来る向き / その逆
        forward = weights.get((prerequisite, concept), 0)
        backward = weights.get((concept, prerequisite), 0)
        if forward + backward == 0:
            continue
        candidates.append(
            Candidate(
                is_prerequisite=is_prerequisite,
                agreement=forward / (forward + backward),
                frequency_gap=math.log1p(occurrences[prerequisite])
                - math.log1p(occurrences[concept]),
                book_gap=float(books[prerequisite] - books[concept]),
                relatedness=float(forward + backward),
            )
        )
    return candidates


def _rows_for(candidates: list[Candidate], subset: str) -> list[dict]:
    labels = [c.is_prerequisite for c in candidates]
    if not (0 < sum(labels) < len(labels)):
        return []

    rows = []
    for name, field in METHODS:
        scores = [getattr(c, field) for c in candidates]
        rows.append(
            {
                "subset": subset,
                "method": name,
                "n": len(candidates),
                **evaluate(labels, scores),
            }
        )
    rows.append(
        {
            "subset": subset,
            "method": "基準: 乱数",
            "n": len(candidates),
            **evaluate(labels, random_scores(len(candidates))),
        }
    )
    return rows


def evaluate_candidates(candidates: list[Candidate]) -> list[dict]:
    """全ペアと, 頻度で判断できないペアの両方で評価する."""
    rows = _rows_for(candidates, "all")

    # 頻度差の小さい下位 1/3. ここで提案が勝てば, 頻度の代理ではないと言える
    magnitudes = sorted(abs(c.frequency_gap) for c in candidates)
    cut = magnitudes[len(magnitudes) // 3]
    rows += _rows_for(
        [c for c in candidates if abs(c.frequency_gap) <= cut], "small_frequency_gap"
    )
    return rows


def format_table(rows: list[dict]) -> str:
    """評価結果を表として整形する."""
    lines = []
    for subset in dict.fromkeys(row["subset"] for row in rows):
        selected = [row for row in rows if row["subset"] == subset]
        label = "全ペア" if subset == "all" else "頻度差が小さいペアのみ"
        lines.append(f"--- {label} (n={selected[0]['n']}) ---")
        lines.append(f"{'手法':26} {'AUC':>7} {'AP':>7} {'正答率':>8}")
        for row in selected:
            lines.append(
                f"{row['method']:26} {row['auc']:7.3f} {row['ap']:7.3f}"
                f" {row['accuracy']:8.3f}"
            )
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    run_dir = Path(sys.argv[1])
    domain = sys.argv[2] if len(sys.argv) > 2 else ACTIVE.name

    graph = json.loads((run_dir / "graph.json").read_text(encoding="utf-8"))
    candidates = build_candidates(graph, domain)
    rows = evaluate_candidates(candidates)

    positives = sum(c.is_prerequisite for c in candidates)
    print(f"分野 {domain} / 評価対象 {len(candidates)} ペア / 正例 {positives}")
    print()
    print(format_table(rows))

    output = run_dir / "metrics.json"
    output.write_text(
        json.dumps({"domain": domain, "rows": rows}, ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    print(f"保存しました: {output}")


if __name__ == "__main__":
    main()
