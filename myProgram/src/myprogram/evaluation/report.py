"""SBM の結果を目次と比較する.

    uv run python -m myprogram.evaluation.report runs/<run_id>
"""

import json
import sys
from pathlib import Path

from myprogram.evaluation.metrics import compare, order_baseline, shuffled_baseline


def _first_occurrences(occurrences: list[dict], book_id: str) -> dict[str, dict]:
    """用語 -> 初出の出現情報."""
    first: dict[str, dict] = {}
    ordered = sorted(
        (o for o in occurrences if o["book_id"] == book_id),
        key=lambda o: (o["doc_order"], o["paragraph"], o["char_offset"]),
    )
    for occurrence in ordered:
        first.setdefault(occurrence["term_id"], occurrence)
    return first


def main() -> None:
    run_dir = Path(sys.argv[1])
    sbm = json.loads((run_dir / "sbm.json").read_text(encoding="utf-8"))
    occurrences = json.loads((run_dir / "occurrences.json").read_text(encoding="utf-8"))

    term_ids = sbm["term_ids"]
    book_id = occurrences[0]["book_id"]
    first = _first_occurrences(occurrences, book_id)

    # 正解ラベル: 初出位置の目次パスを, 粒度ごとに切り出す
    truth = {
        "chapter": [tuple(first[t]["toc_path"][:1]) for t in term_ids],
        "section": [tuple(first[t]["toc_path"][:2]) for t in term_ids],
    }
    positions = [
        (first[t]["doc_order"], first[t]["paragraph"], first[t]["char_offset"])
        for t in term_ids
    ]

    rows = []
    for level, assignment in enumerate(sbm["levels"]):
        k = len(set(assignment))
        if k <= 1:
            continue
        for grain, labels in truth.items():
            rows.append(
                {
                    "partition": f"hSBM level {level}",
                    "n_blocks": k,
                    "truth": grain,
                    **compare(assignment, labels),
                }
            )
            rows.append(
                {
                    "partition": f"baseline order (k={k})",
                    "n_blocks": k,
                    "truth": grain,
                    **compare(order_baseline(positions, k), labels),
                }
            )
            rows.append(
                {
                    "partition": f"baseline shuffled (k={k})",
                    "n_blocks": k,
                    "truth": grain,
                    **compare(shuffled_baseline(assignment), labels),
                }
            )

    print(f"{'分割':28} {'K':>4}  {'正解':8} {'NMI':>7} {'AMI':>7} {'ARI':>7}")
    for row in rows:
        print(
            f"{row['partition']:28} {row['n_blocks']:4}  {row['truth']:8}"
            f" {row['nmi']:7.3f} {row['ami']:7.3f} {row['ari']:7.3f}"
        )

    (run_dir / "metrics.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
