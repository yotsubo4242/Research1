"""N-1 冊の合意が残り 1 冊の導入順をどれだけ言い当てるかを測る.

    uv run python script/leave_one_out.py

実データと, 概念の配置をランダムにした帰無モデルの両方で測り,
runs/leave_one_out.json に保存する.
"""

import json
import random
import statistics

from myprogram.domain import DOMAINS
from myprogram.leaveout import by_book, leave_one_out, shuffle_book
from myprogram.paths import RUNS_DIR
from myprogram.subgraph import load_occurrences

N_NULL = 1000
SEED = 0


def _latest(domain: str):
    runs = [
        p for p in RUNS_DIR.glob(f"*_{domain}") if (p / "occurrences.json").exists()
    ]
    if not runs:
        raise SystemExit(f"{domain}: occurrences.json を持つ実行がありません")
    return max(runs, key=lambda p: p.name)


def main() -> None:
    summary = []
    for domain in DOMAINS:
        occurrences = load_occurrences(_latest(domain))
        books = by_book(occurrences)
        if len(books) < 3:
            print(f"{domain}: {len(books)} 冊では leave-one-out が成立しない")
            continue

        actual = leave_one_out(books)
        accuracies = [r.accuracy for r in actual if r.total]

        # 帰無モデル: 各冊の概念配置をランダムにして同じ測定を繰り返す
        rng = random.Random(SEED)
        null_means = []
        for _ in range(N_NULL):
            shuffled = {b: shuffle_book(occ, rng) for b, occ in books.items()}
            null = leave_one_out(shuffled)
            values = [r.accuracy for r in null if r.total]
            if values:
                null_means.append(statistics.mean(values))

        strength = {name: [0, 0] for name in actual[0].by_strength}
        for r in actual:
            for name, (total, matched) in r.by_strength.items():
                strength[name][0] += total
                strength[name][1] += matched

        entry = {
            "domain": domain,
            "n_books": len(books),
            "per_book": [
                {"book_id": r.book_id, "total": r.total, "accuracy": r.accuracy}
                for r in actual
            ],
            "accuracy_mean": statistics.mean(accuracies),
            "accuracy_sd": statistics.stdev(accuracies) if len(accuracies) > 1 else 0.0,
            "pairs_total": sum(r.total for r in actual),
            "null_mean": statistics.mean(null_means),
            "null_sd": statistics.stdev(null_means) if len(null_means) > 1 else 0.0,
            "by_strength": {
                name: {"total": t, "accuracy": m / t if t else None}
                for name, (t, m) in strength.items()
            },
        }
        summary.append(entry)

        print(f"=== {domain} ({len(books)} 冊) ===")
        print(
            f"  実データ   {entry['accuracy_mean']:.3f} "
            f"(±{entry['accuracy_sd']:.3f}, {entry['pairs_total']:,} ペア)"
        )
        print(f"  帰無モデル {entry['null_mean']:.3f} (±{entry['null_sd']:.3f})")
        for name, info in entry["by_strength"].items():
            if info["accuracy"] is not None:
                print(
                    f"    合意 {name}: {info['accuracy']:.3f} ({info['total']:,} ペア)"
                )
        print()

    output = RUNS_DIR / "leave_one_out.json"
    output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print(f"保存しました: {output}")


if __name__ == "__main__":
    main()
