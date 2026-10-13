"""論文用の図表を figures/ に書き出す.

    uv run python script/figures.py          # 全部
    uv run python script/figures.py 1 4      # 番号を指定

番号は取得したいデータの一覧に対応する.
  1  分野ごとのコーパス規模                 → 表
  2  AL-CPL 語彙の被覆                      → 図
  3  共起グラフの統計                        → 表
  4  辺の重みと順序一致度の分布              → 図
"""

import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from myprogram.alcpl import load_concepts, load_pairs
from myprogram.domain import DOMAINS
from myprogram.figures import splits
from myprogram.figures.style import (
    BASELINE,
    COLUMN_WIDTH,
    DOMAIN_LABELS,
    FULL_WIDTH,
    NEUTRAL,
    PROPOSED,
    save,
    save_table,
    setup,
)
from myprogram.graphstats import (
    majority_edges,
    reciprocity,
    strongly_connected,
)
from myprogram.paths import RUNS_DIR


@dataclass(frozen=True)
class Snapshot:
    """1 分野の最新実行から読み出した, 図表に必要な値."""

    key: str
    label: str
    run: str
    # コーパス
    books: int
    documents: int
    documents_with_term: int
    # 語彙
    concepts: int
    found: int
    # グラフ
    nodes: int
    edges: int
    weights: list[int]
    both_ways: int
    one_way: int
    reciprocity: float
    # 両向きをそのまま使ったときの強連結成分
    scc_all: int
    largest_scc_all: int
    # 多数派の向きだけを残したときの強連結成分
    scc_majority: int
    largest_scc_majority: int
    # 評価
    pairs: int
    positives: int
    evaluated: int


def _latest(domain: str) -> Path:
    runs = [p for p in RUNS_DIR.glob(f"*_{domain}") if (p / "graph.json").exists()]
    if not runs:
        raise SystemExit(f"{domain}: graph.json を持つ実行がありません")
    return max(runs, key=lambda p: p.name)


def load_snapshot(domain: str) -> Snapshot:
    run = _latest(domain)
    graph = json.loads((run / "graph.json").read_text(encoding="utf-8"))
    occurrences = json.loads((run / "occurrences.json").read_text(encoding="utf-8"))

    node_ids = [n["term_id"] for n in graph["nodes"]]
    directed = {(e["source"], e["target"]): e["weight"] for e in graph["edges"]}
    undirected = {tuple(sorted(k)) for k in directed}
    both = sum(1 for a, b in undirected if (a, b) in directed and (b, a) in directed)

    pairs = load_pairs(domain)
    evaluated = sum(
        1
        for concept, prereq, _ in pairs
        if concept in set(node_ids)
        and prereq in set(node_ids)
        and (directed.get((prereq, concept), 0) + directed.get((concept, prereq), 0))
        > 0
    )
    majority = majority_edges(directed)
    scc_all_n, scc_all_big = strongly_connected(node_ids, directed)
    scc_maj_n, scc_maj_big = strongly_connected(node_ids, majority)

    return Snapshot(
        key=domain,
        label=DOMAIN_LABELS[domain],
        run=run.name,
        books=len({o["book_id"] for o in occurrences}),
        documents=len({(o["book_id"], o["doc_order"]) for o in occurrences}),
        documents_with_term=len({(o["book_id"], o["doc_order"]) for o in occurrences}),
        concepts=len(load_concepts(domain)),
        found=len(node_ids),
        nodes=len(node_ids),
        edges=len(directed),
        weights=sorted(directed.values()),
        both_ways=both,
        one_way=len(undirected) - both,
        reciprocity=reciprocity(directed),
        scc_all=scc_all_n,
        largest_scc_all=scc_all_big,
        scc_majority=scc_maj_n,
        largest_scc_majority=scc_maj_big,
        pairs=len(pairs),
        positives=sum(1 for _, _, y in pairs if y),
        evaluated=evaluated,
    )


def agreement_ratios(domain: str, min_votes: int = 3) -> list[float]:
    """無向ペアごとの順序一致度 max(w)/(w_fwd + w_rev)."""
    graph = json.loads((_latest(domain) / "graph.json").read_text(encoding="utf-8"))
    directed = {(e["source"], e["target"]): e["weight"] for e in graph["edges"]}
    ratios = []
    for a, b in {tuple(sorted(k)) for k in directed}:
        forward = directed.get((a, b), 0)
        reverse = directed.get((b, a), 0)
        if forward + reverse >= min_votes:
            ratios.append(max(forward, reverse) / (forward + reverse))
    return ratios


# ---------------------------------------------------------------- 1. corpus


def table_corpus(snaps: list[Snapshot]) -> None:
    rows = [
        f"  {s.label} & {s.books} & {s.documents:,} & "
        f"{s.pairs:,} & {s.positives} & {s.evaluated:,} \\\\"
        for s in snaps
    ]
    total = (
        f"  Total & {sum(s.books for s in snaps)} & {sum(s.documents for s in snaps):,} & "
        f"{sum(s.pairs for s in snaps):,} & {sum(s.positives for s in snaps)} & "
        f"{sum(s.evaluated for s in snaps):,} \\\\"
    )
    save_table(
        [
            "  \\begin{tabular}{lrrrrr}",
            "    \\toprule",
            "    Domain & Books & Sections & Pairs & Positive & Evaluated \\\\",
            "    \\midrule",
            *rows,
            "    \\midrule",
            total,
            "    \\bottomrule",
            "  \\end{tabular}",
        ],
        "tab_corpus",
        "Corpus size and the number of AL-CPL pairs that could be scored. "
        "A pair is evaluated when both concepts occur in the corpus and share at least one section.",
        "tab:corpus",
    )


# ---------------------------------------------------------------- 2. coverage


def figure_coverage(snaps: list[Snapshot]) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH, 2.3), constrained_layout=True)
    y = range(len(snaps))
    labels = [s.label for s in snaps]
    totals = [s.concepts for s in snaps]
    found = [s.found for s in snaps]

    ax.barh(y, totals, color=NEUTRAL, alpha=0.22, height=0.62, label="In AL-CPL")
    ax.barh(y, found, color=PROPOSED, height=0.62, label="Found in corpus")
    for i, (f, t) in enumerate(zip(found, totals)):
        ax.text(
            t + max(totals) * 0.015,
            i,
            f"{f}/{t} ({100 * f / t:.0f}\\%)".replace("\\", ""),
            va="center",
            fontsize=7.2,
            color=NEUTRAL,
        )

    ax.set_yticks(list(y), labels)
    ax.invert_yaxis()
    ax.set_xlabel("Number of concepts")
    ax.set_xlim(0, max(totals) * 1.34)
    ax.grid(axis="y", visible=False)
    # 凡例は軸の外に出す. 内側だと末尾のラベルと重なる
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=2)
    save(fig, "fig_coverage")


# ---------------------------------------------------------------- 3. graph stats


def table_graph(snaps: list[Snapshot]) -> None:
    rows = []
    for s in snaps:
        median = s.weights[len(s.weights) // 2]
        rows.append(
            f"  {s.label} & {s.nodes} & {s.edges:,} & {s.edges / s.nodes:.1f} & "
            f"{s.edges / (s.nodes * (s.nodes - 1)):.3f} & {s.reciprocity:.3f} & "
            f"{median} & {max(s.weights)} & "
            f"{s.largest_scc_all} & {s.largest_scc_majority} \\\\"
        )
    save_table(
        [
            "  \\resizebox{\\linewidth}{!}{%",
            "  \\begin{tabular}{lrrrrrrrrr}",
            "    \\toprule",
            "    & & & Mean & & & Median & Max & \\multicolumn{2}{c}{Largest SCC} \\\\",
            "    \\cmidrule(lr){9-10}",
            "    Domain & Nodes & Edges & degree & Density & Recipr. & weight & weight & both & major. \\\\",
            "    \\midrule",
            *rows,
            "    \\bottomrule",
            "  \\end{tabular}}",
        ],
        "tab_graph",
        "Co-occurrence graph statistics. An edge weight is the number of textbooks that introduce "
        "the source concept before the target; a pair on which the authors disagree carries an edge "
        "in both directions. Reciprocity is the fraction of edges whose reverse also exists, i.e. how "
        "often the authors disagree. The last two columns give the size of the largest strongly "
        "connected component, first on the full graph and then after keeping only the majority "
        "direction of each pair; a value of 1 would mean the consensus order is acyclic.",
        "tab:graph",
    )


# ---------------------------------------------------------------- 4. weights


def figure_weights(snaps: list[Snapshot]) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(
        2, len(snaps), figsize=(FULL_WIDTH, 3.5), constrained_layout=True
    )
    bins = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    for row in axes:
        shared = row[0]
        if shared is not axes[0][0]:
            for ax in row[1:]:
                ax.sharey(shared)

    for col, s in enumerate(snaps):
        # 上段: 辺の重み (= その向きに並べた教科書の冊数)
        top = axes[0][col]
        counts = Counter(s.weights)
        xs = sorted(counts)
        total = len(s.weights)
        top.bar(xs, [100 * counts[x] / total for x in xs], color=NEUTRAL, width=0.7)
        top.set_title(f"{s.label}\n({s.books} books)", fontsize=8)
        top.set_xticks(xs if len(xs) <= 10 else xs[::2])
        if col == 0:
            top.set_ylabel("Edges (\\%)".replace("\\", ""))

        # 下段: 順序一致度 (分野を跨いで比較できるよう 0.5-1.0 に正規化)
        bottom = axes[1][col]
        ratios = agreement_ratios(s.key)
        if ratios:
            hist = Counter(round(r, 1) for r in ratios)
            keys = [b for b in bins if hist.get(b)]
            bottom.bar(
                keys,
                [100 * hist[k] / len(ratios) for k in keys],
                color=PROPOSED,
                width=0.08,
            )
            bottom.set_xlim(0.45, 1.05)
            bottom.set_xticks([0.5, 0.75, 1.0])
        else:
            bottom.text(
                0.5,
                0.5,
                "fewer than\n3 voting books",
                ha="center",
                va="center",
                transform=bottom.transAxes,
                fontsize=7,
                color=BASELINE,
            )
            bottom.set_xticks([])
        if col == 0:
            bottom.set_ylabel("Pairs (\\%)".replace("\\", ""))

    axes[0][0].figure.supxlabel("", fontsize=1)
    for ax in axes[0]:
        ax.set_xlabel("Edge weight (books)", fontsize=7.5)
    for ax in axes[1]:
        ax.set_xlabel("Order agreement", fontsize=7.5)
    save(fig, "fig_weights")


# ---------------------------------------------------------------- 7. splits

USABLE = ["physics", "geometry", "precalculus"]


def figure_coassignment(snaps: list[Snapshot], n_runs: int = 50) -> None:
    """群 A / 群 B の共起行列を同じ並び順で並べる (案 A)."""
    import matplotlib.pyplot as plt

    results = [splits.compare(d, "0", n_runs=n_runs) for d in USABLE]
    results = [r for r in results if r is not None]
    if not results:
        print(f"  {n_runs} 試行の結果がありません")
        return

    fig, axes = plt.subplots(
        2, len(results), figsize=(FULL_WIDTH, 4.6), constrained_layout=True
    )
    for col, result in enumerate(results):
        # 並び順は群 A の構造で決め, 両群に同じ順を使う
        order = splits.order_by_consensus(result["matrix_a"])
        for row, group in enumerate("ab"):
            ax = axes[row][col]
            matrix = result[f"matrix_{group}"][np.ix_(order, order)]
            image = ax.imshow(
                matrix, cmap="Blues", vmin=0, vmax=1, interpolation="nearest"
            )
            ax.set_xticks([])
            ax.set_yticks([])
            books = result[f"books_{group}"]
            ax.set_ylabel(f"Group {group.upper()}\n({books} books)", fontsize=7.5)
            if row == 0:
                ax.set_title(
                    f"{DOMAIN_LABELS[result['domain']]}\n{result['n_terms']} concepts",
                    fontsize=8,
                )
            if row == 1:
                ax.set_xlabel(
                    f"A vs B {result['between']:.2f}  "
                    f"(noise floor {max(result['floor_a'], result['floor_b']):.2f})",
                    fontsize=7,
                )
    fig.colorbar(
        image,
        ax=axes,
        shrink=0.55,
        label=f"Same block (fraction of {n_runs} runs)",
        pad=0.02,
    )
    save(fig, "fig_coassignment" if n_runs == 50 else f"fig_coassignment_{n_runs}")


def figure_coassignment_200(snaps: list[Snapshot]) -> None:
    figure_coassignment(snaps, n_runs=200)


def figure_block_counts(snaps: list[Snapshot]) -> None:
    """各レベルで何ブロックになったかの分布 (案 B)."""
    import matplotlib.pyplot as plt

    rows = []
    for domain in USABLE:
        for group in ("A", "B"):
            trials = splits.load(domain, group)
            if trials is None:
                continue
            for level in ("0", "1"):
                rows.append(
                    (DOMAIN_LABELS[domain], group, level, trials.block_counts[level])
                )

    labels = [f"{d}  {g}  L{lv}" for d, g, lv, _ in rows]
    widest = max(max(c) for *_, c in rows)
    grid = np.zeros((len(rows), widest + 1))
    for i, (*_, counts) in enumerate(rows):
        for c in counts:
            grid[i, c] += 1
    grid = 100 * grid / grid.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(
        figsize=(FULL_WIDTH, 0.26 * len(rows) + 1.1), constrained_layout=True
    )
    image = ax.imshow(grid, cmap="Blues", aspect="auto", interpolation="nearest")
    ax.set_yticks(range(len(rows)), labels, fontsize=7)
    ax.set_xticks(range(0, widest + 1, 2), [str(x) for x in range(0, widest + 1, 2)])
    ax.set_xlabel("Number of blocks")
    ax.grid(visible=False)
    fig.colorbar(
        image, ax=ax, shrink=0.8, label=r"Runs (\%)".replace("\\", ""), pad=0.015
    )
    save(fig, "fig_block_counts")


def table_splits(snaps: list[Snapshot]) -> None:
    rows = []
    for domain in USABLE:
        for level in ("0", "1"):
            for seed in sorted(splits.SPLIT_ROOTS):
                r = splits.compare(domain, level, seed)
                if r is None:
                    continue
                floor = np.nanmax([r["floor_a"], r["floor_b"]])
                name = DOMAIN_LABELS[domain] if (level, seed) == ("0", 0) else ""
                # 分割が 1 ブロックに潰れると相関が定義できない
                cell = "--" if np.isnan(r["between"]) else f"{r['between']:.3f}"
                rows.append(
                    f"  {name} & {level if seed == 0 else ''} & {seed} & "
                    f"{r['books_a']}/{r['books_b']} & {r['n_terms']} & "
                    f"{r['only_a']}/{r['only_b']} & {cell} & "
                    f"{'--' if np.isnan(floor) else f'{floor:.3f}'} \\\\"
                )
            rows.append("  \\addlinespace[2pt]")
    save_table(
        [
            "  \\begin{tabular}{llrrrrrr}",
            "    \\toprule",
            "    & & & Books & Shared & Only & & Noise \\\\",
            "    Domain & Level & Split & A/B & concepts & A/B & A vs B & floor \\\\",
            "    \\midrule",
            *rows[:-1],
            "    \\bottomrule",
            "  \\end{tabular}",
        ],
        "tab_splits",
        "Agreement between the block structures found from two disjoint halves of the textbooks, "
        "repeated over three random splits. Each entry is the Pearson correlation between the two "
        "co-assignment matrices, each built from 50 hSBM runs. Only concepts present in both halves "
        "are compared; ``Only A/B'' counts those dropped. The noise floor is the same correlation "
        "measured between two halves of the runs on a single graph, and bounds the agreement "
        "attainable given the instability of the inference.",
        "tab:splits",
    )


def table_blocks(snaps: list[Snapshot]) -> None:
    """群 A の共起行列から取り出した塊と, その代表的な概念."""
    top_blocks = 6
    top_terms = 5

    rows = []
    for domain in USABLE:
        trials = splits.load(domain, "A")
        if trials is None:
            continue
        columns = np.arange(len(trials.term_ids))
        matrix = splits.coassignment(trials.assignments["0"], columns)
        blocks = splits.consensus_blocks(matrix, trials.term_ids)

        rows.append("    \\midrule")
        rows.append(
            f"    \\multicolumn{{3}}{{l}}{{\\emph{{{DOMAIN_LABELS[domain]}}} "
            f"({len(blocks)} blocks of 3 or more concepts)}} \\\\"
        )
        for i, block in enumerate(blocks[:top_blocks], start=1):
            names = ", ".join(
                term.replace("_", " ").replace("&", "\\&") for term in block[:top_terms]
            )
            tail = ", \\ldots" if len(block) > top_terms else ""
            rows.append(f"    {i} & {len(block)} & {names}{tail} \\\\")

    save_table(
        [
            "  \\begin{tabular}{rrp{0.62\\linewidth}}",
            "    \\toprule",
            "    Block & Size & Most central concepts \\\\",
            *rows,
            "    \\bottomrule",
            "  \\end{tabular}",
        ],
        "tab_blocks",
        "Concept blocks recovered from group A of each domain. A block is a set of concepts that "
        "land together in more than half of the 50 hSBM runs; concepts are listed from the most "
        "central outwards. Only the six largest blocks per domain are shown.",
        "tab:blocks",
    )


# ---------------------------------------------------------------- 8/9. leave-one-out


def _leave_one_out() -> list[dict]:
    path = RUNS_DIR / "leave_one_out.json"
    if not path.exists():
        raise SystemExit(
            "runs/leave_one_out.json がありません. script/leave_one_out.py を先に実行してください"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def figure_leaveout(snaps: list[Snapshot]) -> None:
    """冊ごとの一致率と, 配置をランダムにした帰無分布."""
    import matplotlib.pyplot as plt

    entries = _leave_one_out()
    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH, 2.5), constrained_layout=True)

    for i, entry in enumerate(entries):
        values = [b["accuracy"] for b in entry["per_book"]]
        jitter = np.linspace(-0.13, 0.13, len(values))
        ax.scatter(
            [i + j for j in jitter],
            sorted(values),
            s=13,
            color=PROPOSED,
            zorder=3,
            label="Held-out textbook" if i == 0 else None,
        )
        ax.plot(
            [i - 0.26, i + 0.26],
            [entry["accuracy_mean"]] * 2,
            color=PROPOSED,
            lw=2,
            zorder=4,
            label="Mean" if i == 0 else None,
        )
        # 帰無分布は平均 ± 2 標準偏差の帯で示す
        low = entry["null_mean"] - 2 * entry["null_sd"]
        high = entry["null_mean"] + 2 * entry["null_sd"]
        ax.add_patch(
            plt.Rectangle(
                (i - 0.3, low),
                0.6,
                high - low,
                color=BASELINE,
                alpha=0.3,
                zorder=1,
                label="Null model ($\\pm2$ SD)" if i == 0 else None,
            )
        )

    ax.axhline(0.5, color=NEUTRAL, lw=0.7, ls=":", zorder=0)
    ax.set_xticks(
        range(len(entries)),
        [f"{DOMAIN_LABELS[e['domain']]}\n({e['n_books']} books)" for e in entries],
    )
    ax.set_ylabel("Order agreement with held-out book")
    ax.set_ylim(0.45, 0.9)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2)
    save(fig, "fig_leaveout")


def figure_strength(snaps: list[Snapshot]) -> None:
    """合意の強さ別の一致率. 強いほど当たるなら信頼度として使える."""
    import matplotlib.pyplot as plt

    entries = _leave_one_out()
    bins = ["0.5-0.7", "0.7-0.9", "0.9-1.0"]
    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH, 2.3), constrained_layout=True)

    markers = ["o", "s", "^", "D"]
    for entry, marker in zip(entries, markers):
        values = [entry["by_strength"][b]["accuracy"] for b in bins]
        ax.plot(
            range(len(bins)),
            values,
            marker=marker,
            color=PROPOSED,
            alpha=0.55 + 0.45 * (entry["domain"] == "physics"),
            label=DOMAIN_LABELS[entry["domain"]],
        )
        for x, (b, v) in enumerate(zip(bins, values)):
            if v is not None and entry["domain"] == "physics":
                ax.annotate(
                    f"{entry['by_strength'][b]['total']:,}",
                    (x, v),
                    textcoords="offset points",
                    xytext=(0, 7),
                    ha="center",
                    fontsize=6.5,
                    color=NEUTRAL,
                )

    ax.axhline(0.5, color=NEUTRAL, lw=0.7, ls=":")
    ax.set_xticks(range(len(bins)), bins)
    ax.set_xlabel("Strength of the consensus")
    ax.set_ylabel("Order agreement")
    ax.set_ylim(0.45, 0.95)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="lower right")
    save(fig, "fig_strength")


def table_leaveout(snaps: list[Snapshot]) -> None:
    entries = _leave_one_out()
    rows = []
    for e in entries:
        values = [b["accuracy"] for b in e["per_book"]]
        above = sum(v > 0.5 for v in values)
        strength = " & ".join(
            f"{e['by_strength'][b]['accuracy']:.3f}"
            for b in ("0.5-0.7", "0.7-0.9", "0.9-1.0")
        )
        rows.append(
            f"  {DOMAIN_LABELS[e['domain']]} & {e['n_books']} & "
            f"{e['accuracy_mean']:.3f} & {e['accuracy_sd']:.3f} & "
            f"{min(values):.3f} & {above}/{e['n_books']} & "
            f"{e['null_mean']:.3f} & {e['null_sd']:.3f} & {strength} \\\\"
        )
    save_table(
        [
            "  \\resizebox{\\linewidth}{!}{%",
            "  \\begin{tabular}{lrrrrrrrrrr}",
            "    \\toprule",
            (
                "    & & \\multicolumn{4}{c}{Held-out textbook} "
                "& \\multicolumn{2}{c}{Null model} "
                "& \\multicolumn{3}{c}{By consensus strength} \\\\"
            ),
            "    \\cmidrule(lr){3-6} \\cmidrule(lr){7-8} \\cmidrule(lr){9-11}",
            "    Domain & Books & Mean & SD & Min & $>$0.5 & Mean & SD & 0.5--0.7 & 0.7--0.9 & 0.9--1.0 \\\\",
            "    \\midrule",
            *rows,
            "    \\bottomrule",
            "  \\end{tabular}}",
        ],
        "tab_leaveout",
        "Leave-one-out agreement. Each textbook is held out in turn; the remaining books vote on the "
        "order of every concept pair, and the vote is checked against the held-out book. The null "
        "model reassigns concepts to sections at random within each book, preserving the vocabulary, "
        "the number of sections and the number of concepts per section; it is repeated 1000 times and "
        "the mean and SD are over those repetitions. The last three columns break the agreement down "
        "by how strongly the remaining books agreed.",
        "tab:leaveout",
    )


# ---------------------------------------------------------------- driver

FIGURES = {
    "1": ("corpus size table", table_corpus),
    "2": ("concept coverage figure", figure_coverage),
    "3": ("graph statistics table", table_graph),
    "4": ("weight and agreement figure", figure_weights),
    "7a": ("co-assignment heatmap", figure_coassignment),
    "7a200": ("co-assignment heatmap (200 runs)", figure_coassignment_200),
    "7b": ("block count distribution", figure_block_counts),
    "7c": ("split agreement table", table_splits),
    "7d": ("concept block table", table_blocks),
    "8a": ("leave-one-out figure", figure_leaveout),
    "8b": ("consensus strength figure", figure_strength),
    "8c": ("leave-one-out table", table_leaveout),
}


def main() -> None:
    wanted = sys.argv[1:] or list(FIGURES)
    unknown = [w for w in wanted if w not in FIGURES]
    if unknown:
        raise SystemExit(f"不明な番号: {unknown}. 使えるのは {list(FIGURES)}")

    setup()
    snaps = [load_snapshot(d) for d in DOMAINS]
    for s in snaps:
        print(f"{s.key:13} {s.run}")
    print()

    for number in wanted:
        description, builder = FIGURES[number]
        print(f"[{number}] {description}")
        builder(snaps)


if __name__ == "__main__":
    main()
