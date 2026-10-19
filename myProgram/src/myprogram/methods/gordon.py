"""Gordon et al. (2016) "Modeling Concept Dependencies in a Scientific Corpus".

Wikipedia も目次も使わず, 文書コーパスの出現分布だけから依存関係を決める
2 つの情報理論的手法を実装する.

  cross_entropy    c_i の分布が c_j の分布でよく近似できて逆が成り立たない
                   なら, c_i は c_j に依存する
  information_flow 共起グラフ上のランダムウォークで, 訪問頻度の高い側を
                   前提とみなす

論文では概念をトピックモデルで表すが, 本研究では概念集合が AL-CPL で
与えられているので, 概念を「どの節に出るか」の分布で表す.

向きの読み替えに注意する. 論文の "c_i depends on c_j" は c_j が先修という
意味なので, 本研究の約束 (辺 u -> v は u が先修) では c_j -> c_i になる.
"""

import numpy as np

from myprogram.methods.base import CorpusView, DirectedGraph

# クロスエントロピーの平滑化. 片方にしか出ない節で log 0 になるのを防ぐ.
#
# この値は性能を上げるために選ぶものではなく, alpha -> 0 の極限を数値的に
# 再現するためのもの. 極限では H(c_i;c_j) の主要項が「c_i の出現のうち
# c_j が出ない節に落ちる割合」× log(1/alpha) になり, 非対称性は
# 「はみ出す割合の差」に支配される. これは論文の直感そのもの
# (MLN を見るときはたいてい Probability も見えるが, 逆は成り立たない).
#
# alpha を大きくすると全分布が一様に近づき, 生の出現が多い概念ほど
# エントロピーが低くなるため向きが反転する (alpha=1 で AUC 0.15-0.34).
# 1e-10 まで下げるとペアの順位は極限と spearman 0.999 以上で一致する.
SMOOTHING = 1e-10


def _document_distribution(counts: np.ndarray) -> np.ndarray:
    """概念ごとの「どの節に出るか」の確率分布 p(d|c) を返す."""
    smoothed = counts.astype(np.float64) + SMOOTHING
    return smoothed / smoothed.sum(axis=1, keepdims=True)


def cross_entropy(view: CorpusView) -> DirectedGraph:
    """クロスエントロピーの非対称性で向きを決める.

    H(X;Y) = -sum_d p(d|X) log2 p(d|Y) は「Y の分布が X の分布をどれだけ
    よく近似するか」を測る. 論文の条件は H(c_i;c_j) < H(c_j;c_i) ならば
    c_i が c_j に依存する, つまり c_j が先修.

    辺の重みには非対称性の大きさ H(c_j;c_i) - H(c_i;c_j) を使う. 論文は
    H(c_i;c_j) そのものを重みとするが, それは小さいほど強い量で,
    グラフの重みとしても評価のスコアとしても扱いにくいため.
    """
    probability = _document_distribution(view.counts)
    # H[i, j] = H(c_i ; c_j). 行が近似される側, 列が近似する側.
    entropy = -(probability @ np.log2(probability).T)

    weights: dict[tuple[str, str], float] = {}
    for i, j in view.cooccurring_pairs():
        # H(c_i;c_j) < H(c_j;c_i) なら c_j が先修
        asymmetry = float(entropy[j, i] - entropy[i, j])
        if asymmetry >= 0:
            weights[(view.concepts[j], view.concepts[i])] = asymmetry
        if asymmetry <= 0:
            weights[(view.concepts[i], view.concepts[j])] = -asymmetry

    return DirectedGraph(
        method="gordon_ce",
        domain=view.domain,
        concepts=view.concepts,
        weights=weights,
        normalize="difference",
    )


def _cooccurrence_graph(view: CorpusView) -> np.ndarray:
    """論文 式(4) の重み付き無向共起グラフ w(c_i,c_j) = sum_d p(c_i|d) p(c_j|d).

    p(c|d) は節 d の中で概念 c が占める割合.
    """
    counts = view.counts.astype(np.float64)
    total_per_document = counts.sum(axis=0, keepdims=True)
    # 概念が 1 つも出ない節は寄与しない
    total_per_document[total_per_document == 0] = 1.0
    share = counts / total_per_document

    adjacency = share @ share.T
    np.fill_diagonal(adjacency, 0.0)
    return adjacency


def information_flow(view: CorpusView) -> DirectedGraph:
    """ランダムウォークの定常訪問頻度と遷移確率で向きを決める.

    論文の定義は, 情報スコア I(v) が定常訪問頻度, 情報流 I(u,v) が u から
    v への遷移確率. 無向重み付きグラフではこれらが閉じた形で書けるので,
    Map Equation のモジュール分割は使わずに直接計算する.

      I(c)      = strength(c) / sum(strength)
      I(c_i,c_j) = w(c_i,c_j) / strength(c_i)

    I(c_i) < I(c_j) なら c_i が c_j に依存する, つまり c_j が先修.
    重みには正味の流れ I(c_i,c_j) - I(c_j,c_i) を使う. w が対称なので
    この符号は strength の大小, つまり I(c) の大小と一致する.
    """
    adjacency = _cooccurrence_graph(view)
    strength = adjacency.sum(axis=1)
    # 共起が 1 つも無い概念は孤立点. 0 除算を避けるだけで, 辺は張られない.
    safe_strength = np.where(strength > 0, strength, 1.0)
    flow = adjacency / safe_strength[:, None]

    weights: dict[tuple[str, str], float] = {}
    for i, j in view.cooccurring_pairs():
        net = float(flow[i, j] - flow[j, i])
        # c_i から c_j への流れが強い = c_j の方が訪問されやすい = c_j が先修
        if net >= 0:
            weights[(view.concepts[j], view.concepts[i])] = net
        if net <= 0:
            weights[(view.concepts[i], view.concepts[j])] = -net

    return DirectedGraph(
        method="gordon_if",
        domain=view.domain,
        concepts=view.concepts,
        weights=weights,
        normalize="difference",
    )
