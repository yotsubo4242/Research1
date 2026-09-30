import re
from collections import Counter, defaultdict
from itertools import combinations

from bs4 import BeautifulSoup

from myprogram.alcpl import load_concepts, surface_form
from myprogram.context import RunContext
from myprogram.domain import ACTIVE
from myprogram.types import Corpus, Edge, Occurrence, TermNode, WordGraph

# 語彙として使う AL-CPL の分野. acquire が読む棚と同じ設定から取る.
DOMAIN = ACTIVE.name

# 共起とみなす範囲.
#   "section"   節 (中央値 26 段落). 辺候補が多く, グラフは密になる
#   "paragraph" 段落. より強い関連に絞れるが, 1 段落の概念数は中央値 2 と少ない
COOCCURRENCE_WINDOW = "section"

# 本文でない要素. クラス名で除去する.
DROP_SELECTORS = (
    "section.box-objectives",  # Learning Objectives
    "div.mt-contributor",  # Contributors and Attributions
    "figcaption",  # 図のキャプション
)

# 概念名を含むが概念を指さない慣用句. 概念より長いので先に消費される.
STOP_PHRASES = (
    "in light of",
    "traffic light",
    "power series",
    "in terms of",
    "work out",
)

MATH_PATTERN = re.compile(r"\\\[.*?\\\]|\\\(.*?\\\)", re.DOTALL)
APOSTROPHES = str.maketrans({"’": "'", "‘": "'", "`": "'"})


def _normalize(text: str) -> str:
    """照合のために表記を揃える."""
    return text.translate(APOSTROPHES)


def _to_paragraphs(html: str) -> list[str]:
    """本文 HTML を段落のリストに変換する."""
    soup = BeautifulSoup(html, "html.parser")

    for selector in DROP_SELECTORS:
        for element in soup.select(selector):
            element.decompose()

    paragraphs = []
    for element in soup.find_all(["p", "li"]):
        # 中に <p> を含む <li> は, その <p> 側で拾うので飛ばす
        if element.name == "li" and element.find("p") is not None:
            continue
        text = _normalize(element.get_text(" ", strip=True))
        text = MATH_PATTERN.sub(" ", text)
        text = re.sub(r"\s+", " ", text).strip()
        if text:
            paragraphs.append(text)
    return paragraphs


def _build_lexicon(concepts: list[str]) -> tuple[re.Pattern[str], dict[str, str]]:
    """照合用の正規表現と, 表記 -> 概念名 の対応表を作る."""
    lexicon: dict[str, str] = {}
    for concept in concepts:
        surface = _normalize(surface_form(concept)).lower()
        if surface:
            lexicon.setdefault(surface, concept)

    # 長い表記を優先する. 除外句は概念より長いので, 概念を隠すように働く.
    surfaces = sorted(set(lexicon) | set(STOP_PHRASES), key=len, reverse=True)
    alternation = "|".join(re.escape(s) + r"(?:e?s)?" for s in surfaces)
    return re.compile(rf"\b(?:{alternation})\b", re.IGNORECASE), lexicon


def _lookup(matched: str, lexicon: dict[str, str]) -> str | None:
    """一致した文字列から概念名を引く. 除外句や未知の表記なら None."""
    key = _normalize(matched).lower()
    if key in lexicon:
        return lexicon[key]
    for suffix in ("es", "s"):
        if key.endswith(suffix) and key[: -len(suffix)] in lexicon:
            return lexicon[key[: -len(suffix)]]
    return None


def _collect_occurrences(
    corpus: Corpus,
    pattern: re.Pattern[str],
    lexicon: dict[str, str],
    ctx: RunContext,
) -> list[Occurrence]:
    """1 冊の本から概念の出現をすべて集める."""
    occurrences: list[Occurrence] = []
    n_dropped = 0

    for document in corpus.documents:
        for paragraph_index, paragraph in enumerate(_to_paragraphs(document.html)):
            for match in pattern.finditer(paragraph):
                term_id = _lookup(match.group(), lexicon)
                if term_id is None:
                    n_dropped += 1
                    continue
                occurrences.append(
                    Occurrence(
                        term_id=term_id,
                        book_id=corpus.book_id,
                        doc_order=document.order,
                        toc_path=document.toc_path,
                        paragraph=paragraph_index,
                        char_offset=match.start(),
                    )
                )

    ctx.log(
        "extract.collected",
        book_id=corpus.book_id,
        n_occurrences=len(occurrences),
        n_terms=len({o.term_id for o in occurrences}),
        n_dropped=n_dropped,
    )
    return occurrences


def _position(occurrence: Occurrence) -> tuple[int, int, int]:
    """本の中での位置. 初出順の比較に使う."""
    return (occurrence.doc_order, occurrence.paragraph, occurrence.char_offset)


def first_occurrences(occurrences: list[Occurrence]) -> dict[str, Occurrence]:
    """概念ごとの初出を返す."""
    first: dict[str, Occurrence] = {}
    for occurrence in sorted(occurrences, key=_position):
        first.setdefault(occurrence.term_id, occurrence)
    return first


def _window_key(occurrence: Occurrence) -> tuple[int, ...]:
    """共起ウィンドウを識別するキーを返す."""
    if COOCCURRENCE_WINDOW == "paragraph":
        return (occurrence.doc_order, occurrence.paragraph)
    return (occurrence.doc_order,)


def _cooccurring_pairs(occurrences: list[Occurrence]) -> set[tuple[str, str]]:
    """同じウィンドウに現れた概念のペアを返す. 向きは持たず, 常に (小, 大) の順."""
    terms_by_window: dict[tuple[int, ...], set[str]] = defaultdict(set)
    for occurrence in occurrences:
        terms_by_window[_window_key(occurrence)].add(occurrence.term_id)

    pairs: set[tuple[str, str]] = set()
    for term_ids in terms_by_window.values():
        pairs.update(combinations(sorted(term_ids), 2))
    return pairs


def _count_votes(
    pairs: set[tuple[str, str]],
    first_by_book: dict[str, dict[str, Occurrence]],
) -> Counter[tuple[str, str]]:
    """ペアごとに, どちらを先に導入するかを教科書間で投票する.

    投票できるのは両方の概念を扱っている教科書すべて. 共起した節があるかは
    問わない. 共起を条件にすると投票に使える本が減り, 推定が不安定になるため.
    1 冊 1 票なので, 重みは「そう並べた教科書の数」になる.
    """
    votes: Counter[tuple[str, str]] = Counter()
    for a, b in pairs:
        for first in first_by_book.values():
            if a not in first or b not in first:
                continue
            if _position(first[a]) < _position(first[b]):
                votes[(a, b)] += 1
            else:
                votes[(b, a)] += 1
    return votes


def run(corpora: list[Corpus], ctx: RunContext) -> WordGraph:
    """教科書から概念の共起グラフを構築する."""
    concepts = load_concepts(DOMAIN)
    pattern, lexicon = _build_lexicon(concepts)
    ctx.log(
        "extract.start",
        domain=DOMAIN,
        window=COOCCURRENCE_WINDOW,
        n_concepts=len(concepts),
        n_surfaces=len(lexicon),
        n_books=len(corpora),
        n_docs=sum(len(c.documents) for c in corpora),
    )

    occurrences: list[Occurrence] = []
    cooccurring: set[tuple[str, str]] = set()
    first_by_book: dict[str, dict[str, Occurrence]] = {}

    for corpus in corpora:
        book_occurrences = _collect_occurrences(corpus, pattern, lexicon, ctx)
        # 辺を張るかどうかは共起で決める
        cooccurring |= _cooccurring_pairs(book_occurrences)
        # 向きの投票は教科書ごとの初出順から. 本が違えば順序も違ってよい.
        first_by_book[corpus.book_id] = first_occurrences(book_occurrences)
        occurrences.extend(book_occurrences)

    weights = _count_votes(cooccurring, first_by_book)

    books_by_term: dict[str, set[str]] = defaultdict(set)
    counts_by_term: Counter[str] = Counter()
    for occurrence in occurrences:
        books_by_term[occurrence.term_id].add(occurrence.book_id)
        counts_by_term[occurrence.term_id] += 1

    nodes = [
        TermNode(
            term_id=term_id,
            n_books=len(books_by_term[term_id]),
            n_occurrences=counts_by_term[term_id],
        )
        for term_id in sorted(books_by_term)
    ]
    edges = [
        Edge(source=source, target=target, weight=weight)
        for (source, target), weight in sorted(weights.items())
    ]

    # 辺の端点がすべてノードとして存在することを確認する
    node_ids = {node.term_id for node in nodes}
    unknown = ({e.source for e in edges} | {e.target for e in edges}) - node_ids
    if unknown:
        raise ValueError(f"nodes に無い term_id が辺にある: {sorted(unknown)[:5]}")

    ctx.log(
        "extract.done",
        n_nodes=len(nodes),
        n_edges=len(edges),
        n_occurrences=len(occurrences),
    )
    return WordGraph(nodes=nodes, edges=edges, occurrences=occurrences)
