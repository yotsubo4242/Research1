import re
from bs4 import BeautifulSoup
from collections import Counter, defaultdict
from itertools import combinations
from myprogram.context import RunContext
from myprogram.types import Corpus, Edge, Occurrence, Term, TermNode, WordGraph


# 本文でない要素. クラス名で除去する.
DROP_SELECTORS = (
    "section.box-objectives",     # Learning Objectives
    "div.mt-contributor",         # Contributors and Attributions
    "figcaption",                 # 図のキャプション
)
MATH_PATTERN = re.compile(r"\\\[.*?\\\]|\\\(.*?\\\)", re.S)
APOSTROPHES = str.maketrans({"\u2019": "'", "\u2018": "'", "\u0060": "'"})
PAREN_PATTERN = re.compile(r"\s*\(([^)]*)\)")


def _normalize(text: str) -> str:
    """照合のために表記を揃える."""
    return text.translate(APOSTROPHES)


def _surface_variants(surface: str) -> list[str]:
    """1つの用語から照合に使う表記の候補を返す."""
    surface = _normalize(surface)
    # 括弧を外した本体
    variants = [PAREN_PATTERN.sub("", surface).strip()]
    # 括弧の中身が別名なら, それも候補にする
    for inner in PAREN_PATTERN.findall(surface):
        alias = inner.removeprefix("or ").strip()
        if len(alias) >= 3:
            variants.append(alias)
    return [v for v in variants if v]

def _to_paragraphs(html: str) -> list[str]:
    """本文 HTML を段落のリストに変換する."""
    soup = BeautifulSoup(html, "html.parser")

    for selector in DROP_SELECTORS:
        for element in soup.select(selector):
            element.decompose()

    paragraphs = []
    for p in soup.find_all("p"):
        text = _normalize(p.get_text(" ", strip=True))
        text = MATH_PATTERN.sub(" ", text)
        text = re.sub(r"\s+", " ", text).strip()
        if text:
            paragraphs.append(text)
    return paragraphs

def _build_lexicon(terms: list[Term]) -> tuple[re.Pattern[str], dict[str, str]]:
    """照合用の正規表現と, 表記 -> term_id の対応表を作る."""
    lexicon: dict[str, str] = {}
    for term in terms:
        for variant in _surface_variants(term.surface):
            lexicon.setdefault(variant.lower(), term.term_id)

    # 長い表記を優先し, 語尾の複数形を許容する
    surfaces = sorted(lexicon, key=len, reverse=True)
    alternation = "|".join(re.escape(s) + r"(?:e?s)?" for s in surfaces)
    return re.compile(rf"\b(?:{alternation})\b", re.IGNORECASE), lexicon


def _lookup(matched: str, lexicon: dict[str, str]) -> str | None:
    """一致した文字列から term_id を引く."""
    key = _normalize(matched).lower()
    if key in lexicon:
        return lexicon[key]
    for suffix in ("es", "s"):
        if key.endswith(suffix) and key[: -len(suffix)] in lexicon:
            return lexicon[key[: -len(suffix)]]
    return None


def _collect_occurrences(corpus: Corpus, ctx: RunContext) -> list[Occurrence]:
    """1 冊の本から用語の出現をすべて集める."""
    pattern, lexicon = _build_lexicon(corpus.terms)
    occurrences: list[Occurrence] = []
    n_unresolved = 0

    for document in corpus.documents:
        for paragraph_index, paragraph in enumerate(_to_paragraphs(document.html)):
            for match in pattern.finditer(paragraph):
                term_id = _lookup(match.group(), lexicon)
                if term_id is None:
                    n_unresolved += 1
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
        n_unresolved=n_unresolved,
    )
    return occurrences


def _position(occurrence: Occurrence) -> tuple[int, int, int]:
    """本の中での位置. 初出順の比較に使う."""
    return (occurrence.doc_order, occurrence.paragraph, occurrence.char_offset)


# TODO: evaluationでも使う予定.
def first_occurrences(occurrences: list[Occurrence]) -> dict[str, Occurrence]:
    """用語ごとの初出を返す."""
    first: dict[str, Occurrence] = {}
    for occurrence in sorted(occurrences, key=_position):
        first.setdefault(occurrence.term_id, occurrence)
    return first


def _count_cooccurrences(
    occurrences: list[Occurrence],
    first: dict[str, Occurrence],
) -> Counter[tuple[str, str]]:
    """節ごとに用語のペアを数える. 向きは初出が早いほうから遅いほうへ."""
    terms_by_document: dict[int, set[str]] = defaultdict(set)
    for occurrence in occurrences:
        terms_by_document[occurrence.doc_order].add(occurrence.term_id)

    weights: Counter[tuple[str, str]] = Counter()
    for term_ids in terms_by_document.values():
        for a, b in combinations(sorted(term_ids), 2):
            if _position(first[a]) < _position(first[b]):
                weights[(a, b)] += 1
            else:
                weights[(b, a)] += 1
    return weights



def run(corpora: list[Corpus], ctx: RunContext) -> WordGraph:
    """教科書から用語の共起グラフを構築する."""
    ctx.log("extract.start", n_docs=sum(len(c.documents) for c in corpora))

    occurrences: list[Occurrence] = []
    weights: Counter[tuple[str, str]] = Counter()

    for corpus in corpora:
        book_occurrences = _collect_occurrences(corpus, ctx)
        first = first_occurrences(book_occurrences)
        weights.update(_count_cooccurrences(book_occurrences, first))
        occurrences.extend(book_occurrences)

     # ノードは全冊に現れた用語の和集合
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

