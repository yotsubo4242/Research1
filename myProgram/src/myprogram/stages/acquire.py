import hashlib
import re
import time
from pathlib import Path
from urllib.parse import unquote

import httpx
from bs4 import BeautifulSoup

from myprogram.context import RunContext
from myprogram.paths import DATA_DIR
from myprogram.types import Corpus, Document, Term

BASE_URL = "https://phys.libretexts.org"
SITEMAP_URL = f"{BASE_URL}/sitemap.xml"
REQUEST_INTERVAL = 5.0
CACHE_DIR = DATA_DIR / "raw" / "phys"
USER_AGENT = "Research bot (LibreTexts structure study)"

# University Physics I (OpenStax)
BOOK_ID = "university_physics_i_openstax"
BOOK_TITLE = "University Physics I - Mechanics, Sound, Oscillations, and Waves (OpenStax)"
BOOK_PREFIX = (
    f"{BASE_URL}/Bookshelves/University_Physics/University_Physics_(OpenStax)/"
    "Book%3A_University_Physics_I_-_Mechanics_Sound_Oscillations_and_Waves_(OpenStax)/"
)

# 本文ではないページ.
EXCLUDE_PATTERNS = ("(Exercises)", "Front Matter", "Back Matter", "Index", "Glossary")

# 章末要約ページ. 本文からは除くが Key Terms の供給源として使う.
SUMMARY_MARKER = "(Summary)"


def _cache_path(url: str, suffix: str = ".html") -> Path:
    """URL からキャッシュファイルのパスを作る."""
    last_segment = unquote(url.rstrip("/").rsplit("/", 1)[-1])
    stem = re.sub(r"[^A-Za-z0-9._-]", "_", last_segment)[:60]
    digest = hashlib.sha1(url.encode("utf-8")).hexdigest()[:12]
    return CACHE_DIR / f"{stem}__{digest}{suffix}"


def _fetch(client: httpx.Client, url: str, suffix: str = ".html") -> str:
    """URL の内容を返す. 取得済みならキャッシュから読む."""
    cache_path = _cache_path(url, suffix)
    if cache_path.exists():
        return cache_path.read_text(encoding="utf-8")

    response = client.get(url)
    response.raise_for_status()
    time.sleep(REQUEST_INTERVAL)

    text = response.text
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(text, encoding="utf-8")
    return text


def _leaf_urls(client: httpx.Client) -> list[str]:
    """sitemap から対象書籍の葉ページ URL を, 本の順序で返す."""
    xml = _fetch(client, SITEMAP_URL, suffix=".xml")
    all_urls = re.findall(r"<loc>(.*?)</loc>", xml)
    book_urls = sorted(u for u in all_urls if u.startswith(BOOK_PREFIX))
    # 子を持つ URL は容れ物なので除く
    parents = {u.rsplit("/", 1)[0] for u in book_urls}
    return [u for u in book_urls if u not in parents]


def _toc_path(url: str) -> tuple[str, ...]:
    """URL から目次パスを作る."""
    relative = url[len(BOOK_PREFIX) :]
    return tuple(
        re.sub(r"\s+", " ", unquote(seg).replace("_", " ")).strip()
        for seg in relative.split("/")
    )


def _doc_id(url: str) -> str:
    """URL から安定した文書 ID を作る."""
    return hashlib.sha1(url.encode("utf-8")).hexdigest()[:12]


def _is_excluded(toc_path: tuple[str, ...]) -> bool:
    """本文ページでないなら True を返す."""
    for segment in toc_path:
        for excluded_pattern in EXCLUDE_PATTERNS:
            if excluded_pattern in segment:
                return True
    return False



def _is_summary(toc_path: tuple[str, ...]) -> bool:
    """章末要約ページなら True を返す."""
    return SUMMARY_MARKER in toc_path[-1]


def _extract_body(html: str) -> str:
    """ページ HTML から本文セクションだけを取り出す."""
    soup = BeautifulSoup(html, "html.parser")
    container = soup.find("section", class_="mt-content-container")
    if container is None:
        return ""
    return str(container)


def _extract_key_terms(html: str, chapter: str) -> list[Term]:
    """章末要約ページから Key Terms の表を取り出す."""
    soup = BeautifulSoup(html, "html.parser")
    anchor = soup.find(id="Key_Terms")
    if anchor is None:
        return []
    table = anchor.find_next("table")
    if table is None:
        return []

    terms: list[Term] = []
    for row in table.find_all("tr"):
        cells = row.find_all("td")
        if len(cells) < 2:
            continue
        surface = cells[0].get_text(strip=True)
        if surface:
            terms.append(Term(surface=surface, source_chapter=chapter))
    return terms


def run(ctx: RunContext) -> list[Corpus]:
    """LibreTexts から教科書を取得する."""
    ctx.log("acquire.start", book=BOOK_TITLE)

    documents: list[Document] = []
    terms: list[Term] = []

    with httpx.Client(
        headers={"User-Agent": USER_AGENT},
        timeout=30.0,
        follow_redirects=True,
    ) as client:
        urls = _leaf_urls(client)
        ctx.log("acquire.pages_found", n_pages=len(urls))

        for url in urls:
            toc_path = _toc_path(url)

            if _is_summary(toc_path):
                found = _extract_key_terms(_fetch(client, url), toc_path[0])
                terms.extend(found)
                ctx.log("acquire.key_terms", chapter=toc_path[0], n_terms=len(found))
                continue

            if _is_excluded(toc_path):
                continue

            documents.append(
                Document(
                    doc_id=_doc_id(url),
                    title=toc_path[-1],
                    html=_extract_body(_fetch(client, url)),
                    toc_path=toc_path,
                    order=len(documents),
                )
            )

    corpus = Corpus(
        book_id=BOOK_ID,
        book_title=BOOK_TITLE,
        documents=documents,
        terms=terms,
    )
    ctx.log("acquire.done", n_docs=len(documents), n_terms=len(terms))
    return [corpus]
