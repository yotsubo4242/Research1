import hashlib
import re
import time
from pathlib import Path
from urllib.parse import unquote, urlparse

import httpx
from bs4 import BeautifulSoup

from myprogram.context import RunContext
from myprogram.domain import ACTIVE
from myprogram.paths import DATA_DIR
from myprogram.types import BookSpec, Corpus, Document

REQUEST_INTERVAL = 5.0
USER_AGENT = "Research bot (LibreTexts structure study)"

# 対象は domain.ACTIVE で決まる. 分野の切り替えは環境変数 MYPROGRAM_DOMAIN で.
BASE_URL = ACTIVE.base_url
SITEMAP_URL = ACTIVE.sitemap_url
SHELF_PREFIX = ACTIVE.shelf_prefix

# 教科書でないもの. 棚によっては補助教材が混ざる.
EXCLUDE_BOOKS = ("Supplemental_Modules",)

# 本文ではないページ.
EXCLUDE_PATTERNS = (
    "(Exercises)",
    "(Summary)",
    "Front Matter",
    "Back Matter",
    "Index",
    "Glossary",
)


def _cache_path(url: str, suffix: str = ".html") -> Path:
    """URL からキャッシュファイルのパスを作る.

    保存先はホスト名から決まるので, どのライブラリの URL でも扱える.
    """
    library = urlparse(url).hostname.split(".")[0]
    last_segment = unquote(url.rstrip("/").rsplit("/", 1)[-1])
    stem = re.sub(r"[^A-Za-z0-9._-]", "_", last_segment)[:60]
    digest = hashlib.sha1(url.encode("utf-8")).hexdigest()[:12]
    return DATA_DIR / "raw" / library / f"{stem}__{digest}{suffix}"


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
    # 途中で中断されても壊れたキャッシュが残らないよう, 書き切ってから差し替える
    temporary_path = cache_path.with_name(cache_path.name + ".tmp")
    temporary_path.write_text(text, encoding="utf-8")
    temporary_path.replace(cache_path)
    return text


def _shelf_urls(client: httpx.Client) -> list[str]:
    """棚に属する全 URL を sitemap から返す."""
    xml = _fetch(client, SITEMAP_URL, suffix=".xml")
    return sorted(
        u for u in re.findall(r"<loc>(.*?)</loc>", xml) if u.startswith(SHELF_PREFIX)
    )


def _slug(name: str) -> str:
    """URL の一部から安定した ID を作る."""
    return re.sub(r"[^a-z0-9]+", "_", unquote(name).lower()).strip("_")


def _books(shelf_urls: list[str]) -> list[BookSpec]:
    """棚の URL 一覧から書籍の一覧を作る."""
    names = sorted({u[len(SHELF_PREFIX) :].split("/")[0] for u in shelf_urls})
    return [
        BookSpec(
            book_id=_slug(name),
            title=unquote(name).replace("_", " "),
            prefix=f"{SHELF_PREFIX}{name}/",
        )
        for name in names
        if not any(pattern in name for pattern in EXCLUDE_BOOKS)
    ]


def _leaf_urls(shelf_urls: list[str], book: BookSpec) -> list[str]:
    """その書籍の葉ページ URL を, 本の順序で返す."""
    book_urls = sorted(u for u in shelf_urls if u.startswith(book.prefix))
    # 子を持つ URL は容れ物なので除く
    parents = {u.rsplit("/", 1)[0] for u in book_urls}
    return [u for u in book_urls if u not in parents]


def _toc_path(url: str, book: BookSpec) -> tuple[str, ...]:
    """URL から目次パスを作る."""
    relative = url[len(book.prefix) :]
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


def _is_foreign(soup: BeautifulSoup, book: BookSpec) -> bool:
    """他の書籍から転載されたページなら True を返す."""
    book_path = book.prefix.removeprefix(f"{BASE_URL}/")
    for widget in soup.find_all(attrs={"data-page": True}):
        if not widget["data-page"].startswith(book_path):
            return True
    return False


def _extract_body(soup: BeautifulSoup) -> str:
    """ページ HTML から本文セクションだけを取り出す."""
    container = soup.find("section", class_="mt-content-container")
    if container is None:
        return ""
    return str(container)


def _collect_documents(
    client: httpx.Client,
    book: BookSpec,
    shelf_urls: list[str],
    ctx: RunContext,
) -> list[Document]:
    """1 冊分の本文ページを集める."""
    documents: list[Document] = []
    n_skipped = 0

    for url in _leaf_urls(shelf_urls, book):
        toc_path = _toc_path(url, book)
        if _is_excluded(toc_path):
            n_skipped += 1
            continue

        # パースは1ページにつき1回だけ行い, 判定と抽出で使い回す
        soup = BeautifulSoup(_fetch(client, url), "html.parser")
        if _is_foreign(soup, book):
            n_skipped += 1
            continue

        documents.append(
            Document(
                doc_id=_doc_id(url),
                title=toc_path[-1],
                html=_extract_body(soup),
                toc_path=toc_path,
                order=len(documents),
            )
        )

    ctx.log(
        "acquire.book",
        book_id=book.book_id,
        n_docs=len(documents),
        n_skipped=n_skipped,
    )
    return documents


def run(ctx: RunContext) -> list[Corpus]:
    """棚に並ぶ教科書をすべて取得する."""
    ctx.log("acquire.start", shelf=SHELF_PREFIX)

    with httpx.Client(
        headers={"User-Agent": USER_AGENT},
        timeout=30.0,
        follow_redirects=True,
    ) as client:
        shelf_urls = _shelf_urls(client)
        books = _books(shelf_urls)
        ctx.log("acquire.books_found", n_books=len(books))

        corpora = [
            Corpus(
                book_id=book.book_id,
                book_title=book.title,
                documents=_collect_documents(client, book, shelf_urls, ctx),
            )
            for book in books
        ]

    ctx.log(
        "acquire.done",
        n_books=len(corpora),
        n_docs=sum(len(c.documents) for c in corpora),
    )
    return corpora
