import json
import time
from pathlib import Path
from typing import Any
import httpx
from myprogram.context import RunContext
from myprogram.types import Corpus, Document
from myprogram.paths import DATA_DIR

BASE_URL = "https://chem.libretexts.org/@api/deki"
REQUEST_INTERVAL = 0.5
CACHE_DIR = DATA_DIR / "raw" / "chem"
USER_AGENT = "Research bot"
EXCLUDE_PATTERNS = (
    "Front Matter", "Back Matter", "Appendices",
    "Key Terms", "Key Equations", "Summary", "Exercises",
)
# Chemistry 2e (OpenStax)
BOOK_ID = "414590"

def _as_list(value: Any) -> list[Any]:
    """
    Convert a value to a list. If the value is None, return an empty list.
    """
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _get_json(client: httpx.Client, endpoint: str) -> dict[str, Any]:
    """
    Get JSON data from an endpoint excluding the cache.
    """
    cache_path = CACHE_DIR / f"{endpoint.replace('/', '_')}.json"
    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))

    response = client.get(
        f"{BASE_URL}/{endpoint}",
        params={"dream.out.format": "json"},
    )
    response.raise_for_status()
    time.sleep(REQUEST_INTERVAL)

    data = response.json()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return data


def _subpages(client: httpx.Client, page_id: str) -> list[dict[str, Any]]:
    """
    Get the list of subpages for a given page ID.
    """
    data = _get_json(client, f"pages/{page_id}/subpages")
    return _as_list(data.get("page.subpage"))


def _contents(client: httpx.Client, page_id: str) -> str:
    """
    Get the HTML contents of a page.
    """
    data = _get_json(client, f"pages/{page_id}/contents")
    body = _as_list(data.get("body"))
    if not body:
        return ""
    first = body[0]
    if isinstance(first, str):
        return first
    return ""


def _is_excluded(title: str) -> bool:
    """
    Check if a title matches any of the excluded patterns.
    """
    for excluded_pattern in EXCLUDE_PATTERNS:
        if excluded_pattern in title:
            return True
    return False


def _walk(client: httpx.Client, page_id: str, toc_path: tuple[str, ...], ctx: RunContext) -> list[Document]:
    """
    get the list of documents by walking through the subpages recursively.
    """
    documents: list[Document] = []

    for child in _subpages(client, page_id):
        title = child["title"]
        if _is_excluded(title):
            continue

        child_id = child["@id"]
        child_path = toc_path + (title,)

        if child.get("@subpages") == "true":
            found = _walk(client, child_id, child_path, ctx)
            ctx.log("acquire.chapter", title=title, n_docs=len(found))
            documents.extend(found)
        else:
            documents.append(
                Document(
                    doc_id=child_id,
                    title=title,
                    html=_contents(client, child_id),
                    toc_path=child_path,
                )
            )
    return documents

def run(ctx: RunContext) -> Corpus:
    """
    Acquire a corpus of documents.
    """
    ctx.log("acquire.start", book_id=BOOK_ID)

    with httpx.Client(headers={"User-Agent": USER_AGENT}, timeout=30.0) as client:
        book = _get_json(client, f"pages/{BOOK_ID}")
        documents = _walk(client, BOOK_ID, (), ctx)

    corpus = Corpus(
        book_id=BOOK_ID,
        book_title=book["title"],
        documents=documents,
    )
    ctx.log("acquire.done", n_docs=len(documents))
    return corpus
