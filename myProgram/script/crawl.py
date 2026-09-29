"""LibreTexts のページをまとめて取得し, キャッシュに貯める.

解析は行わない. ネットワークが要る作業だけを先に済ませ,
acquire 側の解析は後からキャッシュ相手に何度でもやり直せるようにする.

robots.txt の Crawl-delay: 5 を守るため, 1ページあたり5秒かかる.

    uv run python script/crawl.py
"""

import re
import sys
import time

import httpx

from myprogram.stages.acquire import USER_AGENT, _cache_path, _fetch

# AL-CPL の4分野に対応する棚. 優先度の高い順に並べる.
TARGETS = [
    ("physics", "phys", "Bookshelves/University_Physics/"),
    ("geometry", "math", "Bookshelves/Geometry/"),
    ("precalculus", "math", "Bookshelves/Precalculus/"),
    ("data mining", "eng", "Bookshelves/Data_Science/"),
]

# 本文でも用語源でもないページ. 取得を省いて負荷と時間を減らす.
SKIP_PATTERNS = (
    "(Exercises)",
    "(Answers)",
    "(Solutions)",
    "Front_Matter",
    "Back_Matter",
    "InfoPage",
    "TitlePage",
    "Table_of_Contents",
    "Licensing",
    "Detailed_Licensing",
)


def _sitemap_urls(client: httpx.Client, library: str) -> list[str]:
    """ライブラリの sitemap から全 URL を返す."""
    url = f"https://{library}.libretexts.org/sitemap.xml"
    return re.findall(r"<loc>(.*?)</loc>", _fetch(client, url, suffix=".xml"))


def _should_skip(url: str) -> bool:
    return any(pattern in url for pattern in SKIP_PATTERNS)


def _crawl_shelf(client: httpx.Client, label: str, library: str, shelf: str) -> None:
    prefix = f"https://{library}.libretexts.org/{shelf}"
    urls = sorted(u for u in _sitemap_urls(client, library) if u.startswith(prefix))
    targets = [u for u in urls if not _should_skip(u)]
    pending = [u for u in targets if not _cache_path(u).exists()]

    print(
        f"[{label}] 棚 {len(urls)} / 対象 {len(targets)} / 未取得 {len(pending)}"
        f"  (推定 {len(pending) * 5 // 60} 分)",
        flush=True,
    )

    for i, url in enumerate(pending, 1):
        try:
            _fetch(client, url)
        except Exception as exc:  # noqa: BLE001  # 1ページの失敗で数時間の作業を止めない
            print(f"[{label}] ERROR {type(exc).__name__} {url}", flush=True)
            time.sleep(5.0)
            continue
        if i % 25 == 0:
            print(f"[{label}] {i}/{len(pending)}", flush=True)

    print(f"[{label}] 完了", flush=True)


def main() -> None:
    started = time.time()
    with httpx.Client(
        headers={"User-Agent": USER_AGENT},
        timeout=60.0,
        follow_redirects=True,
    ) as client:
        for label, library, shelf in TARGETS:
            try:
                _crawl_shelf(client, label, library, shelf)
            except Exception as exc:  # noqa: BLE001  # 1つの棚の失敗で残りを止めない
                print(f"[{label}] 棚ごと失敗: {type(exc).__name__} {exc}", flush=True)
    print(f"全体完了 {(time.time() - started) / 60:.1f} 分", flush=True)


if __name__ == "__main__":
    sys.exit(main())
