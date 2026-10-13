"""扱う分野の設定.

AL-CPL の分野名と, 対応する LibreTexts の棚を 1 箇所で対応づける.
分野を切り替えるときは環境変数 MYPROGRAM_DOMAIN を指定する.

    MYPROGRAM_DOMAIN=geometry uv run myprogram
"""

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class DomainConfig:
    # AL-CPL の分野名. data/alcpl/<name>.pairs に対応する
    name: str
    # LibreTexts のライブラリ (サブドメイン)
    library: str
    # 棚の名前
    shelf: str
    # その棚から外す書籍. URL の書籍名に対する部分一致で判定する.
    # 版違いの重複や, 教科書でない補助教材を除くために使う.
    exclude_books: tuple[str, ...] = ()

    @property
    def base_url(self) -> str:
        return f"https://{self.library}.libretexts.org"

    @property
    def shelf_prefix(self) -> str:
        return f"{self.base_url}/Bookshelves/{self.shelf}/"

    @property
    def sitemap_url(self) -> str:
        return f"{self.base_url}/sitemap.xml"


DOMAINS = {
    config.name: config
    for config in [
        DomainConfig("physics", "phys", "University_Physics"),
        DomainConfig("geometry", "math", "Geometry"),
        DomainConfig(
            "precalculus",
            "math",
            "Precalculus",
            # 1e と 2e は版違いで本文が重複する. 新しい 2e だけを残す.
            # Exercises_(Precalculus) は演習だけの冊子で教科書ではない.
            exclude_books=("Precalculus_1e_(OpenStax)", "Exercises_(Precalculus)"),
        ),
        DomainConfig("data_mining", "eng", "Data_Science"),
    ]
}

ACTIVE = DOMAINS[os.environ.get("MYPROGRAM_DOMAIN", "physics")]
