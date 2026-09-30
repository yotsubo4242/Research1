"""AL-CPL データセットを data/alcpl/ に取得する.

uv run python script/fetch_alcpl.py
"""

import httpx

from myprogram.paths import DATA_DIR

BASE_URL = "https://raw.githubusercontent.com/harrylclc/AL-CPL-dataset/master/data"
DOMAINS = ("physics", "geometry", "precalculus", "data_mining")
SUFFIXES = ("pairs", "preqs")
ALCPL_DIR = DATA_DIR / "alcpl"


def main() -> None:
    ALCPL_DIR.mkdir(parents=True, exist_ok=True)

    with httpx.Client(timeout=60.0, follow_redirects=True) as client:
        for domain in DOMAINS:
            for suffix in SUFFIXES:
                name = f"{domain}.{suffix}"
                path = ALCPL_DIR / name
                if path.exists():
                    print(f"{name:26} 取得済み")
                    continue

                response = client.get(f"{BASE_URL}/{name}")
                response.raise_for_status()
                path.write_text(response.text, encoding="utf-8")
                print(f"{name:26} {len(response.text.splitlines())} 行")


if __name__ == "__main__":
    main()
