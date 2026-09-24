from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Self


@dataclass
class RunContext:
    run_dir: Path

    @classmethod
    def create(cls, name: str) -> Self:
        time_stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
        # Todo: カレントからの相対パスだから後でプロジェクトルートからの相対パスに変更する.
        run_dir = Path("runs") / f"{time_stamp}_{name}"
        run_dir.mkdir(parents=True, exist_ok=True)
        return cls(run_dir=run_dir)

    def log(self, event: str, **fields: object) -> None:
        print(event, fields)
