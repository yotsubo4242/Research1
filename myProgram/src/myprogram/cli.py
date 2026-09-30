from myprogram.domain import ACTIVE
from myprogram.pipeline import run_all


def main() -> None:
    run_dir = run_all(ACTIVE.name)
    print(f"Run completed. Results are saved in: {run_dir}")


if __name__ == "__main__":
    main()
