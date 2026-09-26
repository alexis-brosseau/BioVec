from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import GENOMIC_DATASETS


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the Genomic Benchmarks datasets.")
    parser.add_argument(
        "dataset",
        nargs="?",
        default="all",
        help="Dataset name or 'all'.",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=ROOT / "datasets" / "genomic_benchmarks",
    )
    args = parser.parse_args()

    try:
        from genomic_benchmarks.loc2seq import download_dataset
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency 'genomic-benchmarks'. Install requirements first:\n"
            "  python -m pip install -r requirements.txt"
        ) from exc

    names = GENOMIC_DATASETS if args.dataset == "all" else (args.dataset,)
    args.dest.mkdir(parents=True, exist_ok=True)

    for name in names:
        print(f"Downloading {name} ...")
        path = download_dataset(name, version=0, dest_path=args.dest)
        print(f"Saved to: {path}")


if __name__ == "__main__":
    main()
