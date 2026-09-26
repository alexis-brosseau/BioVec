from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "results" / "crm_distances.tsv"
DEFAULT_REFERENCE = ROOT / "datasets" / "crm" / "ids.json"
DEFAULT_OUTPUT = ROOT / "results" / "crm_score.txt"


def read_distances(path: Path) -> dict[tuple[str, str], float]:
    """Read an AFproject-compatible 3-column distance TSV."""
    distances: dict[tuple[str, str], float] = {}

    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line:
                continue

            fields = line.split()
            if len(fields) != 3:
                raise ValueError(
                    f"{path}:{line_number}: expected 3 columns, got {len(fields)}"
                )

            left, right, value_text = fields
            pair = tuple(sorted((left, right)))

            try:
                value = float(value_text)
            except ValueError as exc:
                raise ValueError(
                    f"{path}:{line_number}: invalid distance {value_text!r}"
                ) from exc

            distances[pair] = value

    if not distances:
        raise ValueError(f"No distances found in {path}")

    return distances


def load_reference(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def score_tissue(
    tissue_data: dict,
    distance_by_pair: dict[tuple[str, str], float],
) -> tuple[int, int, float]:
    """Mirror the original local AFproject-compatible CRM scorer exactly."""
    seqids = tissue_data["seqids"]
    half = len(seqids) // 2

    positives = seqids[:half]
    negatives = seqids[half:]

    ranked: list[tuple[float, int]] = []

    # Preserve the exact deterministic insertion order used by the old scorer.
    # Python's sort is stable, so this also preserves the old tie behavior.
    for a, b in itertools.combinations(positives, 2):
        pair = tuple(sorted((a, b)))
        try:
            distance = distance_by_pair[pair]
        except KeyError as exc:
            raise ValueError(
                f"Input distance file is missing CRM pair: {pair[0]} / {pair[1]}"
            ) from exc
        ranked.append((distance, 1))

    for a, b in itertools.combinations(negatives, 2):
        pair = tuple(sorted((a, b)))
        try:
            distance = distance_by_pair[pair]
        except KeyError as exc:
            raise ValueError(
                f"Input distance file is missing CRM pair: {pair[0]} / {pair[1]}"
            ) from exc
        ranked.append((distance, 0))

    ranked.sort(key=lambda item: item[0])

    # Kept intentionally identical to the old scorer.
    n = min(len(list(itertools.combinations(positives, 2))), 300)
    k = sum(label for _, label in ranked[:n])
    percent = (100.0 * k / n) if n else 0.0

    return k, n, percent


def score(
    distances: dict[tuple[str, str], float],
    reference: dict,
) -> str:
    percentages: list[float] = []
    ns: list[int] = []
    rows: list[tuple[str, int, int, float]] = []

    for tissue, tissue_data in reference.items():
        k, n, percent = score_tissue(tissue_data, distances)
        percentages.append(percent)
        ns.append(n)
        rows.append((tissue, k, n, percent))

    ps = np.asarray(percentages, dtype=np.float64)
    weights = np.asarray(ns, dtype=np.float64)

    average = float(np.mean(ps))
    weighted_average = (
        float(np.average(ps, weights=weights)) if weights.sum() else 0.0
    )
    std = float(np.std(ps))

    lines = [
        "AFproject-compatible CRM scores",
        "-------------------------------",
        "Tissue\tk\tn\tpercent",
    ]

    lines.extend(
        f"{tissue}\t{k}\t{n}\t{percent:.1f}"
        for tissue, k, n, percent in rows
    )

    lines.extend(
        [
            "",
            f"Weighted average:\t{weighted_average:.2f}",
            f"Standard deviat.:\t{std:.2f}",
            f"Average:\t\t{average:.2f}",
        ]
    )

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Score an AFproject CRM distance TSV using the same ranking logic "
            "as the original local VecFuzz CRM scorer."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Distance TSV (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=DEFAULT_REFERENCE,
        help=f"AFproject ids.json (default: {DEFAULT_REFERENCE})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Text score report (default: {DEFAULT_OUTPUT})",
    )
    args = parser.parse_args()

    distances = read_distances(args.input)
    reference = load_reference(args.reference)
    report = score(distances, reference)

    print(report)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report + "\n", encoding="utf-8")
    print(f"\nSaved score report to: {args.output}")


if __name__ == "__main__":
    main()
