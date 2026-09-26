from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.spatial.distance import cdist

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import (
    DISTANCE,
    GENOMIC_DATASETS,
    SEARCH_BATCH_SIZE,
    VECTORIZE_BATCH_SIZE,
    make_vectorizer,
)


def load_split(dataset_dir: Path, split: str):
    split_dir = dataset_dir / split
    if not split_dir.is_dir():
        raise FileNotFoundError(
            f"Missing {split_dir}. Run genomic/download.py first."
        )

    class_dirs = sorted(path for path in split_dir.iterdir() if path.is_dir())
    if not class_dirs:
        raise ValueError(f"No class directories found under {split_dir}")

    sequences: list[str] = []
    labels: list[int] = []
    class_names = [path.name for path in class_dirs]

    for label, class_dir in enumerate(class_dirs):
        for path in sorted(p for p in class_dir.rglob("*") if p.is_file()):
            seq = "".join(path.read_text(encoding="utf-8").split()).lower()
            if seq:
                sequences.append(seq)
                labels.append(label)

    if not sequences:
        raise ValueError(f"No sequences found under {split_dir}")
    return sequences, np.asarray(labels, dtype=np.int64), class_names


def detect_alphabet(*groups: list[str]) -> str:
    chars = "".join(
        sorted({ch for sequences in groups for sequence in sequences for ch in sequence})
    )
    if not chars:
        raise ValueError("Dataset alphabet is empty")
    return chars


def summarize_alphabet(sequences: list[str], label: str) -> None:
    counts = Counter(ch for seq in sequences for ch in seq)
    observed = "".join(sorted(counts))
    print(f"{label} sequences: {len(sequences):,}")
    print(f"{label} alphabet: {observed} ({sum(counts.values()):,} bases)")


def vectorize_batched(sequences: list[str], chars: str, batch_size: int) -> np.ndarray:
    encoder = make_vectorizer(chars)
    parts: list[np.ndarray] = []
    total = len(sequences)
    t0 = perf_counter()

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        parts.append(encoder.transform(sequences[start:end]))
        print(
            f"\rVectorizing {end:,}/{total:,} ({100.0 * end / total:5.1f}%) "
            f"{perf_counter() - t0:7.1f}s",
            end="",
            flush=True,
        )
    print()
    return np.vstack(parts)


def exact_1nn(
    train_vectors: np.ndarray,
    test_vectors: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    neighbors: list[np.ndarray] = []
    total = len(test_vectors)
    t0 = perf_counter()

    train64 = train_vectors.astype(np.float64, copy=False)
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        distances = cdist(
            test_vectors[start:end].astype(np.float64, copy=False),
            train64,
            metric=DISTANCE,
        )
        # Stable sort makes exact ties deterministic by training-row order.
        nearest = np.argmin(distances, axis=1)
        neighbors.append(nearest.astype(np.int64, copy=False))
        print(
            f"\rSearching    {end:,}/{total:,} ({100.0 * end / total:5.1f}%) "
            f"{perf_counter() - t0:7.1f}s",
            end="",
            flush=True,
        )
    print()
    return np.concatenate(neighbors) if neighbors else np.empty(0, dtype=np.int64)


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    matrix = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(matrix, (y_true, y_pred), 1)
    return matrix


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> float:
    values = []
    for cls in range(n_classes):
        tp = int(np.sum((y_true == cls) & (y_pred == cls)))
        fp = int(np.sum((y_true != cls) & (y_pred == cls)))
        fn = int(np.sum((y_true == cls) & (y_pred != cls)))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        values.append(
            2.0 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
    return float(np.mean(values))


def evaluate(dataset_name: str, data_root: Path) -> dict:
    dataset_dir = data_root / dataset_name
    print("\n" + "=" * 78)
    print(f"Dataset: {dataset_name}")
    print(f"Path   : {dataset_dir}")
    print("=" * 78)

    train_seq, train_labels, train_names = load_split(dataset_dir, "train")
    test_seq, test_labels, test_names = load_split(dataset_dir, "test")
    if train_names != test_names:
        raise ValueError(f"Train/test class mismatch: {train_names} != {test_names}")

    summarize_alphabet(train_seq, "Train")
    summarize_alphabet(test_seq, "Test")
    chars = detect_alphabet(train_seq, test_seq)
    print(f"Detected alphabet: {chars} ({len(chars)} symbols)")
    print(f"Classes: {', '.join(train_names)}")

    print("\nVectorizing training split...")
    train_vectors = vectorize_batched(train_seq, chars, VECTORIZE_BATCH_SIZE)
    print("Vectorizing test split...")
    test_vectors = vectorize_batched(test_seq, chars, VECTORIZE_BATCH_SIZE)
    print(
        f"Vector dimension: {train_vectors.shape[1]} "
        f"(train={train_vectors.shape}, test={test_vectors.shape})"
    )

    print(f"\nExact {DISTANCE} 1-NN search...")
    nearest = exact_1nn(train_vectors, test_vectors, SEARCH_BATCH_SIZE)
    pred = train_labels[nearest]
    acc = float(np.mean(pred == test_labels))
    f1 = macro_f1(test_labels, pred, len(train_names))
    cm = confusion_matrix(test_labels, pred, len(train_names))

    print(f"\n1-NN accuracy : {100.0 * acc:.2f}%")
    print(f"1-NN macro-F1 : {100.0 * f1:.2f}%")
    print("Confusion matrix (rows=true, cols=pred):")
    print(cm)

    return {
        "dataset": dataset_name,
        "classes": train_names,
        "train_sequences": len(train_seq),
        "test_sequences": len(test_seq),
        "alphabet": chars,
        "vector_dimension": int(train_vectors.shape[1]),
        "accuracy": acc,
        "macro_f1": f1,
        "confusion_matrix": cm.tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the single config.py representation on Genomic Benchmarks."
    )
    parser.add_argument(
        "dataset",
        nargs="?",
        default="all",
        help="Dataset name or 'all'.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=ROOT / "datasets" / "genomic_benchmarks",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "genomic_results.json",
    )
    args = parser.parse_args()

    names = GENOMIC_DATASETS if args.dataset == "all" else (args.dataset,)
    started = perf_counter()
    datasets = [evaluate(name, args.data_root) for name in names]

    payload = {
        "distance": DISTANCE,
        "datasets": datasets,
        "elapsed_seconds": perf_counter() - started,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(f"{'dataset':32} {'accuracy':>10} {'macro-F1':>10}")
    for dataset in datasets:
        print(
            f"{dataset['dataset']:32} "
            f"{100.0 * dataset['accuracy']:9.2f}% "
            f"{100.0 * dataset['macro_f1']:9.2f}%"
        )
    print(f"\nSaved results to: {args.output}")


if __name__ == "__main__":
    main()
