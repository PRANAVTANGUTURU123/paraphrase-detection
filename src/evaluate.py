"""Accuracy/F1 evaluation and cross-dataset comparison table.

The paraphrase decision threshold is tuned on each dataset's validation
split (best F1), then applied to its evaluation split. Results are
printed as a table and saved to results/comparison.csv + metrics.json.

CLI (small --max-samples runs are CPU-safe locally; full runs belong on Kaggle):
  python -m src.evaluate --model-path results/mrpc-run --datasets mrpc qqp paws
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from src.load_data import DATASET_NAMES, eval_split, load_pairs
from src.model import DEFAULT_MODEL, ParaphraseModel

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def find_best_threshold(similarities: np.ndarray, labels) -> float:
    """Sweep thresholds and return the one maximizing F1.

    Sweeps the observed score range, so it works for any score scale —
    cosine similarities in [-1, 1] as well as unbounded cross-encoder logits.
    """
    lo, hi = float(np.min(similarities)), float(np.max(similarities))
    best_t, best_f1 = lo, -1.0
    for t in np.linspace(lo, hi, 200):
        f1 = f1_score(labels, (similarities >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_t, best_f1 = float(t), f1
    return best_t


def evaluate_on_dataset(
    model: ParaphraseModel, name: str, max_samples: int | None = None
) -> dict:
    """Tune threshold on validation, report accuracy/F1 on the eval split."""
    val = load_pairs(name, split="validation", max_samples=max_samples)
    val_labels = [lbl for _, _, lbl in val]
    threshold = find_best_threshold(model.similarity(val), val_labels)

    test = load_pairs(name, split=eval_split(name), max_samples=max_samples)
    test_labels = [lbl for _, _, lbl in test]
    preds = (model.similarity(test) >= threshold).astype(int)

    return {
        "dataset": name,
        "eval_split": eval_split(name),
        "n_examples": len(test),
        "threshold": round(threshold, 2),
        "accuracy": round(accuracy_score(test_labels, preds), 4),
        "f1": round(f1_score(test_labels, preds), 4),
    }


def evaluate_model(
    model: ParaphraseModel,
    dataset_names=DATASET_NAMES,
    max_samples: int | None = None,
    tag: str = "run",
) -> pd.DataFrame:
    rows = []
    for name in dataset_names:
        print(f"Evaluating on {name} ...")
        rows.append(evaluate_on_dataset(model, name, max_samples=max_samples))
    df = pd.DataFrame(rows)

    RESULTS_DIR.mkdir(exist_ok=True)
    df.to_csv(RESULTS_DIR / f"comparison_{tag}.csv", index=False)
    with open(RESULTS_DIR / f"metrics_{tag}.json", "w") as f:
        json.dump(rows, f, indent=2)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a paraphrase model.")
    parser.add_argument(
        "--model-path",
        default=DEFAULT_MODEL,
        help="Fine-tuned model directory (e.g. results/mrpc-run) or a hub model name",
    )
    parser.add_argument(
        "--datasets", nargs="+", choices=DATASET_NAMES, default=DATASET_NAMES
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Cap examples per split — use a small value (e.g. 500) for local CPU checks",
    )
    parser.add_argument("--tag", default="run", help="Suffix for results filenames")
    args = parser.parse_args()

    model = ParaphraseModel.load(args.model_path)
    print(f"Model: {args.model_path} (device: {model.device})")
    df = evaluate_model(model, args.datasets, max_samples=args.max_samples, tag=args.tag)
    print("\n=== Comparison table ===")
    print(df.to_string(index=False))
    print(f"\nSaved to {RESULTS_DIR / f'comparison_{args.tag}.csv'}")


if __name__ == "__main__":
    main()
