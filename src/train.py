"""Fine-tune Sentence-BERT and compare cross-dataset transfer.

Two modes:
  Single run:  python -m src.train --datasets qqp --sample_size 2000
  Comparison:  python -m src.train --compare --sample_size 2000
    Trains a QQP-only model and a QQP+PAWS-combined model under the same
    TOTAL training budget, evaluates both on QQP and PAWS validation, and
    writes the combined table to results/comparison.csv + comparison.md.

When --sample_size is given with multiple training datasets, the budget is
split EVENLY between them, not proportionally to dataset size: QQP is ~7x
larger than PAWS, so a proportional 2000-pair split would give PAWS only
~250 examples — too few to teach the word-order sensitivity that is the
whole point of adding PAWS.

Threshold protocol: tuned once on the validation split(s) of the dataset(s)
the model was trained on, then frozen for every evaluation. Evaluation
datasets outside the training set get no adaptation of any kind.

Full-dataset training needs a GPU. On CPU (local Windows), pass a small
--sample_size (<= 10000) to run a fast test on a subset.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score

from src.evaluate import find_best_threshold
from src.load_data import DATASET_NAMES, load_pairs
from src.model import (
    DEFAULT_CROSS_MODEL,
    DEFAULT_MODEL,
    CrossEncoderParaphraseModel,
    ParaphraseModel,
)

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def markdown_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "|" + "|".join("---" for _ in cols) + "|",
    ]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    return "\n".join(lines) + "\n"


def run_experiment(label: str, train_names, eval_names, args, model_type: str = "bi") -> list[dict]:
    """Train one model, evaluate on each eval dataset's validation split."""
    tag = "-".join(train_names) + ("-cross" if model_type == "cross" else "")
    output_path = RESULTS_DIR / tag
    budget = None if args.sample_size is None else max(1, args.sample_size // len(train_names))

    print(f"\n### Experiment: {label} "
          f"({'full data' if budget is None else f'{budget} pairs per dataset'})")
    cls = CrossEncoderParaphraseModel if model_type == "cross" else ParaphraseModel
    if args.eval_only:
        model = cls.load(str(output_path))
        print(f"  Loaded checkpoint {output_path} ({model_type}-encoder) on {model.device}")
    else:
        train_pairs = []
        for name in train_names:
            pairs = load_pairs(name, split="train", max_samples=budget)
            print(f"  {name}: {len(pairs)} training pairs")
            train_pairs.extend(pairs)
        print(f"  Total: {len(train_pairs)} pairs")

        model_name = args.model_name or (
            DEFAULT_CROSS_MODEL if model_type == "cross" else DEFAULT_MODEL
        )
        model = cls(model_name)
        print(f"  Training {model_name} ({model_type}-encoder) on {model.device} "
              f"for {args.epochs} epoch(s)")
        model.train(
            train_pairs,
            epochs=args.epochs,
            batch_size=args.batch_size,
            output_path=str(output_path),
        )
        model.save(str(output_path))
        print(f"  Model saved to {output_path}")

    # Encode each needed validation split once; reuse for tuning and metrics.
    encoded = {}
    for name in dict.fromkeys(list(eval_names) + list(train_names)):
        pairs = load_pairs(name, "validation", max_samples=args.eval_sample_size)
        labels = [lbl for _, _, lbl in pairs]
        print(f"  Encoding {name} validation ({len(labels)} pairs) ...")
        encoded[name] = (model.similarity(pairs), labels)

    tune_sims = np.concatenate([encoded[n][0] for n in train_names])
    tune_labels = [lbl for n in train_names for lbl in encoded[n][1]]
    threshold = find_best_threshold(tune_sims, tune_labels)
    print(f"  Threshold tuned on {'+'.join(train_names)} validation: {threshold:.2f}")

    rows = []
    for name in eval_names:
        sims, labels = encoded[name]
        preds = (sims >= threshold).astype(int)
        rows.append(
            {
                "model": label,
                "dataset": f"{name.upper()} (validation)",
                "n": len(labels),
                "pred_positive_rate": round(float(preds.mean()), 4),
                "accuracy": round(accuracy_score(labels, preds), 4),
                "f1": round(f1_score(labels, preds), 4),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Sentence-BERT paraphrase models.")
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Run the QQP-only vs QQP+PAWS-combined comparison",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=DATASET_NAMES,
        default=None,
        help="Which dataset(s) to train on (single-run mode), e.g. --datasets qqp",
    )
    parser.add_argument(
        "--eval-datasets",
        nargs="+",
        choices=DATASET_NAMES,
        default=None,
        help="Datasets whose validation split to evaluate on "
        "(default: qqp+paws in compare mode; training sets + paws otherwise)",
    )
    parser.add_argument(
        "--model-type",
        choices=["bi", "cross"],
        default="bi",
        help="bi = Sentence-BERT bi-encoder (cosine sim); "
        "cross = cross-encoder scoring both sentences jointly",
    )
    parser.add_argument(
        "--model-name",
        default=None,
        help=f"Base model (default: {DEFAULT_MODEL} for bi, {DEFAULT_CROSS_MODEL} for cross)",
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--sample_size",
        "--max-samples",
        dest="sample_size",
        type=int,
        default=None,
        help="TOTAL training-example budget, split evenly across training "
        "datasets (e.g. 2000 for local CPU runs)",
    )
    parser.add_argument(
        "--eval_sample_size",
        type=int,
        default=None,
        help="Cap validation examples per dataset during evaluation",
    )
    parser.add_argument("--tag", default=None, help="Run label (single-run mode)")
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Skip training; load the checkpoint from results/<tag> and re-evaluate",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        if args.sample_size is None or args.sample_size > 10000:
            print(
                "CUDA not available, and no small --sample_size given.\n"
                "Full-dataset training on CPU is impractically slow — pass e.g.\n"
                "  --sample_size 2000\n"
                "for a local CPU test run, or run full training on a GPU."
            )
            raise SystemExit(1)
        print(f"CUDA not available — training on CPU with sample_size={args.sample_size}.")

    if args.compare:
        eval_names = args.eval_datasets or ["qqp", "paws"]
        experiments = [
            ("QQP-only", ["qqp"], "bi"),
            ("QQP+PAWS combined", ["qqp", "paws"], "bi"),
        ]
        out_stem = "comparison"
    else:
        if not args.datasets:
            parser.error("--datasets is required unless --compare is given")
        eval_names = args.eval_datasets or list(dict.fromkeys(args.datasets + ["paws"]))
        default_label = "+".join(args.datasets) + (
            " (cross-encoder)" if args.model_type == "cross" else ""
        )
        experiments = [(args.tag or default_label, args.datasets, args.model_type)]
        stem_tag = "-".join(args.datasets) + ("-cross" if args.model_type == "cross" else "")
        out_stem = f"comparison_{stem_tag}"

    RESULTS_DIR.mkdir(exist_ok=True)
    all_rows = []
    for label, train_names, model_type in experiments:
        all_rows.extend(
            run_experiment(label, train_names, eval_names, args, model_type=model_type)
        )

    df = pd.DataFrame(all_rows)
    print("\n=== Comparison ===")
    print(df.to_string(index=False))
    print("\nNote: each model's threshold is tuned on the validation split(s) of "
          "its own training dataset(s) — those rows are slightly optimistic; "
          "rows for datasets outside the training set are pure transfer.")

    csv_path = RESULTS_DIR / f"{out_stem}.csv"
    df.to_csv(csv_path, index=False)
    with open(RESULTS_DIR / f"{out_stem}.md", "w", encoding="utf-8") as f:
        f.write(markdown_table(df))
    print(f"Saved results to {csv_path} and {csv_path.with_suffix('.md')}")


if __name__ == "__main__":
    main()
