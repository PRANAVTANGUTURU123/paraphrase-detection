"""Inspect PAWS validation errors made by a fine-tuned paraphrase model.

Selects a mix of false negatives (up to 5, most confident first), confident
false positives, and near-threshold false positives, prints them, and saves
them to results/paws_errors.json for write-up.

Usage (threshold should match the training run that produced the model):
  python -m src.error_analysis --model-path results/qqp-paws --threshold 0.66
"""

import argparse
import json
from pathlib import Path

import numpy as np

from src.load_data import load_pairs
from src.model import ParaphraseModel

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze PAWS validation errors.")
    parser.add_argument("--model-path", default=str(RESULTS_DIR / "qqp-paws"))
    parser.add_argument(
        "--threshold",
        type=float,
        required=True,
        help="Decision threshold from the training run being analyzed",
    )
    parser.add_argument("--n", type=int, default=15, help="Total errors to select")
    args = parser.parse_args()

    pairs = load_pairs("paws", "validation")
    labels = np.array([lbl for _, _, lbl in pairs])
    model = ParaphraseModel.load(args.model_path)
    print(f"Scoring {len(pairs)} PAWS validation pairs with {args.model_path} ...")
    sims = model.similarity(pairs)
    preds = (sims >= args.threshold).astype(int)

    fp_idx = np.where((preds == 1) & (labels == 0))[0]
    fn_idx = np.where((preds == 0) & (labels == 1))[0]
    print(
        f"Errors at threshold {args.threshold}: "
        f"{len(fp_idx)} false positives, {len(fn_idx)} false negatives "
        f"({(len(fp_idx) + len(fn_idx)) / len(pairs):.1%} of {len(pairs)})"
    )

    # False negatives: lowest similarity first (most confidently wrong).
    fn_sorted = fn_idx[np.argsort(sims[fn_idx])]
    chosen_fn = list(fn_sorted[: min(5, len(fn_sorted))])

    # False positives: half most-confident (highest sim), half near-threshold.
    n_fp = args.n - len(chosen_fn)
    fp_confident = fp_idx[np.argsort(-sims[fp_idx])]
    fp_near = fp_idx[np.argsort(np.abs(sims[fp_idx] - args.threshold))]
    chosen_fp = list(dict.fromkeys(
        list(fp_confident[: n_fp - n_fp // 2]) + list(fp_near[: n_fp // 2])
    ))
    for i in fp_confident:
        if len(chosen_fp) >= n_fp:
            break
        if i not in chosen_fp:
            chosen_fp.append(i)

    records = []
    for idx in chosen_fn + chosen_fp:
        s1, s2, lbl = pairs[idx]
        records.append(
            {
                "error_type": "FN" if labels[idx] == 1 else "FP",
                "similarity": round(float(sims[idx]), 4),
                "true_label": int(labels[idx]),
                "predicted": int(preds[idx]),
                "sentence1": s1,
                "sentence2": s2,
            }
        )

    for i, r in enumerate(records, 1):
        print(
            f"\n[{i}] {r['error_type']}  sim={r['similarity']:.3f}  "
            f"true={r['true_label']}  pred={r['predicted']}\n"
            f"  s1: {r['sentence1']}\n  s2: {r['sentence2']}"
        )

    out = RESULTS_DIR / "paws_errors.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    print(f"\nSaved {len(records)} errors to {out}")


if __name__ == "__main__":
    main()
