"""Re-score the errors captured in results/paws_errors.json with another model.

Shows side by side: the bi-encoder similarity recorded when the errors were
captured, and the new model's score for the same pairs — the direct test of
whether a cross-encoder distinguishes pairs the bi-encoder saw as identical.

Usage:
  python -m src.retest_errors --model-path results/qqp-paws-cross \
      --model-type cross --threshold 0.5
"""

import argparse
import json
from pathlib import Path

from src.model import CrossEncoderParaphraseModel, ParaphraseModel

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-score captured error pairs.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-type", choices=["bi", "cross"], default="cross")
    parser.add_argument(
        "--threshold",
        type=float,
        required=True,
        help="Decision threshold from the training run of the new model",
    )
    parser.add_argument("--errors-file", default=str(RESULTS_DIR / "paws_errors.json"))
    args = parser.parse_args()

    with open(args.errors_file, encoding="utf-8") as f:
        records = json.load(f)

    cls = CrossEncoderParaphraseModel if args.model_type == "cross" else ParaphraseModel
    model = cls.load(args.model_path)
    pairs = [(r["sentence1"], r["sentence2"], r["true_label"]) for r in records]
    scores = model.similarity(pairs)

    fixed = 0
    print(f"{'#':>2}  {'type':<4} {'true':>4} {'bi_sim':>7} {'new':>7} {'new_pred':>8}  verdict")
    for i, (rec, score) in enumerate(zip(records, scores), 1):
        pred = int(score >= args.threshold)
        ok = pred == rec["true_label"]
        fixed += ok
        rec["retest_score"] = round(float(score), 4)
        rec["retest_pred"] = pred
        rec["retest_correct"] = bool(ok)
        print(
            f"{i:>2}  {rec['error_type']:<4} {rec['true_label']:>4} "
            f"{rec['similarity']:>7.3f} {float(score):>7.3f} {pred:>8}  "
            f"{'FIXED' if ok else 'still wrong'}"
        )
    print(f"\n{fixed}/{len(records)} previously-wrong pairs now correct")

    out = RESULTS_DIR / "paws_errors_retest.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
