"""Load and normalize paraphrase datasets: MRPC, QQP, PAWS.

Every dataset is normalized to (sentence1, sentence2, label) triples,
where label is 1 for paraphrase and 0 for non-paraphrase.

Safe to run locally on CPU (datasets download on first use):
  python -m src.load_data --datasets mrpc     # small, quick first check
  python -m src.load_data                     # all three datasets
"""

import argparse
import json
from pathlib import Path

from datasets import load_dataset

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

DATASET_CONFIGS = {
    "mrpc": {"path": "nyu-mll/glue", "name": "mrpc", "cols": ("sentence1", "sentence2")},
    "qqp": {"path": "nyu-mll/glue", "name": "qqp", "cols": ("question1", "question2")},
    "paws": {
        "path": "google-research-datasets/paws",
        "name": "labeled_final",  # PAWS-Wiki
        "cols": ("sentence1", "sentence2"),
    },
}

DATASET_NAMES = list(DATASET_CONFIGS)
SPLITS = ("train", "validation", "test")

# PAWS-QQP is not distributed on the HF hub: QQP's license requires it to be
# regenerated locally from the raw QQP corpus (see google-research/paws).
# We probe these hub ids anyway so a copy is picked up if one appears.
PAWS_QQP_CANDIDATES = ("google-research-datasets/paws-qqp", "paws-qqp")


def eval_split(name: str) -> str:
    """Best labeled split for final evaluation.

    QQP's GLUE test split has hidden labels (-1), so we evaluate on
    validation. MRPC and PAWS ship real test labels.
    """
    return "validation" if name == "qqp" else "test"


def load_pairs(name: str, split: str = "train", max_samples: int | None = None):
    """Return a list of (sentence1, sentence2, label) triples.

    max_samples: if set, shuffle deterministically and truncate — handy for
    local CPU sanity checks and for taming QQP's ~364k training rows.
    """
    if name not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset {name!r}; choose from {DATASET_NAMES}")
    cfg = DATASET_CONFIGS[name]
    ds = load_dataset(cfg["path"], cfg["name"], split=split, cache_dir=str(DATA_DIR))
    ds = ds.filter(lambda ex: ex["label"] in (0, 1))  # drop hidden-label rows
    if max_samples is not None and max_samples < len(ds):
        ds = ds.shuffle(seed=42).select(range(max_samples))
    s1, s2 = cfg["cols"]
    return [(ex[s1], ex[s2], int(ex["label"])) for ex in ds]


def print_stats(name: str) -> None:
    """Print size and class balance for each split of a dataset."""
    cfg = DATASET_CONFIGS[name]
    print(f"\n=== {name.upper()} ({cfg['path']}/{cfg['name']}) ===")
    for split in SPLITS:
        ds = load_dataset(cfg["path"], cfg["name"], split=split, cache_dir=str(DATA_DIR))
        labels = ds["label"]
        valid = [lbl for lbl in labels if lbl in (0, 1)]
        if not valid:
            print(f"  {split:<12}{len(labels):>8} examples | labels hidden (all -1)")
            continue
        pos = sum(valid)
        pct = 100.0 * pos / len(valid)
        print(
            f"  {split:<12}{len(valid):>8} examples | "
            f"paraphrase {pct:5.1f}% / not-paraphrase {100 - pct:5.1f}%"
        )


def check_paws_qqp() -> bool:
    """Probe the hub for PAWS-QQP; report and skip if unavailable."""
    for hub_id in PAWS_QQP_CANDIDATES:
        try:
            load_dataset(hub_id, split="train", cache_dir=str(DATA_DIR))
            print(f"\nPAWS-QQP: found on the hub as {hub_id!r}")
            return True
        except Exception:
            continue
    print(
        "\nPAWS-QQP: not available on the HF hub — QQP's license requires it "
        "to be generated locally from the raw QQP corpus "
        "(https://github.com/google-research/paws). Skipping."
    )
    return False


def save_samples(names, n: int = 10, path: Path = DATA_DIR / "samples.json") -> None:
    """Save n example (sentence1, sentence2, label) triples per dataset."""
    samples = {}
    for name in names:
        pairs = load_pairs(name, split="train", max_samples=n)
        samples[name] = [
            {"sentence1": s1, "sentence2": s2, "label": lbl} for s1, s2, lbl in pairs
        ]
    path.parent.mkdir(exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)
    print(f"\nSaved {n} sample pairs per dataset to {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download datasets, print stats.")
    parser.add_argument(
        "--datasets", nargs="+", choices=DATASET_NAMES, default=DATASET_NAMES
    )
    args = parser.parse_args()

    for name in args.datasets:
        print_stats(name)
    if "paws" in args.datasets:
        check_paws_qqp()
    save_samples(args.datasets)
