# Paraphrase Detection with Sentence-BERT

Fine-tunes a Sentence-BERT model for paraphrase detection and compares
performance across three datasets: **MRPC**, **QQP**, and **PAWS**.

## Project structure

```
data/               downloaded dataset cache (auto-created)
src/
  load_data.py      loads & normalizes MRPC / QQP / PAWS to (s1, s2, label)
  model.py          Sentence-BERT training/inference wrapper
  evaluate.py       accuracy/F1 + cross-dataset comparison table
  train.py          training script (choose dataset(s) via --datasets)
results/            saved checkpoints, metrics JSON, comparison CSVs
requirements.txt
```

All scripts run as modules **from the project root**, e.g.
`python -m src.load_data` — this works identically on Kaggle and Windows.

## Setup — Part 1: Kaggle GPU notebook (Linux, via VS Code tunnel)

This is where training runs. From the project root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt --break-system-packages
python -c "import torch; print(torch.cuda.is_available())"   # should print True
```

## Setup — Part 2: local Windows (CPU, quick checks only)

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Note: do **not** use `--break-system-packages` here — it's only needed on
the Kaggle/Linux side. If activation is blocked by execution policy, run
`Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` first.

### What's safe to run where

| Command | Local (CPU) | Kaggle (GPU) |
|---|---|---|
| `python -m src.load_data` — download datasets, check formats | ✅ yes | ✅ yes |
| `python -m src.model` — tiny embedding sanity check | ✅ yes | ✅ yes |
| `python -m src.evaluate --max-samples 500` — small eval | ✅ yes (slow-ish) | ✅ yes |
| `python -m src.evaluate` — full evaluation | ⚠️ avoid | ✅ yes |
| `python -m src.train --datasets mrpc --sample_size 2000` — small CPU test run | ✅ yes | ✅ yes |
| `python -m src.train --datasets mrpc` — full fine-tuning | ❌ guarded | ✅ yes |

Without CUDA, `train.py` refuses to run unless you pass `--sample_size`
(max 10000), so you can't accidentally start a full CPU training run.

## Usage (on Kaggle)

Train on one dataset:

```bash
python -m src.train --datasets mrpc
```

Train on several (QQP is ~364k pairs — subsample it):

```bash
python -m src.train --datasets mrpc paws --epochs 2
python -m src.train --datasets qqp --sample_size 50000
```

Quick local CPU test on a subset (works without a GPU):

```bash
python -m src.train --datasets mrpc --sample_size 2000
```

Training saves a checkpoint to `results/<tag>/` and, unless `--skip-eval`
is passed, evaluates on **all three** datasets and writes
`results/comparison_<tag>.csv` + `results/metrics_<tag>.json`.

Evaluate any checkpoint (or a plain hub model as a zero-shot baseline):

```bash
python -m src.evaluate --model-path results/mrpc --tag mrpc
python -m src.evaluate --tag baseline        # un-fine-tuned baseline
```

## Method notes

- Base model: `sentence-transformers/all-MiniLM-L6-v2` (override with `--model-name`).
- Training: `CosineSimilarityLoss` on labeled pairs.
- Inference: cosine similarity between embeddings, thresholded into 0/1.
  The threshold is tuned per dataset on its validation split (best F1).
- Eval splits: MRPC and PAWS use their labeled `test` splits; QQP's test
  labels are hidden, so it is evaluated on `validation`.
