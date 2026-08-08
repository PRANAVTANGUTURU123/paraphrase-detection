# Paraphrase Detection: Bi-Encoder vs Cross-Encoder on QQP and PAWS

Fine-tunes Sentence-BERT-style models for paraphrase detection and asks a
specific question: **when a bi-encoder collapses on adversarial word-swap
paraphrases (PAWS), is the fix more data or a different architecture?**
The answer turned out to be neither cleanly: full-data training lifted the
bi-encoder's PAWS accuracy from 53.5% to 70.2% and the cross-encoder leads
overall (87.5% QQP / 76.1% PAWS), but the most extreme failure cases —
sentence pairs the bi-encoder maps to *identical embeddings* — were fixed
by neither model. For the bi-encoder that ceiling is mathematically
unfixable; for the cross-encoder it is not, yet 25× more data didn't fix
it either, pointing at the training signal as the open suspect.

## Background: why PAWS

Models scored on ordinary paraphrase data (Quora Question Pairs, MRPC) can
succeed via a shortcut: high word overlap ≈ paraphrase. **PAWS**
(Paraphrase Adversaries from Word Scrambling) is built to break that
shortcut: its pairs have near-identical vocabulary, with meaning either
preserved or flipped by word-order swaps — e.g. *"a Brazilian footballer
who plays for the Portuguese association"* vs *"a Portuguese footballer
who plays for the Brazilian Association"* (not paraphrases, same words).
Roughly 44% of PAWS pairs are true paraphrases, so lexical overlap carries
almost no label signal.

## Method

Two architectures, identical data and protocol:

- **Bi-encoder** (`sentence-transformers/all-MiniLM-L6-v2`): encodes each
  sentence separately, fine-tuned with CosineSimilarityLoss; a pair is a
  paraphrase if embedding cosine similarity clears a threshold.
- **Cross-encoder** (`cross-encoder/ms-marco-MiniLM-L-6-v2`): both
  sentences pass through the transformer *jointly*, fine-tuned with BCE
  loss. Same MiniLM backbone capacity, so the comparison isolates
  architecture, not size.

Datasets are loaded from the HuggingFace hub and normalized to
`(sentence1, sentence2, label)` (`src/load_data.py`; MRPC is also
supported). Training on QQP-only or QQP+PAWS combined; with a
`--sample_size` budget the examples are split evenly across datasets. The
decision threshold is tuned once on the validation split(s) of the
training dataset(s), then frozen — evaluation datasets outside the
training set get no adaptation. QQP is evaluated on its validation split
(its test labels are hidden); all evaluations report accuracy, F1, and
the positive-prediction rate, since QQP skews ~63% not-paraphrase and a
degenerate all-positive model gets F1 ≈ 0.61 on PAWS for free.

## Results

**Stage 1 — small-sample CPU runs** (2000 training pairs, 1 epoch): the
QQP-trained bi-encoder works on QQP but collapses on PAWS, predicting
"paraphrase" for 99.5% of pairs:

| model | dataset | pred_positive_rate | accuracy | f1 |
|---|---|---|---|---|
| QQP-only (bi) | QQP (validation) | 0.496 | 0.774 | 0.739 |
| QQP-only (bi) | PAWS (validation) | 0.995 | 0.443 | 0.613 |
| QQP+PAWS combined (bi) | QQP (validation) | 0.523 | 0.766 | 0.738 |
| QQP+PAWS combined (bi) | PAWS (validation) | 0.861 | 0.535 | 0.643 |
| Cross-encoder QQP+PAWS | QQP (validation) | 0.552 | 0.729 | 0.705 |
| Cross-encoder QQP+PAWS | PAWS (validation) | 0.992 | 0.444 | 0.613 |

1000 PAWS training pairs bought 9 points of PAWS accuracy at almost no QQP
cost; the small-sample cross-encoder showed no advantage.

**Stage 2 — full-data GPU runs** (Kaggle; all 363,846 QQP + 49,401 PAWS
training pairs; from `results/comparison.md` and
`results/comparison_qqp-paws-cross.md`):

| model | dataset | pred_positive_rate | accuracy | f1 |
|---|---|---|---|---|
| QQP-only (bi) | QQP (validation) | 0.4217 | 0.8544 | 0.8156 |
| QQP-only (bi) | PAWS (validation) | 0.8768 | 0.5124 | 0.6303 |
| QQP+PAWS combined (bi) | QQP (validation) | 0.4232 | 0.8553 | 0.8172 |
| QQP+PAWS combined (bi) | PAWS (validation) | 0.6571 | 0.7015 | 0.7285 |
| Cross-encoder QQP+PAWS | QQP (validation) | 0.4109 | 0.8747 | 0.8392 |
| Cross-encoder QQP+PAWS | PAWS (validation) | 0.6089 | 0.7608 | 0.7724 |

Full-data training helped everywhere, and the cross-encoder leads on both
datasets. But aggregate numbers hide the interesting part.

## Error analysis: the identical-embedding ceiling

From the small-sample combined bi-encoder's PAWS errors, 15 cases were
captured and tracked across every subsequent model
(`results/error_analysis.md`, `results/paws_errors.json`,
`results/paws_errors_retest.json`): 5 false negatives (true paraphrases
over-penalized for reordering), 5 maximum-confidence false positives —
swap pairs scored at **cosine similarity exactly 1.000**, i.e. identical
embeddings — and 5 moderate (~0.66) false positives.

What full-data training fixed, per model:

| error tier | full-data bi-encoder | full-data cross-encoder |
|---|---|---|
| 5 identical-embedding swaps (sim = 1.000) | 1/5 fixed | 1/5 fixed |
| 5 moderate swaps (sim ≈ 0.66) | 5/5 fixed | 4/5 fixed |
| 5 false negatives (over-penalized reorders) | 0/5 fixed | 1/5 fixed |

Three findings:

1. **The bi-encoder's ceiling is mathematical.** The unambiguous
   sim=1.000 cases still score ≥ 0.98 after 25× more PAWS data. Identical
   vectors cannot be separated by any threshold or any amount of training
   — mean-pooling discards the word-order information before comparison.
   (The single "fixed" case in both models is a pair whose gold label is
   itself arguable — a symmetric predicate where the swap doesn't change
   meaning.)
2. **The cross-encoder shows capability present, correction absent.** It
   assigns those same pairs distinct, unsaturated scores (logits 0.21 to
   2.89) — it can mechanically distinguish them — yet remains confidently
   wrong on 4 of 5, nearly unchanged from its small-sample result.
3. **The over-penalization pattern migrated architectures.** The
   small-sample cross-encoder got all 5 false negatives right; after
   full-data training it rejects 4 of them. Full-strength PAWS pressure
   taught the cross-encoder, too, to punish meaning-preserving reordering
   — evidence that this failure follows the *training signal*, not the
   pooling architecture.

**Bottom line:** more data fixes the moderate failure tier and lifts every
aggregate metric; the extreme tier is provably out of reach for the
bi-encoder and empirically uncorrected in the cross-encoder. Whether the
cross-encoder's residual failures stem from loss design, training
duration, or base-model mismatch is the open question this project
surfaces rather than resolves.

## Limitations

- **Single epoch everywhere**; no learning-rate or epoch sweeps.
- **Cross-encoder base mismatch**: `ms-marco-MiniLM-L-6-v2` is pretrained
  for passage ranking, not sentence-pair classification (chosen for
  capacity parity with the bi-encoder and to fit an 8 GB local machine
  after the originally-planned `stsb-distilroberta-base` was OOM-killed).
- **Threshold caveats**: thresholds are tuned on the same validation
  splits later used for evaluation of in-training-set datasets (slightly
  optimistic for those rows; transfer rows are unaffected). An early
  threshold-sweep bug (cosine-range sweep applied to unbounded logits)
  was found and fixed mid-project; all reported cross-encoder numbers
  post-date the fix.
- QQP's test labels are hidden, so QQP is evaluated on validation;
  PAWS-QQP is not on the HF hub (license requires local generation) and
  was not used — PAWS here means PAWS-Wiki `labeled_final`.
- The 15 tracked error cases are a small, deliberately extreme sample.

## Reproducing

Local setup (Windows/CPU shown; Linux equivalent works):

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m src.load_data              # download datasets, print stats
```

Experiments (small-sample CPU versions; drop `--sample_size` on a GPU for
the full-data runs — `train.py` refuses full-data training without CUDA):

```bash
python -m src.train --compare --sample_size 2000                 # both bi-encoder runs
python -m src.train --datasets qqp paws --model-type cross \
    --tag "Cross-encoder QQP+PAWS" --sample_size 2000            # cross-encoder
python -m src.error_analysis --model-path results/qqp-paws --threshold <t>   # capture errors
python -m src.retest_errors --model-path results/qqp-paws-cross \
    --model-type cross --threshold <t>                           # re-test captured errors
```

`<t>` is the threshold printed by the corresponding training run. For
Kaggle GPU training, upload `setup_and_train.ipynb` (see its cells:
GPU check → VS Code tunnel → clone → env → CUDA check) and run the same
commands without `--sample_size`.

## Future work

- **Isolate the cross-encoder bottleneck**: retrain from an NLI/STS
  sentence-pair base instead of ms-marco, run multiple epochs, and track
  the 15 captured cases' scores across checkpoints to see whether the
  swap tier ever starts to move.
- **Loss design for the false-negative migration**: the BCE/cosine signal
  on PAWS' distribution punishes all reordering; a loss or data mix that
  distinguishes meaning-preserving from meaning-flipping reorders is the
  targeted fix.
- Extend the transfer matrix to MRPC (already supported by the data
  loader) and to locally-generated PAWS-QQP.
