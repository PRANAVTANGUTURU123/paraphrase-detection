# PAWS Validation Error Analysis — QQP+PAWS Combined Model

> **Scope caveat:** these results come from the **small-sample CPU run**
> (2000 training pairs total: 1000 QQP + 1000 PAWS, 1 epoch,
> `all-MiniLM-L6-v2`, threshold 0.66 tuned on QQP+PAWS validation) — **not**
> the eventual full-dataset GPU run. The specific error rate below is not
> final and should improve with full training. What is informative here are
> the **error categories and patterns**, which reflect properties of the
> bi-encoder architecture rather than of the training budget.

## Overall errors at threshold 0.66

| | count |
|---|---|
| PAWS validation pairs | 8000 |
| False positives (pred paraphrase, truly not) | 3537 |
| False negatives (pred not, truly paraphrase) | 187 |
| Total error rate | 46.6% |

The error mass is overwhelmingly false positives — the model still says
"paraphrase" for most high-overlap pairs. 15 examples were selected for
inspection: all 5 most-confident false negatives, 5 most-confident false
positives, and 5 near-threshold false positives (raw data:
`paws_errors.json`).

## The 15 inspected errors

| # | Type | Sim | True | Pred | Sentence 1 | Sentence 2 |
|---|------|-----|------|------|------------|------------|
| 1 | FN | 0.051 | 1 | 0 | Thoranam is a 1987 Indian Malayalam film, directed by Joseph Madappally and produced by V Rajan. | In 1987, a malayalam Indian film, directed by Joseph Madappally and produced by V Rajan. |
| 2 | FN | 0.097 | 1 | 0 | Hardwick appeared in "Coronation Street" in 1997 as Naomi Russell and in 1998 as Sheila Dixon. | In 1997, she appeared as Naomi Russell in "Coronation Street" and in 1998 as Sheila Dixon. |
| 3 | FN | 0.110 | 1 | 0 | In 1997, she appeared as Sheila Dixon in "Coronation Street" and Naomi Russell in 1998. | Hardwick appeared in "Coronation Street" in 1997 as Sheila Dixon and in 1998 as Naomi Russell. |
| 4 | FN | 0.113 | 1 | 0 | Kuykendall served alongside Robert White and Joshua Soule Zimmerman as a Chancery Commissioner for Hampshire County. | Alongside Robert White and Joshua Soule Zimmerman, he served as Chancery Commissioner for Hampshire County. |
| 5 | FN | 0.121 | 1 | 0 | North Abaco is one of the districts of the Bahamas, on the Abaco Islands. It has a population of 9,578. (2010 census) | It is one of the districts of the Bahamas, on the Abaco islands and has a population of 9,578 (census of 2010). |
| 6 | FP | 1.000 | 0 | 1 | A A Khap is a clan or group of related clans, mainly under the jats of the **western** Uttar Pradesh and **eastern** Haryana. | ... mainly under the jats of the **eastern** Uttar Pradesh and **Western** Haryana. |
| 7 | FP | 1.000 | 0 | 1 | Road R205 ... from road R199 in the county of **Leitrim** to the Northern Ireland border in the county of **Fermanagh**, mostly in county **Cavan**. | ... from road R199 in the county of **Cavan** to the border in the county of **Leitrim**, mostly in county **Fermanagh**. |
| 8 | FP | 1.000 | 0 | 1 | Everton Ferreira Guimarães ... is a **Brazilian** footballer who plays for the **Portuguese** association C.D. | ... is a **Portuguese** footballer who plays for the **Brazilian** Association C.D. |
| 9 | FP | 1.000 | 0 | 1 | The Porte de Vincennes is located where the north-eastern corner of the 12th arrondissement meets the south-eastern corner of the 20th arrondissement. | ... where the south-eastern corner of the 20th arrondissement meets the north-eastern corner of the 12th arrondissement. |
| 10 | FP | 1.000 | 0 | 1 | Mount DeVoe ... located southeast from **Rambler Peak** and south of **Gold River**. | ... located southeast of **Gold River** and south of **Rambler Peak**. |
| 11 | FP | 0.661 | 0 | 1 | He eventually published, in the "Western Republican", one of the first botanical collections made in Ohio by a professional botanist. | He finally made one of the first botanical collections in the "Western Republican", which were published in Ohio by a professional botanist. |
| 12 | FP | 0.662 | 0 | 1 | **Sukumar**'s friend Stephen (**Murugan**) helps them in the hour of crisis, and lovers unite in marriage. | **Murugan**'s friend Stephen (**Sukumar**) helps them in their hour of crisis and the lovers unite in marriage. |
| 13 | FP | 0.662 | 0 | 1 | The race rubber dish was used by the National Guard of the United States as a base during the historic Wooster Avenue Riots of 1968. | The historic Rubber Bowl was used by the National Guard of the United States as the base during the racist Wooster Avenue Riots of 1968. |
| 14 | FP | 0.663 | 0 | 1 | 3 October: **Phil Tufnell** defeated **James Hewitt** (winning Dart Double 1) | 3 October: **James Hewitt** beat **Phil Tufnell** (Dart Double 1) |
| 15 | FP | 0.663 | 0 | 1 | **Norman Melancton Geddes**, born **Norman Bel Geddes** (27 April 1893 – 8 May 1958), was an American theatre and industrial designer. | **Norman Bel Geddes**, born **Norman Melancton Geddes** (April 27, 1893 – May 8, 1958), was an American theatrical and industrial designer. |

## Error categories

### A. Entity/attribute swap flips meaning, words identical — 7/15 (#6, 7, 8, 10, 12, 14, 15) — **most common**

Two entities or attributes trade places (east↔west, Leitrim↔Cavan,
Brazilian↔Portuguese, winner↔loser, name↔birth-name) and the meaning
flips while the word multiset stays identical. The five most-confident
cases (#6–10) all have **similarity exactly 1.000**: the two sentences map
to *literally identical embeddings*. Three more (#12, 14, 15) share the
same mechanism with the model slightly less certain (sim ≈ 0.662).

### B. Meaning-preserving restructure over-penalized — 5/15 (#1–5, all FNs)

The mirror image of A. True paraphrases that rearrange syntax or swap a
name for a pronoun ("Kuykendall served alongside X" → "Alongside X, he
served") get pushed to *near-zero* similarity (0.05–0.12) — far below
even unrelated-sentence territory. The 1000 PAWS training pairs taught the
model "reordering ⇒ not paraphrase", and lacking the machinery to tell
meaning-changing swaps from harmless ones, it over-applies the penalty.
This is an overcorrection artifact: more PAWS data pressure produces more
of it, not less.

### C. Genuinely ambiguous / questionable gold label — 2/15 (#9, 13)

\#9 uses the symmetric predicate "meets": "corner A meets corner B" vs
"corner B meets corner A" describe the same location; the gold label of 0
is arguable. #13 contains corrupted text ("The race rubber dish" for "the
Rubber Bowl"), making the pair noise rather than signal.

### D. Model uncertain, wrong side of threshold — 1/15 primary (#11; also secondary for #12–15)

All five near-threshold FPs sit within 0.003 of the 0.66 cutoff, so
threshold placement matters at the margin — but note the sims cluster
*just above* threshold rather than spreading below it, i.e., the model
isn't finely ranking these; it simply has no signal to push them down.

## Verdict: architecture, not more data

Category A — the most common — is the decisive evidence, and it points to
**a different architecture (cross-encoder), not more data**:

1. **Similarity 1.000 is unfixable by training.** For #6–10 the bi-encoder
   produces identical embeddings for the two sentences. No threshold, no
   additional epochs, and no amount of PAWS data can separate two identical
   vectors — the mean-pooled embedding has discarded the word-order
   information the label depends on before the comparison ever happens.
2. **Category B shows what "more data" actually does.** The model *did*
   respond to PAWS training pressure — by learning an indiscriminate
   "reordering ⇒ different" penalty that now destroys true paraphrases
   (sim 0.05 for a sentence whose clauses were merely rearranged). A
   bi-encoder can move its one similarity dial, but it cannot learn *which*
   reorderings matter, because each sentence is encoded without seeing the
   other.
3. A **cross-encoder** feeds both sentences jointly through the
   transformer, letting attention align "Leitrim...Fermanagh" against
   "Cavan...Leitrim" token-by-token — exactly the mechanism categories A
   and B demand. On the published PAWS benchmark this is also the known
   remedy (cross-encoders score ~10–15 points higher than bi-encoders).

Full-dataset GPU training is still worth running: it will shrink the noise
and sharpen threshold placement (category D) and may soften B somewhat.
But categories A and B are structural, so the expected outcome is a
plateau well below QQP-level performance — at which point a cross-encoder
baseline (e.g. fine-tuning `bert-base` with paired-sentence input) is the
right next experiment.

## Postscript: cross-encoder re-test of these 15 errors

A cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`, same backbone
capacity as the bi-encoder) was fine-tuned on the identical 1000+1000
QQP+PAWS sample and re-scored these 15 pairs (scores are logits; decision
threshold −0.43; raw data: `paws_errors_retest.json`):

| Finding | Result |
|---|---|
| Category B false negatives (#1–5) | **All 5 fixed** — meaning-preserving restructures now score as paraphrases |
| Category A/D false positives (#6–15) | All 10 still wrong — swaps still score 1.0–3.1, above threshold |
| Saturated scores? | **No** — the five bi-encoder sim=1.000 pairs now get *distinct* scores (1.74–2.57) |

Interpretation: at this training budget the cross-encoder's headline PAWS
metrics are no better than the bi-encoder's (44.4% accuracy, 99.2%
positive rate — see `comparison.md`), so the architecture hypothesis is
**not yet confirmed by aggregate numbers**. But the failure mode differs
in a structurally important way: the bi-encoder mapped swap pairs to
*identical* representations (no amount of training can separate identical
vectors), while the cross-encoder assigns them varied, unsaturated scores
— wrong today, but trainable in principle. Combined with the elimination
of category B, the mechanism evidence still favors the cross-encoder;
whether it cashes out in metrics is exactly what the full-data GPU run
must decide. Note also the ms-marco base model is pretrained for passage
ranking, not paraphrase detection, and 2000 pairs × 1 epoch (train loss
0.93) is likely too little to repurpose its head.

**Deployment tradeoff observed on CPU**: the cross-encoder needed one full
transformer forward per *pair* (48,430 forwards for the two validation
splits) and cannot precompute or index embeddings — for search-style
workloads its cost scales with candidates × queries, vs. the bi-encoder's
one-time encoding per sentence plus cheap cosine comparisons. Training was
also ~2× slower per example than the bi-encoder despite equal backbone
size.

## Full-data bi-encoder retest verdict

The combined bi-encoder was retrained on Kaggle GPU with the **full
datasets** (including all 49,401 PAWS train pairs; decision threshold 0.60
from that run; aggregate results in `comparison.md` — PAWS accuracy rose
from 53.5% to 70.2%). Re-scoring the same 15 captured pairs:

| # | Type | True | Small-sample sim | Full-data sim | Pred | Verdict |
|---|------|------|------------------|---------------|------|---------|
| 1 | FN | 1 | 0.051 | 0.510 | 0 | still wrong |
| 2 | FN | 1 | 0.097 | 0.125 | 0 | still wrong |
| 3 | FN | 1 | 0.110 | 0.134 | 0 | still wrong |
| 4 | FN | 1 | 0.113 | 0.067 | 0 | still wrong |
| 5 | FN | 1 | 0.121 | 0.395 | 0 | still wrong |
| 6 | FP | 0 | 1.000 | 0.981 | 1 | still wrong |
| 7 | FP | 0 | 1.000 | 0.996 | 1 | still wrong |
| 8 | FP | 0 | 1.000 | 0.999 | 1 | still wrong |
| 9 | FP | 0 | 1.000 | 0.528 | 0 | FIXED |
| 10 | FP | 0 | 1.000 | 0.999 | 1 | still wrong |
| 11 | FP | 0 | 0.661 | 0.479 | 0 | FIXED |
| 12 | FP | 0 | 0.662 | 0.506 | 0 | FIXED |
| 13 | FP | 0 | 0.662 | 0.414 | 0 | FIXED |
| 14 | FP | 0 | 0.663 | 0.013 | 0 | FIXED |
| 15 | FP | 0 | 0.663 | 0.256 | 0 | FIXED |

Three findings, refining the small-sample analysis:

1. **The extreme identical-embedding cases are confirmed as an
   architectural ceiling, not a data-volume problem.** 4 of the 5 true
   sim=1.000 swap cases (#6–8, #10) are still wrong, with similarities
   ≥ 0.98 even after a ~25× increase in PAWS training data — full-scale
   training separated the embeddings by epsilon, not by anything usable.
   The one exception, #9 (0.528), is precisely the pair category C flagged
   as genuinely ambiguous (symmetric predicate "meets", gold label
   arguable), so it is weak evidence of real swap sensitivity.
2. **More data fixed every moderate-overlap swap case.** All five ~0.66
   false positives (#11–15) now score 0.01–0.51, below threshold. Where
   the embeddings had *any* separation for training to widen, data volume
   was the right lever — consistent with the aggregate PAWS gain.
3. **The false negatives (overcorrection) remain unfixed, but the pattern
   is insufficient correction, not deepened overcorrection.** All five are
   still below threshold, but two moved substantially toward correct
   (0.051→0.510, 0.121→0.395), two are roughly flat, and one slipped
   slightly (0.113→0.067). Full training did not worsen the indiscriminate
   reorder penalty; it partially relaxed it without crossing the line.

**Refined verdict**: more data helps broadly — it lifted PAWS accuracy 17
points and cleared the entire moderate tier of errors — but the most
extreme identical-embedding failures did not move meaningfully and appear
to be a genuine ceiling of the bi-encoder architecture. Those specific
cases are exactly what the cross-encoder's joint encoding addresses, and
its full-data run (still pending) is the matching test: if it separates
#6–8/#10 where the full-data bi-encoder could not, the architecture
hypothesis is confirmed on the strongest possible evidence.

## Full-data cross-encoder retest verdict

The cross-encoder was also retrained on Kaggle GPU with the full datasets.
At the aggregate level it now **leads the full-data bi-encoder on both
datasets** — QQP 87.5% vs 85.5%, PAWS 76.1% vs 70.2%
(`comparison_qqp-paws-cross.md`) — so joint encoding does pay off overall.
Re-scoring the same 15 pairs (scores are logits; raw data:
`paws_errors_retest.json`):

| # | Type | True | Small-sample bi sim | Full-data cross score | Pred | Verdict |
|---|------|------|--------------------|-----------------------|------|---------|
| 1 | FN | 1 | 0.051 | −1.889 | 0 | still wrong |
| 2 | FN | 1 | 0.097 | −0.503 | 0 | still wrong |
| 3 | FN | 1 | 0.110 | −0.414 | 0 | still wrong |
| 4 | FN | 1 | 0.113 | 1.196 | 1 | FIXED |
| 5 | FN | 1 | 0.121 | −0.981 | 0 | still wrong |
| 6 | FP | 0 | 1.000 | 2.293 | 1 | still wrong |
| 7 | FP | 0 | 1.000 | 1.571 | 1 | still wrong |
| 8 | FP | 0 | 1.000 | 0.207 | 1 | still wrong |
| 9 | FP | 0 | 1.000 | −2.983 | 0 | FIXED |
| 10 | FP | 0 | 1.000 | 2.885 | 1 | still wrong |
| 11 | FP | 0 | 0.661 | −2.223 | 0 | FIXED |
| 12 | FP | 0 | 0.662 | −1.562 | 0 | FIXED |
| 13 | FP | 0 | 0.662 | −2.813 | 0 | FIXED |
| 14 | FP | 0 | 0.663 | −3.197 | 0 | FIXED |
| 15 | FP | 0 | 0.663 | 1.349 | 1 | still wrong |

Findings:

1. **4 of the 5 true sim=1.000 cases remain wrong (#6–8, #10), matching
   the small-sample cross-encoder result almost exactly — full-scale
   training did not fix this failure mode.** The only fix, #9, is again
   the ambiguous symmetric-predicate pair, the same one the full-data
   bi-encoder "fixed". Unlike the bi-encoder, though, the cross-encoder
   assigns each case a distinct, unsaturated score (0.21 to 2.89, with #8
   pushed nearly to the boundary): it *can* mechanically distinguish these
   pairs but is confidently, consistently wrong about them.
2. **The moderate tier is mostly fixed** (4 of 5; #11–14 now strongly
   negative). The exception, #15 (the birth-name swap), remains wrong.
3. **The overcorrection pattern migrated into the cross-encoder.** The
   small-sample cross-encoder got all 5 false negatives right; the
   full-data version now rejects 4 of them (#1–3, #5, scores −0.4 to
   −1.9). Full-strength PAWS training pressure taught *this* architecture,
   too, to penalize meaning-preserving reorderings — evidence that this
   failure tracks the training signal rather than the pooling
   architecture.

**Revised verdict — the original architecture hypothesis is complicated,
not confirmed.** The two failure modes now have provably different
characters. The bi-encoder's failure on #6–8/#10 is mathematically
unfixable: identical vectors cannot be separated by any training. The
cross-encoder's failure on the *same* cases is not mathematically forced —
its scores are distinct and it leads on every aggregate metric — yet it
was not corrected empirically by 25× more training data and a full epoch.
That pattern (capability present, correction absent), together with
finding 3, points at the training side as the live suspect: the BCE
signal on PAWS' label distribution, training duration, or the ms-marco
passage-ranking pretraining mismatch — rather than pooling architecture
alone. The project surfaces this as a genuinely open question rather than
resolving it; natural next probes are an NLI-pretrained cross-encoder
base, more epochs, and inspecting whether the swap cases' scores move
across checkpoints during training.
