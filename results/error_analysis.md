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
