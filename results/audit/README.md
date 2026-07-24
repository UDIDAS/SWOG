# Audit — random-masking OAKG 0.403 vs 0.407

Requested resolution of the two reported random-masking OAKG nDCG@10 values:

- **0.402694** — union-ablation table (`union_ablation_method_summary.csv`).
- **0.407438** — main benchmark / policy-selection (`ranking_policy_selection.csv`,
  `retrieval_summary.csv`, `publication_ready_retrieval_table.csv`).

Both were **recomputed from the real data under the frozen paper configuration**
(`configs/aaai27_paper.yaml`) through the two code paths.

## Audit table

`random_oakg_0403_vs_0407_audit.csv` — field-by-field. Summary:

| Field | 0.403 (union) | 0.407 (main) |
|---|---|---|
| number of queries | 111 (444 query×seed rows) | 111 (444 query×seed rows) |
| query-ID set hash | `aeb7664de3f74cdb` | `aeb7664de3f74cdb` (**identical**) |
| query-set difference | none | none |
| candidate pool | `candidate_pool(qc,corpus)` | same |
| relevance / eligibility | graded; ≥1 graded-relevant | same |
| seeds / realizations | random .2/.4/.6/.8, seeds 2028–2031 | same |
| seeds per query | 4 | 4 |
| averaging order | mean over 444 rows | mean over 444 rows |
| ranking policy / params | lexicographic; bins .25/.50/.75 | same |
| scoring function | `_pair_similarity` (group-mean) | `oakg_scores` (group-mean) |
| **incomparable handling** | **ranked at BOTTOM; ideal over FULL pool** | **DROPPED; ideal over ELIGIBLE only** |

## Cause (checked in the requested order)

1. query-ID membership — **identical** (same hash). ✗ not the cause
2. averaging order / per-query seed counts — **identical** (4 seeds each). ✗
3. relevance eligibility — **identical**. ✗
4. realization / seed selection — **identical**. ✗
5. **incomparable handling — DIFFERS. ← root cause**
6. ranking configuration — identical policy/scoring. ✗

Because the query sets are identical, the earlier "served-query vs all-query"
explanation was **incorrect** and has been removed. The real cause: the main
pipeline **drops** incomparable candidates before scoring (ideal-DCG over eligible
candidates only), while the union ablation **ranks them at the bottom** (ideal-DCG
over the full pool, per the declared `incomparable_policy: bottom`). The two agree
on 426/444 query×seed rows and differ on the **18 queries** that have a
relevant-but-incomparable candidate (`random_oakg_affected_queries.csv`; mean gap
+0.117 there).

## Consequence

The declared frozen policy is `incomparable_policy: bottom`, which the union
ablation implements and the main pipeline did **not** (it dropped). **Resolved:**
the main pipeline now ranks incomparable candidates at the bottom (commit that
adds this folder's follow-up), so the main random OAKG moved from 0.407438 to
**0.402694**, identical to the union ablation. Other main-table OAKG cells shifted
too — small under uniform/random (~0.005) but larger under the one-sided regimes
(asymmetric up to ~0.16, dataset-style ~0.09), exactly where dropping incomparable
candidates had been inflating OAKG. There is now **one** random-regime OAKG number
(0.402694) everywhere.

> **Two later corrections superseded the asides in this note.** (1) Under the **all-111**
> query convention OAKG beats masked cosine in **2 of 4** regimes (uniform, random), not
> "all four" — see `CHANGELOG.md` Correction 2. (2) The union-ablation **Δ_obs was
> *not* unaffected**: that module scored abstained queries by bottom-tie order, not 0;
> corrected to all-111, its Δ_obs are negative (OAKG-Union ≥ OAKG) — see `CHANGELOG.md`
> Correction 3.

Regenerate this audit: see `oakg`-based recomputation in the commit that added this
folder.
