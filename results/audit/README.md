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
ablation implements and the main pipeline does **not** (it drops). Reconciling to
one consistent pipeline means adopting bottom-ranking in the main pipeline too,
which would move the main random OAKG from 0.407438 to 0.402694 and shift other
main-table OAKG cells by a similar small margin. This is a headline-number change
and is left as an explicit decision (see repo discussion), not applied unilaterally.
The paired **Δ_obs** in the union ablation is unaffected by the convention.

Regenerate this audit: see `oakg`-based recomputation in the commit that added this
folder.
