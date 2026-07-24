# Bottom-category tie audit (union ablation)

Answers the reviewer's Item 2: how are candidates tied in the *bottom category*
(observation-incomparable candidates, all sharing the floor score) ordered, and does
that ordering materially affect the OAKG − OAKG-Union Δ_obs?

## Secondary ordering and tie-awareness
- **Secondary key = candidate/insertion order** (`corpus.case_order`). The main pipeline
  sorts with `np.argsort(-scores, kind="mergesort")` (stable → original order preserved on
  ties, `oakg/metrics.py`); the ablation sorts by `(rank_key, -candidate_index)`
  (`oakg/union_ablation.py`). Both are deterministic insertion order.
- **`case_order` IS blocked by dataset** (Pancreas → LiTS → FLARE), so the secondary order
  *does* correlate with dataset identity — the reviewer's concern is valid to check.
- **nDCG is not tie-aware**: it scores the resulting deterministic order directly (no
  tie-averaging). So in principle a dataset-correlated tie order could bias the metric.

## Exposure — why it does not matter (per regime, evaluable query×seed)
| Regime | Method | % queries <10 comparable | % top-10 positions from bottom category | # relevant candidates in bottom category |
|---|---|---|---|---|
| uniform | OAKG | 0.0% | **0.0%** | 0 |
| uniform | OAKG-Union | 0.0% | 0.0% | 0 |
| random | OAKG | 0.0% | **0.0%** | 4,236 |
| random | OAKG-Union | 0.0% | 0.0% | 0 |
| asymmetric | OAKG | 27.0% | **0.0%** | 18,868 |
| asymmetric | OAKG-Union | 0.0% | 0.0% | 2,105 |
| dataset-style | OAKG | 38.3% | **0.0%** | 25,230 |
| dataset-style | OAKG-Union | 0.0% | 0.0% | 22,682 |

Key fact: **`% queries <10 comparable` equals the abstention rate exactly** (0/0/27/38%) —
i.e. the *only* queries with fewer than 10 comparable candidates are the **fully-abstained**
ones (0 comparable), which score **0** under all-111 regardless of any ordering. Every
*served* query has ≥10 comparable candidates, so **the top-10 is always filled from the
comparable set and the tied bottom category never enters the top-10 (0.0%)**. Relevant
candidates that fall in the bottom category (organ-incomparable but relevant) sit at rank
>10; they lower OAKG's nDCG through the ideal-DCG (correct support-restriction penalty) but
their *tie order among themselves* is irrelevant to nDCG@10.

## Randomized-tie sensitivity of Δ_obs (20 random bottom-category orders)
| Regime | Δ_obs deterministic | rand mean | rand min | rand max | max shift |
|---|---|---|---|---|---|
| uniform | +0.0008 | +0.0008 | +0.0008 | +0.0008 | **0.0000** |
| random | −0.0097 | −0.0097 | −0.0097 | −0.0097 | **0.0000** |
| asymmetric | −0.0875 | −0.0875 | −0.0875 | −0.0875 | **0.0000** |
| dataset-style | −0.0437 | −0.0437 | −0.0437 | −0.0437 | **0.0000** |

## Conclusion
Exposure is **negligible** and Δ_obs is **exactly tie-insensitive** (0.0000 shift, every
regime, 20 random orders) — because the tied bottom category lives entirely below rank 10.
The reviewer's condition ("if exposure is negligible or the results are insensitive, a brief
audit table is sufficient") is met: **no randomized-tie protocol or additional experiment is
required**; the deterministic Δ_obs values stand as reported.

Reproduce: `oakg`-based recomputation over `data/` (build_default_realizations →
observation_overlap ranking; deterministic vs `rng.shuffle` on the incomparable set).
