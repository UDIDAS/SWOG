# Changelog — how the results evolved (incomparable-candidate correction)

**What changed and why.** The main pipeline originally **dropped** observation-
incomparable candidates before scoring (ideal-DCG over eligible candidates only).
The declared paper policy is `incomparable_policy: bottom` — keep them in the pool,
ranked at the bottom (ideal-DCG over the full pool), as the OAKG-Union ablation and
native analysis already did. The main pipeline (`oakg/benchmark.py`) and the strata
(`oakg/strata_results.py`) were corrected to bottom-ranking. All final outputs now
come from the corrected implementation only.

## Before → after (authoritative, computed identically on both sides)

All values ref track, OAKG-Lexicographic primary.

| Result | BEFORE (drop) | AFTER (bottom) | Note |
|---|---|---|---|
| Policy-selection OAKG-Lex (random) | **0.4074** | **0.4027** | the 0.407→0.403 headline |
| Policy-selection OAKG-Similarity | 0.3227 | 0.3179 | ablation |
| Absolute OAKG-Lex — **uniform** | 0.399 | 0.399 | ~unchanged (no incomparable) |
| Absolute OAKG-Lex — **random** | 0.407 | **0.403** | −0.005 |
| Absolute OAKG-Lex — **asymmetric** | 0.403 | **0.352** | **−0.051** (largest move) |
| Absolute OAKG-Lex — **dataset-style** | 0.339 | 0.339 | served-mean ~unchanged |
| Adversarial hard-distractor OAKG-threshold | NaN (abstained) | 0.000 (serve-at-bottom) | only cell that changed there |

**Why the one-sided regimes move most:** dropping incomparable candidates let OAKG
skip relevant cases it declined to rank, inflating nDCG exactly where one-sided
coverage is common (asymmetric, dataset-style). Bottom-ranking removes that free pass.

## Unchanged by the correction
| Result | Value (both before & after) |
|---|---|
| Hard-distractor OAKG − ZeroImp | **+0.068 [0.029, 0.113]** (broadly-covered → no incomparable) |
| FLARE tumor (exploratory) OAKG − ZeroImp | ≈ +0.043 |
| Union-ablation Δ_obs, native cross-dataset | unchanged (already bottom-ranking) |
| Policy tie (lexicographic == product), lex > similarity | still holds |

## What stayed true (headline claims survive)
- OAKG beats masked cosine **significantly in all 4 regimes** (+0.165 … +0.352).
- OAKG still only **ties** strong imputation on aggregate (Hyp 3 negative), separating
  only on the hard-distractor stratum.
- Random-regime OAKG is now **0.403 (0.402694) everywhere** — main pipeline == union
  ablation. **No primary output uses the old 0.407.**

## Superseded / removed
- The earlier **"served-query vs all-query"** explanation of 0.403-vs-0.407 was
  **wrong** (the query sets are identical) and has been removed. Root cause =
  incomparable-candidate handling; see `random_oakg_0403_vs_0407_audit.csv`.

Exact paper-text substitutions: `paper_number_changes.md`. Corrected code: commit
that changed `oakg/benchmark.py` (see `git log`).
