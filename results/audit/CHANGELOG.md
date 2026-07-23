# Changelog — how the results evolved

Two corrections, in order. Both apply the same principle — **don't let OAKG escape
hard cases by abstaining** — first at the candidate level, then at the query level.

## Correction 1 — incomparable candidates: drop → bottom
The main pipeline originally **dropped** observation-incomparable candidates (ideal-DCG
over eligible only). Fixed to `incomparable_policy: bottom` (keep them, ranked at the
bottom; ideal-DCG over the full pool). Effect: random OAKG-Lex **0.4074 → 0.4027**.

## Correction 2 — query averaging: served-only → all-111
The pipeline still averaged nDCG over **served** queries only (excluding queries where
OAKG had no comparable candidate). Fixed to **all 111** queries, same set for every
method: a query OAKG cannot serve scores **0** (it abstains rather than fabricate a
comparison). This is the fair, comparable convention and — because every method now
covers the same 111 queries — it makes the pooled and mean-of-realization aggregations
coincide, so **every main-pipeline table agrees exactly** (enforced by the consistency
gate in `validate_checklist.py`).

### Before → after (drop/served-only → bottom/all-111), OAKG-Lexicographic, ref
| Regime | drop, served-only | bottom, served-only | **bottom, all-111 (final)** |
|---|---|---|---|
| uniform | 0.399 | 0.399 | **0.399** |
| random | 0.407 | 0.403 | **0.403** |
| asymmetric | 0.403 | 0.352 | **0.263** |
| dataset-style | 0.339 | 0.339 | **0.206** |

### Headline consequence (owned explicitly)
Under all-111, **OAKG beats masked cosine in 2 of 4 regimes** (uniform +0.352, random
+0.298), and **ties** in asymmetric (+0.019 ns) and dataset-style (+0.002 ns) — where
OAKG abstains on 40–56% of queries while masked cosine ranks via residual global
features. The served-only "beats in all 4 regimes" was an artifact of hiding OAKG's
abstentions. The **patient-level hard-distractor result is unchanged** (+0.068 [0.029,
0.113]) — it is broadly-covered, so all-111 does not touch it.

## Unchanged by the corrections
| Result | Value (both before & after) |
|---|---|
| Hard-distractor OAKG − ZeroImp | **+0.068 [0.029, 0.113]** (broadly-covered → no incomparable) |
| FLARE tumor (exploratory) OAKG − ZeroImp | ≈ +0.043 |
| Union-ablation Δ_obs, native cross-dataset | unchanged (already bottom-ranking) |
| Policy tie (lexicographic == product), lex > similarity | still holds |

## What stayed true
- OAKG beats masked cosine **significantly in uniform (+0.352) and random (+0.298)**;
  ties in asymmetric/dataset-style (all-111 — see headline consequence above).
- OAKG still only **ties** strong imputation on aggregate (Hyp 3 negative), separating
  on the patient-level hard-distractor stratum (unchanged).
- Random-regime OAKG is **0.403 (0.402694) everywhere**. **No primary output uses the
  old 0.407**, and every main-pipeline table agrees on all four regimes.

## Superseded / removed
- The earlier **"served-query vs all-query"** explanation of 0.403-vs-0.407 was
  **wrong** (the query sets are identical). Root cause of that gap = incomparable-
  candidate handling (Correction 1); see `random_oakg_0403_vs_0407_audit.csv`. The
  served-vs-all distinction was instead the real issue in the *aggregation* layer,
  now resolved by Correction 2 (all-111).

Exact paper-text substitutions: `paper_number_changes.md`. Corrected code: commit
that changed `oakg/benchmark.py` (see `git log`).
