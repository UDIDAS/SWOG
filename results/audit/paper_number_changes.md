# Paper-text number changes (apply to the manuscript)

Two corrections changed several OAKG numbers: (1) incomparable candidates drop→bottom,
(2) query averaging served-only→all-111. The repo tables, README, snapshot, and tests
are updated; the **manuscript is not in this repo**, so apply the FINAL substitutions
below (OLD = original served-only; NEW = final all-111, `incomparable_policy: bottom`).

## Find → replace

**P0 headline (OAKG > masked cosine).**
- OLD: `+0.303 [0.239, 0.368]`, "significant in all 4 masking regimes"
- NEW: "significant in **uniform (+0.352) and random (+0.298)**; **ties** masked cosine in asymmetric (+0.019, ns) and dataset-style (+0.002, ns) — OAKG abstains on 40–56% of one-sided-regime queries under the all-111 convention", p<0.001

**Policy ablation (coverage-aware > coverage-blind).**
- OLD: `0.407 > 0.322 (+0.085)`
- NEW: `0.403 > 0.318 (+0.085)`

**Random-regime OAKG (single value).** `0.407` → **`0.403`** (0.402694).

**Per-stratum: OAKG-Lexicographic − masked cosine (nDCG@10, ref, all-111).**

| Regime | OLD (served-only) | NEW (all-111, final) |
|---|---|---|
| uniform | +0.352 [0.284, 0.420] | +0.352 [0.284, 0.420] *(unchanged)* |
| random | +0.303 [0.239, 0.368] | **+0.298 [0.264, 0.333]** |
| dataset-style | +0.144 [0.100, 0.190] | **+0.002 [−0.040, 0.043] (ns)** |
| asymmetric | +0.190 [0.120, 0.263] | **+0.019 [−0.039, 0.076] (ns)** |

**Absolute OAKG-Lexicographic (ref, all-111):** uniform 0.399, random 0.403, asymmetric
**0.263**, dataset-style **0.206** (was 0.352 / 0.339 under served-only). Same value in
`master_nDCG_table.csv`, `publication_ready_retrieval_table.csv`, `retrieval_summary.csv`.

**By missingness level (nDCG@10, ref).**

| Missing | vs masked-cosine OLD→NEW | vs zero-imp OLD→NEW | vs missingness-ind OLD→NEW |
|---|---|---|---|
| 20% | +0.359 → **+0.362** | −0.036 → **−0.026** | −0.038 → **−0.028** |
| 40% | +0.320 → **+0.332** | −0.015 → **−0.032** | −0.017 → **−0.034** |
| 60% | +0.320 → **+0.291** | −0.018 → **−0.021** | −0.021 → **−0.024** |
| 80% | +0.233 → **+0.206** | −0.054 → **−0.068** | −0.057 → **−0.070** |

**Absolute per-regime OAKG nDCG with CIs (asymmetric/dataset moved most).** Do NOT
hand-copy from here — read the regenerated
`results/tables/publication_ready_retrieval_table.csv` directly, so the CI matches
the exact table/track the paper cites.

## Unchanged — safe to keep as-is
- **uniform** vs-masked-cosine (+0.352) — no incomparable candidates under uniform.
- **hard-distractor** OAKG − zero-imp = **+0.068 [0.029, 0.113]** (patient-level).
- **real-GT tumor** OAKG − zero-imp = **+0.043 [0.019, 0.066]**.
- **adversarial hard-distractor** OAKG-similarity = **1.000, +0.366 [0.220, 0.512]**
  (only OAKG-threshold changed: NaN/abstain → 0.000 serve-at-bottom).
- **union-ablation Δ_obs** and the whole **native cross-dataset** table (both already
  used bottom-ranking): unchanged.
- Policy tie (lex == product) and lex > similarity: still hold.

## One-line rationale for the paper/rebuttal
OAKG's evaluation ranks observation-incomparable candidates at the bottom
(`incomparable_policy: bottom`), so the ideal-DCG is over the full candidate pool;
this is applied uniformly across the main benchmark, the OAKG-Union ablation, and
the native analysis. (Earlier the main tables dropped incomparable candidates,
which inflated OAKG in the one-sided regimes.)
