# Native (unmasked) cross-dataset analysis

Tests whether the observation-boundary effect appears under the **original
annotation scopes** of MSD Pancreas, LiTS, and FLARE22 — i.e. real source-protocol
differences, **no added masking**. Answers a likely reviewer question: does the
effect exist natively, or only under simulated masks?

**Primary comparison: OAKG vs OAKG-Union**, stratified by source direction.
Scores: OAKG = support-restricted similarity over the **intersection** of native
observed organs (incomparable when no organ is shared → ranked at the bottom);
OAKG-Union = similarity over the **union** with one-sided anatomy completed as
absent. Same-source is the built-in negative control (matched scope → OAKG=Union).

OAKG-Union is a *controlled twin* of OAKG (same γ, similarity, policy — only the
boundary rule differs), **not** the zero-imputation baseline: it 0-fills only
one-sided organs inside the union and never scores organs neither case observed.
See `results/union_ablation/README.md` for the full distinction.

**Ranking policy — primary = lexicographic (validation-selected).** These tables
are generated under the primary lexicographic policy. We verified the native
result is **exactly policy-invariant**: regenerating under similarity-only
reproduces every stratum's nDCG@10 **bit-for-bit** (max |Δ| = 0.0 across all 45
stratum×method cells). The reason is that within each native candidate pool the
γ-category tiering only reshuffles the *non-relevant tail*; graded-relevant
candidates dominate on both γ and similarity, so both policies place them
identically in the top-10. The similarity-only ablation is provided verbatim in
[`similarity_policy/`](similarity_policy/) and is identical to the primary tables
here.

## How to reproduce

```bash
# 1. export native candidate-level scores from OUR benchmark (this repo)
#    --policy lexicographic is the validation-selected PRIMARY policy.
python -m oakg.native_export --out results/native_inputs --policy lexicographic
# 2. run the provided analysis package (docs/OAKG_Native_Cross_Dataset_Analysis.zip)
PYTHONPATH=<pkg>/src python <pkg>/scripts/run_native_cross_dataset_analysis.py \
  --cases results/native_inputs/native_cases.csv \
  --queries results/native_inputs/native_queries.csv \
  --relevance results/native_inputs/native_relevance.csv \
  --scores results/native_inputs/native_scores.csv \
  --out-dir results/native_cross_dataset --n-boot 10000 --n-permutations 100000
```

Conventions (per the analysis-package documentation): query is the bootstrap unit; a query is included in a
stratum only if its stratum candidate pool has ≥1 nonzero graded-relevant candidate;
relevance labels and candidate pools are identical across methods; OAKG-incomparable
pairs stay in the pool at the bottom (never dropped).

## Primary result — OAKG vs OAKG-Union (nDCG@10, 10k paired bootstrap, Holm)

| Stratum | N_q | OAKG | OAKG-Union | Δ_obs [95% CI] | Holm p | OAKG incomp. |
|---|---|---|---|---|---|---|
| same-source (neg. control) | 111 | 0.399 | 0.398 | **+0.001** [0.000, 0.002] | 1.00 | 0.0% |
| **cross-source pooled** | 21 | 0.312 | 0.143 | **+0.170** [0.045, 0.315] | 0.108 | 31.9% |
| flare22 → others | 8 | 0.179 | 0.000 | +0.179 [0.026, 0.422] | 0.245 | 0.0% |
| msd_pancreas → others | 10 | 0.307 | 0.215 | +0.092 [−0.028, 0.228] | 0.749 | 56.7% |
| lits → others | 3 | 0.687 | 0.284 | +0.404 [−0.257, 0.936] | 0.994 | 73.8% |

> **All-111 convention — verified robust.** Unlike the masked-regime `union_ablation`
> (where masking removed the shared organ, so OAKG fully abstained on 27–38% of queries),
> here **OAKG's served rate is 1.000 in every stratum** — the FLARE hub shares an organ
> with every case, so no included query is left with zero comparable candidates. The
> all-111 zeroing of fully-abstained queries therefore never triggers, and these Δ_obs are
> identical under the all-111 and served-query conventions (replicated from
> `results/native_inputs/native_scores.csv`; OAKG 0.312 / Union 0.143 / Δ +0.170 both
> ways). The incomparable *pairs* (31.9%) are ranked at the bottom, not dropped — that is
> the `bottom` candidate policy, not query-level abstention.

## What it shows

- **The observation boundary is real under native source protocols.** OAKG marks
  **31.9%** of cross-source query–candidate pairs *incomparable* (up to **73.8%**
  for LiTS queries; **0%** for FLARE queries, which observe all organs and so share
  anatomy with everything). This is genuine one-sided coverage from the annotation
  protocols, not a simulated mask.
- **OAKG beats OAKG-Union on cross-source in every direction** (Δ = +0.09 … +0.40,
  all positive), pooled **+0.170** — and **same-source is a clean ≈0 control**. So
  the boundary mechanism helps exactly where source protocols create one-sidedness.
- **Significance is limited by query count.** Only 21 queries have cross-source
  relevant candidates (organ-consistent relevance makes most cross-source pairs
  non-relevant), so the pooled effect is raw-significant (p=0.022) but not
  Holm-significant (0.108) across the primary family. Direction is consistent.

## Honest caveat (also in the full CSVs)

On cross-source, the **imputation baselines outrank OAKG** (missingness-indicators
0.626, zero-imputation 0.585 vs OAKG 0.312) — consistent with the paper's main
finding. OAKG's demonstrated advantage here is specifically over **OAKG-Union**
(the observation-boundary ablation) and in the **incomparability behaviour**, not
in beating imputation on raw cross-source nDCG.

## Files

`native_paired_effects.csv` (all pairwise, primary + exploratory source-pairs),
`native_method_summary.csv` (per-method nDCG@10 + CIs), `native_incomparability_summary.csv`
(incomparable counts/rates), `native_query_eligibility.csv` / `native_query_coverage.csv`
/ `native_query_level_metrics.csv`, and `native_*_table.tex` (paper-ready).

Analysis code: the provided `oakg_benchmark` package (docs/OAKG_Native_Cross_Dataset_Analysis.zip).
Score export: `oakg/native_export.py` (this repo). Inputs regenerate to
`results/native_inputs/` (git-ignored; 12 MB scores).
