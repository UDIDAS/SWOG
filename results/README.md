# OAKG results — what each file is and which table it fills

These are the **committed, review-friendly** outputs (all < ~150 KB). The large
raw per-query dumps under `query_level/` are git-ignored (regenerable, multi-MB).
Everything here regenerates from `data/` via:

```bash
PYTHONPATH=src python -m oakg.pipeline            # tables/ + figures/ + config.json
PYTHONPATH=src python -m oakg.strata_results      # strata/
```

`config.json` records the exact settings (seed, bootstrap count, paths) behind
this snapshot. Current benchmark: **512 patient-level cases** (281 Pancreas +
131 LiTS + 100 FLARE), 111 test queries, 17 features.

## Which file fills which paper table

| File | Paper output | What it is |
|------|--------------|------------|
| `tables/publication_ready_retrieval_table.csv` | **Table A** | Strong baselines: P@10, R@10, mAP, nDCG@10, served rate, with 95% bootstrap CIs, by track × masking regime × method |
| `tables/retrieval_summary.csv` | **Table B** | Same metrics, ref & pred tracks across all masking regimes (means; the raw material for A/B and the per-stratum deltas) |
| `tables/ranking_policy_selection.csv` | **Table C** | OAKG policy ablation (similarity / product / threshold / lexicographic) by validation nDCG@10 + served rate |
| `tables/cross_backbone_observability.csv` | **Table D** | Base vs +Obs per backbone (phenotype, WL): Δ_obs |
| `tables/ranking_consistency.csv` | **Table E** | Full-to-partial ranking consistency: Kendall τ, Spearman ρ, top-10 overlap (per track) |
| `tables/structured_query_summary.csv` | **Table F** | Closed / open / OAKG three-valued semantics: T/F/U precision-recall-F1, macro-F1, indeterminate & unsupported-negative rates |
| `figures/risk_coverage_curve.png` + `tables/risk_coverage_curve.csv` | **Figure 1** | Selective risk vs served-query coverage; AURC |
| `figures/masking_stress.png` | **Figure 2** | nDCG@10 vs missing-coverage level (20/40/60/80%) for key methods, ref track |
| `tables/upstream_degradation.csv` | Section 3 | ref−pred degradation (Δ_upstream) per method/metric/masking |
| `tables/query_diagnostics.csv` | Section 11.1 | Per-query pool size, #relevant, prevalence, zero-relevant flags |

## Supplementary strata (`strata/`) — the OAKG-vs-imputation story

The main benchmark shows OAKG **beats masked cosine decisively** but only **ties
strong imputation** on aggregate. These strata isolate where OAKG's mechanism
separates from imputation. Each CSV: `method, nDCG@10, delta_vs_ZeroImp, ci_low,
ci_high, served_rate, sig` (paired 95% bootstrap).

| File | Result | Meaning |
|------|--------|---------|
| `strata/hard_distractor.csv` | OAKG − ZeroImp **+0.068** [0.028, 0.115] SIG (41 patient-level queries) | On a stratum where imputed zeros mislead, OAKG's support-restriction separates from imputation. WL +0.064 SIG; masked cosine collapses. |
| `strata/hard_distractor_adversarial.csv` | **OAKG-similarity +0.366** [0.220, 0.512] SIG; OAKG-product/lexicographic tie ZeroImp; threshold abstains | Distractors chosen to fool imputation. **Pure support-restriction (OAKG-similarity) is near-perfect; the γ-weighted policies BACKFIRE** (they down-weight narrow relevant cases). Key policy finding. |
| `strata/flare_tumor_realgt.csv` | OAKG − ZeroImp **+0.043** [0.019, 0.066] SIG (364 queries) | Same separation on **real ground-truth** FLARE cross-organ tumor data. WL +0.068 SIG. |

**Slice-level caveat (tumor stratum only):** `flare_tumor_realgt.csv` is built
from real class-14 tumor GT, but at **slice granularity** (patient identity lost
in the source). It is **evaluation-only** (thresholds frozen from the patient
benchmark) and reported as **paired deltas** — absolute nDCG runs optimistic due
to slice correlation, but paired comparisons stay valid. Kept separate from the
patient-level corpus (`dataset = FLARE_tumor`). Built by `oakg.build_tumor_stratum`.

## The one-paragraph takeaway

OAKG's **support-restriction** is the mechanism that matters: it beats masked
cosine everywhere and beats imputation on the hard-distractor and real-GT tumor
strata. Its **γ-weighting policy** is a double-edged knob — it helps on the broad
multi-organ benchmark but **backfires on narrow-relevant distractors**, so the
primary policy should be **OAKG-similarity** (support-restriction, no γ), with
γ-product retained as an ablation. WL (graph kernel) is consistently the strongest
single method. Structured semantics add a qualitative win no retrieval metric
captures: a **0.000 unsupported-negative rate**.
