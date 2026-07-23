# OAKG results — what each file is and which table it fills

These are the **committed, review-friendly** outputs (all < ~150 KB). The large
raw per-query dumps under `query_level/` are git-ignored (regenerable, multi-MB).
Everything here regenerates from `data/` via:

```bash
python -m oakg.pipeline            # tables/ + figures/ + config.json
python -m oakg.strata_results      # strata/
```

`config.json` records the exact settings (seed, bootstrap count, paths) behind
this snapshot. Current benchmark: **512 patient-level cases** (281 Pancreas +
131 LiTS + 100 FLARE), 111 test queries, 17 features.

## Result categories

All outputs come from the corrected implementation (`incomparable_policy: bottom`,
OAKG-Lexicographic primary); the authoritative snapshot is `../paper_snapshot.json`.

- **PRIMARY (paper tables/figures):** `tables/` (**master_nDCG_table**, publication_ready,
  retrieval_summary, ranking_policy_selection, cross_backbone, structured_query_summary,
  ranking_consistency, query_diagnostics), `figures/masking_stress.png`.
- **SUPPLEMENTARY (patient-level analyses):** `union_ablation/` (observation-boundary),
  `native_cross_dataset/` (native source protocols), `strata/hard_distractor.csv` +
  `strata/hard_distractor_adversarial.csv` (patient-level diagnostic), `strata/backbone_comparison.csv`.
  Also **documented limitations, not primary claims:** `tables/upstream_degradation.csv`
  (ref→pred inverted — partial pred coverage) and the selective/risk-coverage output
  (`figures/risk_coverage_curve.png` + `tables/risk_coverage_curve.csv`) which shows a
  **flat AURC / no abstention range** and is therefore **not a primary figure**.
- **EXPLORATORY (not confirmatory):** `strata/flare_tumor_realgt.csv` — **slice-level**,
  364 slices ≠ 364 patients, patient clustering unreconstructable; CIs/p not confirmatory.
- **DEPRECATED / corrected:** none tracked. The 0.403-vs-0.407 correction (the main
  pipeline previously *dropped* incomparable candidates) is documented in
  `audit/` — no stale 0.407 or "served-vs-all-query" explanation remains in the
  primary outputs.

## Which file fills which paper table

| File | Paper output | What it is |
|------|--------------|------------|
| `tables/publication_ready_retrieval_table.csv` | **Table A** | Strong baselines: P@10, R@10, mAP, nDCG@10, served rate, with 95% bootstrap CIs, by track × masking regime × method |
| `tables/retrieval_summary.csv` | **Table B** | Same metrics, ref & pred tracks across all masking regimes (means; the raw material for A/B and the per-stratum deltas) |
| `tables/ranking_policy_selection.csv` | **Table C** | OAKG policy ablation (similarity / product / threshold / lexicographic) by validation nDCG@10 + served rate |
| `tables/cross_backbone_observability.csv` | **Table D** | Base vs +Obs per backbone (phenotype, WL): Δ_obs |
| `union_ablation/` (see its README) | **OAKG vs OAKG-Union** | Observation-boundary ablation: intersection vs union-completion, per masking regime (asymmetric +0.068, dataset-style +0.112, both sig; uniform ≈0 control) |
| `native_cross_dataset/` (see its README) | **Native cross-dataset** | OAKG vs OAKG-Union under the *original unmasked* source scopes, by source direction. Cross-source pooled +0.170; same-source ≈0 control; 31.9% of cross-source pairs incomparable natively |
| `tables/ranking_consistency.csv` | **Table E** | Full-to-partial ranking consistency: Kendall τ, Spearman ρ, top-10 overlap (per track) |
| `tables/structured_query_summary.csv` | **Table F** | Closed / open / OAKG three-valued semantics: T/F/U precision-recall-F1, macro-F1, indeterminate & unsupported-negative rates. **`recall_F` = supported-negative recall** (recall of *observed* absences); T=PRESENT, F=ABSENT, U=UNOBSERVED (U P/R are 0 — "unobserved" is a prediction state, not a ground-truth class). **Key contrast:** OAKG reaches **unsupported-negative rate 0.000** (vs closed-world 0.014) at the cost of supported-negative recall 0.49 (vs 0.97) and indeterminate rate 0.43 — it declines to assert absence it cannot support. OAKG and generic open-world coincide on this eval (both 0.000). |
| `figures/risk_coverage_curve.png` + `tables/risk_coverage_curve.csv` | *exploratory (not primary)* | Selective risk vs served-query coverage; **flat AURC, no abstention range** — not a primary figure |
| `figures/masking_stress.png` | **Figure 2** | nDCG@10 vs missing-coverage level (20/40/60/80%) for key methods, ref track |
| `tables/upstream_degradation.csv` | *supplementary limitation* | ref−pred degradation (Δ_upstream); **inverted (pred>ref) from partial pred coverage — documented limitation, not a primary result** |
| `tables/query_diagnostics.csv` | Section 11.1 | Per-query pool size, #relevant, prevalence, zero-relevant flags |

## Supplementary strata (`strata/`) — the OAKG-vs-imputation story

The main benchmark shows OAKG **beats masked cosine in uniform/random** (ties it in
the one-sided regimes, where it abstains) and only **ties strong imputation** on
aggregate. These strata isolate where OAKG's mechanism separates from imputation. Each CSV: `method, nDCG@10, delta_vs_ZeroImp, ci_low,
ci_high, served_rate, sig` (paired 95% bootstrap).

| File | Result | Meaning |
|------|--------|---------|
| `strata/hard_distractor.csv` | OAKG − ZeroImp **+0.068** [0.028, 0.115] SIG (41 patient-level queries) | On a stratum where imputed zeros mislead, OAKG's support-restriction separates from imputation. WL +0.064 SIG; masked cosine collapses. |
| `strata/hard_distractor_adversarial.csv` | **OAKG-similarity +0.366** [0.220, 0.512] SIG; OAKG-Lexicographic (and product) tie ZeroImp; threshold serves at bottom (0.000) | Distractors chosen to fool imputation. Pure support-restriction (OAKG-similarity) is near-perfect here while the γ-weighted policies tie ZeroImp — evidence that **shared-evidence weighting is task-dependent** and can hurt when relevant cases have narrow coverage. Reported as an ablation; the primary policy stays **lexicographic** (validation-selected). |
| `strata/flare_tumor_realgt.csv` | OAKG − ZeroImp ≈ **+0.043** (364 *slice* queries) — **EXPLORATORY** | Directional real-label corroboration on FLARE cross-organ tumor. **Slice-level, not confirmatory**: 364 slices ≠ 364 patients, patient clustering unreconstructable; CIs/p-values not treated as confirmatory. |
| `strata/backbone_comparison.csv` | WL 0.49 > CompGCN 0.43 ≈ ZeroImp 0.43 ≈ OAKG 0.41 ≫ MaskedCos 0.09 (random 40%, ref) | **CompGCN** (Section 8.2 relational graph encoder, trained here) lands mid-pack — statistically tied with WL/imputation/OAKG. Confirms: graph/vector methods cluster; masked cosine collapses. WL alone significantly beats imputation. **Observed-region CT (Section 8.3) not included** — it needs a 3D CT encoder over 512 volumes (heavier pipeline). |

**EXPLORATORY — slice-level caveat (tumor stratum only):** `flare_tumor_realgt.csv`
is built from real class-14 tumor GT, but at **slice granularity**. It is an
**exploratory** analysis, **not confirmatory patient-level validation**:
- it is **slice-level**, not patient-level;
- the **364 slices are not 364 independent patients**;
- **patient-level clustering cannot be reconstructed** (patient identity was lost
  in the class-stack source format);
- absolute nDCG runs optimistic due to slice correlation, so only **paired deltas**
  are shown and the **CIs / p-values are not treated as confirmatory evidence**.

It is evaluation-only (policy frozen from the patient benchmark) and kept entirely
separate from the patient-level corpus (`dataset = FLARE_tumor`). Built by
`oakg.build_tumor_stratum`.
To make this patient-level, we would need per-patient FLARE volumes with a tumor
label (the **FLARE 2023 pan-cancer** dataset; see the main README "Data facts").

## Qualitative retrieval examples (`qualitative/`)

Complex multi-condition queries and OAKG's top-5 retrievals — browseable on GitHub
and **runnable by anyone**:

| File | What it is |
|------|-----------|
| `qualitative/qualitative_retrieval.md` | Human-readable table: each query → OAKG top-5 (✓ = all conditions matched) |
| `qualitative/qualitative_retrieval.csv` | Same, with per-candidate matched-condition counts and the actual phenotype values |
| `qualitative/queries.json` | The query definitions — **edit this to try your own**, then re-run |
| `qualitative/imputation_vs_oakg.md` / `.csv` | **Where zero-imputation fails** — side-by-side on hard-distractor queries: imputation's top-5 vs OAKG's, ✓/✗ |

```bash
python -m oakg.qualitative              # regenerate from queries.json
python -m oakg.qualitative --queries my_queries.json
python -m oakg.qualitative --contrast   # + the imputation-vs-OAKG contrast
```

Content-based retrieval (rank all cases by OAKG-Lexicographic similarity to an anchor
case that satisfies the query); relevance graded by conditions matched. Current
set: 8 queries, mean conditions-matched@5 = 0.86.

**`imputation_vs_oakg`** isolates *where* imputation fails: on **hard-distractor**
queries (a broad multi-organ query for a large target organ, with narrow true
matches + broad small-target distractors), imputation ranks the broad distractors
high because they share *other* organs — **top-5 relevant: imputation 0% vs OAKG
100%**. Note this is the stratum where observability matters; on easy/natural
queries imputation is competitive (see the aggregate tables).

## The one-paragraph takeaway

OAKG's **support-restriction** is the mechanism that matters: it beats masked
cosine in uniform/random (ties under extreme one-sidedness, where it abstains) and
beats imputation on the patient-level hard-distractor
stratum (with a directional, exploratory slice-level tumor corroboration). Its
**γ-weighting policy** is a double-edged knob — it helps on the broad
multi-organ benchmark but can **backfire on narrow-relevant distractors**. The
**primary policy is lexicographic** (validation-selected, tied with product and
frozen before test evaluation); similarity-only and product are **ablations**. We
do not promote similarity-only from the test hard-distractor result — instead we
report that shared-evidence weighting is task-dependent and may hurt when relevant
cases have narrow coverage. WL (graph kernel) is consistently the strongest
single method. Structured semantics add a qualitative win no retrieval metric
captures: a **0.000 unsupported-negative rate**.
