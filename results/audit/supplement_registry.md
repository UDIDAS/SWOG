# OAKG supplement registry (verified values)

Concrete values for the AAAI supplement `[[FILL]]` placeholders, generated from the frozen benchmark. Regenerate: `python -m oakg.export_supplement_registry`.

## 1. Phenotype + observability registry (17 variables)

| Feature | Type | Support organs | Normalization range |
|---|---|---|---|
| `has_tumor` | binary | (global) |  |
| `left_kidney_present` | binary | left_kidney |  |
| `left_kidney_volume_cm3` | numeric | left_kidney | 971.5925 |
| `lesion_multiplicity` | numeric | (global) | 74.0 |
| `liver_present` | binary | liver |  |
| `liver_tumor_containment` | binary | liver |  |
| `liver_tumor_present` | binary | liver |  |
| `liver_volume_cm3` | numeric | liver | 4116.9554 |
| `pancreas_present` | binary | pancreas |  |
| `pancreas_tumor_containment` | binary | pancreas |  |
| `pancreas_tumor_present` | binary | pancreas |  |
| `pancreas_volume_cm3` | numeric | pancreas | 181.1618 |
| `right_kidney_present` | binary | right_kidney |  |
| `right_kidney_volume_cm3` | numeric | right_kidney | 821.3436 |
| `spleen_present` | binary | spleen |  |
| `spleen_volume_cm3` | numeric | spleen | 773.2535 |
| `tumor_burden_cm3` | numeric | (global) | 732.3881 |

Normalization: numeric features scaled by the per-feature range above (min–max over the corpus); component similarity uses `max(0, 1 − |Δ|/range)`. Top-k = 10.

## 2. Component definitions and weights

- **numeric** component: mean over jointly-observed numeric features of `max(0, 1 − |x_q − x_c| / range_f)`.
- **categorical_relational** component: mean exact-match over jointly-observed set-like / binary features.
- **Weights: uniform.** The pair similarity is the unweighted mean of the present component group(s) — no learned or per-feature weights (`oakg.oakg.oakg_similarity_components`).

## 3. Thresholds and ranking parameters

- Lesion threshold: a tumor connected component must exceed **10 voxels** to count (`MIN_LESION_VOXELS`).
- Tumor-in-organ containment bbox margin: **±2 voxels**.
- Eligibility threshold η (γ_min, threshold policy): **0.25**.
- Lexicographic evidence-bin boundaries: **[0.25, 0.5, 0.75]** (γ-category = number of boundaries met, 0–3).
- Bootstrap B: **10000** paired resamples; Holm-corrected; seed **2027**.
- γ (shared-evidence coefficient) = Jaccard overlap of the two cases' observed-organ sets.

## 4. Masking seeds and realizations

- Base seed: **2027**.
- uniform: seed 2027 (no additional masking).
- random: fractions 0.20/0.40/0.60/0.80, seeds 2028–2031.
- dataset-style: pancreas_only / liver_only / kidney_only / multi_organ, seed 2027.
- asymmetric: broad_to_narrow + narrow_to_broad, seed 2027.
- Full machine-readable catalogue: `benchmark/masking.json`.

## 5. Predicted-mask cohort coverage

Reference (GT) track covers all cases; the predicted track is partial:
- Pancreas / LiTS: SSL/SAM3 predictions paired with GT (pred track).
- FLARE: predictions for **20 of 100** cases (`sam3_delivery`); the other 80 use GT on both tracks.
- Cases by source: Pancreas=281, LiTS=131, FLARE=100.

## 6. Precision@10 / Recall@10 / mAP (primary lexicographic, ref)

| Regime | P@10 | R@10 | mAP | nDCG@10 |
|---|---|---|---|---|
| asymmetric | 0.281 [0.232, 0.333] | 0.046 [0.039, 0.054] | 0.270 [0.234, 0.310] | 0.360 [0.307, 0.416] |
| dataset_style | 0.275 [0.236, 0.315] | 0.048 [0.041, 0.056] | 0.328 [0.295, 0.362] | 0.333 [0.293, 0.375] |
| random | 0.330 [0.296, 0.364] | 0.053 [0.047, 0.059] | 0.347 [0.318, 0.377] | 0.403 [0.368, 0.438] |
| uniform | 0.323 [0.253, 0.394] | 0.051 [0.038, 0.064] | 0.376 [0.320, 0.435] | 0.399 [0.328, 0.470] |

Full table (all methods/regimes/tracks): `results/tables/publication_ready_retrieval_table.csv`.

## 7. Ontology concepts and identifiers

Organ/lesion/phenotype concepts map to SNOMED CT / NCIt via the T-Box in the upstream handoff (`ontology_schema/schema.owl` + `ontology_mappings.json`). These source files are not redistributed here; the organ vocabulary and per-feature support sets are in `benchmark/anatomy.json`. **[verify the OWL/mappings ship separately if the supplement cites specific SNOMED/NCIt codes].**

## 8. Hardware and software environment

- OS: Linux-4.18.0-553.126.1.el8_10.x86_64-x86_64-with-glibc2.28
- CPU: 128 logical cores
- Memory: 755 GB
- GPU: not required for the retrieval pipeline (CPU-only); optional for the CompGCN backbone and upstream segmentation.
- Python: 3.11.15
- Libraries: numpy 1.26.4, pandas 3.0.2, scipy 1.17.1, sklearn 1.8.0, networkx 3.6.1, statsmodels 0.14.6, torch 2.5.1+cu121

- Runtime: the full paper pipeline (`make reproduce-paper`, 10k bootstrap) completes in a few minutes on the above CPU; storage for tracked results is < 5 MB (raw masks/embeddings excluded).
