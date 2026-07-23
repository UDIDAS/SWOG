# OAKG — all final paper tables (one place)

Generated from the committed result CSVs (corrected pipeline: `incomparable_policy: bottom`, OAKG-Lexicographic primary; snapshot `paper_snapshot.json`). Regenerate: `python -m oakg.export_paper_tables`.

> Absolute nDCG@10 is over each method's **served** queries (OAKG's served-rate is < 1 in one-sided regimes), so **do not subtract absolute columns across methods** to get an effect — use the **paired** tables (union-ablation, hard-distractor) for method contrasts.

---
## PRIMARY

### Master nDCG@10 (all methods × regimes) — `tables/master_nDCG_table.csv`

| method | uniform | random | asymmetric | dataset_style |
|---|---|---|---|---|
| OAKG-Lexicographic | 0.399 | 0.403 | 0.352 | 0.339 |
| Masked cosine | 0.047 | 0.105 | 0.244 | 0.204 |
| Zero imputation | 0.440 | 0.439 | 0.434 | 0.404 |
| Missingness indicators | 0.442 | 0.442 | 0.435 | 0.418 |
| WL graph kernel | 0.495 | 0.475 | 0.406 | 0.471 |
| OAKG-Union | 0.398 | 0.412 | 0.351 | 0.249 |


### Ranking-policy selection (Table C, validation) — `tables/ranking_policy_selection.csv`

| method | mean_metric | served_rate | n_queries |
|---|---|---|---|
| OAKG-lexicographic [ref] | 0.4027 | 1.0000 | 111 |
| OAKG-product [ref] | 0.4027 | 1.0000 | 111 |
| OAKG-similarity [ref] | 0.3179 | 1.0000 | 111 |
| OAKG-threshold [ref] | 0.3163 | 1.0000 | 111 |


### Strong baselines, OAKG-Lexicographic (ref), with CIs (Table A) — `tables/publication_ready_retrieval_table.csv`

| masking_regime | P@10 | R@10 | mAP | nDCG@10 | ServedRate |
|---|---|---|---|---|---|
| asymmetric | 0.281 [0.232, 0.333] | 0.046 [0.039, 0.054] | 0.270 [0.234, 0.310] | 0.360 [0.307, 0.416] | 0.730 |
| dataset_style | 0.275 [0.236, 0.315] | 0.048 [0.041, 0.056] | 0.328 [0.295, 0.362] | 0.333 [0.293, 0.375] | 0.617 |
| random | 0.330 [0.296, 0.364] | 0.053 [0.047, 0.059] | 0.347 [0.318, 0.377] | 0.403 [0.368, 0.438] | 1.000 |
| uniform | 0.323 [0.253, 0.394] | 0.051 [0.038, 0.064] | 0.376 [0.320, 0.435] | 0.399 [0.328, 0.470] | 1.000 |

(full baselines/tracks in the CSV.)


### Structured-query semantics (Table F) — `tables/structured_query_summary.csv`

| semantics | recall_F | macro_F1 | indeterminate_rate | unsupported_negative_rate |
|---|---|---|---|---|
| closed_world | 0.969 | 0.609 | 0.000 | 0.014 |
| oakg | 0.490 | 0.503 | 0.430 | 0.000 |
| open_world | 0.490 | 0.503 | 0.430 | 0.000 |

`recall_F` = supported-negative recall. OAKG reaches unsupported-negative rate 0.000 (vs closed-world 0.014) at the cost of supported-negative recall and a 0.43 indeterminate rate.


---
## SUPPLEMENTARY

### Observation-boundary ablation — OAKG vs OAKG-Union (paired) — `union_ablation/union_ablation_paired_comparison.csv`

| regime | OAKG | OAKG_Union | delta_obs | ci_low | ci_high | holm_p |
|---|---|---|---|---|---|---|
| asymmetric | 0.419 | 0.351 | 0.068 | 0.022 | 0.115 | 0.000 |
| dataset_style | 0.361 | 0.249 | 0.112 | 0.045 | 0.182 | 0.000 |
| random | 0.403 | 0.412 | -0.010 | -0.023 | 0.000 | 0.000 |
| uniform | 0.399 | 0.398 | 0.001 | 0.000 | 0.002 | 1.000 |


### Native cross-dataset — OAKG vs OAKG-Union by source direction — `native_cross_dataset/native_paired_effects.csv`

| stratum | n_queries | delta | ci_low | ci_high | p_value | holm_p |
|---|---|---|---|---|---|---|
| cross-source-pooled | 21 | 0.170 | 0.045 | 0.315 | 0.022 | 0.108 |
| flare22-to-other-sources | 8 | 0.179 | 0.026 | 0.422 | 0.061 | 0.245 |
| lits-to-other-sources | 3 | 0.404 | -0.257 | 0.936 | 0.497 | 0.994 |
| msd_pancreas-to-other-sources | 10 | 0.092 | -0.028 | 0.228 | 0.250 | 0.749 |
| same-source | 111 | 0.001 | 0.000 | 0.002 | 1.000 | 1.000 |


### Patient-level hard-distractor (paired vs zero-imp) — `strata/hard_distractor.csv`

| method | nDCG@10 | delta_vs_ZeroImp | ci_low | ci_high | served_rate | sig |
|---|---|---|---|---|---|---|
| OAKG | 0.9419 | 0.0677 | 0.0286 | 0.1127 | 1.0000 | SIG |
| WL | 0.9383 | 0.0641 | 0.0308 | 0.1045 | 1.0000 | SIG |
| MissInd | 0.8741 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | ns |
| ZeroImp | 0.8741 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | nan |
| MeanImp | 0.8426 | -0.0315 | -0.0640 | -0.0042 | 1.0000 | SIG |
| MaskedCos | 0.5298 | -0.3443 | -0.4278 | -0.2593 | 1.0000 | SIG |


---
## EXPLORATORY (not confirmatory)

### FLARE cross-organ tumor — **slice-level** — `strata/flare_tumor_realgt.csv`

| method | nDCG@10 | delta_vs_ZeroImp | ci_low | ci_high | served_rate | sig |
|---|---|---|---|---|---|---|
| WL | 0.6407 | 0.0677 | 0.0412 | 0.0945 | 1.0000 | SIG |
| OAKG | 0.6157 | 0.0427 | 0.0194 | 0.0668 | 1.0000 | SIG |
| ZeroImp | 0.5730 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | nan |
| MissInd | 0.5728 | -0.0002 | -0.0010 | 0.0005 | 1.0000 | ns |
| MeanImp | 0.5147 | -0.0582 | -0.0822 | -0.0341 | 1.0000 | SIG |
| MaskedCos | 0.1912 | -0.3817 | -0.4204 | -0.3424 | 1.0000 | SIG |

**Exploratory only:** slice-level; 364 slices ≠ 364 patients; patient clustering unreconstructable; CIs/p not treated as confirmatory.

