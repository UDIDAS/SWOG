# Master nDCG@10 table (final, `incomparable_policy: bottom`, lexicographic primary)

Absolute nDCG@10 by masking regime. Rows 1–5 are the MAIN pipeline (ref track);
OAKG-Union is ablation-internal (the only place it exists). Regenerate:
`python -m oakg.export_master_table`.

| Method | Uniform | Random | Asymmetric | Dataset-style |
|---|---|---|---|---|
| OAKG-Lexicographic | 0.399 | 0.403 | 0.352 | 0.339 |
| Masked cosine | 0.047 | 0.105 | 0.244 | 0.204 |
| Zero imputation | 0.440 | 0.439 | 0.434 | 0.404 |
| Missingness indicators | 0.442 | 0.442 | 0.435 | 0.418 |
| WL graph kernel | 0.495 | 0.475 | 0.406 | 0.471 |
| OAKG-Union | 0.398 | 0.412 | 0.351 | 0.249 |

> **Do NOT subtract the OAKG-Union row from the OAKG-Lexicographic row.**
> Rows 1–5 are the main pipeline; **OAKG-Union comes from the separate observation-boundary ablation** and is a *paired twin of OAKG within that ablation*, not of the main-pipeline OAKG-Lexicographic. The valid OAKG-vs-Union effect is the **paired Δ** in `results/union_ablation/` (e.g. asymmetric Δ=+0.068, dataset-style Δ=+0.112), NOT the difference of these two absolute rows. OAKG-Union is listed here only for completeness of absolute values, per request.

Sources: rows 1–5 `results/tables/retrieval_summary.csv` (ref); OAKG-Union `results/union_ablation/union_ablation_method_summary.csv`.
