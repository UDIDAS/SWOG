# Master nDCG@10 table (final, `incomparable_policy: bottom`, lexicographic primary)

Absolute nDCG@10 by masking regime. Rows 1–5 are the MAIN pipeline (ref track);
OAKG-Union is ablation-internal (the only place it exists). Regenerate:
`python -m oakg.export_master_table`.

| Method | Uniform | Random | Asymmetric | Dataset-style |
|---|---|---|---|---|
| OAKG-Lexicographic | 0.399 | 0.403 | 0.263 | 0.206 |
| Masked cosine | 0.047 | 0.105 | 0.244 | 0.204 |
| Zero imputation | 0.440 | 0.439 | 0.434 | 0.404 |
| Missingness indicators | 0.442 | 0.442 | 0.435 | 0.418 |
| WL graph kernel | 0.495 | 0.475 | 0.406 | 0.471 |
| OAKG-Union | 0.398 | 0.412 | 0.351 | 0.249 |

> **All-111 convention:** every value is averaged over the same 111 queries (a query a method cannot serve scores 0). So rows **1–5 (main pipeline) ARE directly subtractable** — e.g. OAKG-Lexicographic − Masked cosine gives the paired effect.
> **Exception — do NOT subtract the OAKG-Union row.** OAKG-Union comes from the separate observation-boundary ablation (its own accounting), not the main pipeline. The valid OAKG-vs-Union effect is the **paired Δ** in `results/union_ablation/` (asymmetric +0.068, dataset-style +0.112), not the difference of these two absolute rows.

Sources: rows 1–5 `results/tables/retrieval_summary.csv` (ref); OAKG-Union `results/union_ablation/union_ablation_method_summary.csv`.
