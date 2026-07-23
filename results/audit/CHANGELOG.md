# Changelog — incomparable-candidate correction

The main pipeline previously **dropped** observation-incomparable candidates before
scoring; the declared paper policy is `incomparable_policy: bottom` (keep them in the
pool, ranked at the bottom). This was corrected — all final outputs now come from the
corrected implementation only.

## Code
- `oakg/benchmark.py` — OAKG scoring now ranks incomparable candidates at the bottom
  (finite floor below all eligible scores); ideal-DCG is over the full pool.
- `oakg/strata_results.py` — same convention for the strata (`_ndcg` helper).

## Results affected (regenerated from the corrected pipeline)
| Output | Effect of the correction |
|---|---|
| `tables/retrieval_summary.csv`, `publication_ready_retrieval_table.csv` | OAKG rows shifted; largest under one-sided regimes (asymmetric, dataset-style) |
| `tables/ranking_policy_selection.csv` | random OAKG-Lexicographic **0.407438 → 0.402694** |
| `tables/cross_backbone_observability.csv`, `upstream_degradation.csv` | OAKG-derived cells shifted |
| `figures/masking_stress.png` | regenerated |
| `strata/hard_distractor_adversarial.csv` | only OAKG-threshold changed (NaN → 0.000 serve-at-bottom) |

## Results NOT affected
- `strata/hard_distractor.csv` (**CI stays [0.029, 0.113]**), `strata/flare_tumor_realgt.csv`
  (broadly-covered / no incomparable candidates → drop == bottom).
- `union_ablation/*`, `native_cross_dataset/*` — already used bottom-ranking.

## Headline
- Random-regime OAKG is now **0.403 (0.402694) everywhere** — main pipeline == union
  ablation. No primary output uses the old **0.407**.
- Exact paper-text substitutions: `paper_number_changes.md`.
- Root-cause audit (query sets identical; the earlier "served-vs-all-query" explanation
  was wrong and is removed): `random_oakg_0403_vs_0407_audit.csv` + `README.md`.
