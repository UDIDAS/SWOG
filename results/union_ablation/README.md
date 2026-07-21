# OAKG vs OAKG-Union — observation-boundary ablation

Isolates **one** mechanism: how one-sided anatomical evidence is treated. Everything
else is identical (graph, phenotypes, component similarities, feature weights,
ranking policy, queries, candidates, relevance, masks, seeds, γ).

- **OAKG** (intersection): compare only anatomy observed in **both** cases; a pair
  with no jointly-observed anatomy is **incomparable** (ranked at the bottom).
- **OAKG-Union**: compare over the **union** of observed anatomy; anatomy observed
  on only one side is **completed as absent (0)** on the other, and a numerical
  similarity is always returned.

γ is computed from the **original (masked) observation scopes** for both methods.
Implementation: `src/oakg/union_ablation.py` (adds `support_mode` + one-sided
completion to our existing group-mean similarity — *not* the reference module's
simplified similarity).

## Reproduce

```bash
PYTHONPATH=src python -m oakg.union_ablation --policy lexicographic --n-boot 10000
```

- **Ranking policy:** `lexicographic` — the **validation-selected** policy
  (nDCG@10 on the random-masking validation family; lexicographic and product tie
  at 0.407, both above similarity/threshold). The *same* policy is used for the
  masked-cosine table, pooled-baseline table, this ablation, and the
  hard-distractor evaluation.
- **Query convention:** nDCG@10 uses **graded** relevance, under which **all 111
  queries** have ≥1 graded-relevant candidate and are evaluable. (The 5
  "zero-relevant" queries are *binary*-zero but retain graded-partial matches; a
  binary convention would give 106. Applied identically to every method.)
- **Incomparable pairs:** ranked at the bottom (`bottom` policy), for both methods.

## Main paper table (OAKG − OAKG-Union, nDCG@10, ref; 10k paired bootstrap, Holm)

| Regime | OAKG | OAKG-Union | Δ_obs | 95% CI | Holm p |
|---|---|---|---|---|---|
| Uniform | 0.399 | 0.398 | **+0.001** | [0.000, 0.002] | 1.00 |
| Random | 0.403 | 0.412 | −0.010 | [−0.023, 0.000] | <0.001 |
| Asymmetric | 0.419 | 0.351 | **+0.068** | [0.022, 0.115] | <0.001 |
| Dataset-style | 0.361 | 0.249 | **+0.112** | [0.045, 0.182] | <0.001 |

```latex
\begin{tabular}{lccc}
\toprule
Regime & OAKG & OAKG-Union & $\Delta_{\mathrm{obs}}$ \\
\midrule
Uniform       & 0.399 & 0.398 & $+0.001$ \\
Random        & 0.403 & 0.412 & $-0.010$ \\
Asymmetric    & 0.419 & 0.351 & $\mathbf{+0.068}$ \\
Dataset-style & 0.361 & 0.249 & $\mathbf{+0.112}$ \\
\bottomrule
\end{tabular}
```

## What it shows

- **Negative control (uniform): Δ ≈ 0** — when query and candidates share observation
  scope, intersection = union, so the two methods are (as required) essentially
  identical (residual +0.001 comes from inherently one-sided cross-*dataset* pairs
  that exist even under uniform because the sources annotate different organs).
- **The observation-boundary mechanism helps exactly where one-sided coverage is
  created:** **asymmetric (+0.068)** and **dataset-style (+0.112)** are significant.
  Completing one-sided anatomy as *absent* (OAKG-Union) mis-ranks cases whose
  relevant organ was observed on only one side; OAKG's explicit boundary avoids it.
- **Random: −0.010** (tiny, significant) — random masking creates less *systematic*
  one-sided coverage, and there union-completion is marginally ahead.
- **Hard-distractor (natural 41-query stratum): Δ = 0** under the lexicographic
  policy — the relevant set contains broadly-covered cases that both methods rank
  identically (γ-category dominates), so this stratum does not isolate the boundary.

## Files

| File | Contents |
|---|---|
| `union_ablation_paired_comparison.csv` | `regime, OAKG, OAKG_Union, delta_obs, ci_low, ci_high, p_value, holm_p, …` |
| `union_ablation_method_summary.csv` | `regime, method, nDCG@10, ci_low, ci_high, n_queries, n_query_seed_rows` |
| `union_ablation_hard_distractor.csv` | the 41-query stratum, reported separately |
| `union_ablation_query_level.csv` | per (regime, seed, query, method) nDCG@10 |
| `union_ablation_config.json` | policy, seeds, query convention, bins |

`seed` indexes the masking realization within a regime (fraction / style / direction).

## Supplementary — similarity policy (NOT the primary; documented for transparency)

Files in `similarity_policy/`. We ran the same ablation under the `similarity`
policy (rank by S only, no γ). **It is not a valid control here:** without γ, OAKG
sends disjoint cross-dataset pairs to the bottom while OAKG-Union scores them via
global features, so the two diverge even under uniform — the **negative control
fails** (Δ = −0.158). Results (Δ_obs = OAKG − OAKG-Union):

| Regime | OAKG | OAKG-Union | Δ_obs |
|---|---|---|---|
| uniform | 0.240 | 0.398 | −0.158 (control fails) |
| asymmetric | 0.406 | 0.351 | +0.055 |
| dataset-style | 0.322 | 0.248 | +0.073 |
| random | 0.318 | 0.411 | −0.093 |
| hard-distractor | 0.369 | 0.942 | −0.573 |

Because the negative control fails under `similarity`, the **lexicographic**
(validation-selected) policy above is the one used for the paper. This table is
supplementary evidence for that choice, not a result table.
