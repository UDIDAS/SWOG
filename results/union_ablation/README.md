# OAKG vs OAKG-Union — observation-boundary ablation

Isolates **one** mechanism: how one-sided anatomical evidence is treated. Everything
else is identical (graph, phenotypes, component similarities, feature weights,
ranking policy, queries, candidates, relevance, masks, seeds, γ).

- **OAKG** (intersection): compare only anatomy observed in **both** cases; a pair
  with no jointly-observed anatomy is **incomparable** (ranked at the bottom), and a
  query with **no** comparable candidate **abstains** (scores nDCG@10 = 0).
- **OAKG-Union**: compare over the **union** of observed anatomy; anatomy observed
  on only one side is **completed as absent (0)** on the other, and a numerical
  similarity is **always** returned — OAKG-Union never abstains.

γ is computed from the **original (masked) observation scopes** for both methods.
Implementation: `oakg/union_ablation.py` (adds `support_mode` + one-sided
completion to our existing group-mean similarity).

> **Not the same as the zero-imputation baseline.** OAKG-Union also fills a missing
> organ with 0, but it is a *controlled twin* of OAKG, not the external baseline.
> It (i) 0-fills **only one-sided organs inside the union** — organs *neither* case
> observed are excluded; (ii) uses **OAKG's own group-mean similarity**, not plain
> cosine; and (iii) **keeps γ and the ranking policy**. It changes exactly one thing
> vs. OAKG — intersection→union at the boundary — which is what lets Δ_obs isolate the
> boundary mechanism.

## Reproduce

```bash
python -m oakg.union_ablation --policy lexicographic --n-boot 10000
```

- **Ranking policy:** `lexicographic` — the **validation-selected** policy (ties
  product at 0.403 on the random-masking validation family). The *same* policy is
  used for the masked-cosine table, the pooled-baseline table, and this ablation.
- **Query convention — all-111 (identical to the main pipeline,
  `oakg.metrics.query_metric_row`):** a query a method **cannot serve** (0 comparable
  candidates) scores **nDCG@10 = 0**; the metric is NaN only if a query has no
  relevant candidate at all. **All 111 queries are retained in every regime (0 NaN).**
  OAKG-Union always serves; OAKG abstains in the one-sided regimes (served rates
  below). This is the **same abstention rule as the main benchmark** — so the ablation
  OAKG absolute equals the main-pipeline OAKG-Lexicographic per regime (enforced by
  `validate_checklist.py`).
- **Incomparable candidates:** ranked at the bottom (`bottom` policy), ideal-DCG over
  the full candidate pool, for both methods.

## Main paper table (OAKG − OAKG-Union, nDCG@10, ref; 10k paired bootstrap, Holm)

| Regime | OAKG | OAKG-Union | Δ_obs | 95% CI | Holm p | OAKG served |
|---|---|---|---|---|---|---|
| Uniform | 0.399 | 0.398 | +0.001 | [0.000, 0.002] | 1.00 | 1.00 |
| Random | 0.403 | 0.412 | −0.010 | [−0.023, 0.000] | <0.001 | 1.00 |
| Asymmetric | 0.263 | 0.351 | **−0.088** | [−0.127, −0.052] | <0.001 | 0.73 |
| Dataset-style | 0.206 | 0.249 | **−0.044** | [−0.072, −0.021] | <0.001 | 0.62 |

```latex
\begin{tabular}{lccc}
\toprule
Regime & OAKG & OAKG-Union & $\Delta_{\mathrm{obs}}$ \\
\midrule
Uniform       & 0.399 & 0.398 & $+0.001$ \\
Random        & 0.403 & 0.412 & $-0.010$ \\
Asymmetric    & 0.263 & 0.351 & $\mathbf{-0.088}$ \\
Dataset-style & 0.206 & 0.249 & $\mathbf{-0.044}$ \\
\bottomrule
\end{tabular}
```

> **Reads honestly:** under the main-benchmark all-111 convention, **OAKG-Union ≥ OAKG
> in every regime**. In the one-sided regimes OAKG abstains on 27% (asymmetric) / 38%
> (dataset-style) of queries — those score 0, while OAKG-Union always returns a ranking
> — so union-completion is ahead by −0.044 to −0.088. Ablation OAKG absolute equals the
> main-pipeline OAKG-Lexicographic per regime (uniform 0.399, random 0.403, asymmetric
> 0.263, dataset-style 0.206), verified by `validate_checklist.py`'s ablation-convention
> gate. *(A previous version scored abstained queries by their arbitrary bottom-tie
> order instead of 0; that inflated OAKG to 0.419/0.361 and produced spurious
> Δ_obs = +0.068/+0.112. Corrected — see `results/audit/CHANGELOG.md`.)*

## What it shows (honest reading)

- **The observation-boundary ablation does NOT show an aggregate-nDCG advantage** for
  intersection+abstention (OAKG) over union-completion (OAKG-Union). Where one-sided
  coverage is created (asymmetric, dataset-style), OAKG's abstention **costs** on
  aggregate ranking: a scored-0 abstention loses to any returned ranking. This mirrors
  the main benchmark, where OAKG ties/does not beat the imputation baselines on
  aggregate one-sided regimes.
- **Uniform (control): Δ ≈ 0** — intersection = union when scopes match.
- **Random: −0.010** (tiny) and **hard-distractor: 0.000** (both 0.9419) — union
  completion is not misled on these; the boundary is not isolated by them.
- **Where OAKG's mechanism actually pays off is not aggregate ranking**, but (i) its
  **abstention semantics** — it returns *Unknown* on one-sided anatomy instead of a
  false-absent completion (see the semantic table), and (ii) versus the **cruder
  zero-imputation** baseline on the patient-level **hard-distractor** stratum
  (**+0.068** vs zero-imputation — a *different*, external comparison; note OAKG-Union
  itself ties OAKG there). The OAKG-vs-OAKG-Union contrast is a null/negative result on
  aggregate nDCG and should be reported as such.

## Masking regimes (what each one is)

- **Uniform** — nothing dropped; near-zero one-sided anatomy — the **negative control**.
- **Random** — each case independently drops 20/40/60/80% of organs; mild, unsystematic.
- **Dataset-style** — corpus restricted to one organ pattern at a time; heavy systematic
  one-sided coverage.
- **Asymmetric** — query and candidate sides given deliberately different breadth — the
  extreme one-sided case.

## Files

| File | Contents |
|---|---|
| `union_ablation_paired_comparison.csv` | `regime, OAKG, OAKG_Union, delta_obs, ci_low, ci_high, p_value, oakg_served_rate, holm_p, …` |
| `union_ablation_method_summary.csv` | `regime, method, nDCG@10, ci_low, ci_high, served_rate, n_queries, n_query_seed_rows` |
| `union_ablation_hard_distractor.csv` | the 41-query stratum, reported separately |
| `union_ablation_query_level.csv` | per (regime, seed, query, method): `nDCG@10, served` |
| `union_ablation_config.json` | policy, seeds, query convention (all-111), bins |

`seed` indexes the masking realization within a regime (fraction / style / direction).

## Supplementary — similarity policy (NOT the primary; documented for transparency)

Files in `similarity_policy/`. Same ablation under the `similarity` policy (rank by S
only, no γ), all-111 convention. Under it the negative control **fails** (uniform
Δ = −0.158), confirming why the **lexicographic** (validation-selected) policy above is
the one used. Δ_obs = OAKG − OAKG-Union:

| Regime | OAKG | OAKG-Union | Δ_obs |
|---|---|---|---|
| uniform | 0.240 | 0.398 | −0.158 (control fails) |
| asymmetric | 0.250 | 0.351 | −0.100 |
| dataset-style | 0.166 | 0.248 | −0.082 |
| random | 0.318 | 0.411 | −0.093 |
| hard-distractor | 0.369 | 0.942 | −0.573 |
