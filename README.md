# OAKG — Observability-Aware Knowledge Graph Reasoning

Reproducible retrieval-evaluation code for the AAAI 2027 study on heterogeneous
medical imaging data. OAKG restricts pairwise case comparison to jointly
observed, anatomically supported evidence and modulates ranking by a shared-
evidence coefficient. The experiments establish that this observability
mechanism adds value beyond simple missing-data handling (e.g. masked cosine).

## Current validated outcomes

> Living section — refreshed every commit. Claims that fail to replicate are
> removed and replaced by the validated alternative, not kept as history.

**Benchmark (validated):** 512 real cases — 281 Pancreas + 131 LiTS (single-organ,
organ+tumor) + 100 FLARE (multi-organ hub, 5-organ morphometry, no tumor;
predictions for the 20 in `sam3_delivery`). 111 test queries, 17 features,
patient-level splits (288/113/111). FLARE22 label map validated at Dice=1.000.
**Relevance is organ-consistent** — a candidate can only be relevant if it shares
an annotated organ with the query case (cross-organ "matches" are false positives,
not hits). Env: `llmft` (Python 3.11).

**Acceptance test and supporting hypotheses** (nDCG@10, random masking, ref track;
paired 95% bootstrap CI, Holm-corrected):

| # | Hypothesis | Expected | Obtained (512-case) | Verdict |
|---|---|---|---|---|
| 1 | **OAKG > masked cosine** (P0 acceptance test) | positive, significant | **+0.303 [0.239, 0.368]**, p<0.001; **significant in all 4 masking regimes and all 4 missingness levels** | ✅ pass (strong, robust) |
| 2 | Coverage-aware policy > coverage-blind | product/lex ≥ similarity | 0.407 > 0.322 (**+0.085**) | ✅ confirmed |
| 3 | OAKG > strong imputation baselines | ≥ zero/mean/missingness/Gower | beats Gower +0.213, mean +0.034 (ns); **never beats** zero-imp / missingness-indicators, **significantly worse at 80% missing** (see below) | ❌ **not met (settled)** |
| 4 | Upstream degradation ref > pred | positive Δ | OAKG-product **−0.035** (pred > ref) — inverted | ⚠️ anomaly (partial pred coverage) |
| 5 | OAKG semantics: no unsupported negatives | ≈0 unsupported-neg rate | **0.000** vs closed-world 0.014; indeterminate 0.43 | ✅ pass |
| 6 | Selective retrieval trades coverage for reliability | risk ↓ as served-rate ↓ | AURC 0.000; served-rate flat at 1.0 | ❌ no abstention range |

**Per-stratum — OAKG-product − masked cosine by masking regime** (nDCG@10, ref;
the aggregate is a floor — OAKG wins in *every* regime):

| Masking regime | Δ nDCG@10 | 95% CI |
|---|---|---|
| uniform | +0.352 | [0.284, 0.420] |
| random | +0.303 | [0.239, 0.368] |
| asymmetric | +0.190 | [0.120, 0.263] |
| dataset-style | +0.144 | [0.100, 0.190] |

**By missingness level — OAKG-product vs baselines** (nDCG@10, ref): OAKG beats
masked cosine at every level, but never beats imputation and loses at 80%:

| missing | vs masked-cosine | vs zero-imp | vs missingness-ind |
|---|---|---|---|
| 20% | **+0.359** ✓ | −0.036 (ns) | −0.038 (ns) |
| 40% | **+0.320** ✓ | −0.015 (ns) | −0.017 (ns) |
| 60% | **+0.320** ✓ | −0.018 (ns) | −0.021 (ns) |
| 80% | **+0.233** ✓ | **−0.054** ✗ | **−0.057** ✗ |

**Findings — what they mean:**

- **The central claim holds and is robust.** OAKG beats masked cosine — the
  guidelines' "most important simple baseline" — significantly in *every* masking
  regime and at *every* missingness level. The graph-based observability mechanism
  (support-restriction + γ-eligibility) clearly adds value over restricting
  comparison to jointly-observed features.
- **Hyp 3 is settled negative: OAKG does not beat strong imputation.** We traced
  *why* zero-imputation is so strong: (i) OAKG's γ is organ-set overlap, so it
  abstains on 100% of cross-organ pairs — but those candidates never reach the
  top-10 anyway, so the gap is actually in *same-dataset* ranking under masking;
  (ii) making relevance organ-consistent (removing cross-organ hits) did **not**
  close it (−0.025 → −0.032); (iii) the gap does **not** shrink with missingness —
  OAKG is *significantly worse* than zero-imputation at 80% missing. So on raw
  ranking OAKG is competitive-but-not-superior to imputation. Its distinctive
  value is the masked-cosine dominance, the policy ablation, and the semantics
  below — **not** beating a well-behaved imputed cosine.
- **Structured-semantic honesty is real.** OAKG's three-valued reasoning has a
  0.000 unsupported-negative rate: it abstains (U) on unobserved anatomy instead
  of asserting "absent" (closed-world's 0.014), at the cost of a 0.43 indeterminate
  rate. No retrieval metric captures this.
- **Two open anomalies:** upstream degradation is *inverted* (pred slightly beats
  ref) — an artifact of partial predicted coverage (only 20/100 FLARE and
  tumor-only LiTS have preds); and selective retrieval has *no* range (organ
  overlap is near-universal, so the threshold policy never abstains, AURC≈0).

**Direction — where we are headed:**

1. **Fix the upstream-degradation comparison** — restrict to cases with matched
   ref+pred coverage so ref vs pred is apples-to-apples.
2. **P1 neural baselines** — CompGCN + observed-region CT embeddings, hybrid
   image–graph retrieval (`run_neural_baselines`); the graph/image structure is
   where OAKG may separate from vector imputation.
3. **Full statistical protocol** — 10k paired bootstrap over queries + masking
   seeds for the final tables.

## Repository layout

```
configs/        experiment configuration (YAML); mirrors src/oakg/config.py
data_schema/    schema notes for the five benchmark tables (see below)
notebooks/      thin driver notebook — imports oakg, reproduces all outputs
src/oakg/       the package: all experiment logic lives here
embeddings/     precomputed CompGCN/CT embeddings (.npz)   [git-ignored]
results/        query_level/, tables/, figures/ deliverables [git-ignored]
checkpoints/    model checkpoints                            [git-ignored]
docs/           guidelines / manuscript                      [git-ignored]
```

Only code (`src/`, `notebooks/`) and text configs are version-controlled. Data,
embeddings, results, checkpoints, and documents (PDF/Word) are git-ignored.

## Environment

Python 3.10+ with the dependencies in [`requirements.txt`](requirements.txt):

```bash
pip install -r requirements.txt        # or: pip install -e .
```

## Reproducing every table and figure

**Notebook:** open [`notebooks/OAKG_AAAI2027_Experiment_Notebook.ipynb`](notebooks/OAKG_AAAI2027_Experiment_Notebook.ipynb)
and run all cells.

**Command line:**

```bash
PYTHONPATH=src python -m oakg.pipeline
```

Both drive `oakg.pipeline.run_all(config)`, which writes to `results/`:

| File | Guideline output |
|------|------------------|
| `query_level/query_level_retrieval_results.csv` | per-query scores (reproduce any aggregate) |
| `tables/retrieval_summary.csv` | Tables A/B — strong baselines & masking tracks |
| `tables/upstream_degradation.csv` | ref→pred degradation (Section 3) |
| `tables/ranking_policy_selection.csv` | Table C — S, γ·S, threshold, lexicographic |
| `tables/risk_coverage_curve.csv` + `figures/risk_coverage_curve.png` | Figure 1 — selective retrieval |
| `tables/ranking_consistency.csv` | Table E — Kendall τ, Spearman ρ, top-10 overlap |
| `tables/structured_query_summary.csv` + `query_level/structured_query_predictions.csv` | Table F — T/F/U semantics |
| `tables/query_diagnostics.csv` | Section 11.1 diagnostics |
| `tables/publication_ready_retrieval_table.csv` | Section 12 — metrics with 95% bootstrap CIs |
| `config.json` | exact configuration behind the run |

## Demo vs. real data

By default `Config.use_demo_data=True` runs a reproducible synthetic benchmark so
the full pipeline executes anywhere. For the real study, set `use_demo_data=False`
and place the five CSVs in `data/`.

### Building the real benchmark from segmentation masks

`oakg.build_benchmark` derives the five CSVs from paired ground-truth and
predicted NIfTI masks across three heterogeneous sources:

- **Pancreas** → pancreas organ + tumor (single-organ)
- **LiTS** → liver organ + tumor (single-organ)
- **FLARE** → 5-organ morphometry, no tumor (**multi-organ hub**)

Per-source organ coverage forms the observability structure; the FLARE cases
share organs with both single-organ datasets, so the shared-evidence coefficient
γ is non-degenerate and cross-organ retrieval is possible (the OAKG ranking
policies only differentiate once multi-organ cases are present). Reference (GT)
masks define relevance and the `ref` track; predicted masks define the
end-to-end `pred` track.

```bash
PYTHONPATH=src python -m oakg.build_benchmark --out data          # full
PYTHONPATH=src python -m oakg.build_benchmark --out data --limit 8  # quick debug
```

Phenotypes (`oakg.phenotypes`): organ presence/volume, tumor presence, tumor
burden (cm³), lesion multiplicity (connected components), and tumor-in-organ
containment — aligned with the imaging-KG ontology (SNOMED CT / NCIt coded).

The five CSVs it writes have this schema:

| File | Columns |
|------|---------|
| `cases.csv` | `case_id, dataset, split, available_organs` (`\|`-joined organs) |
| `phenotypes_ref.csv` | `case_id, feature, value, feature_type, support_organs` |
| `phenotypes_pred.csv` | same schema, from predicted segmentations |
| `queries.csv` | `query_id, query_case_id, primary_predicates, secondary_predicates` (JSON) |
| `relevance.csv` | `query_id, candidate_id, binary_relevance, graded_relevance` |

Splits are patient-level; no masked realization of a patient appears in two
splits. Reference annotations define relevance labels and the image-access
boundary only — they never supply reference-derived features to end-to-end
predicted-track models.

## Package modules

| Module | Role |
|--------|------|
| `config` | fixed `Config` (seeds, thresholds, paths) |
| `data` | `ExperimentData`, demo generator, loader, validation, `Corpus` state |
| `masking` | uniform / random / dataset-style / asymmetric regimes |
| `metrics` | P@10, R@10, mAP, nDCG@10, served rate |
| `baselines` | zero/mean imputation, missingness indicators, masked cosine, Gower |
| `oakg` | support-restricted similarity + ranking policies |
| `benchmark` | full sweep over regimes × tracks |
| `stats` | paired bootstrap CIs, Holm correction, policy selection, upstream degradation |
| `selective` | risk–coverage / AURC |
| `consistency` | full-to-partial ranking consistency |
| `semantics` | closed / open / OAKG three-valued query reasoning |
| `diagnostics` | query & candidate-pool statistics |
| `graph` | Weisfeiler–Lehman kernel baseline |
| `neural` | CompGCN/CT embedding I/O, observed-region CT, hybrid, cross-backbone |
| `tables` | publication-ready CSV export |
| `pipeline` | `run_all(config)` — regenerates every deliverable |
