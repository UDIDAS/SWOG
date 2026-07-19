# OAKG — Observability-Aware Knowledge Graph Reasoning

Reproducible retrieval-evaluation code for the AAAI 2027 study on heterogeneous
medical imaging data. OAKG restricts pairwise case comparison to jointly
observed, anatomically supported evidence and modulates ranking by a shared-
evidence coefficient. The experiments establish that this observability
mechanism adds value beyond simple missing-data handling (e.g. masked cosine).

## Current validated outcomes

> Living section — refreshed every commit. Claims that fail to replicate are
> removed and replaced by the validated alternative, not kept as history.

**Benchmark (validated):** 432 real cases — 281 Pancreas + 131 LiTS (single-organ,
organ+tumor) + 20 FLARE (multi-organ hub, 5-organ morphometry, no tumor).
91 test queries, 17 features, patient-level splits. Env: `llmft` (Python 3.11).
*In flight:* FLARE ref track enriching 20 → 100 cases (see Direction).

**Acceptance test and supporting hypotheses** (nDCG@10, random masking, ref track;
paired 95% bootstrap CI, Holm-corrected):

| # | Hypothesis | Expected | Obtained | Verdict |
|---|---|---|---|---|
| 1 | **OAKG > masked cosine** (P0 acceptance test) | positive, significant | **+0.283 [0.220, 0.348]**, p<0.001 | ✅ pass (strong) |
| 2 | Coverage-aware policy > coverage-blind | product/lex ≥ similarity | 0.471 > 0.427 (+0.044) — needs multi-organ | ✅ conditional |
| 3 | OAKG > strong imputation baselines | ≥ zero/mean/missingness/Gower | beats Gower +0.195, mean +0.103; **tied** w/ zero-imp & missingness (ns) | ⚠️ partial |
| 4 | Upstream degradation ref > pred | positive Δ | OAKG-product +0.017; masked cosine −0.045 | ✅ measured |
| 5 | OAKG semantics: no unsupported negatives | ≈0 unsupported-neg rate | **0.000** vs closed-world 0.001; indeterminate 0.26 | ✅ pass |
| 6 | Selective retrieval trades coverage for reliability | risk ↓ as served-rate ↓ | AURC 0.006; served-rate 0.99–1.0 (little abstention) | ⚠️ limited range |

**Findings — what they mean:**

- **The central claim holds.** OAKG beats masked cosine — the guidelines' "most
  important simple baseline" — decisively and significantly. The graph-based
  observability mechanism (support-restriction + γ-eligibility) adds value beyond
  restricting comparison to jointly-observed features.
- **Honest nuance (hyp 3).** On *aggregate* random-masking nDCG, OAKG only *ties*
  a well-behaved zero-imputed / missingness-indicator cosine. Its edge is
  specifically over comparison-restriction (masked cosine) and mixed-similarity
  (Gower) — not "graphs beat vectors everywhere."
- **Where OAKG's real advantage lives (not in aggregate nDCG):** (a) structured
  semantics — 0.000 unsupported-negative rate: OAKG abstains (U) on unobserved
  anatomy instead of fabricating "absent"; (b) the harder strata (cross-dataset,
  asymmetric), where imputation should hallucinate similarity across incompatible
  coverage. Aggregate averages dilute both.
- **The policy ablation only comes alive with multi-organ heterogeneity** — γ is
  degenerate (0/1) for single-organ sources; FLARE's multi-organ cases are what
  make product/lexicographic separate from similarity. This validates the
  three-dataset MMKG design.
- **Selective retrieval has little range here** — organ overlap almost always
  exists, so the threshold policy rarely abstains (AURC≈0). Meaningful abstention
  needs the higher-missingness / asymmetric regimes.

**Direction — where we are headed:**

1. **Enrich FLARE ref track 20 → 100 cases** (13-label GT, mapping validated at
   Dice=1.000) — more multi-organ mass to sharpen γ and cross-organ retrieval. *(in progress)*
2. **Per-stratum deltas** — break OAKG−baseline out by matched / cross-dataset /
   asymmetric direction; the aggregate is a floor, not the story.
3. **P1 neural baselines** — CompGCN + observed-region CT embeddings, hybrid
   image–graph retrieval (`run_neural_baselines`).
4. **Full statistical protocol** — 10k paired bootstrap over queries + masking
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
