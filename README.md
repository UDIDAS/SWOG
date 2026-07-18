# OAKG — Observability-Aware Knowledge Graph Reasoning

Reproducible retrieval-evaluation code for the AAAI 2027 study on heterogeneous
medical imaging data. OAKG restricts pairwise case comparison to jointly
observed, anatomically supported evidence and modulates ranking by a shared-
evidence coefficient. The experiments establish that this observability
mechanism adds value beyond simple missing-data handling (e.g. masked cosine).

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
and place the five CSVs in `data/`:

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
