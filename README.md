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

**Masking regimes** (how partial observation is simulated — each hides different
anatomy per case): **uniform** = keep full annotation (matched-scope control);
**random** = each case independently drops a random fraction (20/40/60/80%) of its
organs (evidence decreases); **dataset-style** = restrict the whole corpus to one
organ pattern at a time (pancreas-only / liver-only / kidney-only / multi-organ,
mimicking source-specific annotation); **asymmetric** = give the query and
candidate sides different breadth (broad↔narrow, the extreme one-sided case).

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
- **WL graph kernel is the strongest retriever** (now a first-class method).
  Cosine over WL embeddings of the masked case-graph significantly beats OAKG
  (+0.07–0.08) *and* zero-imputation (+0.06–0.14), on both ref and pred tracks
  (paired, all SIG). A graph representation that structurally *omits* unobserved
  evidence wins; OAKG's pairwise γ-eligibility adds little on top.
- **Cross-backbone observability (Table D): +Obs gains are small/mixed.** Adding
  OAKG eligibility to a backbone helps WL slightly (+0.011–0.014) but *hurts* the
  phenotype backbone on the ref track (−0.038). Observability-awareness adds
  little once the representation already drops unobserved evidence.
- **Hard-distractor stratum — OAKG separates from imputation.** On a stratum where
  imputed zeros mislead (broad queries, narrow relevant + broad distractors),
  OAKG − zero-imputation is significant: **+0.185 [0.096, 0.272]** (n=15 prototype),
  holding at scale **+0.068 [0.028, 0.115]** (41 patient-level queries, ref).
  Masked cosine collapses. First clean OAKG-vs-imputation separation.
- **Policy finding — support-restriction wins, γ-weighting backfires.** On the
  *adversarial* hard-distractor (distractors chosen to fool imputation),
  **OAKG-similarity = 1.000 → +0.366 [0.220, 0.512] vs zero-imp, SIG** (perfect
  discrimination), but **OAKG-product/lexicographic tie zero-imp** and threshold
  *abstains*. The γ term down-weights narrow-coverage relevant cases — exactly the
  ones that should rank high. **Primary policy should be OAKG-similarity** (pure
  support-restriction); γ-product is an ablation that helps only on the broad
  multi-organ benchmark. See `results/strata/hard_distractor_adversarial.csv`.
- **Real-GT FLARE cross-organ tumor stratum — OAKG > imputation (real labels).**
  Using the actual class-14 tumor GT (merged with organs; slice-level, so
  evaluation-only and reported as *paired deltas*), on 364 queries with the
  cross-organ tumor phenotype: **OAKG − zero-imputation = +0.043 [0.019, 0.066],
  SIG** (WL +0.068 SIG; masked cosine collapses −0.382). Confirms the separation on
  *real ground-truth* tumor data, not predictions. Build: `oakg.build_tumor_stratum`.
- **Structured-semantic honesty is real.** OAKG's three-valued reasoning has a
  0.000 unsupported-negative rate: it abstains (U) on unobserved anatomy instead
  of asserting "absent" (closed-world's 0.014), at the cost of a 0.43 indeterminate
  rate. No retrieval metric captures this.
- **Two open anomalies:** upstream degradation is *inverted* (pred slightly beats
  ref) — an artifact of partial predicted coverage (only 20/100 FLARE and
  tumor-only LiTS have preds); and selective retrieval has *no* range (organ
  overlap is near-universal, so the threshold policy never abstains, AURC≈0).

**Data facts (for composition decisions):** Patient-level FLARE GT is capped at
**100** (50 labelsTr + 50 validation) and is **tumor-free** (FLARE22 = 13 organs,
no tumor class). FLARE tumor GT *does* exist (class 14) but only as slice-level
stacks with patient identity lost — usable as a real-GT **slice-level** stratum
(above), not merged into the patient-level corpus. Predicted patient-level tumor
was rejected: autonomous inference is noisy (107 false-positive blobs/case) and
unmeasurable on FLARE (no GT to score) — real GT was preferred over unvalidatable
predictions. **Disjoint-coverage query–candidate pairs = 14,885** (26.2% of
56,721), *all* LiTS⊥Pancreas single-organ; FLARE shares all organs so contributes 0.

**To upgrade tumor to patient-level:** we need per-patient FLARE volumes that carry
a tumor label alongside the organs — i.e. `case_ct.nii.gz` + `case_seg.nii.gz` with
labels `0=bg, 1–13 organs, 14=tumor`, patient IDs preserved. The direct source is
the **FLARE 2023 pan-cancer challenge dataset** (13 organs + pan-cancer tumor,
per-volume); the local class-14 slice stacks' per-patient source is **not on disk**
(only the pre-sliced `.npy` remains). Those FLARE23 tumor cases would enter as
*new* multi-organ+tumor patients (a different set than the 100 organ-only FLARE22
cases), giving a patient-level cross-organ tumor stratum to replace the slice-level one.

*Fetching FLARE23:* it is **not** on Google Drive (checked — only the slice-level
`class_*.npy` is) and has **no open HuggingFace/Zenodo mirror**. The only source is
the challenge platform, which requires **registration with real name/affiliation**:
CodaLab `competitions/12239` (MICCAI FLARE 2023). After registering and accepting
the data-use agreement, download the labeled training set and point
`oakg.build_benchmark` at the per-volume masks (organs 1–13 + tumor 14). Note:
`/scratch` currently has ~116 GB free, so fetch the labeled subset, not all 4000 cases.

**Direction — where we are headed:**

1. **Build out the hard-distractor stratum** (highest priority) — the first clean
   OAKG-vs-imputation separation; expand beyond the n=15 prototype (both organs,
   more queries) for tight CIs and a proper benchmark stratum.
2. **Fix the upstream-degradation comparison** — restrict to cases with matched
   ref+pred coverage so ref vs pred is apples-to-apples.
3. **Full statistical protocol** — 10k paired bootstrap over queries + masking
   seeds for the final tables.

## Repository layout

```
oakg/           the package: all experiment logic lives here
configs/        experiment configuration (YAML); mirrors oakg/config.py
notebooks/      thin driver notebook — imports oakg, reproduces all outputs
results/        query_level/, tables/, figures/ deliverables [partly git-ignored]
embeddings/     precomputed CompGCN/CT embeddings (.npz)   [git-ignored]
checkpoints/    model checkpoints                            [git-ignored]
docs/           guidelines / manuscript                      [git-ignored]
```

Only code (`oakg/`, `notebooks/`) and text configs are version-controlled. Data,
embeddings, results, checkpoints, and documents (PDF/Word) are git-ignored.

## Environment

Python 3.10+ with the dependencies in [`requirements.txt`](requirements.txt):

```bash
pip install -r requirements.txt        # or: pip install -e .
```

## Reproducing every table and figure

**Tutorial (start here):** [`notebooks/OAKG_Tutorial_Walkthrough.ipynb`](notebooks/OAKG_Tutorial_Walkthrough.ipynb)
is an interactive, meeting-ready walkthrough of the whole pipeline — input masks →
phenotypes → MMKG schema → masking → OAKG scoring → evaluation → findings — with
the design decisions that make up the novel contribution. Runs on demo data, so it
executes anywhere.

**Notebook (driver):** open [`notebooks/OAKG_AAAI2027_Experiment_Notebook.ipynb`](notebooks/OAKG_AAAI2027_Experiment_Notebook.ipynb)
and run all cells.

**Command line:**

```bash
python -m oakg.pipeline
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
python -m oakg.build_benchmark --out data          # full
python -m oakg.build_benchmark --out data --limit 8  # quick debug
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
