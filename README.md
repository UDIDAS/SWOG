# OAKG — Observability-Aware Knowledge Graph Reasoning

Reproducible retrieval-evaluation code for the AAAI 2027 study on heterogeneous
medical imaging data. OAKG restricts pairwise case comparison to jointly
observed, anatomically supported evidence and modulates ranking by a shared-
evidence coefficient. The experiments establish that the complete OAKG framework
outperforms masked cosine where it can support the comparison (uniform, random) and
**ties** it under extreme one-sidedness (asymmetric, dataset-style), where OAKG
abstains rather than compare over unshared anatomy (all-111 convention). The
*observation-boundary mechanism* itself is isolated not by the masked-cosine contrast
but by the **OAKG vs OAKG-Union** ablation (see `results/union_ablation/`).

## Current validated outcomes

> Living section — refreshed every commit. Claims that fail to replicate are
> removed and replaced by the validated alternative, not kept as history.

**Benchmark (validated):** 512 real cases — 281 Pancreas + 131 LiTS (single-organ,
organ+tumor) + 100 FLARE (multi-organ hub, 5-organ morphometry, no tumor;
predictions for the 20 held-out FLARE cases). 111 test queries, 17 features,
patient-level splits (288/113/111). FLARE22 label map verified against the organizer GT.
**Relevance is organ-consistent** — a candidate can only be relevant if it shares
an annotated organ with the query case (cross-organ "matches" are false positives,
not hits). Python 3.11; see `requirements.txt`.

**Acceptance test and supporting hypotheses** (nDCG@10, random masking, ref track;
paired 95% bootstrap CI, Holm-corrected):

| # | Hypothesis | Expected | Obtained (512-case) | Verdict |
|---|---|---|---|---|
| 1 | **OAKG > masked cosine** (P0 acceptance test) | positive, significant | **significant in uniform (+0.352) and random (+0.298)**; **ties** in asymmetric (+0.019 ns) and dataset-style (+0.002 ns) — where OAKG *abstains* on 40–56% of queries rather than compare over unshared organs, while masked cosine still ranks via residual global features (all-111 convention) | ✅ pass (uniform/random); ⚠️ **ties under extreme one-sidedness** |
| 2 | Coverage-aware policy > coverage-blind | product/lex ≥ similarity | 0.403 > 0.318 (**+0.085**) | ✅ confirmed |
| 3 | OAKG > strong imputation baselines | ≥ zero/mean/missingness/Gower | beats Gower +0.213, mean +0.034 (ns); **never beats** zero-imp / missingness-indicators, **significantly worse at 80% missing** (see below) | ❌ **not met (settled)** |
| 4 | Upstream degradation ref > pred | positive Δ | OAKG-Lexicographic **−0.035** (pred > ref) — inverted | ⚠️ anomaly (partial pred coverage) |
| 5 | OAKG semantics: no unsupported negatives | ≈0 unsupported-neg rate | **0.000** vs closed-world 0.014; indeterminate 0.43 | ✅ pass |
| 6 | Selective retrieval trades coverage for reliability | risk ↓ as served-rate ↓ | AURC 0.000; served-rate flat at 1.0 | ❌ no abstention range |

**Masking regimes** (how partial observation is simulated — each hides different
anatomy per case): **uniform** = no *additional* masking (each case keeps its full
native annotation; the residual cross-source scope differences from the source
protocols still remain) — the matched-scope control;
**random** = each case independently drops a random fraction (20/40/60/80%) of its
organs (evidence decreases); **dataset-style** = restrict the whole corpus to one
organ pattern at a time (pancreas-only / liver-only / kidney-only / multi-organ,
mimicking source-specific annotation); **asymmetric** = give the query and
candidate sides different breadth (broad↔narrow, the extreme one-sided case).

**Evaluation convention — incomparable at the bottom, averaged over all 111 queries.**
Two rules, applied identically to every method:
1. **Candidate level:** a candidate OAKG cannot compare (shares no observed organ)
   is kept in the pool and ranked at the **bottom**, not dropped (`incomparable_policy:
   bottom`) — so ideal-DCG is over the *full* pool.
2. **Query level:** every query is averaged (**all 111**, same set for every method).
   A query where OAKG has **no** comparable candidate scores **0** — OAKG abstains
   rather than fabricate a comparison over unshared organs. It is *not* excluded.

Averaging over served queries only (excluding OAKG's abstentions) inflates OAKG in the
one-sided regimes; all-111 is the fair, comparable convention. Because every method
now covers the same 111 queries, the pooled and mean-of-realization aggregations
coincide, so **every main-pipeline table reports one identical value per regime**
(enforced by the consistency gate in `validate_checklist.py`). Trade-off owned
explicitly: under all-111 OAKG *ties* masked cosine in asymmetric/dataset-style (it
abstains a lot there) and wins in uniform/random. See [results/audit/CHANGELOG.md](results/audit/CHANGELOG.md).

**Per-stratum — OAKG-Lexicographic − masked cosine by masking regime** (nDCG@10, ref;
all-111 — significant in uniform/random, ties in the one-sided regimes where OAKG abstains):

| Masking regime | Δ nDCG@10 | 95% CI |
|---|---|---|
| uniform | +0.352 | [0.284, 0.420] |
| random | +0.298 | [0.264, 0.333] |
| dataset-style | +0.002 | [−0.040, 0.043] (ns) |
| asymmetric | +0.019 | [−0.039, 0.076] (ns) |

**By missingness level — OAKG-Lexicographic vs baselines** (nDCG@10, ref): OAKG beats
masked cosine at every level, but never beats imputation and loses at 80%:

| missing | vs masked-cosine | vs zero-imp | vs missingness-ind |
|---|---|---|---|
| 20% | **+0.362** ✓ | −0.026 (ns) | −0.028 (ns) |
| 40% | **+0.332** ✓ | −0.032 (ns) | −0.034 (ns) |
| 60% | **+0.291** ✓ | −0.021 (ns) | −0.024 (ns) |
| 80% | **+0.206** ✓ | **−0.068** ✗ | **−0.070** ✗ |

**Findings — what they mean:**

- **OAKG beats masked cosine where it can support the comparison; ties where it
  abstains.** Significantly beats masked cosine in **uniform (+0.352)** and
  **random (+0.298, every missingness level)** — where OAKG serves ~all queries. In
  the **extreme one-sided regimes (asymmetric, dataset-style)** OAKG *ties* masked
  cosine: it **abstains on 40–56% of queries** (no candidate shares its target organ,
  so it scores 0 under all-111) while masked cosine still ranks those via residual
  *global* features. This is the honest all-111 result — OAKG never does *worse* than
  masked cosine, and its value is precisely that it declines to compare over unshared
  anatomy rather than fabricate a ranking. (Served-only averaging previously hid the
  abstentions and made this look like a 4/4 win.)
- **Hyp 3 is settled negative: OAKG does not beat strong imputation.** We traced
  *why* zero-imputation is so strong: (i) OAKG's γ is organ-set overlap, so it
  marks 100% of cross-organ pairs incomparable (ranked at the bottom) — but those
  candidates never reach the
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
- **Policy finding — shared-evidence weighting is task-dependent.** On the
  *adversarial* hard-distractor (distractors chosen to fool imputation),
  **OAKG-similarity = 1.000 → +0.366 [0.220, 0.512] vs zero-imp, SIG** (perfect
  discrimination), while **OAKG-Lexicographic (and product) tie zero-imp**. The γ
  (shared-evidence) term down-weights narrow-coverage relevant cases, which hurts
  on this stratum. **The primary policy remains lexicographic** — it was selected
  on the *validation* family (tied with product; both above similarity/threshold)
  and frozen before any test evaluation. We do **not** promote similarity-only on
  the basis of this test-set stratum; instead we report that shared-evidence
  weighting is task-dependent and can hurt when relevant cases have narrow
  coverage. Similarity-only and product are reported as **ablations**. See
  `results/strata/hard_distractor_adversarial.csv`.
- **FLARE cross-organ tumor stratum — EXPLORATORY (real labels, slice-level).**
  Using the actual class-14 tumor GT (merged with organs), on 364 *slice* queries
  with the cross-organ tumor phenotype, OAKG − zero-imputation ≈ **+0.043** (WL
  ≈ +0.068; masked cosine collapses ≈ −0.382) — directionally consistent with the
  patient-level hard-distractor separation. **This is an exploratory check, not
  confirmatory:** it is **slice-level**, the **364 slices are not 364 independent
  patients**, adjacent slices of a case are correlated, and **patient-level
  clustering cannot be reconstructed** (patient identity was lost in the class-stack
  format). Its confidence intervals / p-values are **not** treated as confirmatory
  evidence; it is reported only as a directional, real-label corroboration and is
  kept entirely separate from the 512-case patient-level benchmark. Build:
  `oakg.build_tumor_stratum`.
- **Observation-boundary mechanism (OAKG vs OAKG-Union).** Isolating the boundary
  rule alone, OAKG beats union-completion where source protocols create one-sided
  coverage: **dataset-style +0.112 [0.045, 0.182]**, **asymmetric +0.068 [0.022,
  0.115]** (Holm-sig), uniform ≈0 (control). See `results/union_ablation/`.
- **Native (unmasked) cross-source.** Under the original source scopes, OAKG beats
  OAKG-Union on cross-source pooled by **+0.170 nDCG@10, 95% CI [0.045, 0.315],
  Holm p=0.108** — a substantial positive *directional* effect that is **not
  statistically significant after Holm** correction (only 21 cross-source-eligible
  queries). Note: **missingness-indicators (0.626) and zero-imputation (0.585)
  achieve stronger raw pooled cross-source nDCG@10 than OAKG (0.312)**; OAKG's
  demonstrated advantage here is specifically over OAKG-Union and in its
  incomparability behaviour, not in beating imputation on raw ranking. See
  `results/native_cross_dataset/`.
- **Structured-semantic honesty is real.** OAKG's three-valued reasoning has a
  0.000 unsupported-negative rate: it returns Unknown (U) on unobserved anatomy
  instead of asserting "absent" (closed-world's 0.014), at the cost of a 0.43
  indeterminate rate. No retrieval metric captures this.
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

**FLARE evaluations:** [`notebooks/OAKG_FLARE_Evaluation.ipynb`](notebooks/OAKG_FLARE_Evaluation.ipynb) — all FLARE analyses (dataset roles, FLARE22 UNOBSERVED-tumor demo, contribution to the benchmark, FLARE23 exploratory tumor stratum, FLARE24 audit-pending), runs on the committed artifacts.

**Notebook (driver):** open [`notebooks/OAKG_AAAI2027_Experiment_Notebook.ipynb`](notebooks/OAKG_AAAI2027_Experiment_Notebook.ipynb)
and run all cells.

**Authoritative paper command (reviewers start here):**

```bash
make reproduce-paper          # == python -m oakg.pipeline --config configs/aaai27_paper.yaml + strata/ablations
python validate_checklist.py  # verify every artifact + snapshot hashes + headline numbers
```

This regenerates the **one authoritative snapshot** recorded in
[`paper_snapshot.json`](paper_snapshot.json) (frozen config, `incomparable_policy:
bottom`, OAKG-Lexicographic primary, 10k bootstrap; commit-pinned + SHA-256'd) and
the headline [`results/tables/master_nDCG_table.md`](results/tables/master_nDCG_table.md).
`python -m oakg.pipeline` with **no** config is a synthetic **smoke test only** — it
prints a banner and does **not** produce paper results.

`run_all(config)` writes to `results/`:

| File | Guideline output |
|------|------------------|
| `query_level/query_level_retrieval_results.csv` | per-query scores (reproduce any aggregate) |
| `tables/retrieval_summary.csv` | Tables A/B — strong baselines & masking tracks |
| `tables/upstream_degradation.csv` | ref→pred degradation (Section 3) |
| `tables/ranking_policy_selection.csv` | Table C — S, γ·S, threshold, lexicographic |
| `tables/risk_coverage_curve.csv` + `figures/risk_coverage_curve.png` | selective retrieval — **exploratory, not a primary figure** (flat AURC, no abstention range) |
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
containment. The MMKG schema grounds these entities/values in standard medical
terminologies (SNOMED CT / NCIt) — see `benchmark/ontology_mappings.json` and
`benchmark/kg_schema.owl`.

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
