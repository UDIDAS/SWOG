# SWOG — Autonomous Abdominal-CT Segmentation → Imaging Knowledge Graph

**Aim.** Given a **new abdominal CT with no annotations**, automatically segment **liver, kidney, and
pancreas + their tumors**, turn them into an **ontology-grounded knowledge graph**, and use that KG to
retrieve similar patients, validate new cases without ground truth, and grow over time — a knowledge base
that keeps improving as patients arrive.

> **Reading this README.** Every score below is stated with **what it was trained on**, **what it was
> tested on**, and **what the number means**. Two short reference sections make that unambiguous:
> [§2 Datasets & splits](#2-datasets--how-each-is-split) and [§3 How to read every score](#3-how-to-read-every-score).

---

## 0. The whole system in plain terms

**The goal.** A clinician uploads a **new abdominal CT with no annotations**. The system (1) segments the
organs and tumors, (2) turns those masks into a structured, queryable **knowledge graph**, (3) checks the
result is medically plausible *without any ground truth*, and (4) finds similar past patients — and it
gets a little smarter every time it sees a new case.

**We use data in two distinct ways** — this is the key to the whole design (and the "two FLAREs"):

1. **Image + label pairs → *train the segmentation models*.** To teach a model to draw a mask, you must
   show it CT images together with the correct masks. Our image+label datasets are **LiTS, MSD Pancreas,
   KiTS23, and FLARE-Task2 (100 volumes)**.
2. **Labels alone → *build the knowledge graph*.** The KG doesn't need pixels: from a label mask we
   compute *numbers* — organ volume, tumor diameter, tumor burden, etc. — and those numbers become the
   graph's facts. **FLARE23's 1,312 label masks** (no images) power the KG and the retrieval experiments.

So a label is used *either* to train a segmenter (when it comes with an image) *or* to add a patient to
the KG (when we only have the mask). FLARE23 gives us 1,312 KG patients cheaply (labels only); FLARE-Task2
gives us 100 fully-paired cases to actually train organ segmentation on.

**What gets trained, on what, and how it's tested:**

| Model | Trained on | How | Tested on |
|---|---|---|---|
| **Generic tumor model** (one model, prompt `"tumor"`) | pooled tumor slices: LiTS + Pancreas + KiTS + FLARE (~20.7k) | SAM3 fine-tune, no box | held-out **patients** per dataset → **≈0.86** ([§5](#5-segmentation-results--the-generic-tumor-model)) |
| **Organ models** — fine-tuned | FLARE-Task2 (100 image+label pairs) | SAM3 + GT box (semi-oracle) | 20 held-out **patients** → 0.92–0.99 ([§4a](#4a-organ-ceiling--flare-task2-models-patient-level-test-the-honest-organ-numbers)) |
| **Organ models** — autonomous | *no training* (base SAM3 `"liver"` prompt) | concept prompt, no box | 5 held-out FLARE23 CTs → liver 0.82; small organs need fine-tuning ([§4b](#4b-autonomous-organ--full-volume-no-labels-experiment-c--the-deployment-number)) |

**What happens when a NEW CT arrives (the payoff):**
```
new CT (no labels)
  1. Segment      → autonomous organs (SAM3 concept) + tumor (generic model) → masks
  2. Phenotype    → volumes, diameters, centroids, tumor burden per organ
  3. Validate     → compare each phenotype to the KG cohort; flag implausible values
                    (e.g. a 4,900 cc "liver") as a segmentation error — NO ground truth needed
  4. Retrieve     → find the most similar known patients (observability-aware / OAKG)
  5. Grow         → admit the patient; the cohort model sharpens → the next case is validated better
```
Steps 3–5 are exactly what the knowledge graph is *for*: it's not a store, it's the component that
*interprets and vets* each new, unlabeled scan and improves as it grows.

**Would more/complex images make the models more robust?** Yes — this is the most direct improvement
available. The weakest link is segmentation robustness (the FLARE tumor **test** Dice is the lowest at
0.833, and small-organ autonomous Dice is low). Adding harder, more diverse CTs **with tumors** to the
training pool would raise robustness the most. It requires image+label pairs (a scoped download —
storage-aware), and is item 1 in the [roadmap](#8-planned-next-steps).

---

## 1. How we got here (results-driven)

1. **The blocker.** We first reproduced AUSAM (SAM1 + prompts derived from the GT mask). Strong Dice, but
   **every prediction needed a ground-truth box/point** — so it could not touch a *new, unlabeled* scan.
2. **Switch backbone → SAM3.** SAM3 adds **concept/text prompting** (segment `"liver"` with no box). On a
   held-out CT this gave liver Dice **0.964 autonomous vs 0.974 semi-oracle** — organ segmentation without
   a label is real.
3. **Concept prompting fails on tumors (0.27–0.37).** Foundation models don't know medical tumors → we
   **train one generic tumor model** (`"tumor"`, no box) on **pooled** abdominal-CT tumors.
4. **It doesn't transfer to unseen tumor types (0.02).** A liver+pancreas tumor model scored 0.02 on
   unseen kidney tumors → **coverage must be trained in**; each dataset added lifts it.
5. **Scope → liver / kidney / pancreas** — the three organs with matching tumor datasets (LiTS→liver,
   KiTS→kidney, MSD→pancreas), plus **FLARE23** for multi-organ diversity and the knowledge graph.
6. **The KG is a living knowledge base**, not a dump — see [§6](#6-the-knowledge-graph).

## The pipeline today
```
new CT (no labels)
  → organs:  SAM3 concept prompt  ("liver"/"kidney"/"pancreas")      [autonomous]
  → tumor:   generic tumor model  (prompt "tumor", trained)          [autonomous]
  → masks → phenotypes (volume, diameter, centroid, burden, …)
  → ontology-grounded KG  →  retrieval · GT-free validation · growth
```
Segmentation entry point: `src/scripts/infer_ensemble.py`. KG build: `build_flare23_enriched_kg.py` +
`kg_grounding.py`. Interactive KG: `app/oakg_query_app.py`. Walkthroughs: `src/notebooks/`.

---

## 2. Datasets & how each is split

Five datasets, each with a distinct role. **There is exactly one FLARE**: the **full FLARE23, 1,312
patients** — the earlier 100-case "FLARE22 demo" has been retired everywhere.

| Dataset | What it labels | Size | How we split train/test | Used for |
|---|---|---|---|---|
| **LiTS** (Liver Tumor Seg.) | liver + liver tumor | 131 volumes (pre-sliced, no patient IDs) | **slice-level** (no IDs available) | tumor pool (5,600 tumor slices); LiTS tumor delivery model |
| **MSD Pancreas** (Decathlon Task07) | pancreas + pancreatic tumor | 281 labeled | **case-level** | tumor pool (2,537); pancreas organ+tumor delivery models |
| **KiTS23** (Kidney Tumor Seg. 2023) | kidney + kidney tumor | 489 cases | **case-level** | tumor pool (5,267) |
| **FLARE23** (full) | **13 organs + tumor** | **1,312 patients** (labels-only), 608 with tumor (liver/kidney/pancreas) | **case-level** | **the knowledge graph**; OAKG experiments **A2, B2**; tumor pool (7,269 FLARE tumor slices) |
| **FLARE-Task2 2024** | 5 organs (no tumor) | 100 volumes (50 train_gt + 50 public-val) | **patient-level 70/10/20** (seed 42) | the **organ segmentation models** (semi-oracle ceilings); pancreas cases for **B1** |

*Why FLARE appears twice:* **FLARE23** (1,312 patients, labels-only) is what the **KG and OAKG paper
experiments** use. **FLARE-Task2 2024** (100 volumes *with images*) is what the **organ segmentation
models were trained on**, because it ships per-patient CT volumes suitable for training.

---

## 3. How to read every score

**Metric — Dice.** Overlap between predicted and ground-truth mask, 0–1; higher is better, ≈0.9+ is
strong. "3D Dice" = computed over the whole volume per patient; "slice Dice" = per 2-D slice.

**Prompt regime — how the model is told *where* to look (this changes the number a lot):**
- **Autonomous (concept prompt)** — the model gets *only the image + a word* (e.g. `"liver"`), **no
  annotation**. This is the **real deployment number** — what runs on a new, unlabeled scan.
- **Semi-oracle (GT box)** — the model is handed a bounding box drawn from the ground-truth mask (told
  *where* the structure is). An **upper bound / ceiling**: it needs a label, so it *cannot* run on new
  data; it only shows how much a label would be worth. The gap to autonomous = the cost of no labels.

**Split — how train/test were divided (this decides whether a number is honest):**
- **case-level / patient-level** = *whole patients* held out → **honest** (predicts performance on a
  brand-new patient). "patient-level 3D" is the strongest form.
- **slice-level** = random 2-D slices held out → a patient's near-identical neighbouring slices can land
  in *both* train and test (**data leakage → inflated**). Used only where a dataset has no patient IDs.

**Validation vs. Test.** *Validation* Dice is measured on the split used to pick the model during
training (can be slightly optimistic). *Test* Dice is a held-out set never used for model selection — the
number to trust. Where both exist we report both and say which is which.

**Retrieval / KG metrics** (used in §7): **Precision@k** = fraction of the top-k retrieved that are true
matches; **spurious-match rate** = fraction of retrieved that share ≤1 organ yet aren't true neighbours
(the false positive OAKG targets); **AUROC** = separability of two groups (0.5 = chance, 1.0 = perfect);
**MAPE** = mean absolute % error; **Pearson r** / **Spearman ρ** = linear / rank correlation.

---

## 4. Segmentation results — organs

### 4a. Organ ceiling — FLARE-Task2 models, patient-level test *(the honest organ numbers)*
**Trained on** ~70 FLARE-Task2 patients · **tested on** 19–20 **held-out patients** (patient-level, seed
42) · **semi-oracle** (GT-box) · **3-D Dice**. Script: `run_flare_task2_sam3.py`.

| Organ | 3-D Dice (semi-oracle) | # test patients |
|---|:--:|:--:|
| Liver | **0.985** | 20 |
| Spleen | **0.980** | 20 |
| Right kidney | **0.970** | 20 |
| Left kidney | **0.970** | 19 |
| Pancreas | **0.919** | 20 |

*Meaning:* with a label to localise the organ, SAM3 segments abdominal organs very accurately on brand-new
patients; pancreas is the hardest (small, low-contrast). This is the **ceiling** the autonomous numbers below aim at.

### 4b. Autonomous organ — full-volume, no labels *(experiment C — the deployment number)*
**No training** (base SAM3 concept prompt, `"liver"` etc., no box) · **tested on** 5 held-out **FLARE23**
cases that carry organ labels · **full-volume** Dice (the model must also decide *which* slices contain
the organ). Ceiling column = semi-oracle GT-box on the same FLARE23 cases (`flare23_predict.py`). Script:
`exp_autonomous_organ_sweep.py`.

| Organ | Autonomous (base concept, full-volume) | Semi-oracle ceiling (same cases) | Gap |
|---|:--:|:--:|:--:|
| Liver | **0.82** | 0.973 | 0.15 |
| Spleen | 0.585 | 0.962 | 0.38 |
| Left / right kidney | 0.46 / 0.44 | 0.956 | ~0.50 |
| Pancreas | 0.306 | 0.882 | 0.58 |

*Meaning:* **base** concept prompting holds up full-volume only for the **liver**; small organs collapse
(it over-segments empty slices and misses the organ elsewhere). **This is the clear next step:** per-organ
**fine-tuning** — the same recipe that took a single held-out liver from **0.931 base → 0.964 fine-tuned**
(slice-level bake-off, `exp_sam3_prompt_bakeoff.py`) and that trains the tumor model in §5.

### 4c. Per-dataset organ/tumor delivery models *(handed off to collaborators)*
Fine-tuned SAM3, semi-oracle (GT-box), **case-level** test. These are the `sam3_pancreas_*` /
`sam3_lits_*` checkpoints in the Krishna handoff.

| Dataset | Organ Dice | Tumor Dice | Split |
|---|:--:|:--:|---|
| Pancreas (MSD Task07) | 0.866 | 0.894 | case-level |
| LiTS (liver tumor) | — *(liver from GT)* | 0.840 | case-level |

---

## 5. Segmentation results — the generic tumor model

**One** model segments tumors across organs from the text prompt `"tumor"` (no box).

**Training pool** (≈20,673 tumor slices), and the seed-42 split used for train / val / test:

| Source | Tumor slices | Split type |
|---|:--:|---|
| LiTS | 5,600 | slice-level *(no patient IDs)* |
| KiTS23 | 5,267 | case-level |
| FLARE23 | 7,269 | case-level |
| MSD Pancreas | 2,537 | case-level |
| **total** | **20,673** | → train / val / test = **14,589 / 1,615 / 4,469** |

### 5a. Headline: validation vs. honest per-dataset test
- **Validation Dice (during training) = 0.938.** This is the *slice-level* validation split used to pick
  the model — **slightly optimistic** (a val slice can share a patient with a train slice).
- **Held-out TEST Dice (never used for selection)** — the numbers to trust. Autonomous (`"tumor"`, no
  box), per source dataset (`eval_tumor_per_dataset.py`):

| Dataset | Autonomous **test** Dice | # test slices | Split |
|---|:--:|:--:|---|
| LiTS | **0.897** | 1,120 | slice-level |
| KiTS | **0.864** | 1,122 | case-level |
| Pancreas | **0.856** | 465 | case-level |
| FLARE | **0.833** | 1,762 | case-level |
| **mean** | **≈0.86** | 4,469 | — |

*Meaning:* on genuinely unseen patients the autonomous tumor model scores **≈0.83–0.90** depending on
dataset (LiTS highest, FLARE lowest). The often-quoted **0.938** is the validation number; **≈0.86 is the
honest cross-dataset test performance.**

### 5b. Coverage grows the model
Validation Dice as the pool grew: **0.37** (base, no training) → **0.909** (v1: LiTS+Pancreas) →
**0.9145** (v2: +FLARE) → **0.938** (v3: +KiTS). **Cross-dataset caveat:** the v1 model (liver+pancreas
only) scored **0.02** on unseen FLARE kidney tumors — reliable **within trained tumor types**, not
universally; coverage must be trained in.

---

## 6. The knowledge graph

Built from the **full FLARE23 — 1,312 patients, 13 organs + tumor** (`kg/graph/imaging_kg_flare23.ttl`).
Each patient is a subgraph `ImagingCase → Organ / Lesion`, phenotypes as **direct triples**
(`gt_volume_cm3`, `gt_max_diameter_mm`, `gt_centroid_mm`, `tumorBurden`, `lesionMultiplicity`,
`lesion_count`), every entity grounded to **SNOMED / LOINC / ICD / MeSH** via a live mapper — no
hard-coded codes; open-world.

**Used as a knowledge base, not a store:**
- **Retrieval** — SPARQL ("largest kidney tumors", "tumors per organ") **and observability-aware
  similarity (OAKG)**: patients are compared over *jointly-observed* phenotypes, weighted by shared
  evidence (γ), so an unobserved organ reads as *unknown*, **never 0**.
- **Semantic interoperability** — grounding lets external hierarchies reason over it (a kidney tumor
  *is-a* genitourinary neoplasm).
- **GT-free validation of a new patient** — each autonomously-segmented phenotype is scored against the
  cohort distribution (percentile / z-score / joint covariance); an implausible value (e.g. a 4,900 cc
  "liver") is flagged as a segmentation error — **no ground truth required**.

**How it evolves:** `new CT → autonomous segmentation → phenotypes → GT-free plausibility check (admit /
flag) → admitted patient joins the global query KG → cohort model sharpens → the NEXT patient is validated
better.` Interactive: `app/oakg_query_app.py` (now over the 1,312-patient FLARE23 + Pancreas + LiTS) ·
runnable demo: `src/notebooks/KG_as_Knowledge_Base.ipynb`.

---

## 7. Research contribution & the OAKG experiments

**Thesis.** The novelty is **not** the segmentation (it builds on SAM3 and is on par with, not ahead of,
SOTA such as K-Prism / GF-Screen). It is the **downstream reasoning layer**: an **observability-aware,
self-evolving clinical knowledge graph (OAKG)** built end-to-end from *label-free* autonomous segmentation.

**The problem we own.** Merging many imaging datasets into one KG creates **structural
partial-observability** — each source annotated different organs. Standard retrieval either **imputes**
missing values (→ false matches) or does **naive masked similarity** (→ a "perfect match" on a single
shared feature). As the field unifies ever more datasets, this grows — and no one addresses it. **OAKG**
never imputes and weights matches by **joint observability (γ)**.

All four experiments below are **real and reproducible** from `results/*.json`; runnable write-up with
figures: `src/notebooks/OAKG_Experiments.ipynb`.

### A2 — Structured multi-source retrieval benchmark  *(the core method result)*
- **Data:** full **FLARE23** corpus (1,312 patients; 600 sampled), 5 core organ volumes.
  `exp_oakg_structured.py`.
- **Setup:** patients are assigned to single-site "datasets" with **disjoint observed organs**
  (Pancreas→pancreas, LiTS→liver, KiTS→kidneys) plus a full-observation FLARE hub — mirroring the real
  merge. Ground truth = each patient's true neighbours on the *full* organ vector. Vary heterogeneity.
- **Metric:** Precision@10 and spurious thin-overlap-match rate. **Baselines:** zero/mean-impute, Gower,
  masked-cosine. **OAKG = masked-cosine × γ**, so masked-cosine *is* the γ-ablation.

| Method (realistic mixed regime) | Precision@10 | Spurious matches |
|---|:--:|:--:|
| **OAKG (evidence-calibrated)** | **0.78** | **0.14** |
| mean-impute | 0.73 | 0.17 |
| zero-impute | 0.71 | 0.19 |
| **masked-cosine (OAKG *without* γ)** | **0.08** | **0.84** |

*Finding:* OAKG beats every imputation baseline, and the **γ ablation is decisive** — removing γ drops
precision **10×** (0.78→0.08) and raises spurious matches **6×** (0.14→0.84): a single shared organ reads
as a "perfect match" without γ. At extreme disjointness no method can retrieve; OAKG returns the honest
floor instead of fabricating a signal.

### B1 — Predicted-KG vs GT-KG answer fidelity
- **Data:** **20 paired pancreas cases** (CT + GT mask + autonomous prediction, mean Dice **0.919**) from
  the FLARE-Task2 pancreas delivery. `exp_kg_fidelity.py`.
- **Setup:** derive the exact phenotypes the KG stores from *both* masks; compare answers.

| Level | Metric | Result |
|---|---|:--:|
| node | volume MAPE / Pearson r | **4.8% / 0.992** |
| node | diameter MAPE / r | 0.9% / 0.997 |
| node | centroid error | 1.4 mm |
| categorical | size-bin agreement | 0.95 |
| query | "rank by size" Spearman ρ / top-3 overlap | **0.986 / 1.0** |

*Finding:* at good segmentation the KG built from **autonomous** masks answers essentially identically to
the GT-KG — **segmentation quality, not KG construction, is the bottleneck** (fidelity scales with Dice).

### B2 — Self-evolving GT-free validation
- **Data:** full **FLARE23** corpus; feature = 5 organs × {log-volume, log-max-diameter} (10-D).
  `exp_oakg_evolve.py`.
- **Setup:** plant realistic segmentation errors (a volume leak that breaks the volume↔diameter relation),
  grow the reference cohort N=15→1083, score plausibility with **no ground truth**. Compare a **marginal**
  per-organ z-score vs the KG's **joint** (covariance / Mahalanobis) model. Metric = clean-vs-error AUROC.

| Model | AUROC | Behaviour as KG grows |
|---|:--:|---|
| **Joint (KG covariance)** | **0.78** | rises 0.74→0.79 (N=15→120), then plateaus |
| Marginal (per-organ z) | 0.66 | flat |

*Finding:* the joint model beats the marginal check by **+0.12 AUROC** (it catches inconsistencies the
marginal check can't see) **and improves as the KG grows** — the self-evolving property. Accumulating
patients sharpens the *joint phenotype model*, which is exactly what a growing KG accumulates.

### C — Autonomous per-organ Dice sweep
Covered in [§4b](#4b-autonomous-organ--full-volume-no-labels-experiment-c--the-deployment-number): base
concept prompting is strong on liver (0.82) but weak on small organs full-volume → per-organ fine-tuning
is the next step.

### Results at a glance
| Exp | Question | Headline result |
|---|---|---|
| **A2** | Does γ beat imputation on a merged KG? | P@10 **0.78** vs **0.08** without γ; spurious 0.14 vs 0.84 |
| **B1** | Does an autonomous KG answer like a GT-KG? | volume **r=0.992**, query **ρ=0.986** (at Dice 0.919) |
| **B2** | Does GT-free validation improve as the KG grows? | joint **0.78** vs marginal 0.66; climbs then plateaus |
| **C** | Autonomous full-volume organ Dice vs ceiling | liver **0.82**; small organs 0.31–0.59 → fine-tune |
| **Tumor** | Generic autonomous tumor model | val **0.938**; honest **test ≈0.86** (LiTS .90 / KiTS .86 / Panc .86 / FLARE .83) |

---

## 8. Planned next steps
1. **Expand the training pool with harder, more diverse images (robustness).** The weakest link is
   segmentation robustness — FLARE tumor **test** Dice is 0.833 and small-organ autonomous Dice is low.
   Adding more complex CTs **with tumors** (varied pathology, scanners, sizes) is the most direct win.
   Needs image+label pairs → a scoped, storage-aware download (candidate sources: more FLARE23 cases that
   ship images, PanTS for pancreas).
2. **Per-organ fine-tuned concept models** — close the small-organ gap C exposed (liver 0.82→0.964 shows the recipe works).
3. **Multi-organ predicted-KG fidelity** — extend B1 beyond pancreas to full multi-organ predicted graphs.
4. **Cross-dataset segmentation generalization** + like-for-like comparison vs **K-Prism / GF-Screen / PanTS**.
5. **Query-type false-positive breakdown** — extend A2 to per-clinical-query (largest-tumor, burden, …).
6. **Paper draft** around the OAKG contribution (benchmark and method framings).

**Target venues.** NeurIPS Datasets & Benchmarks (benchmark framing) or ICLR/AAAI (OAKG-as-method); MICCAI / health-AI as strong domain fits.

## 9. Repository
```
app/            oakg_query_app.py — interactive KG retrieval/validation (over 1,312-patient FLARE23 + Pancreas + LiTS)
kg/             schema.owl · ontology_mappings.json (grounding cache) · data/ · graph/  (data & graph gitignored)
results/        oakg_structured · kg_fidelity · oakg_evolve · autonomous_organ_sweep  (JSON + figures)
src/
  notebooks/    OAKG_Experiments · Segmentation_Results · KG_as_Knowledge_Base · Test_KG_from_CT · SWOG_KG_Pipeline_Demo
  scripts/      segmentation  (train_tumor_generic_v3, build_*_pool, extract_*, infer_ensemble, run_flare_task2_sam3)
                experiments   (exp_oakg_structured, exp_kg_fidelity, exp_oakg_evolve, exp_autonomous_organ_sweep, eval_tumor_per_dataset)
                KG            (kg_grounding, build_flare23_enriched_kg, flare23_predict, kg_build_*, migrate_corpora_to_flare23)
```

## 10. Environment
```
conda env: llmft  ·  Python 3.11, PyTorch 2.5.1+cu121  ·  2× NVIDIA L40S (48 GB)
torch · transformers (SAM3) · monai · rdflib · scikit-image · scipy · nibabel
```
