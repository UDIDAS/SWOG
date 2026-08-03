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
| **Generic tumor model** (one model, prompt `"tumor"`) | pooled tumor slices: LiTS + Pancreas + KiTS + FLARE (~20.7k) | SAM3 fine-tune, no box | held-out **patients** per dataset → **≈0.85** ([§5](#5-segmentation-results--the-generic-tumor-model)) |
| **Organ models** — fine-tuned | FLARE-Task2 (100 image+label pairs) | SAM3 + GT box (semi-oracle) | 20 held-out **patients** → 0.92–0.99 ([§4a](#4a-organ-ceiling--flare-task2-models-patient-level-test-the-honest-organ-numbers)) |
| **Organ models** — autonomous | *no training* (base SAM3 `"liver"` prompt) | concept prompt, no box | 5 held-out FLARE23 CTs → liver 0.82; small organs need fine-tuning (§4b) |

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
| **LiTS** (Liver Tumor Seg.) | liver + liver tumor | 131 per-patient volumes | **case-level** for the dedicated LiTS model (patient-held-out); slice-level only inside the *pooled* tumor model | tumor pool (5,600 slices); dedicated case-level LiTS tumor model |
| **MSD Pancreas** (Decathlon Task07) | pancreas + pancreatic tumor | 281 labeled | **case-level** | tumor pool (2,537); pancreas organ+tumor delivery models |
| **KiTS23** (Kidney Tumor Seg. 2023) | kidney + kidney tumor | 489 cases | **case-level** | tumor pool (5,267) |
| **FLARE23** (full) | **13 organs + tumor** | **1,312 patients** (labels-only), 608 with tumor (liver/kidney/pancreas) | **case-level** | **the knowledge graph**; OAKG experiments **A and C**; tumor pool (7,269 FLARE tumor slices) |
| **FLARE-Task2 2024** | 13 organs (no tumor) | 100 volumes (50 train_gt + 50 public-val) | **patient-level 70/10/20** (seed 42) | the **organ segmentation models** (semi-oracle ceilings); pancreas cases for **Experiment B** |

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

### 4b. Autonomous organ — full-volume, no labels *(Experiment D — the no-labels number)*
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
| LiTS | 5,600 | slice-level in the *pooled* model *(honest patient-level LiTS = 0.840, see §5a †)* |
| KiTS23 | 5,267 | case-level |
| FLARE23 | 7,269 | case-level |
| MSD Pancreas | 2,537 | case-level |
| **total** | **20,673** | → train / val / test = **14,589 / 1,615 / 4,469** |

### 5a. Headline: validation vs. honest per-dataset test
- **Validation Dice (during training) = 0.938.** This is the *slice-level* validation split used to pick
  the model — **slightly optimistic** (a val slice can share a patient with a train slice).
- **Held-out TEST Dice (never used for selection)** — the numbers to trust. Autonomous (`"tumor"`, no
  box), per source dataset (`eval_tumor_per_dataset.py`):

| Dataset | Autonomous **test** Dice | Split |
|---|:--:|---|
| LiTS | **0.840** † | **case-level** (patient-held-out) |
| KiTS | **0.864** | case-level |
| Pancreas | **0.856** | case-level |
| FLARE | **0.833** | case-level |
| **mean** | **≈0.85** | all patient/case-level |

† **LiTS honesty note.** The *pooled* model's own LiTS split was slice-level, which reads **0.897** —
inflated by adjacent-slice leakage (neighbouring slices of one patient in both train and test). The
**honest patient-held-out** LiTS number is **0.840**, from a dedicated case-level LiTS tumor model
(`lits_sam3_v3_caselevel_tumor`, the 131 raw volumes split **by patient**, seed 42) — the same model whose
per-patient reconstructions were delivered to collaborators. The ~0.07 drop (0.910/0.897 slice → 0.840
patient) *is* the leakage the slice-level split hid. **All four datasets are now reported case/patient-level.**

*Meaning:* on genuinely unseen **patients** the autonomous tumor model scores **≈0.83–0.86** depending on
dataset (KiTS highest, FLARE lowest). The often-quoted **0.938** is the *validation* number (slice-level,
optimistic); **≈0.85 is the honest patient-level test performance.**

### 5b. Coverage grows the model
Validation Dice as the pool grew: **0.37** (base, no training) → **0.909** (v1: LiTS+Pancreas) →
**0.9145** (v2: +FLARE) → **0.938** (v3: +KiTS). **Cross-dataset caveat:** the v1 model (liver+pancreas
only) scored **0.02** on unseen FLARE kidney tumors — reliable **within trained tumor types**, not
universally; coverage must be trained in.

---

## 6. The knowledge graph

**The graph spans all three training datasets — 1,724 patients:** **FLARE23** (1,312, 13 organs + tumor),
**MSD Pancreas** (281, pancreas + tumor), and **LiTS** (131, liver + tumor). FLARE23 is the largest
component and the richest (up to 13 organs per patient); Pancreas and LiTS add organ+tumor detail for
their sites. (`corpus_perpatient.json` is the unified cohort; `imaging_kg_flare23.ttl` is the FLARE23
component in RDF/Turtle.) Each patient is a subgraph `ImagingCase → Organ / Lesion`, phenotypes as
**direct triples** (`gt_volume_cm3`, `gt_max_diameter_mm`, `gt_centroid_mm`, `tumorBurden`,
`lesionMultiplicity`, `lesion_count`), every entity grounded to **SNOMED / LOINC / ICD / MeSH** via a live
mapper — no hard-coded codes; open-world.

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

### Experiment A — Does it retrieve the *right* similar patients on a merged graph?  *(the core result)*
**In one line:** we take real patients, hide organs the way the real datasets do (each "source" labeled
different organs), then ask each method to find each patient's true look-alikes. Data: the 1,312-patient
FLARE23 corpus. Script: `exp_oakg_structured.py`.

| Method | Found the right look-alikes *(0–1, higher = better)* | Returned junk matches *(0–1, lower = better)* |
|---|:--:|:--:|
| **OAKG — ours** | **0.78** | **0.14** |
| "fill the missing organs with the average" | 0.73 | 0.17 |
| "fill the missing organs with zero" | 0.71 | 0.19 |
| **ours, but with the γ weighting switched off** | **0.08** | **0.84** |

**What the two columns mean.** *Found the right look-alikes* — of the 10 patients it called "most
similar," how many genuinely are (the textbook name is **Precision@10**). *Junk matches* — of those 10,
how many were ranked similar despite sharing almost nothing (the false matches we're trying to kill).

**Reading it:** ours finds the most right matches and the fewest junk ones. The last row is the key test —
take our method and **switch off the γ weighting**, and it falls apart (right 0.78 → 0.08, junk 0.14 →
0.84). So **the γ weighting is the thing that works**: without it, one shared organ fakes a "perfect match."

### Experiment B — Is a graph built from *AI* masks as trustworthy as one from *doctor* masks?
**In one line:** for 20 patients we build the graph twice — once from the AI's segmentation, once from the
doctor's ground-truth mask — and check whether it gives the same answers. Script: `exp_kg_fidelity.py`.

| What we check | Result | Plain meaning |
|---|:--:|---|
| tumor **volume**: AI vs doctor | within **4.8%** | the AI-derived volume is ~5% off the true value, on average |
| do the volumes line up? | **0.99** | AI volume tracks the true volume almost perfectly (1.0 = identical) |
| "who has the biggest pancreas?" | **identical top-3** | the query returns the same patients in the same order (rank agreement 0.99) |

**Reading it:** when the AI segments well, its graph answers *the same* as a doctor-labeled graph — so the
graph is trustworthy, and **the only thing that limits it is segmentation quality**, not the graph.

### Experiment C — Does the graph get *better at catching bad masks* as it grows?
**In one line:** we feed in masks with realistic errors and ask the graph to flag them using **no ground
truth** — just how the new patient compares to everyone already in the graph — then grow the graph and
watch. Script: `exp_oakg_evolve.py`.

| The graph's "is this mask believable?" check | Score *(0.5 = coin-flip, 1.0 = perfect)* | As the graph grows… |
|---|:--:|---|
| **using learned organ relationships** (ours) | **0.78** | **improves** (0.74 → 0.79), then levels off |
| a plain per-organ size range (baseline) | 0.66 | stays flat |

**What the score means.** How well the check separates good masks from broken ones (textbook name:
**AUROC** — 0.5 is pure guessing, 1.0 is flawless).

**Reading it:** the graph catches bad masks better than a plain size check, **and it improves as more
patients join** — because it learns how organs relate (a normal liver beside a tiny pancreas is suspicious
even if each looks fine alone). That's the "self-improving" property.

### Experiment D — How well does it segment organs with **no** labels?
Covered in §4b: with no training and no labels, base SAM3 segments the **liver** well over the whole volume
(0.82) but struggles on small organs → per-organ fine-tuning is the next step.

### The experiments at a glance
| | The question (plain English) | The answer |
|---|---|---|
| **A** | Does it find the right similar patients on a merged graph? | Yes — 0.78 right / 0.14 junk; **collapses to 0.08 / 0.84 without our γ weighting** |
| **B** | Is an AI-built graph as good as a doctor-built one? | Yes — volumes within ~5%, identical query rankings |
| **C** | Does bad-mask detection improve as the graph grows? | Yes — 0.78 vs 0.66 baseline, and it climbs with size |
| **D** | Can it segment organs with no labels? | Liver yes (0.82); small organs need fine-tuning |
| **Tumor** | How good is the tumor model on unseen patients? | ≈0.85, all patient-level (LiTS 0.84 / KiTS 0.86 / Pancreas 0.86 / FLARE 0.83) |

---

## 8. Planned next steps
1. **Expand the training pool with harder, more diverse images (robustness).** The weakest link is
   segmentation robustness — FLARE tumor **test** Dice is 0.833 and small-organ autonomous Dice is low.
   Adding more complex CTs **with tumors** (varied pathology, scanners, sizes) is the most direct win.
   Needs image+label pairs → a scoped, storage-aware download (candidate sources: more FLARE23 cases that
   ship images, PanTS for pancreas).
2. **Per-organ fine-tuned concept models** — close the small-organ gap Experiment D exposed (liver 0.82→0.964 shows the recipe works).
3. **Multi-organ predicted-KG fidelity** — extend Experiment B beyond pancreas to full multi-organ predicted graphs.
4. **Cross-dataset segmentation generalization** + like-for-like comparison vs **K-Prism / GF-Screen / PanTS**.
5. **Query-type false-positive breakdown** — extend Experiment A to per-clinical-query (largest-tumor, burden, …).
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
