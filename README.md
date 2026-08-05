# SWOG — Autonomous Abdominal-CT Segmentation → Self-Evolving Imaging Knowledge Graph

**Aim.** Take a **new abdominal CT with no annotations**, automatically segment **liver, kidney, pancreas
and their tumors**, turn those masks into an **ontology-grounded knowledge graph**, and use that graph to
validate the result *without ground truth*, retrieve similar patients, and get smarter with every new case.

This README describes the **current approach and status**, with headline results tables (the baseline organ and
tumor models, [§6](#6-status--roadmap)); the full set of numbers lives in [`results/`](results/) (JSON + figures)
and the walkthrough notebooks in [`src/notebooks/`](src/notebooks/).

---

## 1. The system

A clinician uploads a **new, unlabeled abdominal CT**. The pipeline:

```
new CT (no labels)
  1. Segment    organs → generic organ model  (text prompt "liver"/"kidney"/"pancreas", no box)
                tumor  → generic tumor model   (text prompt "tumor", no box)
  2. Phenotype  per structure: volume, max diameter, centroid, tumor burden, lesion count …
  3. Repair     KG anatomical atlas cleans each mask (keep the plausible component, drop spurious blobs)
  4. Validate   compare each phenotype to the KG cohort; flag the implausible (a 4,900 cc "liver")
                as a likely segmentation error — with NO ground truth
  5. Retrieve   find the most similar known patients (observability-aware retrieval, OAKG)
  6. Grow       admit the patient; the cohort sharpens → the next case is validated better
```

Two ideas carry the whole design:

1. **Two generic, label-free segmenters** — one for tumors, one for organs — each driven only by a **text
   prompt**, no box and no ground truth at inference (SAM3 backbone).
2. **A knowledge graph that closes the loop in both directions** — it is not a passive store. It *shapes
   training* (an anatomical-plausibility loss) **and** *interprets inference* (repairs masks, validates
   phenotypes with no labels, retrieves similar patients, and grows as patients arrive).

**Two ways we use data — the key to the design.**
- **Image + label pairs → train the segmenters.** A model can only learn to draw a mask from images paired
  with correct masks. Training sources: **LiTS, MSD Pancreas, KiTS23, FLARE-Task2, and the imaged FLARE23
  cases** — ~950 of FLARE23's 1,312 have retrievable CT, and we extract organ/tumor slices from them (organ:
  449 cases → 10,759 slices; tumor: 270 cases → 7,269 slices).
- **Labels alone → build the graph.** The KG needs only *numbers* derived from a mask (volume, diameter,
  burden), not pixels — so **all 1,312 FLARE23 labels** populate the KG (including the ~360 with no image, which
  can't train a segmenter but still add a KG patient for free).

Segmentation entry point: [`src/scripts/infer_ensemble.py`](src/scripts/infer_ensemble.py). KG build:
`build_flare23_enriched_kg.py` + `kg_grounding.py`. Interactive KG: [`app/oakg_query_app.py`](app/oakg_query_app.py).

---

## 2. The two label-free models

Both are single generic models prompted by text (no per-organ weights, no boxes, no GT at inference), trained
from the shared SAM3 recipe (partial-freeze, Dice+Focal, discriminative LR, cosine schedule, DDP).

- **Generic tumor model** — one model, prompt `"tumor"`, trained on pooled tumor slices from LiTS + MSD +
  KiTS + FLARE23. Evaluated strictly **patient-level**, and **cross-dataset** (leave-one-dataset-out) to test
  generalization to entirely unseen tumor types. `train_tumor_incremental.py`, `eval_tumor_per_dataset.py`.
- **Generic organ model** — one model, prompt `"liver"`/`"kidney"`/`"pancreas"`, trained on a pooled,
  balanced, strictly patient-level organ pool (LiTS + KiTS + MSD + FLARE-Task2 + FLARE23).
  `build_organ_pool_lkp.py`, `train_organ_generic.py`. *(Currently training — see [§6](#6-status--roadmap).)*

**Why generic and label-free.** A prior AUSAM/SAM1 pipeline needed a GT-derived box for every prediction, so
it could never touch a *new, unlabeled* scan. SAM3 concept prompting removes that dependency; the two models
above are the autonomous, deployable form.

**Why fine-tune a pretrained model at all?** SAM3 is pretrained on *natural* images — an excellent
"segment-what-I-point-at" engine, but with **no notion of medical structures on CT**. Fine-tuning
(partial-freeze: keep the backbone, train only the last ~20 blocks + mask decoder) teaches it three things
pretraining can't: (1) the **concepts** — what liver/kidney/pancreas/tumor look like on CT; (2) to **answer the
text prompt on CT** (outline the liver from the word alone); (3) the **CT domain** — HU-windowed grayscale
cross-sections, nothing like natural photos. The lift is largest where SAM3 knows least: base concept prompting
scores ~0.93 on the big, obvious **liver** but only **0.27–0.37 on tumors (essentially fails)** — fine-tuning
takes tumors to 0.70–0.94. So the training is what makes tumor segmentation *exist at all*; the KG loss then
layers anatomical plausibility on top.

---

## 3. Datasets & splits

| Dataset | Images? | Role | Structures |
|---|:--:|---|---|
| **LiTS** | ✅ | train tumor + organ (liver) | liver, liver tumor |
| **MSD Pancreas** (Task07) | ✅ | train tumor + organ (pancreas) | pancreas, pancreas tumor |
| **KiTS23** | ✅ | train tumor + organ (kidney) | kidney, kidney tumor |
| **FLARE-Task2** (100 vols) | ✅ | train organ (l/k/p) **+ KG records** | 13 organs |
| **FLARE23** (1,312 labels; ~950 imaged) | ✅ ~950 | **train** organ (449 cases) + tumor (270 cases) from image slices; **KG** from all 1,312 labels | 13 organs + tumor |

**Splitting.** Every training pool is split **by whole patient** (seed 42), roughly **70 / 10 / 20**
train / val / test: validation and test are held-out *patients*, never other slices from a training patient —
no leakage. Cross-dataset evaluation trains on some datasets and tests on an entirely unseen one. Exact counts
(**patients / slices**):

**Organ model** (`organ_pool_lkp` — one pooled model):

| Dataset | Train (pat / slices) | Val | Test |
|---|:--:|:--:|:--:|
| LiTS | 25 / 3,883 | 3 / 409 | 6 / 708 |
| KiTS23 | 70 / 2,822 | 10 / 447 | 20 / 552 |
| MSD Pancreas | 116 / 3,521 | 16 / 469 | 33 / 1,010 |
| FLARE-Task2 | 70 / 9,505 | 10 / 1,503 | 20 / 2,921 |
| FLARE23 | 316 / 7,567 | 44 / 1,056 | 89 / 2,136 |
| **Total** | **597 / 27,298** | **83 / 3,884** | **168 / 7,327** |

**Tumor model** (incremental LODO — per-dataset *test* patients are fixed across all stages, so the coverage
curve is a fair comparison):

| Dataset | Train (pat / slices) | Val | Test |
|---|:--:|:--:|:--:|
| LiTS | 76 / 3,704 | 10 / 747 | 21 / 1,149 |
| MSD Pancreas | 197 / 1,811 | 28 / 261 | 56 / 465 |
| KiTS23 | 126 / 3,901 | 18 / 436 | 36 / 930 |
| FLARE23 | 189 / 5,208 | 27 / 453 | 54 / 1,608 |
| **Total** | **588 / 14,624** | **83 / 1,897** | **167 / 4,152** |

**Coverage — available vs. used (patients).** We do not use every case; the reasons differ by dataset:

| Dataset | Available | Organ used | Tumor used | Why not all |
|---|:--:|:--:|:--:|---|
| LiTS | 131 | 34 | 107 | organ: 5k-slice cap · tumor: 107 have a tumor |
| MSD Pancreas | 281 | 165 | 281 | organ: 5k cap · tumor: **all** (every case has a tumor) |
| KiTS23 | ~489 | 100 | 180 | **extraction subset** (we pulled 100–180, not all) |
| FLARE-Task2 | 100 | 100 | — | organ: **all** · tumor: no tumor labels in this set |
| FLARE23 | 1,312 labeled (~950 imaged) | 449 | 270 | organ: **extraction stopped at 449/950** · tumor: 270 tumor-bearing imaged |

Four reasons cases drop out: **(1)** a **5,000-slice balance cap** per (organ, dataset) so no source dominates
a class; **(2)** the tumor pool is **tumor-bearing only**; **(3)** FLARE-Task2 has **no tumor labels**; **(4)**
**extraction cutoffs** (FLARE23 organ 449/950 — cluster restart; KiTS a subset of ~489). Only (4) is
unintended — resumable, usable data left on the table (the clearest lever for the hard classes, pancreas/tumor).

**FLARE23 label note.** The local label store holds ~2,200 FLARE23 files, but **888 are empty** (the unlabeled
portion of the challenge) — only **1,312 carry real annotations**, which is the number we use everywhere.

**Label scope.** For FLARE, segmentation trains on **liver / kidney / pancreas (+ tumor) only** — spleen and the
other eight organs are dropped from *training*; the **KG still records all 13** (free anatomy, richer retrieval).

**Split vs. score — read this carefully.** The *split* is patient-level, but the automated test metric is a
**slice-level 2-D Dice** (averaged over the held-out patients' organ-present slices). That is patient-**split**,
slice-**scored** — it does *not* test whether the model finds the right slices in a whole volume or fires on
organ-*absent* ones, so it reads **optimistically**. A true **patient-level (3-D / whole-volume) Dice** comes
only from the separate **full-volume evaluation** (run the model over every slice of a patient and score the
assembled 3-D mask). Treat slice-level numbers as an upper read; the full-volume numbers are the deployment truth.

---

## 4. The knowledge graph

**Structure.** Direct (non-reified) triples per case: an `ImagingCase` `depicts_organ` each `Organ`
(`gt_volume_cm3`, `gt_max_diameter_mm`, `gt_centroid_mm`, `has_lesion`, `mapped_to_concept`), and each
`Lesion` (`is_tumor`, volume, diameter, `located_in`, `tumorBurden`, `lesion_count`). Organs and lesions are
**grounded to ontologies** (SNOMED / LOINC / ICD / MeSH) via `kg_grounding.py`.

**Coverage.** FLARE23 (1,312 labels) + Pancreas + LiTS + KiTS + FLARE-Task2 records → a pooled corpus and an
**anatomical atlas** (`build_kg_atlas.py`): per-organ plausible size / diameter / location priors. FLARE-Task2
adds 100 fully-labelled organ patients (13 organs, no tumor) so every training source is also represented in
the graph.

**The KG closes the loop in two places — this is the core contribution:**
- **At training** — from the *training patients only* we derive a leakage-free atlas (each organ's typical
  location + plausible size band) and add a differentiable **plausibility / consistency loss**
  (`kg_consistency_loss` in `run_pancreas_sam3.py`). Each model can be trained **baseline vs KG-in-training**;
  the difference isolates the KG's contribution to *learning* (`train_organ_generic.py --kg`).
- **At inference** — the same anatomical knowledge **repairs** masks (`kg_guided_segment.py`: keep the
  plausible connected component, drop spurious blobs), **validates** phenotypes against the cohort with no
  labels, and drives **retrieval** and **growth**.

---

## 5. Research contribution

The contribution is a full **label-free, self-improving pipeline** for abdominal-CT interpretation, and its
novelty is the **bidirectional coupling** of autonomous segmentation with an ontology-grounded, self-evolving
knowledge graph. Prior knowledge-guided systems act at the perception / data layer (e.g. K-Prism, GF-Screen,
PanTS); ours adds the **reasoning layer** that *trains from, vets, and grows on* segmentation with no ground
truth. Four pillars:

1. **Label-free autonomous segmentation.** Two generic text-prompted models (tumor + organ) segment new,
   unlabelled CTs with no box and no GT — the deployable form ([§2](#2-the-two-label-free-models)).
2. **Knowledge graph in the loop — both directions.** The KG *shapes training* (a leakage-free
   anatomical-plausibility loss) **and** *vets inference* (atlas-guided repair, GT-free validation):
   segmentation feeds the KG, and the KG's accumulated anatomy improves segmentation. This closed loop is the
   central idea, and it distinguishes us from one-directional knowledge-guided segmentation.
3. **OAKG — reasoning under partial observability.** On a merged graph where datasets label different organs,
   retrieval and validation are **evidence-calibrated**: never impute missing structures, weight similarity by
   joint observability (γ). This is the graph's *reasoning engine* — one pillar, not the whole contribution.
4. **Self-evolving.** The graph validates new masks against the cohort with no labels and sharpens as patients
   are admitted.

The experiment suite probes these (scripts `src/scripts/exp_*.py`, numbers in `results/*.json`, walkthrough in
`src/notebooks/OAKG_Experiments.ipynb`):

- **A — Retrieval.** Does γ-weighting retrieve the *right* similar patients on a merged graph, and suppress
  spurious matches from missing-not-shared structures?
- **B — Fidelity.** Is a graph built from *predicted* masks as trustworthy as one from expert masks
  (phenotype agreement, query-ranking agreement)?
- **C — Self-evolving.** Does the graph get *better at catching implausible masks* as the cohort grows
  (joint vs marginal outlier detection)?
- **D — Autonomous + KG-repair.** How well does the label-free pipeline segment with no labels, and how much
  does inference-time KG repair recover?

---

## 6. Status & roadmap

**The KG-in-training ablation.** Each model is trained twice — **baseline** vs **`--kg`** (plausibility loss) —
giving **four checkpoints** (organ ± KG, tumor ± KG). Comparing each pair isolates the KG's contribution to
*learning*; `compare_kg_ablation.py` writes the deltas to `results/kg_ablation_summary.json`. The organ
plausibility term uses size **and** location priors; the tumor term is **size-only** (a tumor's location is not
stable). Priors are computed from the training split only (`build_organ_train_priors.py`, `build_tumor_train_priors.py`).

**Now running** — organ baseline trained (converged); organ **+KG** and tumor **+KG** training at ≤6 epochs
(fast; the models converge by ~epoch 5), then the auto-comparison. All of this is scored **slice-level**
(patient split). **Ablation matching:** the **organ** pair is matched at convergence (baseline plateaus by
epoch 4); the **tumor** KG@6 is compared to the existing **14-epoch** baseline, so its delta is a
**directional first look, not epoch-matched** — `compare_kg_ablation.py` self-flags this, and a matched tumor
baseline@6 is a cheap follow-up if the directional result warrants it.

**Baseline generic organ model — held-out per-dataset test Dice** *(slice-level, patient-split; the
patient-level 3-D number will be lower, see [§3](#3-datasets--splits))*:

| Organ | FLARE23 | FLARE-Task2 | LiTS | KiTS | MSD | **mean** |
|---|:--:|:--:|:--:|:--:|:--:|:--:|
| **Liver** | 0.898 | 0.962 | 0.949 | — | — | **0.941** |
| **Kidney** | 0.885 | 0.938 | — | 0.889 | — | **0.909** |
| **Pancreas** | 0.756 | 0.836 | — | — | 0.771 | **0.787** |

Pancreas is the hard organ (the KG's clearest target); FLARE-Task2 is cleanest across the board. The **+KG**
deltas fill in when that run completes (`results/organ_generic_kg.json`). Source: `results/organ_generic.json`.

**Baseline generic tumor model** — one model, prompt `"tumor"`, no box, trained incrementally
(LiTS → +Pancreas → +KiTS → +FLARE), strictly patient-level. Two views:

*(a) Deployment (all 4 datasets) — within-dataset test Dice:*

| LiTS | KiTS | MSD Pancreas | FLARE23 | **mean** |
|:--:|:--:|:--:|:--:|:--:|
| 0.671 | 0.788 | 0.648 | 0.708 | **0.704** |

*(b) Cross-dataset generalization — FLARE23 held out until the last stage; its Dice as coverage grows (a
**cross-dataset** number until the last row, then in-distribution):*

| Trained on | FLARE23 Dice | |
|---|:--:|---|
| LiTS | 0.348 | cross-dataset |
| + Pancreas | 0.405 | cross-dataset |
| + KiTS | 0.514 | cross-dataset |
| + FLARE (all 4) | 0.708 | in-distribution |

**Coverage must be trained in:** each dataset added lifts the held-out FLARE23 number (**0.35 → 0.51**), and
folding FLARE in reaches 0.71 — the evidence that a generic model generalizes only to tumor types it has seen.
Same slice-level caveat as above. Source: `results/tumor_incremental.json`.

**Established** — generic tumor baseline (patient-split + cross-dataset generalization); OAKG experiments A–D
(component-level, on curated / GT-derived inputs). *(Numbers: `results/`.)*

**The end-to-end evaluation — the payoff (a deliberate next step, not automatic).** Segmentation Dice is only
layer 1. Once the trained models exist, a single **full-volume pass over held-out cases** produces the substrate
for the real system-level test:
1. **Patient-level (3-D / whole-volume) Dice** — the honest deployment number. It **supersedes** the slice-level
   figures *and* the base-SAM3 *0.49 → 0.61* repair result, and yields the up-to-date **KG-repair** delta on the
   trained model. (Note: more epochs is *not* the lever for a weak patient-level number — the slice→volume gap is
   full-volume false positives from training on organ-*present* slices only; the levers are **negative slices**
   and **KG-repair**, not epochs.)
2. **Confusion matrices** — per model, {liver, pancreas, kidney, others} for organs and {tumor, others} for tumor
   (`confusion_eval.py`).
3. **OAKG at inference — the contribution — end-to-end on the pipeline's *own predicted* output:**
   **GT-free validation** (does OAKG flag bad segmentations with no labels? scored by AUROC of flagged vs
   actually-low-Dice, GT used only to grade the validator), **retrieval on predicted phenotypes** (does
   γ-weighting survive real pipeline noise?), and **growth** (admitting predicted cases sharpens the cohort).
   Experiments A–D validated these as components; this is the **system-level** claim — the reasoning layer working
   on the messy output of the autonomous segmenters.

**Next** — expand the pool with harder tumor-bearing CTs (segmentation robustness is the weakest link);
multi-organ predicted-KG fidelity; like-for-like comparison vs **K-Prism / GF-Screen / PanTS**; paper draft
around the full contribution — **label-free segmentation coupled bidirectionally with a self-evolving KG**,
with OAKG as the reasoning layer (benchmark and method framings).

**Target venues.** NeurIPS Datasets & Benchmarks (benchmark framing) or ICLR/AAAI (the KG-coupled method);
MICCAI / health-AI as domain fits.

---

## 7. Repository

```
app/            oakg_query_app.py — interactive KG retrieval / GT-free validation
kg/             schema.owl · ontology_mappings.json · data/ · graph/        (data & graph gitignored)
results/        oakg_structured · kg_fidelity · oakg_evolve · autonomous_organ_sweep · tumor_incremental  (JSON + figures)
src/
  notebooks/    OAKG_Experiments · Segmentation_Results · KG_as_Knowledge_Base · Test_KG_from_CT · SWOG_KG_Pipeline_Demo
  scripts/
    tumor model    train_tumor_incremental [--kg] · eval_tumor_per_dataset · build_tumor_pool · build_tumor_train_priors · rebuild_lits_pool_patientlevel
    organ model    build_organ_pool_lkp · build_organ_train_priors · train_organ_generic [--kg | --eval-only]
    ablation       compare_kg_ablation  (baseline vs KG deltas -> kg_ablation_summary.json)
    shared trainer run_pancreas_sam3 (SAM3 partial-freeze + Dice/Focal + kg_consistency_loss) [+ base: run_flare, run_pancreas_nifti]
    KG             kg_grounding · build_flare23_enriched_kg · build_flare_task2_kg_records · kg_build_* · build_kg_atlas · build_kits_kg_records · flare23_predict
    inference      infer_ensemble · kg_guided_segment · kg_guided_eval
    experiments    exp_oakg_structured · exp_kg_fidelity · exp_oakg_evolve · exp_autonomous_organ_sweep
```

## 8. Environment
```
conda env: llmft · Python 3.11 · PyTorch 2.5.1+cu121 · 2× NVIDIA L40S (48 GB)
torch · transformers (SAM3) · monai · rdflib · scikit-image · scipy · nibabel
```
