# SWOG — Autonomous Abdominal-CT Segmentation → Self-Evolving Imaging Knowledge Graph

**One-line aim.** Take a **new abdominal CT with no annotations**, automatically segment **liver, kidney,
pancreas and their tumors**, turn those masks into an **ontology-grounded knowledge graph**, and use that
graph to validate the result *without ground truth*, retrieve similar patients, and get smarter with every
new case.

Two ideas carry the whole system:

1. **Two generic, label-free segmenters.** One model for tumors, one for organs — each driven only by a
   **text prompt** (`"tumor"`, `"liver"`, …), no box, no ground truth at inference. (SAM3 backbone.)
2. **A knowledge graph that closes the loop — in both directions.** The KG is not a passive store. It
   *shapes training* (an anatomical-plausibility loss) **and** *interprets inference* (repairs masks,
   validates phenotypes with no labels, retrieves similar patients, and grows as patients arrive).

> **How to read this README.** Every number states **what it was trained on**, **what it was tested on**,
> and **what it means**. Splits are always **by whole patient** (val/test held out per patient) unless
> marked otherwise. See [§2 Datasets & splits](#2-datasets--splits) and [§3 Reading the scores](#3-reading-the-scores).

---

## 1. The system in plain terms

A clinician uploads a **new, unlabeled abdominal CT**. The system:

```
new CT (no labels)
  1. Segment    organs → generic organ model  (prompt "liver"/"kidney"/"pancreas", trained, no box)
                tumor  → generic tumor model   (prompt "tumor", trained, no box)
  2. Phenotype  per structure: volume, max diameter, centroid, tumor burden, lesion count …
  3. Repair     KG anatomical atlas cleans each mask (keep the plausible component, drop spurious blobs)
  4. Validate   compare each phenotype to the KG cohort; flag the implausible (a 4,900 cc "liver")
                as a segmentation error — with NO ground truth
  5. Retrieve   find the most similar known patients (observability-aware retrieval, OAKG)
  6. Grow       admit the patient; the cohort sharpens → the next case is validated better
```

Steps 3–6 are what the graph is *for*: it vets and interprets each unlabeled scan and improves as it grows.

**Two ways we use data — the key to the design.**
- **Image + label pairs → train the segmenters.** You can only teach a model to draw a mask by showing it
  images with correct masks. Training sources: **LiTS, MSD Pancreas, KiTS23, FLARE-Task2**.
- **Labels alone → build the graph.** The KG needs only *numbers* derived from a mask (volume, diameter,
  burden), not pixels. **FLARE23's 1,312 label masks** (no images) populate the KG and retrieval cheaply.

**The KG helps in two places, not one** — this is the core contribution:
- **At training** — from the *training patients only* we compute a leakage-free atlas (each organ's typical
  location + plausible size band) and add a differentiable **plausibility/consistency loss**. Each model is
  trained **baseline vs KG-in-training**; the gap isolates the KG's contribution to *learning*.
- **At inference** — the same anatomical knowledge **repairs** masks and **validates** phenotypes without
  labels ([§4b](#4b-autonomous-segmentation--the-kg-repair-ablation), [§6](#6-the-knowledge-graph)).

---

## 2. Datasets & splits

| Dataset | Has images? | What we use it for | Structures |
|---|:--:|---|---|
| **LiTS** | ✅ | train tumor + organ (liver) | liver, liver tumor |
| **MSD Pancreas** (Task07) | ✅ | train tumor + organ (pancreas) | pancreas, pancreas tumor |
| **KiTS23** | ✅ | train tumor + organ (kidney) | kidney, kidney tumor |
| **FLARE-Task2** (100 vols) | ✅ | train organ (l/k/p) | 13 organs |
| **FLARE23** (1,312 masks) | labels only | build the KG; organ slices for training | 13 organs + tumor |

**Splitting.** Every training pool is split **by whole patient** (seed 42): val and test are held-out
*patients*, never slices from a training patient. This is the honest setting — no slice-level leakage.

---

## 3. Reading the scores

- **Dice** ∈ [0,1], overlap of prediction vs reference; 1 = perfect.
- **Semi-oracle** — the model is given a GT-derived box to localize the organ. Measures segmentation
  *quality* given perfect localization → an **upper ceiling**, not an autonomous number.
- **Autonomous** — text prompt only, no box, no GT. The real deployment number (the model must also find
  *which slices* contain the structure). Always lower than semi-oracle.
- **Full-volume Dice** — scored over the whole 3D volume (includes slice selection), the strictest organ metric.
- **Patient-level** — val/test are whole held-out patients. **Cross-dataset** — trained on some datasets,
  tested on an entirely unseen one (generalization).

---

## 4. Segmentation results

### 4a. Organ ceiling — semi-oracle, patient-level *(the upper bound)*
FLARE-Task2 fine-tuned SAM3, GT-box, 3-D Dice, ~70 train / 19–20 held-out **patients**. `run_flare_task2_sam3.py`.

| Organ | Dice (semi-oracle) | Organ | Dice |
|---|:--:|---|:--:|
| Liver | **0.985** | Left kidney | **0.970** |
| Spleen | 0.980 | Pancreas | **0.919** |
| Right kidney | 0.970 | | |

*With a localizing box, SAM3 segments organs very accurately on new patients; pancreas is hardest. This is
the ceiling the autonomous numbers below aim at.*

### 4b. Autonomous segmentation & the KG-repair ablation
Fully autonomous (base SAM3 concept-prompt organs + generic tumor model, **no boxes, no labels**) on **40
held-out full FLARE CTs**, full-volume Dice. **+KG-repair** applies the anatomical atlas (§6) to each mask.
`exp_autonomous_organ_sweep.py`, `kg_guided_segment.py`, `kg_guided_eval.py`.

| Structure | Autonomous **raw** | **+ KG-repair** | Δ | Ceiling |
|---|:--:|:--:|:--:|:--:|
| Liver | 0.80 | **0.85** | +0.04 | 0.985 |
| Spleen | 0.61 | **0.76** | +0.16 | 0.980 |
| Left kidney | 0.49 | **0.77** | +0.28 | 0.970 |
| Right kidney | 0.47 | **0.56** | +0.09 | 0.970 |
| Pancreas | 0.30 | **0.42** | +0.12 | 0.919 |
| Tumor (full-volume) | 0.28 | 0.27 | −0.01 | — |
| **mean** | **0.49** | **0.61** | **+0.11** | |

**This ablation isolates the KG at inference.** Base concept prompting over-segments small organs and picks
wrong slices; the atlas rule ("keep the plausible component, drop spurious blobs") recovers **+0.11 mean, up
to +0.28** (left kidney), concentrated exactly where autonomous segmentation is messiest. `results/kg_guided_eval.json`.

### 4c. Generic tumor model — one model, prompt `"tumor"`, no box
Pooled tumor slices from **LiTS + MSD + KiTS + FLARE23** (~20.7k), patient-level. `train_tumor_incremental.py`,
`eval_tumor_per_dataset.py`.

- **Deployment (all 4 datasets, strict patient-level): ≈0.70.**
- **Cross-dataset generalization (leave-one-dataset-out).** Training incrementally and always testing on the
  held-out datasets, mean cross-dataset tumor Dice climbs **0.35 → 0.51** as coverage grows — evidence that
  *coverage must be trained in*, not assumed (a liver+pancreas model scored 0.02 on unseen kidney tumors).

| Stage trained on | held-out cross-dataset mean |
|---|:--:|
| LiTS only | 0.35 |
| + Pancreas | 0.44 |
| + KiTS | ↑ |
| + FLARE (all 4) | **0.51** |

*(`results/tumor_incremental.json` has the full per-stage / per-dataset matrix.)*

### 4d. Generic organ model — one model, prompt `"liver"`/`"kidney"`/`"pancreas"` *(in training)*
The proper autonomous organ path (replacing base concept prompting in §4b): one trained, label-free model.
Pooled, balanced, strictly patient-level. `build_organ_pool_lkp.py`, `build_organ_train_priors.py`,
`train_organ_generic.py [--kg]`.

| Organ | LiTS | KiTS | MSD | FLARE-Task2 | FLARE23 | **total** |
|---|:--:|:--:|:--:|:--:|:--:|:--:|
| Liver | 5,000 | — | — | 5,000 | 3,584 | **13,584** |
| Kidney | — | 3,821 | — | 5,000 | 3,592 | **12,413** |
| Pancreas | — | — | 5,000 | 3,929 | 3,583 | **12,512** |

**38,509 slices · 848 patients · 5 datasets** → train 27,298 / val 3,884 / test 7,327 (by patient). Trained
**twice for the ablation**: (i) baseline `0.7·Dice + 0.3·Focal`; (ii) **+KG-in-training** `+0.1·KG_consistency`
using the leakage-free train-split atlas (organ centroids + plausible area bands). **Held-out per-organ /
per-dataset Dice fills in on completion** (`results/organ_generic.json` vs `organ_generic_kg.json`).

### 4e. Per-dataset delivery models *(handed off)*
Fine-tuned SAM3, semi-oracle, case-level — the checkpoints in the collaborator handoff.

| Dataset | Organ Dice | Tumor Dice |
|---|:--:|:--:|
| Pancreas (MSD Task07) | 0.866 | 0.894 |
| LiTS (liver tumor) | — | 0.840 |

---

## 5. The knowledge graph

**What it is.** Direct (non-reified) triples per case: an `ImagingCase` `depicts_organ` each `Organ`
(with `gt_volume_cm3`, `gt_max_diameter_mm`, `gt_centroid_mm`, `has_lesion`, `mapped_to_concept`), and each
`Lesion` (`is_tumor`, volume, diameter, `located_in`, `tumorBurden`, `lesion_count`). Organs and lesions are
**grounded to ontologies** (SNOMED/LOINC/ICD/MeSH) via `kg_grounding.py`.

**What it spans.** FLARE23 (1,312 labels) + Pancreas + LiTS + KiTS records → a pooled corpus and an
**anatomical atlas** (`build_kg_atlas.py`, per-organ plausible size/diameter priors over ~2,100 patients).

**What it does.**
- **Trains** — the train-split atlas drives the KG-in-training plausibility loss (§4d).
- **Repairs** — the atlas cleans autonomous masks (§4b).
- **Validates** — flags phenotypes outside the cohort's plausible range as likely segmentation errors, no GT.
- **Retrieves & grows** — observability-aware retrieval (OAKG, §7) that admits new patients and sharpens.

Build: `build_flare23_enriched_kg.py`, `kg_build_*`, `build_kits_kg_records.py`. Interactive: `app/oakg_query_app.py`.

---

## 6. OAKG — the research contribution

Retrieval and validation on a **merged, partially-observed** graph (different datasets label different
organs). OAKG is **evidence-calibrated**: it never imputes missing structures and weights similarity by
**joint observability** (γ, a Jaccard term). Experiments (`src/scripts/exp_*.py`, `results/*.json`):

| Exp | Question | Result |
|---|---|---|
| **A** — retrieval | Does it retrieve the *right* similar patients on a merged graph? | With γ: **P@10 0.78, spurious 0.14**. Without (masked-cosine): 0.08 / 0.84. γ is decisive. |
| **B** — fidelity | Is a graph from *AI* masks as trustworthy as one from *doctor* masks? | Volume **r = 0.992**, query ranking **Spearman 0.986** (20 pancreas cases). |
| **C** — self-evolving | Does it get better at catching bad masks as it grows? | Joint (Mahalanobis) **AUROC 0.78** vs marginal 0.66. |
| **D** — autonomous | How well does it segment with *no* labels? | Organs mean **0.49 → 0.61** with KG-repair (§4b). |

---

## 7. 3D reconstruction handoff *(application resources)*

App-ready 3D surfaces of the segmented organs/tumors, built **CPU-only** from data already on disk —
**171 reconstructions**: 12 with a CT underlay, 9 ground-truth-vs-model-prediction pairs, 150 meshed from the
local FLARE23 label store. Each case ships a CT-aligned segmentation NIfTI, a colored **`.glb`** (loads in
`<model-viewer>` / three.js / Unity), per-organ **`.stl`**, and a preview PNG. Headroom: **2,200** full 3D
label volumes are local and **950** have retrievable CT, so the library scales far past 171.

Tools: `meshify.py` (label volume → meshes), `reconstruct_3d.py` (raw CT → segmentation → meshes via the
trained models). Guide: `results/3D_Reconstruction_Resources.docx`. Delivered to the Drive handoff folder.

---

## 8. Roadmap

**In flight** — (1) generic organ model **baseline** *(training)*; (2) **KG-in-training** organ rerun →
the learning-contribution ablation; (3) same KG-in-training term for the tumor model.
**Done** — generic tumor model (≈0.70 + cross-dataset 0.35→0.51); KG-repair ablation (0.49→0.61); OAKG A–D;
3D reconstruction handoff.
**Later** — expand the pool with harder tumor-bearing CTs (robustness is the weakest link); multi-organ
predicted-KG fidelity; like-for-like vs **K-Prism / GF-Screen / PanTS**; paper draft (benchmark + method framings).

**Target venues.** NeurIPS D&B (benchmark) or ICLR/AAAI (OAKG-as-method); MICCAI / health-AI domain fits.

---

## 9. Repository

```
app/            oakg_query_app.py  — interactive KG retrieval / GT-free validation
kg/             schema.owl · ontology_mappings.json · data/ · graph/   (data & graph gitignored)
results/        oakg_structured · kg_fidelity · oakg_evolve · autonomous_organ_sweep · tumor_incremental (JSON + figures)
src/
  notebooks/    OAKG_Experiments · Segmentation_Results · KG_as_Knowledge_Base · Test_KG_from_CT · SWOG_KG_Pipeline_Demo
  scripts/
    tumor model   train_tumor_incremental · eval_tumor_per_dataset · build_tumor_pool · rebuild_lits_pool_patientlevel
    organ model   build_organ_pool_lkp · build_organ_train_priors · train_organ_generic [--kg]
    shared trainer run_pancreas_sam3 (SAM3 partial-freeze + Dice/Focal + kg_consistency_loss)  [+ base: run_flare, run_pancreas_nifti]
    KG            kg_grounding · build_flare23_enriched_kg · kg_build_* · build_kg_atlas · build_kits_kg_records · flare23_predict
    inference     infer_ensemble · kg_guided_segment · kg_guided_eval
    experiments   exp_oakg_structured · exp_kg_fidelity · exp_oakg_evolve · exp_autonomous_organ_sweep
    3D handoff    meshify · reconstruct_3d · build_krishna_3d_pack · build_krishna_3d_guide
    extraction    extract_flare_* · extract_kits_* · build_flare23_image_index
archive/v0/     superseded early work (AUSAM/SAM1, prompt bake-offs, pre-Task2 runners, v1/v2 pools) — kept for provenance
```

## 10. Environment
```
conda env: llmft · Python 3.11 · PyTorch 2.5.1+cu121 · 2× NVIDIA L40S (48 GB)
torch · transformers (SAM3) · monai · rdflib · scikit-image · scipy · nibabel · trimesh
```
