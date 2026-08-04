# SWOG — Autonomous Abdominal-CT Segmentation → Self-Evolving Imaging Knowledge Graph

**Aim.** Take a **new abdominal CT with no annotations**, automatically segment **liver, kidney, pancreas
and their tumors**, turn those masks into an **ontology-grounded knowledge graph**, and use that graph to
validate the result *without ground truth*, retrieve similar patients, and get smarter with every new case.

This README describes the **current approach and status**. It intentionally carries **no result tables** —
every number lives in [`results/`](results/) (JSON + figures) and the walkthrough notebooks in
[`src/notebooks/`](src/notebooks/).

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
  with correct masks. Training sources: **LiTS, MSD Pancreas, KiTS23, FLARE-Task2**.
- **Labels alone → build the graph.** The KG needs only *numbers* derived from a mask (volume, diameter,
  burden), not pixels. **FLARE23's 1,312 label masks** (no images) populate the KG cheaply.

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

---

## 3. Datasets & splits

| Dataset | Images? | Role | Structures |
|---|:--:|---|---|
| **LiTS** | ✅ | train tumor + organ (liver) | liver, liver tumor |
| **MSD Pancreas** (Task07) | ✅ | train tumor + organ (pancreas) | pancreas, pancreas tumor |
| **KiTS23** | ✅ | train tumor + organ (kidney) | kidney, kidney tumor |
| **FLARE-Task2** (100 vols) | ✅ | train organ (l/k/p) | 13 organs |
| **FLARE23** (1,312 masks) | labels only | build the KG; organ/tumor slices for training | 13 organs + tumor |

**Splitting.** Every training pool is split **by whole patient** (seed 42): validation and test are held-out
*patients*, never other slices from a training patient. No slice-level leakage. Cross-dataset evaluation
trains on some datasets and tests on an entirely unseen one.

---

## 4. The knowledge graph

**Structure.** Direct (non-reified) triples per case: an `ImagingCase` `depicts_organ` each `Organ`
(`gt_volume_cm3`, `gt_max_diameter_mm`, `gt_centroid_mm`, `has_lesion`, `mapped_to_concept`), and each
`Lesion` (`is_tumor`, volume, diameter, `located_in`, `tumorBurden`, `lesion_count`). Organs and lesions are
**grounded to ontologies** (SNOMED / LOINC / ICD / MeSH) via `kg_grounding.py`.

**Coverage.** FLARE23 (1,312 labels) + Pancreas + LiTS + KiTS records → a pooled corpus and an **anatomical
atlas** (`build_kg_atlas.py`): per-organ plausible size / diameter / location priors.

**The KG closes the loop in two places — this is the core contribution:**
- **At training** — from the *training patients only* we derive a leakage-free atlas (each organ's typical
  location + plausible size band) and add a differentiable **plausibility / consistency loss**
  (`kg_consistency_loss` in `run_pancreas_sam3.py`). Each model can be trained **baseline vs KG-in-training**;
  the difference isolates the KG's contribution to *learning* (`train_organ_generic.py --kg`).
- **At inference** — the same anatomical knowledge **repairs** masks (`kg_guided_segment.py`: keep the
  plausible connected component, drop spurious blobs), **validates** phenotypes against the cohort with no
  labels, and drives **retrieval** and **growth**.

---

## 5. OAKG — the research contribution

Retrieval and validation on a **merged, partially-observed** graph (different datasets label different
organs). OAKG is **evidence-calibrated**: it never imputes missing structures and weights similarity by
**joint observability** (γ, a Jaccard term). The experiment suite (scripts `src/scripts/exp_*.py`, numbers in
`results/*.json`, walkthrough in `src/notebooks/OAKG_Experiments.ipynb`):

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

**Now training** — the generic **organ model**: (1) baseline (`0.7·Dice + 0.3·Focal`); next (2) the
**KG-in-training** variant (`+ 0.1·KG_consistency`) → the learning-contribution ablation; then (3) the same
plausibility term for the tumor model.

**Established** — generic tumor model (patient-level + cross-dataset generalization); inference-time
KG-repair ablation; OAKG experiments A–D. *(Numbers: `results/`.)*

**Next** — expand the pool with harder tumor-bearing CTs (segmentation robustness is the weakest link);
multi-organ predicted-KG fidelity; like-for-like comparison vs **K-Prism / GF-Screen / PanTS**; paper draft
around the OAKG contribution (benchmark and method framings).

**Target venues.** NeurIPS Datasets & Benchmarks (benchmark framing) or ICLR/AAAI (OAKG-as-method); MICCAI /
health-AI as domain fits.

---

## 7. Repository

```
app/            oakg_query_app.py — interactive KG retrieval / GT-free validation
kg/             schema.owl · ontology_mappings.json · data/ · graph/        (data & graph gitignored)
results/        oakg_structured · kg_fidelity · oakg_evolve · autonomous_organ_sweep · tumor_incremental  (JSON + figures)
src/
  notebooks/    OAKG_Experiments · Segmentation_Results · KG_as_Knowledge_Base · Test_KG_from_CT · SWOG_KG_Pipeline_Demo
  scripts/
    tumor model    train_tumor_incremental · eval_tumor_per_dataset · build_tumor_pool · rebuild_lits_pool_patientlevel
    organ model    build_organ_pool_lkp · build_organ_train_priors · train_organ_generic [--kg]
    shared trainer run_pancreas_sam3 (SAM3 partial-freeze + Dice/Focal + kg_consistency_loss) [+ base: run_flare, run_pancreas_nifti]
    KG             kg_grounding · build_flare23_enriched_kg · kg_build_* · build_kg_atlas · build_kits_kg_records · flare23_predict
    inference      infer_ensemble · kg_guided_segment · kg_guided_eval
    experiments    exp_oakg_structured · exp_kg_fidelity · exp_oakg_evolve · exp_autonomous_organ_sweep
```

## 8. Environment
```
conda env: llmft · Python 3.11 · PyTorch 2.5.1+cu121 · 2× NVIDIA L40S (48 GB)
torch · transformers (SAM3) · monai · rdflib · scikit-image · scipy · nibabel
```
