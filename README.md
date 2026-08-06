# MMKG — AUSAM Abdominal-CT Segmentation → Ontology-Grounded Knowledge Graph

A **Multi-Modal Knowledge Graph (MMKG)** built on autonomous abdominal-CT segmentation: segment liver / kidney /
pancreas and their tumors, turn the masks into **ontology-grounded phenotypes**, and use the graph to **validate**
results without labels and **retrieve** similar patients. This README describes the **whole pipeline** and reports
the current results — **AUSAM segmentation is complete; the knowledge-graph stage is next.**

---

## 1. The pipeline

Four stages take a CT to a queryable, ontology-grounded graph entry:

```
CT (test patient)
  1. Segment    AUSAM: per-dataset SAM3 + text prompt + GT box → organ & tumor masks   [DONE — §4]
  2. Phenotype  per-patient 3-D mask → volume, max diameter, centroid, tumor burden,   [next stage]
                lesion count                              (kg_extract_phenotypes.py)
  3. Represent  phenotypes → ontology-grounded KG node   (SNOMED / NCIt, kg_grounding) [machinery exists]
  4. Reason     the KG at inference:
                 · Validate — flag implausible phenotypes vs the cohort, no labels
                 · Retrieve — find the most similar known patients   (OAKG, γ-weighted)
```

**How we test (this stage — the sidetrack).** Everything is scored on **held-out test patients**, in-distribution
per dataset (a test patient's dataset is known, so its per-dataset model is used). Segmentation is scored two ways:
per-slice **2-D Dice** (§4) and, next, **3-D whole-volume Dice** — run the 2-D model over *every* slice of a test
volume, stack the outputs into a 3-D mask, and score the assembled shape (this also catches false positives on
organ-*absent* slices, which the 2-D score can't). The KG stage is then tested on AUSAM's **predicted** phenotypes:
**fidelity** (predicted-KG vs GT-KG), **GT-free validation** (AUROC of flagged vs actually-low-Dice), and
**retrieval** quality.

**How the KG is used at inference.** The graph is not a passive store. At inference it (a) **validates** a new
patient's phenotype against the cohort with *no ground truth* — a 4,900 cc "liver" is flagged as a likely
segmentation error — and (b) **retrieves** the most similar known patients for context.

**OAKG — many datasets / many graphs under one roof.** Each dataset labels *different* structures (LiTS→liver,
KiTS→kidney, MSD→pancreas, FLARE→multi-organ), so a merged graph has **partial observability**: most patients are
missing most organs **by construction, not by absence**. Naïve similarity then invents spurious matches. **OAKG**
(observability-aware KG) fixes this — it **never imputes** a missing structure and weights patient-to-patient
similarity by their **joint observability** (γ), so a liver-only LiTS patient and a kidney-only KiTS patient aren't
called "similar" merely because both are missing everything else. This is what lets the per-dataset graphs coexist
as **one merged knowledge graph** and be queried together.

**Status.** Stage 1 (AUSAM segmentation) is complete (§2–§5). Stages 2–4 reuse existing OAKG machinery
(`kg_extract_phenotypes`, `kg_grounding`, `kg_retrieval_v2`, the validator/atlas, the GT knowledge graph) and are
the **next stage**, run on AUSAM's test-set outputs.

---

## 2. Training datasets

Five public abdominal-CT datasets provide the **image + label pairs** used to train the segmenters. Each
contributes the organs (and, where annotated, the tumors) it labels:

| Dataset | Images | Structures | Organ slices | Tumor slices |
|---|:--:|---|--:|--:|
| **LiTS** | ✅ | liver (+ liver tumor) | 5,000 | 5,600 |
| **MSD Pancreas** (Task07) | ✅ | pancreas (+ pancreas tumor) | 5,000 | 2,537 |
| **KiTS23** | ✅ | kidney (+ kidney tumor) | 3,821 | 5,267 |
| **FLARE-Task2** | ✅ | liver / kidney / pancreas | 13,929 | — |
| **FLARE23** | ✅ | liver / kidney / pancreas (+ tumor) | 10,759 | 7,269 |

Slice counts are the extracted, class-balanced training slices per dataset (organ pool and tumor pool
respectively). FLARE-Task2 carries no tumor labels.

---

## 3. Training paradigm — AUSAM (per-dataset, GT-box)

**AUSAM** is a SAM3 segmenter fine-tuned **per dataset** and prompted, at inference, with the target's
**text** (organ name / `"tumor"`) **plus a ground-truth-derived bounding box** (the box of the target mask). It is
a **semi-oracle, prompt-guided** segmenter — faithful to the original AUSAM's mask-derived prompting.

- **One model per dataset — 5 checkpoints total.** Each dataset gets its own SAM3 checkpoint. LiTS, KiTS23 and
  MSD are single-organ (liver / kidney / pancreas), so their model segments that one organ; FLARE-Task2 and
  FLARE23 are multi-organ, so a **single model segments all three** (liver / kidney / pancreas), selected at
  inference by the text prompt. Each is evaluated in-distribution.
- **Backbone & fine-tuning.** SAM3 with **partial freeze** (backbone frozen; last ~20 encoder blocks + mask
  decoder trained), **Dice + Focal** loss (0.7 / 0.3), discriminative learning rates, cosine schedule, DDP on
  2 GPUs, ≤12 epochs with early stopping.
- **Prompt.** Text + GT-derived box (`bbox_from_mask`, 3-px pad); SAM3 returns mask proposals and the
  highest-confidence proposal is taken.
- **Split.** Every dataset is split **by whole patient** (seed 42), ≈ **70 / 10 / 20** train / val / test —
  validation and test are held-out *patients*, never other slices from a training patient (no leakage).
- **Metric.** Per-slice **2-D Dice**, averaged over the held-out test patients' target-present slices.

*What these numbers are and aren't.* Because the box is derived from ground truth, they are the **interactive
upper bound** (semi-oracle); and being slice-level, they read optimistically relative to a 3-D whole-volume score.

### Train / val / test split per dataset (by patient)

<!-- SPLITS:START -->
**Organ models** (`organ_pool_lkp`):

| Dataset | Total patients | Train (pt / slices) | Val (pt / slices) | Test (pt / slices) |
|---|:--:|:--:|:--:|:--:|
| **LiTS** | 34 | 25 / 3,883 | 3 / 409 | 6 / 708 |
| **KiTS23** | 100 | 70 / 2,789 | 10 / 377 | 20 / 655 |
| **MSD Pancreas** | 165 | 116 / 3,530 | 16 / 462 | 33 / 1,008 |
| **FLARE-Task2** | 100 | 70 / 10,417 | 10 / 1,166 | 20 / 2,346 |
| **FLARE23** | 449 | 316 / 7,568 | 44 / 1,055 | 89 / 2,136 |

**Tumor models** (`tumor_pool` + `flare_tumor_pool` + `kits_tumor_pool`):

| Dataset | Total patients | Train (pt / slices) | Val (pt / slices) | Test (pt / slices) |
|---|:--:|:--:|:--:|:--:|
| **LiTS** | 107 | 76 / 3,825 | 10 / 539 | 21 / 1,236 |
| **MSD Pancreas** | 281 | 197 / 1,811 | 28 / 261 | 56 / 465 |
| **KiTS23** | 180 | 126 / 3,589 | 18 / 492 | 36 / 1,186 |
| **FLARE23** | 270 | 189 / 5,041 | 27 / 615 | 54 / 1,613 |

Split **by whole patient** (seed 42), 20% test / 10% val; val and test are held-out patients (no slice-level leakage).
<!-- SPLITS:END -->

---

## 4. Results — AUSAM per-dataset test Dice

Held-out **patient-level** test Dice, per dataset (auto-generated from
`results/{organ_generic_ausam_*,tumor_ausam_*}.json` by `build_readme_ausam_tables.py`):

<!-- AUSAM_BASELINE:START -->
**Organ AUSAM** — held-out per-dataset test Dice (GT-box, per dataset):

| Organ | FLARE23 | FLARE-Task2 | LiTS | KiTS | MSD | **mean** |
|---|:--:|:--:|:--:|:--:|:--:|:--:|
| **Liver** | 0.938 | 0.970 | 0.957 | — | — | **0.955** |
| **Kidney** | 0.914 | 0.951 | — | 0.927 | — | **0.930** |
| **Pancreas** | 0.856 | 0.904 | — | — | 0.871 | **0.877** |

**Tumor AUSAM** — held-out per-dataset test Dice (GT-box, per dataset):

| LiTS | MSD Pancreas | KiTS | FLARE23 | **mean** |
|:--:|:--:|:--:|:--:|:--:|
| 0.819 | 0.882 | 0.925 | … | **0.875** |

_9 organ (dataset,organ) cells from 5/5 datasets · 3/4 tumor datasets. '…' = still training, '—' = organ not in that dataset._
<!-- AUSAM_BASELINE:END -->

---

## 5. The tumor model

The tumor model follows the **same AUSAM paradigm** as the organ models: one SAM3 checkpoint **per dataset**,
fine-tuned on that dataset's tumor slices, prompted at inference with `"tumor"` + a GT-derived box, same
patient-level split and per-slice Dice metric. Four models — **LiTS** (liver tumor), **MSD Pancreas**,
**KiTS23** (kidney tumor), **FLARE23** (pan-cancer) — on the tumor slice counts in [§2](#2-training-datasets).

Tumors are smaller, sparser, and more variable than organs, so they are the harder target; they are trained
**separately per dataset** for the same reason the organs are. Their numbers fill into the table in
[§4](#4-results--ausam-per-dataset-test-dice) as each run completes (`results/tumor_ausam_<ds>.json`).

---

## 6. Reproduce

```
env:           conda llmft · Python 3.11 · PyTorch 2.5.1+cu121 · 2× NVIDIA L40S · transformers (SAM3)
organ AUSAM:   python src/scripts/train_organ_generic.py --ausam --dataset <lits|kits|msd|flare_task2|flare23>
tumor AUSAM:   python src/scripts/train_tumor_ausam.py --dataset <lits|pancreas|kits|flare>
results table: python src/scripts/build_readme_ausam_tables.py     # regenerate §4 from results/*.json
split tables:  python src/scripts/build_readme_splits.py           # regenerate §3 splits from pool metas
```
