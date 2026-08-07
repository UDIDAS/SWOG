# MMKG — AUSAM Abdominal-CT Segmentation → Ontology-Grounded Knowledge Graph

A **Multi-Modal Knowledge Graph (MMKG)** built on autonomous abdominal-CT segmentation: segment liver / kidney /
pancreas and their tumors, turn the masks into **ontology-grounded phenotypes**, and use the graph to **validate**
results without labels and **retrieve** similar patients. **The contribution is the *pipeline*** — multi-source
segmentation → phenotypes → a KG that integrates datasets labeling *different* structures → observability-aware
retrieval. Segmentation is a competent **component**, not a segmentation-SOTA claim. All evaluation is on held-out
test sets under a consistent **semi-oracle (GT-box)** setting, so every comparison here is like-for-like.

---

## 1. The pipeline

Four stages take a CT to a queryable, ontology-grounded graph entry:

```
CT (test patient)
  1. Segment    AUSAM: per-dataset SAM3 + text prompt + GT box → organ & tumor masks   [DONE — §4, §5]
  2. Phenotype  per-patient 3-D mask → volume, tumor burden, multiplicity, location    [DONE — §6]
                                                          (kg_extract_phenotypes.py)
  3. Represent  phenotypes → ontology-grounded KG node   (SNOMED / NCIt, kg_grounding) [DONE — §6]
  4. Reason     the KG at inference:
                 · Validate — flag implausible phenotypes vs the cohort, no labels      [DONE — null on AUSAM]
                 · Retrieve — find the most similar known patients   (OAKG, γ-weighted) [completing — §6]
```

**How we test (this stage — the sidetrack).** Everything is scored on **held-out test patients**, in-distribution
per dataset (a test patient's dataset is known, so its per-dataset model is used). Segmentation is scored two ways:
per-slice **2-D Dice** (§4) and **3-D whole-volume** DSC / NSD / HD95 (§5) — the 2-D model run over *every* slice
of a test volume, stacked into a 3-D mask. The KG stage is tested on AUSAM's **predicted** phenotypes (§6):
**fidelity** (predicted-KG vs GT-KG), **GT-free validation**, and **retrieval** quality.

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

**Status.** Segmentation is complete — 2-D per-dataset (§4) **and** patient-level 3-D / DSC·NSD·HD95 (§5). The KG
stage runs on AUSAM's predicted outputs (§6): **node fidelity done** (a KG from predicted masks ≈ one from GT),
**GT-free validation done** (a null on AUSAM — it is too accurate to flag, which is itself the finding), and
**observability-aware retrieval on predicted phenotypes** — the headline experiment — completing.

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
| 0.819 | 0.882 | 0.925 | 0.803 | **0.857** |

_9 organ (dataset,organ) cells from 5/5 datasets · 4/4 tumor datasets. '…' = still training, '—' = organ not in that dataset._
<!-- AUSAM_BASELINE:END -->

---

## 5. Patient-level 3-D (whole-volume)

The §4 scores are per-slice on organ-present slices. The **patient-level** number runs the same 2-D model over
*every* slice of a test volume and scores the assembled 3-D mask — reported three ways (challenge-standard):
**DSC** (volume overlap), **NSD@2 mm** (surface agreement), **HD95** (95th-percentile boundary distance, mm).
Same semi-oracle setting, so these are the whole-volume *interactive ceiling* — not an autonomous number.

<!-- RESULTS3D:START -->
**Patient-level 3-D (whole-volume, semi-oracle setting)** — DSC = volume overlap, NSD@2mm = surface agreement, HD95 = 95th-percentile boundary distance (mm):

| Dataset | Organ | DSC | NSD@2mm | HD95 (mm) | n |
|---|---|:--:|:--:|:--:|:--:|
| FLARE-Task2 | liver | 0.981 | 0.947 | 3.82 | 20 |
| FLARE-Task2 | kidney | 0.964 | 0.946 | 2.97 | 20 |
| FLARE-Task2 | pancreas | 0.916 | 0.919 | 3.08 | 20 |
| FLARE23 | liver | 0.971 | 0.925 | 4.52 | 10 |
| FLARE23 | kidney | 0.954 | 0.908 | 5.22 | 10 |
| FLARE23 | pancreas | 0.884 | 0.869 | 4.74 | 10 |
| KiTS23 | kidney | 0.938 | 0.907 | 5.61 | 20 |
| MSD | pancreas | 0.881 | 0.848 | 4.49 | 33 |
| LiTS | liver | 0.956 | — | — | 6 |

_LiTS is DSC-only — its `.npy` volumes did not retain mm spacing, which NSD/HD95 require._
<!-- RESULTS3D:END -->

### 5.1 Cross-dataset organ generalization

Each per-dataset AUSAM organ model applied to *every* dataset's test split — the transfer view that motivates the
KG's multi-source integration. On-diagonal is in-distribution; off-diagonal is zero-shot transfer to a dataset the
model never trained on.

<!-- CROSSDATASET:START -->
**Cross-dataset organ generalization** — rows = training dataset, columns = test dataset, slice-level Dice (GT box). **Bold** = in-distribution (diagonal); off-diagonal = zero-shot transfer.

_liver_
| model \ test | FLARE23 | FLARE-Task2 | LiTS |
|---|:--:|:--:|:--:|
| FLARE23 | **0.939** | 0.964 | 0.941 |
| FLARE-Task2 | 0.935 | **0.970** | 0.939 |
| LiTS | 0.925 | 0.955 | **0.957** |

_kidney_
| model \ test | FLARE23 | FLARE-Task2 | KiTS23 |
|---|:--:|:--:|:--:|
| FLARE23 | **0.922** | 0.948 | 0.849 |
| FLARE-Task2 | 0.894 | **0.955** | 0.802 |
| KiTS23 | 0.858 | 0.899 | **0.911** |

_pancreas_
| model \ test | FLARE23 | FLARE-Task2 | MSD |
|---|:--:|:--:|:--:|
| FLARE23 | **0.870** | 0.890 | 0.835 |
| FLARE-Task2 | 0.852 | **0.913** | 0.841 |
| MSD | 0.854 | 0.879 | **0.877** |

Mean in-distribution Dice **0.924** vs zero-shot transfer **0.892** (Δ +0.032). Liver transfers cleanly (all ≥0.92); the widest gap is transfer *into* KiTS (FLARE→KiTS kidney 0.80–0.85), reflecting KiTS's tumor-distorted kidneys and a harder test split — a data-distribution effect that motivates integrating datasets at the KG level rather than expecting one model to cover all.
<!-- CROSSDATASET:END -->

---

## 6. Knowledge-graph stage (the contribution)

The KG stage runs on AUSAM's **predicted** phenotypes (organ + tumor volumes → burden / multiplicity / containment
/ location, derived exactly as the GT corpus derives them). Two questions:

- **Node fidelity** — is a KG built from *predicted* masks as trustworthy as one from GT? Measured as
  predicted-vs-GT organ-volume agreement.
- **OAKG retrieval on predicted phenotypes** — rank patients by *predicted*-phenotype similarity, score relevance
  against *GT* tumor features. The **γ (observability) ablation** shows that weighting by shared observed organs
  suppresses spurious cross-dataset matches — the multi-source-integration claim, now on the pipeline's own output.

GT-free validation is wired but returns a **null on AUSAM** — the semi-oracle segmenter is too accurate to produce
implausible phenotypes, so there is nothing to flag (a finding: the validator's real test is a noisier autonomous arm).

<!-- KG:START -->
**Node fidelity** — is a KG built from *predicted* masks as trustworthy as one from GT? Predicted-vs-GT organ-volume agreement:

| organ / dataset | volume corr | MAPE |
|---|:--:|:--:|
| kidney/flare23 | 0.9938 | 4.1% |
| kidney/flare_task2 | 0.9992 | 1.7% |
| kidney/kits | 0.9965 | 1.9% |
| liver/flare23 | 0.99 | 2.1% |
| liver/flare_task2 | 0.997 | 0.9% |
| liver/lits | 0.9967 | 3.6% |
| pancreas/flare23 | 0.9671 | 6.6% |
| pancreas/flare_task2 | 0.9943 | 4.4% |
| pancreas/msd | 0.981 | 10.5% |

**OAKG retrieval on predicted phenotypes** — ranked by predicted-phenotype similarity, relevance scored against GT tumor features:

| Setting | P@5 | P@10 | mAP | nDCG | spurious@10 |
|---|:--:|:--:|:--:|:--:|:--:|
| GT phenotypes gamma (upper bound) | 1.0 | 0.996 | 0.992 | 0.999 | 0.0 |
| PREDICTED phenotypes gamma | 1.0 | 0.992 | 0.972 | 0.996 | 0.0 |
| PREDICTED no gamma (coverage blind) | 0.681 | 0.673 | 0.714 | 0.676 | 0.32 |
| PREDICTED base (organ-agnostic) | 0.598 | 0.535 | 0.524 | 0.558 | 0.465 |
<!-- KG:END -->

---

## 7. The tumor model

The tumor model follows the **same AUSAM paradigm** as the organ models: one SAM3 checkpoint **per dataset**,
fine-tuned on that dataset's tumor slices, prompted at inference with `"tumor"` + a GT-derived box, same
patient-level split and per-slice Dice metric. Four models — **LiTS** (liver tumor), **MSD Pancreas**,
**KiTS23** (kidney tumor), **FLARE23** (pan-cancer) — on the tumor slice counts in [§2](#2-training-datasets).

Tumors are smaller, sparser, and more variable than organs, so they are the harder target; they are trained
**separately per dataset** for the same reason the organs are. Their numbers fill into the table in
[§4](#4-results--ausam-per-dataset-test-dice) as each run completes (`results/tumor_ausam_<ds>.json`).

---

## 8. Reproduce

```
env:              conda llmft · Python 3.11 · PyTorch 2.5.1+cu121 · 2× NVIDIA L40S · transformers (SAM3)
organ AUSAM:      python src/scripts/train_organ_generic.py --ausam --dataset <lits|kits|msd|flare_task2|flare23>
tumor AUSAM:      python src/scripts/train_tumor_ausam.py --dataset <lits|pancreas|kits|flare>
3-D DSC/NSD/HD95: python src/scripts/eval_ausam_3d.py --dataset <ds>
predicted corpus: python src/scripts/build_predicted_corpus.py --dataset <msd|lits|kits> [--gt]
OAKG retrieval:   python src/scripts/retrieval_on_predicted.py
README tables:    python src/scripts/build_readme_{ausam_tables,splits,results}.py   # §4 / §3 / §5–6
```

### Walkthrough notebooks
`src/notebooks/Patient_Level_3D_Eval.ipynb` (how the 3-D DSC/NSD are obtained) ·
`src/notebooks/Two_Model_Paradigm.ipynb` (why tumors are pooled — single-dataset vs pooled on real cases).
