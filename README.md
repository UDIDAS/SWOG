# AUSAM Reproduction

Reimplementation of the AUSAM (Adaptive Unified Segmentation Anything Model) pipeline across multiple abdominal CT segmentation datasets. The original implementation spans multiple Jupyter notebooks per dataset -- this branch consolidates them into a unified pipeline with shared utilities and per-experiment sections.

## Final Results

All Dice scores below are **test-set only** — computed on held-out cases the model never saw during training. All evaluation uses GT-derived prompts (DBSCAN point prompts, and bounding boxes where noted), matching the original AUSAM oracle prompting regime. Deployment without ground truth is an open problem addressed by CRISP-SAM.

### FLARE Dataset

All original notebook targets **beaten** using the same core method with minimal code changes. Slice-level split (data is pre-sliced without patient metadata; same protocol as original AUSAM). No augmentation or box prompts applied.

| Target Structure | Original Dice | Our Test Dice | Method | Verdict |
|-----------------|---------------|----------|--------|---------|
| **Pancreas** (class 4) | 0.822 | **0.839** | H2E curriculum (patience=20) | **BEAT** |
| **Liver** (class 1) | 0.941 | **0.965** | Single-stage fine-tune | **BEAT** |
| **Tumor** (class 14) | 0.839 | **0.855** | Traditional Increments | **BEAT** |
| Duodenum (class 12) | -- | **0.890** | Transfer learning (Liver -> Duodenum) | Done |

### LiTS Dataset (Liver Tumor Segmentation)

131 abdominal CT scans. Case-level split: train 91 / val 13 / test 27 volumes. HU window [-100, 400]. Evaluated on tumor-bearing slices (≥50 tumor pixels).

| Method | Split Type | Test Dice |
|--------|-----------|-----------|
| Single-stage 100% data (no aug/box) | Slice-level | 0.901 |
| **Single-stage + augmentation + box prompts** | **Case-level** | **0.845** |

Best honest result: **Test Dice 0.845** on case-level split. The 0.901 used a slice-level random split where slices from the same patient can appear in both train and test, inflating the number.

### Pancreas CT Dataset (MSD Task07)

281 abdominal CT scans with voxel-level organ and tumor labels. Case-level split: train 196 / val 28 / test 57. HU window [-100, 300].

| Target | Without Aug+Box | **With Aug+Box** | Improvement | Method |
|--------|-----------------|-------------------|-------------|--------|
| **Organ** (pancreas) | 0.741 | **0.834** | +0.093 | H2E curriculum (patience=20) |
| **Tumor** (cancer) | 0.775 | **0.904** | +0.129 | Traditional Increments + transfer |

Best result: **Organ 0.834, Tumor 0.904** on held-out test cases with augmentation + box prompts. The initial run without augmentation had a large train/test gap (classic overfitting on ~3K organ slices). Augmentation closed the gap; box prompts gave SAM the spatial cue it was pretrained with.

### Pancreas Ablation: Augmentation vs Box Prompts

Isolates the individual contributions. All four configs evaluated via `evaluate_test` on 57 held-out test cases using DBSCAN point prompts. Same case-level split as above.

| Config | Augment | Box | Organ Test Dice | Tumor Test Dice |
|--------|:-:|:-:|:-:|:-:|
| Aug only | Yes | No | 0.689 | 0.789 |
| **Box only** | No | Yes | **0.828** | **0.914** |

For context, the main Pancreas table reports organ 0.741→0.834 and tumor 0.775→0.904, measured from the NIfTI delivery manifest (centroid prompts, volume-level Dice). The ablation uses DBSCAN slice-level Dice, so the absolute numbers differ but the relative ranking is consistent.

Box prompts are the dominant factor. Augmentation alone did not help organ — the H2E curriculum only reached 55% data utilization with augmented samples, suggesting the added noise disrupted entropy-based ordering. For tumors, augmentation alone (0.789) trails box-only (0.914) by a wide margin, confirming that the bounding box provides a spatial prior that a single DBSCAN centroid cannot match for small, irregular structures.

### SAM3 Comparison (Pancreas CT)

SAM3 (facebook/sam3, 840M params) tested as a potential backbone for CRISP-SAM. Unlike SAM1 which uses DBSCAN point + box prompts, SAM3 uses text + box prompts via a DETR-based architecture. All configs use GT-derived box prompts on the same case-level split as above.

| Config | Encoder Strategy | Params Trained | Organ Dice | Tumor Dice |
|--------|-----------------|:--------------:|:----------:|:----------:|
| SAM1 box+points (reference) | Full finetune (93.7M) | 100% | 0.828 | **0.914** |
| SAM3 full finetune | All 840M trainable | 100% | 0.861 | 0.888 |
| SAM3 frozen encoder | Encoder frozen, decoder only | 46% | 0.826 | 0.853 |
| **SAM3 partial freeze** | Blocks 0-19 frozen; 20-31 + FPN + decoder trainable | 67% | **0.866** | *in progress* |

SAM3 full finetune beats SAM1 on organ (+0.033) but loses on tumor (-0.026). Freezing the encoder entirely eliminates overfitting but underperforms both — SAM3's ViT-Large was pretrained on natural images and cannot learn CT-specific features without fine-tuning.

The **partial freeze** config is the best SAM3 strategy: it reaches organ Dice **0.866** (best of any SAM3 variant, beating SAM1 by +0.038) while keeping a healthy train-val gap (~0.03 vs full-finetune's 0.064 overfit). It freezes the low-level transformer blocks and patch/position embeddings, then fine-tunes the high-level blocks + FPN neck + decoder with **discriminative learning rates** (encoder 1e-5, decoder 1e-4), a **warmup + cosine-annealing** schedule, and a combined **Dice+Focal loss**. Nearly all validation gains arrived as the cosine schedule dropped the LR below 50%, confirming the schedule — not just the freeze — drives convergence.

Key findings for CRISP-SAM:
- **SAM3 wins on organ, SAM1 wins on tumor.** SAM3's larger capacity helps organ segmentation but is a liability on small structures (~700 tumor slices), where the lightweight SAM1 generalizes better.
- **A frozen encoder does not work for CT** — the natural-image encoder must be partially adapted.
- The partial-freeze SAM3 backbone is the recommended choice for organ-level work, since it provides a stable, CT-adapted encoder for the downstream prompt generator to build on.

## 3D NIfTI Deliverables

Per-case `ct.nii.gz` / `gt.nii.gz` / `pred.nii.gz` for 3D reconstruction. Dice below is averaged across **all** cases (train + val + test), since deliverables are generated for every patient. Test-only numbers are reported in the results tables above.

| Dataset | Cases | Organ Dice (all) | Tumor Dice (all) | Location |
|---------|-------|------------|------------|----------|
| **Pancreas** (MSD Task07) | 281 | 0.846 | 0.917 | `/scratch/.../Pancreas/delivery_v2/` |
| **LiTS** (Liver Tumor) | 131 | 0.996 (liver from GT) | 0.675 | `/scratch/.../LiTS_delivery/delivery/` |
| **FLARE** | — | — | — | Blocked (source data is pre-sliced, no per-patient volumes) |

## Best Approach Across Datasets

The recipe that consistently produced the best results:

1. **Box prompts** — bounding box from GT mask (padded ±3px) alongside DBSCAN point prompts. The single largest contributor in ablation (box-only organ 0.828 vs aug-only 0.689). SAM was pretrained with both prompt types; using only points leaves half the prompt architecture untrained
2. **Data augmentation** — random horizontal flip + brightness jitter ±15 (uint8 RGB). Helps generalization when combined with box prompts, though ablation shows box prompts alone nearly match the combined result
3. **Transfer learning** for tumors — initialize from organ weights, since organ features provide useful low-level representations
4. **Case-level data splits** — hold out entire patients, not random slices. Slice-level splits leak patient features and inflate reported Dice (LiTS: 0.901 slice-level vs 0.845 case-level)
5. **Single-stage training** with all data from epoch 1 — beats entropy curriculum on tumors, matches or beats on organs

## How We Beat the Original Results

Our reproduction uses the **same AUSAM method** with only necessary code fixes -- no architectural changes, no new loss functions, no additional data. The improvements came from three insights discovered during reproduction:

**1. Curriculum data utilization was the bottleneck, not the method itself**

The original notebooks use `early_stopping_patience=10` and `data_increment_patience=5`, meaning the curriculum gets exactly one data expansion before early stopping (expand at epoch 5 of no improvement, stop at epoch 10). With 2 GPUs instead of 4, each expansion was less impactful. By adjusting to `patience=20, increment=3`, the curriculum could expand 6+ times before stopping, reaching 100% data utilization. This alone pushed FLARE Pancreas from 0.792 to **0.839** (beating the 0.822 target).

**2. Random sampling beats entropy-based ordering for tumors**

The original Tumor notebook achieved its best result (0.839) using "Traditional With Increments" -- fixed percentage steps with random sampling -- not the entropy-based E2H/H2E curriculum. We confirmed this: entropy measures image complexity (CT anatomy), not tumor difficulty. A high-entropy slice may have complex anatomy but a simple tumor. Random sampling at each percentage step preserves morphological diversity, which is critical for tumors that vary wildly in size (22 to 6,917 pixels), shape, and location.

**3. DDP synchronization prevents silent training degradation**

The original notebooks don't synchronize the "improved" decision across GPU ranks. With `mp.spawn`, floating-point rounding causes ranks to occasionally disagree -- one rank saves a checkpoint while the other continues training, or one triggers data expansion while the other doesn't. This doesn't crash immediately but causes subtle training instability. Broadcasting the decision from rank 0 ensures all ranks take identical paths, producing cleaner training curves and better final models.

## Implementation Notes

Changes from the original notebooks, with justification:

**1. SAM API update (required compatibility fix)** -- `point_annotations` removed in transformers 5.8.1, updated to `input_points` 4D tensor API.

**2. DDP synchronization fix (required stability fix)** -- Broadcast `improved` flag from rank 0 to prevent NCCL crashes from rank divergence.

**3. Label handling (equivalent, not different)** -- Original `bitwise_not` + `invert_black_white` was an identity on 0/255 data. Data is now 0/1, so we use `(labels > 0)` directly (same net effect).

**4. Configurable curriculum patience** -- Original hardcodes patience=10. Made configurable to compensate for 2 GPUs vs 4.

**5. Traditional Increments no_improve reset** -- Reset no-improvement counter when stepping to the next percentage level, allowing the model to fully benefit from each data expansion step.

**6. Data augmentation** -- Random horizontal flip (image + label + point coords) and brightness jitter ±15 on uint8 RGB. Eliminates the 18-point train/val Dice gap observed in Pancreas organ training.

**7. Box prompts** -- Bounding box from GT mask (padded ±3px, scaled to SAM's 1024x1024 input) passed alongside DBSCAN point prompts. SAM was pre-trained with box prompts; using them during fine-tuning provides a stronger spatial cue and faster convergence.

## Original Notebooks

### FLARE (5 notebooks in `FLARE.zip`)

| Notebook | What it does | Data | Best Result |
|----------|-------------|------|-------------|
| `SAM-DBSCAN_FLARE-paper1.ipynb` | E2H/H2E entropy curriculum + augmentations | Class 4 (Pancreas) | H2E Dice 0.822 |
| `SAM-DBSCAN_FLARE-paper1-aggregation.ipynb` | Transfer learning: Liver -> Duodenum | Class 1 -> 12 | *(not saved)* |
| `SAM-DBSCAN_FLARE-paper1-sam2.ipynb` | SAM2 Hiera architecture comparison | Class 1 (Liver) | Dice 0.941 |
| `SAM-DBSCAN_FLARE-Tumor.ipynb` | Multiple strategies for tumor segmentation | Class 14 (Tumor) | Traditional Dice 0.839 |
| `SAM-DBSCAN_FLARE-Multi-Class0-3d.ipynb` | 3D volume-level evaluation | Class 0 (all) | Train Dice 0.975 |

### LiTS (2 notebooks in `LiTS.zip`)

| Notebook | What it does | Data | Best Result |
|----------|-------------|------|-------------|
| `SAM-DBSCAN_LiTS_Tumor.ipynb` | Multiple strategies: entropy curriculum, single-stage, random increments, full training | Tumor | RPSF Dice 0.927 |
| `SAM-DBSCAN_LiTS_Tumor_TF_125pixels.ipynb` | Transfer learning variant with 125px minimum tumor size | Tumor | -- |

## Repository Structure

```
results/
  flare/                           # FLARE segmentation results
    viz/                           # Best/worst prediction images + training curves
    metrics/                       # Per-epoch CSV metrics
  lits/                            # LiTS segmentation results
  pancreas/                        # Pancreas CT results (organ + tumor metrics, delivery manifest)

src/scripts/
  run_flare.py                     # All shared code + DDP training functions (aug + box prompts)
  run_lits.py                      # LiTS reproduction runner (slice-level split)
  run_lits_v3.py                   # LiTS with aug+box (case-level split)
  run_pancreas_nifti.py            # Pancreas CT train + NIfTI delivery pipeline
  run_pancreas_ablation.py         # Pancreas ablation: aug vs box prompt contribution
  run_pancreas_sam3.py             # SAM3 vs SAM1 comparison (full/frozen/partial freeze)
  run_lits_nifti.py                # LiTS train + NIfTI delivery pipeline
  run_remaining.py                 # FLARE Round 1 runner
  run_round2.py                    # FLARE Round 2 runner
  run_tumor_fix.py                 # Standalone Tumor Traditional rerun
  flare_sam_finetune.py            # Initial standalone fine-tuning script
```

## How to Run

```bash
# FLARE
nohup python -u src/scripts/run_round2.py > runs/log 2>&1 &

# LiTS
nohup python -u src/scripts/run_lits_v3.py > runs/log 2>&1 &

# Pancreas (train + NIfTI deliverables)
cd src/scripts && nohup python -u run_pancreas_nifti.py > runs/log 2>&1 &
```

## Environment

```
conda env: llmft
Python 3.11, PyTorch 2.5.1+cu121
GPU: 2x NVIDIA L40S (48GB each)
Dependencies: torch, transformers, monai, scikit-learn, scikit-image, scipy
```
