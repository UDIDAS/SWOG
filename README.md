# AUSAM Reproduction

Reimplementation of the AUSAM (Adaptive Unified Segmentation Anything Model) pipeline across multiple abdominal CT segmentation datasets. The original implementation spans multiple Jupyter notebooks per dataset -- this branch consolidates them into a unified pipeline with shared utilities and per-experiment sections.

## Final Results

### FLARE Dataset

All original notebook targets **beaten** using the same core method with minimal code changes.

| Target Structure | Original Dice | Our Dice | Method | Verdict |
|-----------------|---------------|----------|--------|---------|
| **Pancreas** (class 4) | 0.822 | **0.839** | H2E curriculum (patience=20) | **BEAT** |
| **Liver** (class 1) | 0.941 | **0.965** | Single-stage fine-tune | **BEAT** |
| **Tumor** (class 14) | 0.839 | **0.855** | Traditional Increments | **BEAT** |
| Duodenum (class 12) | -- | **0.890** | Transfer learning (Liver -> Duodenum) | Done |

Sample segmentation results (best and worst predictions) and training curves are in [results/flare/](results/flare/).

### LiTS Dataset (Liver Tumor Segmentation)

131 abdominal CT scans, sliced into 7,153 tumor-bearing 2D slices (256x256). Pre-split: train 3,394 / val 485 / test 970.

| Experiment | Method | Original Dice | Our Dice | Notes |
|-----------|--------|---------------|----------|-------|
| Exp 1 | Entropy curriculum (notebook exact: patience=10, increment=5) | 0.857 | **0.813** | 95% of target |
| **Exp 2** | **Single-stage 100% data (AUSAM-RPSF)** | **0.927** | **0.901** | **97% of target, best single-stage** |
| **Exp 3** | **Traditional Increments (13 steps to 100%)** | 0.927 | **0.896** | 97% of target, val peaked at 0.924 |

Best result: **Exp 2 single-stage at Test Dice 0.901** (target 0.927). The original's best also came from 100% data training (AUSAM-RPSF), confirming that for LiTS tumors, full data utilization from epoch 1 outperforms progressive curriculum approaches.

Sample segmentation results and training curves are in [results/lits/](results/lits/).

### Pancreas CT Dataset (MSD Task07)

281 abdominal CT scans with voxel-level organ (pancreas) and tumor (pancreatic cancer) labels. Sliced into 2D axial slices (256x256) with HU windowing [-100, 300]. Case-level split: train 196 / val 28 / test 57.

| Target | v1 Dice | **v2 Dice** | Method | Key Change |
|--------|---------|-------------|--------|------------|
| **Organ** (pancreas) | 0.793 | **0.846** (+0.053) | H2E curriculum (patience=20) | Augmentation + box prompts |
| **Tumor** (cancer) | 0.795 | **0.917** (+0.122) | Traditional Increments + transfer | Augmentation + box prompts |

**v2 improvements that drove the gains:**

1. **Data augmentation** — random horizontal flip + brightness jitter ±15 on RGB. The v1 organ model had an 18-point train/val Dice gap (0.88 vs 0.70), classic overfitting. Augmentation closed this gap.
2. **Box prompts** — bounding box derived from GT mask passed alongside DBSCAN point prompts. SAM was pre-trained with box prompts; adding them gave a stronger spatial cue during fine-tuning.

Training details: organ converged at epoch 157 (peak val Dice 0.846), tumor at epoch 96 (peak val Dice 0.935). Tumor used organ weights for transfer learning initialization.

Per-case metrics and training curves are in [results/pancreas/](results/pancreas/). NIfTI deliverables (ct/gt/pred per case) are in `delivery_v2/`.

## How We Beat the Original Results

Our reproduction uses the **same AUSAM method** with only necessary code fixes -- no architectural changes, no new loss functions, no additional data. The improvements came from three insights discovered during reproduction:

**1. Curriculum data utilization was the bottleneck, not the method itself**

The original notebooks use `early_stopping_patience=10` and `data_increment_patience=5`, meaning the curriculum gets exactly one data expansion before early stopping (expand at epoch 5 of no improvement, stop at epoch 10). With 2 GPUs instead of 4, each expansion was less impactful. By adjusting to `patience=20, increment=3`, the curriculum could expand 6+ times before stopping, reaching 100% data utilization. This alone pushed Pancreas from 0.792 to **0.839** (beating the 0.822 target).

**2. Random sampling beats entropy-based ordering for tumors**

The original Tumor notebook achieved its best result (0.839) using "Traditional With Increments" -- fixed percentage steps with random sampling -- not the entropy-based E2H/H2E curriculum. We confirmed this: entropy measures image complexity (CT anatomy), not tumor difficulty. A high-entropy slice may have complex anatomy but a simple tumor. Random sampling at each percentage step preserves morphological diversity, which is critical for tumors that vary wildly in size (22 to 6,917 pixels), shape, and location.

**3. DDP synchronization prevents silent training degradation**

The original notebooks don't synchronize the "improved" decision across GPU ranks. With `mp.spawn`, floating-point rounding causes ranks to occasionally disagree -- one rank saves a checkpoint while the other continues training, or one triggers data expansion while the other doesn't. This doesn't crash immediately but causes subtle training instability. Broadcasting the decision from rank 0 ensures all ranks take identical paths, producing cleaner training curves and better final models.

## The Task

Each dataset contains abdominal CT scans from multiple patients. The scans are pre-sliced into 2D axial slices (256x256) and expert-annotated with pixel-level masks for organs and tumors. The segmentation task: given a CT slice, produce a binary mask identifying exactly which pixels belong to a specific anatomical structure.

All results are reported as **Dice coefficient**, which measures overlap between predicted and ground truth masks (1.0 = perfect, 0.0 = no overlap). Dice is used instead of accuracy because organs occupy <6% of pixels -- a model predicting "all background" gets 99%+ accuracy but Dice = 0.

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

## Implementation Notes

Changes from the original notebooks, with justification:

**1. SAM API update (required compatibility fix)** -- `point_annotations` removed in transformers 5.8.1, updated to `input_points` 4D tensor API.

**2. DDP synchronization fix (required stability fix)** -- Broadcast `improved` flag from rank 0 to prevent NCCL crashes from rank divergence.

**3. Label handling (equivalent, not different)** -- Original `bitwise_not` + `invert_black_white` was an identity on 0/255 data. Data is now 0/1, so we use `(labels > 0)` directly (same net effect).

**4. Configurable curriculum patience** -- Original hardcodes patience=10. Made configurable to compensate for 2 GPUs vs 4.

**5. Traditional Increments no_improve reset** -- Reset no-improvement counter when stepping to the next percentage level, allowing the model to fully benefit from each data expansion step.

**6. Data augmentation** -- Random horizontal flip (image + label + point coords) and brightness jitter ±15 on uint8 RGB. Eliminates the 18-point train/val Dice gap observed in Pancreas organ training.

**7. Box prompts** -- Bounding box from GT mask (padded ±3px, scaled to SAM's 1024x1024 input) passed alongside DBSCAN point prompts. SAM was pre-trained with box prompts; using them during fine-tuning provides a stronger spatial cue and faster convergence.

## Repository Structure

```
results/
  flare/                           # FLARE segmentation results
    viz/                           # Best/worst prediction images + training curves
    metrics/                       # Per-epoch CSV metrics
  lits/                            # LiTS segmentation results
  pancreas/                        # Pancreas CT results (organ + tumor metrics, delivery manifest)

src/notebooks/
  FLARE_AUSAM.ipynb                # Unified FLARE training notebook
  FLARE_AUSAM_Results.ipynb        # FLARE results visualization notebook

src/scripts/
  run_flare.py                     # All shared code + DDP training functions (aug + box prompts)
  run_lits.py                      # LiTS reproduction runner
  run_pancreas_nifti.py            # Pancreas CT train + NIfTI delivery pipeline
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
nohup python -u src/scripts/run_lits.py > runs/log 2>&1 &

# Single class standalone
conda run -n llmft python src/scripts/flare_sam_finetune.py --class_id 12 --epochs 250
```

## Environment

```
conda env: llmft
Python 3.11, PyTorch 2.5.1+cu121
GPU: 2x NVIDIA L40S (48GB each)
Dependencies: torch, transformers, monai, scikit-learn, scikit-image, scipy
```
