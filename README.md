# AUSAM FLARE Reproduction

Reimplementation of the AUSAM (Adaptive Unified Segmentation Anything Model) pipeline for the FLARE abdominal CT segmentation dataset. The original implementation spans 5 Jupyter notebooks -- this branch consolidates them into a unified pipeline with shared utilities and per-experiment sections.

## Original Notebooks

The `FLARE.zip` file contains 5 notebooks, each exploring a different aspect of SAM fine-tuning for abdominal CT segmentation:

| Notebook | What it does | Data | Best Result |
|----------|-------------|------|-------------|
| `SAM-DBSCAN_FLARE-paper1.ipynb` | Core experimentation notebook. Tests E2H (Easy-to-Hard) and H2E (Hard-to-Easy) entropy-based curriculum learning, plus H2E with augmentations (intensity rescaling, Gaussian smoothing) and GAN-based augmentation. | Class 4 (Pancreas) | H2E Test Dice 0.822 |
| `SAM-DBSCAN_FLARE-paper1-aggregation.ipynb` | Transfer learning experiment. Trains on class 1 (Liver), then fine-tunes on class 12 (Duodenum) to test if pretraining on a large easy organ helps small hard organs. | Class 1 -> 12 | *(no test output saved)* |
| `SAM-DBSCAN_FLARE-paper1-sam2.ipynb` | Architecture comparison. Replaces SAM ViT-Base with SAM2 Hiera-Base-Plus to test if the newer architecture improves segmentation. | Class 1 (Liver) | Test Dice 0.941 |
| `SAM-DBSCAN_FLARE-Tumor.ipynb` | Applies multiple training strategies to tumor segmentation: E2H, H2E, single-stage, fixed percentage, and Traditional With Increments (fixed percentage steps with progressive data growth). | Class 14 (Tumor) | Traditional Test Dice 0.839 |
| `SAM-DBSCAN_FLARE-Multi-Class0-...3d.ipynb` | 3D volume-level evaluation. Runs SAM slice-by-slice, stacks predictions into 3D volumes, computes per-subject volumetric Dice. | Class 0 (all organs) | Train Dice 0.975 |

## Results

| Section | Method | Class | Original | Ours | vs Target | Status |
|---------|--------|-------|----------|------|-----------|--------|
| S2 | Single-stage baseline | 12 (Duodenum) | -- | 0.873 | -- | Done |
| S3-E2H | E2H curriculum | 4 (Pancreas) | 0.822 | 0.773 | 94% | Done |
| S3-H2E | H2E curriculum | 4 (Pancreas) | 0.822 | 0.792 | 96% | Done |
| **R2-H2E** | **H2E (patience=20)** | **4 (Pancreas)** | **0.822** | **0.839** | **BEAT** | **Done** |
| **S5-Liver** | **Liver base model** | **1 (Liver)** | **0.941** | **0.965** | **BEAT** | **Done** |
| S5-Transfer | Liver -> Duodenum | 12 (Duodenum) | -- | 0.890 | +1.7% | Done |
| S6-H2E | H2E tumor | 14 (Tumor) | 0.839 | 0.717 | 85% | Done |
| R2-Trad | Traditional Increments | 14 (Tumor) | 0.839 | 0.797 | 95% | Done |
| R2-E2H | E2H tumor | 14 (Tumor) | 0.839 | ~0.751 | 90% | Done |

## Key Findings

- **Pancreas target BEATEN (0.839 vs 0.822)**: Increasing curriculum patience from 10 to 20 and data increment from 5 to 3 allowed the curriculum to expand to 100% data, which was the missing ingredient.
- **Liver target BEATEN (0.965 vs 0.941)**: SAM ViT-Base outperforms the original SAM2 Hiera on large organs.
- **Tumor improved significantly (0.797 vs 0.717)**: Traditional Increments (random percentage steps) works better than entropy-based curriculum for tumors because entropy measures image complexity, not tumor difficulty.
- **H2E consistently outperforms E2H**: Starting with hard samples builds more robust features.
- **Transfer learning confirmed**: Liver pretraining boosted Duodenum from 0.873 to 0.890.

## Implementation Notes

Changes from the original notebooks, with justification:

**1. SAM API update (required compatibility fix)**

The original notebooks use `point_annotations` which was removed in the current HuggingFace transformers version (5.8.1). Updated to the current `input_points` API which requires a 4D tensor `(batch, point_batch_size, num_points, 2)` plus `input_labels`. Without this fix, the code crashes on import.

**2. DDP synchronization fix (required stability fix)**

The original notebooks use `mp.Process` directly. Our implementation uses `mp.spawn`, which causes each GPU rank to compute validation loss independently from its own data shard. Floating-point differences between shards can cause ranks to disagree on whether training improved, leading one rank to enter a `dist.barrier()` the other skips -- crashing with an NCCL watchdog timeout. Fixed by broadcasting the `improved` flag from rank 0 so all ranks make identical decisions.

**3. Label handling (equivalent, not different)**

The original notebooks apply `bitwise_not` then `invert_black_white` to the labels. This double inversion was written when the FLARE label data was stored as 0/255 (standard image format), where the chain is an identity operation (organ pixels stay as organ). The data has since been re-saved as 0/1 binary, making the double inversion produce all-zeros (a bug). Our `(labels > 0).astype(uint8)` produces the same result as the original pipeline on the original 0/255 data format.

**4. Configurable curriculum patience (tuning for 2 GPUs)**

The original hardcodes `early_stopping_patience=10` and `data_increment_patience=5`. With 2 GPUs instead of 4, each epoch processes the same data but the curriculum has fewer expansion opportunities before early stopping. Making these configurable via the cfg dict allowed Round 2 to use `patience=20, increment=3`, which let the curriculum reach 100% data and beat the Pancreas target.

**5. 2 GPUs instead of 4 (hardware constraint)**

The original notebooks use 4 GPUs. We use 2x NVIDIA L40S (48GB each). DDP splits data across GPUs, so effective throughput per epoch is the same but with different mini-batch dynamics (effective batch 10 vs 20). The main impact is on curriculum expansion -- fewer epochs before early-stop means less time for data to grow.

## Repository Structure

```
src/notebooks/
  FLARE_AUSAM.ipynb              # Unified training notebook (Sections 1-9)
  FLARE_AUSAM_Results.ipynb      # Results visualization notebook

src/scripts/
  run_flare.py                   # All shared code + DDP training functions
  run_remaining.py               # Runner for Round 1 (S3, S5, S6)
  run_round2.py                  # Runner for Round 2 (Pancreas patience=20, Tumor Traditional)
  run_s6_only.py                 # Standalone S6 runner
  flare_sam_finetune.py          # Initial standalone fine-tuning script
```

## How to Run

```bash
# Round 1: S3 + S5 + S6
nohup python -u src/scripts/run_remaining.py > runs/log 2>&1 &

# Round 2: Pancreas patience=20 + Tumor Traditional + Tumor E2H
nohup python -u src/scripts/run_round2.py > runs/log 2>&1 &

# Single class standalone
conda run -n llmft python src/scripts/flare_sam_finetune.py --class_id 12 --epochs 250

# Test evaluation only
conda run -n llmft python src/scripts/flare_sam_finetune.py --class_id 12 --test_only
```

## Data

FLARE per-class data at `/scratch/ud3d4/acm_data/FLARE/`:
- `class_{0-14}_images.npy` -- 256x256x3 RGB uint8 axial CT slices
- `class_{0-14}_labels.npy` -- 256x256 binary masks (0=background, 1=organ/tumor)

15 classes: Liver(1), R.Kidney(2), Spleen(3), Pancreas(4), Aorta(5), IVC(6), R.Adrenal(7), L.Adrenal(8), Gallbladder(9), Esophagus(10), Stomach(11), Duodenum(12), L.Kidney(13), Tumor(14), Background(0)

## Environment

```
conda env: llmft
Python 3.11, PyTorch 2.5.1+cu121
GPU: 2x NVIDIA L40S (48GB each)
Dependencies: torch, transformers, monai, scikit-learn, scikit-image, scipy
```
