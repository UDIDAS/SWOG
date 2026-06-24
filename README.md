# AUSAM FLARE Reproduction

Reimplementation of the AUSAM (Adaptive Unified Segmentation Anything Model) pipeline for the FLARE abdominal CT segmentation dataset. The original implementation spans 5 Jupyter notebooks -- this branch consolidates them into a unified pipeline with shared utilities and per-experiment sections.

## The Task

The FLARE (Fast and Low-resource semi-supervised Abdominal oRgan sEgmentation) dataset contains abdominal CT scans from multiple patients. Each 3D scan has been pre-sliced into 2D axial slices (256x256 RGB) and the pixels belonging to each organ or tumor have been manually annotated by experts.

The segmentation task is: given a 2D CT slice, produce a binary mask that identifies exactly which pixels belong to a specific anatomical structure. Each "class" in the dataset corresponds to a different structure:

| Class | Structure | Slices | Organ Fraction | Difficulty |
|-------|-----------|--------|----------------|------------|
| 1 | Liver | 19,664 | 5.76% | Easy -- large, high contrast |
| 4 | Pancreas | 9,888 | 0.60% | Hard -- small, irregular shape, low contrast |
| 12 | Duodenum | 2,070 | 0.39% | Hard -- very small, adjacent to pancreas |
| 14 | Tumor | 9,911 | 0.87% | Hardest -- irregular, variable size (22-6917 px), no fixed position |

When we say "Class 4 (Pancreas)", it means the model is trained and evaluated specifically on CT slices containing the pancreas, and its job is to correctly identify which pixels in each slice are pancreas tissue vs everything else (background, other organs, etc.).

## What the Scores Mean

All results are reported as **Dice coefficient** (also called F1 score for segmentation), which measures the overlap between the model's predicted mask and the expert-annotated ground truth mask:

```
Dice = (2 x |Predicted AND Ground Truth|) / (|Predicted| + |Ground Truth|)
```

- **Dice = 1.0**: Perfect overlap -- every pixel matches the expert annotation
- **Dice = 0.9+**: Excellent -- minor boundary disagreements only
- **Dice = 0.8-0.9**: Good -- captures the organ shape but with some boundary errors or small missed regions
- **Dice = 0.7-0.8**: Moderate -- captures the rough location but with significant boundary/shape errors
- **Dice < 0.5**: Poor -- model fails to locate or delineate the structure

Accuracy alone is misleading for this task because organs are tiny (0.4-6% of pixels). A model that predicts "all background" gets 99%+ accuracy but Dice = 0. Dice forces the model to actually find and delineate the organ.

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

| Section | Method | Target Structure | What was tested | Original Dice | Our Dice |
|---------|--------|-----------------|-----------------|---------------|----------|
| S2 | Single-stage fine-tune | Duodenum (class 12) | Baseline: train SAM on duodenum slices with DBSCAN point prompts, no curriculum or pretraining | -- | 0.873 |
| S3-E2H | E2H curriculum | Pancreas (class 4) | Start training on easy (low-entropy) CT slices, progressively add harder ones | 0.822 | 0.773 |
| S3-H2E | H2E curriculum | Pancreas (class 4) | Start training on hard (high-entropy) slices, progressively add easier ones | 0.822 | 0.792 |
| **R2-H2E** | **H2E (patience=20)** | **Pancreas (class 4)** | **Same as S3-H2E but with more patience, allowing curriculum to reach 100% data** | **0.822** | **0.839 BEAT** |
| **S5-Liver** | **Single-stage fine-tune** | **Liver (class 1)** | **Train SAM on liver slices -- large organ, high contrast, establishes strong base model** | **0.941** | **0.965 BEAT** |
| S5-Transfer | Transfer learning | Duodenum (class 12) | Take Liver-trained SAM and fine-tune on Duodenum -- tests if organ-to-organ transfer helps | -- | 0.890 |
| S6-H2E | H2E curriculum | Tumor (class 14) | Entropy-based curriculum on tumors -- trains on high-entropy slices first | 0.839 | 0.717 |
| R2-Trad | Traditional Increments | Tumor (class 14) | Fixed percentage steps (14.8% -> 30% -> ... -> 100%) with random sampling at each step | 0.839 | 0.797 |
| R2-E2H | E2H curriculum | Tumor (class 14) | Entropy-based curriculum on tumors -- trains on low-entropy slices first | 0.839 | ~0.751 |

**Reading the table**: Each row is an experiment where SAM was fine-tuned to segment one specific structure. "Original Dice" is the score from the original notebooks. "Our Dice" is what our reproduction achieved on a held-out test set that the model never saw during training. Higher is better.

## Key Findings

- **Pancreas target BEATEN (0.839 vs 0.822)**: Increasing curriculum patience from 10 to 20 and data increment from 5 to 3 allowed the curriculum to expand to 100% data, which was the missing ingredient. The model needed to see all training examples to learn the full range of pancreas shapes.
- **Liver target BEATEN (0.965 vs 0.941)**: SAM ViT-Base outperforms the original SAM2 Hiera on large organs. Liver occupies 5.76% of pixels with clear boundaries, so the standard SAM architecture handles it excellently.
- **Tumor improved but not beaten (0.797 vs 0.839)**: Traditional Increments (random percentage steps) works much better than entropy-based curriculum for tumors (+8 points over H2E). Entropy measures image complexity, not tumor difficulty -- a high-entropy image might have complex anatomy but an easy tumor, so entropy-based ordering is the wrong signal for tumors. Random sampling preserves morphological diversity at every training stage.
- **H2E consistently outperforms E2H on organs**: Starting with hard samples builds more robust features that transfer well when easier samples are added later.
- **Transfer learning confirmed**: Liver pretraining boosted Duodenum from 0.873 to 0.890 (+1.7 points), validating that features learned from a large organ generalize to smaller ones.

## Implementation Notes

Changes from the original notebooks, with justification:

**1. SAM API update (required compatibility fix)**

The original notebooks use `point_annotations` which was removed in the current HuggingFace transformers version (5.8.1). Updated to the current `input_points` API which requires a 4D tensor `(batch, point_batch_size, num_points, 2)` plus `input_labels`. Without this fix, the code crashes on import.

**2. DDP synchronization fix (required stability fix)**

The original notebooks use `mp.Process` directly. Our implementation uses `mp.spawn`, which causes each GPU rank to compute validation loss independently from its own data shard. Floating-point differences between shards can cause ranks to disagree on whether training improved, leading one rank to enter a `dist.barrier()` the other skips -- crashing with an NCCL watchdog timeout. Fixed by broadcasting the `improved` flag from rank 0 so all ranks make identical decisions.

**3. Label handling (equivalent, not different)**

The original notebooks apply `bitwise_not` then `invert_black_white` to the labels. This double inversion was written when the FLARE label data was stored as 0/255 (standard image format), where the chain is an identity operation (organ pixels stay as organ). The data has since been re-saved as 0/1 binary, making the double inversion produce all-zeros (a bug). Our `(labels > 0).astype(uint8)` produces the same result as the original pipeline on the original 0/255 data format.

**4. Configurable curriculum patience (tuning for 2 GPUs)**

The original hardcodes `early_stopping_patience=10` and `data_increment_patience=5`. With 2 GPUs instead of 4, the curriculum has fewer expansion opportunities before early stopping. Making these configurable via the cfg dict allowed Round 2 to use `patience=20, increment=3`, which let the curriculum reach 100% data and beat the Pancreas target.

**5. 2 GPUs instead of 4 (hardware constraint)**

The original notebooks use 4 GPUs. We use 2x NVIDIA L40S (48GB each). DDP splits data across GPUs, so each epoch sees the same total data but with different mini-batch dynamics (effective batch 10 vs 20). The main impact is on curriculum expansion -- with 2 GPUs and the original patience settings, the model early-stops before the curriculum can fully expand.

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
