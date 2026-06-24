# AUSAM FLARE Reproduction

Reimplementation of the AUSAM (Adaptive Unified Segmentation Anything Model) pipeline for the FLARE abdominal CT segmentation dataset. The original implementation spans 5 Jupyter notebooks -- this branch consolidates them into a unified pipeline with shared utilities and per-experiment sections.

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
| R2-E2H | E2H tumor | 14 (Tumor) | 0.839 | ~0.733 | 87% | Done |

## Key Findings

- **Pancreas target BEATEN (0.839 vs 0.822)**: Increasing curriculum patience from 10 to 20 and data increment from 5 to 3 allowed the curriculum to expand to 100% data, which was the missing ingredient.
- **Liver target BEATEN (0.965 vs 0.941)**: SAM ViT-Base outperforms the original SAM2 Hiera on large organs.
- **Tumor improved significantly (0.797 vs 0.717)**: Traditional Increments (random percentage steps) works better than entropy-based curriculum for tumors because entropy measures image complexity, not tumor difficulty.
- **H2E consistently outperforms E2H**: Starting with hard samples builds more robust features.
- **Transfer learning confirmed**: Liver pretraining boosted Duodenum from 0.873 to 0.890.

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

## Key Implementation Differences from Original

1. **SAM API**: Updated from deprecated `point_annotations` to `input_points` (4D tensor) + `input_labels`
2. **DDP sync**: Added `dist.broadcast` for improved flag to prevent NCCL watchdog crashes
3. **Configurable patience**: Curriculum patience and data_increment_patience read from cfg dict
4. **Traditional Increments**: New trainer matching the exact Tumor notebook strategy (fixed percentage steps)
5. **Label handling**: Direct binary (organ=1) without the original inversion step
6. **2 GPUs** (L40S) instead of original 4 GPUs

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
- `class_{0-14}_labels.npy` -- 256x256 binary masks

15 classes: Liver(1), R.Kidney(2), Spleen(3), Pancreas(4), Aorta(5), IVC(6), R.Adrenal(7), L.Adrenal(8), Gallbladder(9), Esophagus(10), Stomach(11), Duodenum(12), L.Kidney(13), Tumor(14), Background(0)

## Environment

```
conda env: llmft
Python 3.11, PyTorch 2.5.1+cu121
GPU: 2x NVIDIA L40S (48GB each)
Dependencies: torch, transformers, monai, scikit-learn, scikit-image, scipy
```
