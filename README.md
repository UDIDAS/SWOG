# AUSAM FLARE Reproduction

Reimplementation of the AUSAM (Adaptive Unified Segmentation Anything Model) pipeline for the FLARE abdominal CT segmentation dataset. The original implementation spans 5 Jupyter notebooks — this branch consolidates them into a single notebook with shared utilities and per-experiment sections.

## Original Notebooks (in `FLARE.zip`)

| Notebook | Class | Method | Reported Test Dice |
|---|---|---|---|
| `SAM-DBSCAN_FLARE-paper1.ipynb` | 4 (Pancreas) | H2E curriculum | 0.822 |
| `SAM-DBSCAN_FLARE-paper1-aggregation.ipynb` | 12 (Duodenum) | Transfer learning (class 1 -> 12) | *(not saved)* |
| `SAM-DBSCAN_FLARE-paper1-sam2.ipynb` | 1 (Liver) | SAM2 Hiera | 0.941 |
| `SAM-DBSCAN_FLARE-Tumor.ipynb` | 14 (Tumor) | E2H / H2E curriculum | 0.839 |
| `SAM-DBSCAN_FLARE-Multi-Class0-...3d.ipynb` | 0 (all) | Entropy curriculum + 3D eval | 0.975 (train) |

## Our Reproduction

### File Structure

```
FLARE_AUSAM.ipynb     # Unified notebook (Sections 1-9)
run_flare.py          # Standalone runner — all shared code + training functions
run_remaining.py      # Runner for pending sections (S3, S5, S6)
flare_sam_finetune.py # Initial single-stage fine-tuning script
FLARE.zip             # Original notebooks + coordinate files
```

### Key Implementation Differences from Original

1. **SAM API**: Updated from deprecated `point_annotations` to `input_points` (4D tensor: batch, point_batch_size, num_points, 2) + `input_labels`
2. **DDP sync**: Added `dist.broadcast` for the `improved` flag so all ranks agree on early-stop/expand decisions (fixes NCCL watchdog crashes)
3. **Label handling**: Original notebooks invert labels; we use `(labels > 0).astype(uint8)` directly (organ=1, background=0)
4. **2 GPUs** (L40S) instead of original's 4 GPUs

### Reproduction Status

| Section | Notebook | Class | Method | Original Dice | Our Dice | Status |
|---|---|---|---|---|---|---|
| S2 | paper1-aggregation | 12 (Duodenum) | Single-stage baseline | -- | **0.873** | Done |
| S3 | paper1 | 4 (Pancreas) | E2H curriculum | 0.822 | -- | Pending (fixed) |
| S3 | paper1 | 4 (Pancreas) | H2E curriculum | 0.822 | -- | Pending (fixed) |
| S5 | paper1-aggregation | 1 (Liver) | Train base model | -- | 0.970 (val) | Trained, needs test eval |
| S5 | paper1-aggregation | 12 (Duodenum) | Transfer from Liver | -- | -- | Pending |
| S6 | Tumor | 14 (Tumor) | H2E curriculum | 0.839 | -- | Pending |
| S7 | paper1-sam2 | 1 (Liver) | SAM2 Hiera | 0.941 | -- | Pending (needs `sam2` pkg) |
| S8 | Multi-Class0-3d | 0 (all) | 3D volume eval | 0.975 (train) | -- | Blocked (no NIfTI data) |

### How to Run

```bash
# Full pipeline (S3 + S5 + S6 sequentially)
nohup /home/ud3d4/.conda/envs/llmft/bin/python -u run_remaining.py \
    > /scratch/ud3d4/acm_data/FLARE/runs/remaining_run.log 2>&1 &

# Or run individual sections via the standalone script
conda run -n llmft python flare_sam_finetune.py --class_id 12 --epochs 250

# Test evaluation only (if model exists)
conda run -n llmft python flare_sam_finetune.py --class_id 12 --test_only
```

### Data

FLARE per-class data at `/scratch/ud3d4/acm_data/FLARE/`:
- `class_{0-14}_images.npy` — 256x256x3 RGB uint8 axial CT slices
- `class_{0-14}_labels.npy` — 256x256 binary masks

15 classes: Liver (1), R.Kidney (2), Spleen (3), Pancreas (4), Aorta (5), IVC (6), R.Adrenal (7), L.Adrenal (8), Gallbladder (9), Esophagus (10), Stomach (11), Duodenum (12), L.Kidney (13), Tumor (14), Background (0).

### Dependencies

```
torch>=2.5, transformers, monai, scikit-learn, scikit-image, scipy
# For SAM2 section: pip install sam2
```

### Environment

```
conda env: llmft
Python 3.11, PyTorch 2.5.1+cu121
GPU: 2x NVIDIA L40S (48GB each)
```
