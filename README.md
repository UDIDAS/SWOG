# SEMIR: Semantic Minor-Induced Representation Learning

Reproduction of SEMIR graph-minor segmentation for medical images, targeting LiTS liver tumor benchmark.

## Goal

Replace AuSAM (SAM2.1 + DBSCAN) with SEMIR's learned graph minors in the ACM MMKG VKG pipeline, eliminating dependency on ground-truth CSV files for tumor phenotype extraction.

## Repository Structure

```
fastloops/
  src/lib.rs          — Rust crate: graph-minor pooling via PyO3
  Cargo.toml          — Rust dependencies (pyo3, numpy, ndarray)

notebooks/
  semir_lits_twostage.ipynb — Current: two-stage liver-crop + protected coarsening + GINE
  semir_lits_v7.py          — Previous: full-volume oracle search + GINE training
  run_semir_jepa.py         — DINO + IG-JEPA self-supervised pre-training exploration
  rust_crate.ipynb          — Luke's original crate reference + feature extraction

results/
  semir_lits_twostage/ — Two-stage pipeline results (oracle diagnostics + GINE Dice)
  semir_lits_v7/       — Full-volume pipeline results
  archive/             — Earlier experiment results (v5, v6)
```

## Rust Crate (`fastloops`)

Binary tensor-based graph-minor pooling on 3D medical volumes.

**Build:**
```bash
source ~/.cargo/env
cd fastloops
VIRTUAL_ENV= CONDA_PREFIX=~/.conda/envs/llmft maturin develop --release
```

**API:**
```python
import fastloops
import numpy as np

# Input: uint8 CT volume, C-contiguous, channel-last
ct_u8 = np.ascontiguousarray(volume[..., np.newaxis])  # (D, H, W, 1)

# Standard coarsening
node_feats, edge_index, edge_feats, labels, adj = fastloops.merge_and_cut(
    ct_u8,
    merge_distance=5,       # ψ: edge contraction threshold (uint8 scale)
    cut_distance=25,        # α: edge deletion threshold
    connectivity="faces",   # 6-connected
)

# Protected coarsening (prevents merging/deletion of candidate tumor voxels)
protect_mask = suspicious_region.astype(np.uint8)  # same shape as volume
node_feats, edge_index, edge_feats, labels, adj = fastloops.merge_and_cut_protected(
    ct_u8, protect_mask,
    merge_distance=5, cut_distance=25, connectivity="faces",
)
```

**Returns:**
- `node_feats` (N, 20): per-supernode statistics (area, coord sums, covariances, channel sums, boundary, canonical)
- `edge_index` (2, E): supernode adjacency
- `edge_feats` (E, 4): boundary count, summed/max distance, cut fraction
- `labels` (D, H, W): voxel-to-supernode mapping (-1 = deleted)
- `adj`: binary tensor with bitflags

**Protection mechanism:** `merge_and_cut_protected` takes a binary mask where nonzero voxels are "protected." During edge contraction, merging is blocked across the protected/non-protected boundary. During node deletion, any supernode containing a protected voxel is preserved regardless of size/intensity thresholds.

## SEMIR Pipeline

**Node features** (per supernode):
| Feature | Formula |
|---------|---------|
| Volume | aᵤ (voxel count) |
| Boundary length | bᵤ |
| Compactness | bᵤ / aᵤ^(2/3) |
| Elongation | λ_max / (λ_min + ε) |
| Dominant axis | Principal eigenvector (3 components) |
| Mean intensity | Channel sum / volume / 255 |
| Intensity std | √(E[I²] - E[I]²) |

**Edge features**: size contrast, boundary fractions, mean contrast, shape dissimilarity, axis alignment, boundary contrast, cut fraction.

**GNN**: 3-layer GINE (hidden 128), binary tumor-vs-rest, Adam lr=1e-3, early stopping on lifted voxel Dice.

## Current Approach: Two-Stage Protected Coarsening

The key insight: standard coarsening destroys tumor information because default deletion removes 77% of tumor voxels (small tumor supernodes get killed). The two-stage approach fixes this:

1. **Liver ROI crop** — bounding box from organ mask + 32-voxel margin, reduces volume by ~50%
2. **Intensity-based protection** — identify hypodense voxels inside liver (< mean - 0.5σ), dilate by 2. This catches tumor candidates without using GT labels. Achieves 100% recall on tumor voxels.
3. **Protected coarsening** — run fastloops with default deletion, but protected supernodes survive. Result: 0% tumor deletion, oracle Dice ~0.90.
4. **GINE classification** — train on the protected graph, lift predictions to voxel level.

### Results (10-volume diagnostic)

| Mode | Description | Oracle@0.25 | Tumor Del% | Supernodes |
|------|------------|-------------|------------|------------|
| A | Full CT, default deletion | 0.282 | 77.4% | 5K |
| B | Liver crop, default deletion | 0.281 | 77.2% | 5K |
| **C** | **Liver crop + intensity protection** | **0.892** | **0.0%** | **241K** |
| D | Liver crop + GT protection (UB) | 0.895 | 0.0% | 27K |

### Training Results (118 volumes, Mode C)

| Split | Voxel Dice | Recall | Precision | N |
|-------|-----------|--------|-----------|---|
| Train | 0.176 | 0.235 | 0.240 | 82 |
| Val | 0.197 | 0.216 | 0.343 | 17 |
| Test | 0.233 | 0.300 | 0.343 | 19 |

**Best val Dice: 0.321** — up from <1% with unprotected coarsening. The remaining gap to the paper's 0.891 is primarily due to the large graph size (~241K nodes avg) which makes GINE training difficult. Next step: reduce protected region to compress graphs closer to ~10K-50K nodes.

## Data

- **LiTS**: `/scratch/ud3d4/acm_data/Data/` (118 volumes with tumor, .npy format)
- **Pancreas (MSD Task07)**: `/scratch/ud3d4/acm_data/Pancreas/` (281 volumes, .nii.gz)
- **GT CSVs**: `/home/ud3d4/Desktop/Projects/acm_mmkg/data/`

## Environment

```
conda env: llmft
Python 3.11, PyTorch 2.5.1+cu121, PyG 2.7
GPU: NVIDIA L40S (49GB)
```
