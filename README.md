# SEMIR: Semantic Minor-Induced Representation Learning

Reproduction of SEMIR graph-minor segmentation for medical images, targeting LiTS liver tumor benchmark.

## Goal

Replace AuSAM (SAM2.1 + DBSCAN) with SEMIR's learned graph minors in the ACM MMKG VKG pipeline, eliminating dependency on ground-truth CSV files.

## Repository Structure

```
fastloops/
  src/lib.rs          — Rust crate: graph-minor pooling via PyO3
  Cargo.toml          — Rust dependencies (pyo3, numpy, ndarray)

notebooks/
  rust_crate.ipynb    — Luke's original crate reference + feature extraction
  semir_lits_v7.py    — Current pipeline: oracle search → diagnostics → GINE training
  semir_pancreas_detections.ipynb — Pancreas dataset exploration

results/
  semir_lits_v7/      — Latest experiment results
  archive/            — Earlier experiment results (v5, v6)
```

## Rust Crate (`fastloops`)

Binary tensor-based graph-minor pooling on 3D medical volumes.

**Build:**
```bash
source ~/.cargo/env
cd fastloops
maturin build --release --interpreter ~/.conda/envs/llmft/bin/python
pip install --no-deps --force-reinstall target/wheels/fastloops-*.whl
```

**Usage:**
```python
import fastloops
import numpy as np

# Input: uint8 CT volume, C-contiguous, channel-last
ct_u8 = np.ascontiguousarray(volume[..., np.newaxis])  # (D, H, W, 1)

node_feats, edge_index, edge_feats, labels, adj = fastloops.merge_and_cut(
    ct_u8,
    merge_distance=3,       # ψ: edge contraction threshold (uint8 scale)
    cut_distance=15,        # α: edge deletion threshold
    delete_small_node_max_size=0,   # β_min (0 = no deletion)
    delete_large_node_min_size=n,   # β_max
    delete_value_min=0,     # m_min
    delete_value_max=255,   # m_max
    connectivity="faces",   # 6-connected
)
```

**Returns:**
- `node_feats` (N, 20): per-supernode statistics (area, coord sums, covariances, channel sums, boundary, canonical)
- `edge_index` (2, E): supernode adjacency
- `edge_feats` (E, 4): boundary count, summed/max distance, cut fraction
- `labels` (D, H, W): voxel-to-supernode mapping (-1 = deleted)
- `adj`: binary tensor with bitflags

**Current algorithm:** Kruskal-like canonical-anchored edge contraction:
1. Bucket-sort all adjacent voxel pairs by intensity distance
2. Process smallest-first with Union-Find
3. Before each merge, check canonical intensities of both supernodes (not raw edge distance)
4. Canonical = intensity of largest component's representative voxel (prevents drift)

## SEMIR Pipeline

**Node features** (per supernode, paper Section 3.2):
| Feature | Formula |
|---------|---------|
| Volume | aᵤ (voxel count) |
| Boundary length | bᵤ |
| Compactness | 36πaᵤ²/(bᵤ³ + ε) |
| Elongation | λ_max / (λ_min + ε) |
| Dominant axis | Principal eigenvector (3 components) |
| Mean intensity | Channel sum / volume / 255 |
| Intensity std | √(E[I²] - E[I]²) |

**Edge features**: size contrast, boundary fractions, mean contrast, shape dissimilarity, axis alignment, boundary contrast, cut fraction.

**GNN**: 3-layer GINE (hidden 128), binary tumor-vs-rest, Adam lr=1e-3, early stopping on lifted voxel Dice.

## Current Status

**Paper target:** LiTS tumor Dice 0.891 ± 0.007, ~1,075 supernodes.

### What works
- Oracle Dice 0.94 at overlap threshold 0.10 — the graph representation preserves tumor boundaries
- Full-graph GPU training at 7-30s/epoch on L40S (49GB)
- Feature extraction matching the paper's 7 node + 6 edge features

### Blocking issue: graph compression
Our crate produces **~1.7M supernodes** vs the paper's **~1K**. This 1000x gap causes:
- Features degenerate at single-voxel resolution (Cohen's d < 0.54)
- Extreme class imbalance (143:1 tumor:background)
- GNN cannot learn (val Dice = 0.000 with standard CE)

Best training result: **val Dice 0.112** on consolidated 300K-node graphs (oracle ceiling 0.47).

### Approaches tried
| Approach | Supernodes | Oracle | Training Dice | Issue |
|----------|-----------|--------|--------------|-------|
| Luke's BFS (ψ=3) | 1.7M | 0.94 | 0.000 | Too many nodes, features degenerate |
| Iterative consolidation (canonical, no-transitivity) | 300K | 0.47 | 0.112 | Oracle too low |
| Kruskal canonical-anchored | 1.9M | 0.94 | — | Same as BFS: 1 giant blob + fragments |
| Higher ψ (8-20) | 250K-840K | 0.07-0.80 | — | Tumor merges with liver |

### Next steps
- Awaiting the paper author's LiTS implementation code
- The contraction algorithm is the sole remaining gap — all other components match the paper

## Data

- **LiTS**: `/scratch/ud3d4/acm_data/Data/` (118 volumes with tumor, .npy)
- **Pancreas (MSD Task07)**: `/scratch/ud3d4/acm_data/Pancreas/` (281 volumes, .nii.gz)
- **GT CSVs**: `/home/ud3d4/Desktop/Projects/acm_mmkg/data/`

## Environment

```
conda env: llmft
Python 3.11, PyTorch 2.5.1+cu121, PyG 2.7
GPU: NVIDIA L40S (49GB)
```
