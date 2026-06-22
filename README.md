# SEMIR: Semantic Minor-Induced Representation Learning

Reproducing the SEMIR paper's binary-tensor approach for LiTS liver tumor segmentation, with the goal of replacing GT-dependent CSV pipelines in the ACM MMKG VKG system.

## Goal

Replace AuSAM (SAM2.1 + DBSCAN) with SEMIR's learned graph minors in the ACM MMKG VKG pipeline. SEMIR derives tumor phenotypes (volume, compactness, elongation, intensity) directly from graph structure — no ground-truth CSV files needed.

## How It Works

SEMIR takes a 3D CT scan and **grows regions from seed points**. Each region expands until it hits a tissue boundary (where intensity changes sharply), producing ~1,000 supernodes that each cover a meaningful tissue region.

A few-shot boundary search (5 labeled examples) finds the exact threshold where supernode boundaries align with real tissue boundaries. A GINE then classifies each supernode as tumor or background, and predictions are lifted back to voxels.

## Pipeline

```
Raw CT → Liver crop → merge_and_cut (Rust binary tensor, ~1K supernodes)
→ Few-shot boundary search (6 params, Eq. 1) → Voxel reassignment
→ Feature extraction (7 node + 6 edge) → 3-layer GINE → Voxel lifting
```

## Repository Structure

```
notebooks/
  v18_paper_exact.ipynb          -- Paper-exact SEMIR (current)
  v16_luke_pipeline.ipynb        -- v16b: Luke's band-flood + GINE (val Dice 0.59)

scripts/
  v18_paper_exact.py             -- Same as notebook, script form

fastloops/                       -- Rust crate: merge_and_cut (paper's binary tensor)
fastloops_band/                  -- Rust crate: band_build (Luke's learned bands)

results/
  v18_paper_exact/               -- Current run
  v16b_band_protected/           -- Best previous (val Dice 0.59)

docs/                            -- Papers + presentations
archive/                         -- All previous experiments (v1-v17)
```

## Results

| Version | Approach | Nodes | Oracle | Val Dice | Status |
|---------|----------|-------|--------|----------|--------|
| v16b | band_build + protection | 240K | 0.98 | **0.59** | Best so far |
| v18 | **paper-exact merge_and_cut** | ~1K | TBD | TBD | Running |
| Paper | SEMIR (reference) | 1,075 | — | 0.891 | Target |

## Key Findings

1. **Graph size is the bottleneck.** 240K-node graphs cap GINE at 0.59 Dice despite 0.98 oracle. The paper gets 0.891 on ~1K nodes where 3 hops covers the entire graph.

2. **merge_and_cut and band_build are the same algorithm** with different merge predicates. `merge_and_cut`: `|I_seed - I_neighbor| ≤ ψ` (paper-exact). `band_build`: `band_of[I_seed] == band_of[I_neighbor]` (learned non-uniform bands). Luke's 196-band dictionary fragmented into 1.6M single-voxel supernodes because the bands were too narrow.

3. **Boundary alignment is the real objective.** The paper's few-shot search (Eq. 1) maximizes boundary Dice — not oracle, not compression. This is what makes ~1K nodes meaningful.

4. **Deleted voxels must be reassigned** to nearest survivor. Without this, oracle drops from 0.87 to 0.34.

5. **A hybrid approach is promising:** coarser `band_build` dictionary (~20-30 bands) could combine Luke's non-uniform boundaries with actual merging.

## Data

- **LiTS**: 131 training volumes (118 with tumor, 13 liver-only), .npy
- **Pancreas (MSD Task07)**: 281 volumes, .nii.gz

## Environment

```
conda env: llmft
Python 3.11, PyTorch 2.5.1+cu121, PyG 2.7
GPU: NVIDIA L40S (49GB)
```
