# SEMIR: Semantic Minor-Induced Representation Learning

Graph-minor segmentation for 3D medical images. Building on the learned intensity-band coarsening approach for LiTS liver tumor segmentation.

## Goal

Replace AuSAM (SAM2.1 + DBSCAN) with SEMIR's learned graph minors in the ACM MMKG VKG pipeline, eliminating dependency on ground-truth CSV files for tumor phenotype extraction.

## Pipeline Architecture

```
Stage 1a  — Learned intensity-band flood contraction (cheap, safe, transferable)
Stage 1b  — Certified junk deletion (oracle-budgeted intensity band removal)
Stage 2   — Learned faithful contraction (edge scoring + shape-faithfulness metric)
Stage 3   — Task GNN (tumor node classification + voxel lifting)
```

## Repository Structure

```
notebooks/
  NewSemirStage1 (2).ipynb       — Luke's Stage 1: band-flood coarsening + certified deletion
  stage2_selective_contraction.ipynb — Our Stage 2: selective contraction + SMBO + GINE (v14)

fastloops/                       — Our Rust crate (merge_and_cut + merge_and_cut_protected)
fastloops_band/                  — Luke's Rust crate (band_build for band-flood)

docs/
  SEMIR-MedicalImages.pdf        — SEMIR paper (MICCAI submission)
  SEMIR_Semantic_Minor_Ind.pdf   — SEMIR supplementary

results/
  stage2_selective/              — Stage 2 selective contraction results (SMBO + GINE)

archive/                         — Previous experiments (v1-v13, see archive/CHANGELOG.md)
```

## Current Status (v14)

**Stage 2 Selective Contraction + SMBO + GINE**

SMBO (Optuna, 40 trials) optimized 7 parameters jointly against oracle Dice on val set:
- psi=9, alpha=83, std_mult=1.86, dilate_iter=1
- bg_threshold=5, prot_threshold=9, n_rounds=1
- Compressed to ~52K nodes (from 172K), oracle 0.856

| Split | Dice | Recall | Precision | Oracle | Nodes | N |
|-------|------|--------|-----------|--------|-------|---|
| Train | 0.090 | 0.096 | 0.223 | 0.815 | 54K | 82 |
| Val | 0.057 | 0.035 | 0.231 | 0.856 | 53K | 17 |
| Test | 0.050 | 0.047 | 0.204 | 0.853 | 62K | 19 |

**Finding**: Selective contraction compressed graphs effectively (172K → 52K) but GINE performance dropped significantly (val 0.48 → 0.13). The contraction changes graph topology in ways that hurt GINE's message passing — likely hub-and-spoke structure replacing the original grid-like connectivity.

**Previous best (v11, archive)**: val Dice 0.48, test 0.32 on 172K-node protected graphs without contraction.

## Key Learnings

See [archive/CHANGELOG.md](archive/CHANGELOG.md) for full experiment history (v1-v13).

The core challenge remains graph compression: the paper achieves 0.891 Dice on ~1K nodes. Our best oracle is 0.90 on 172K nodes. Reducing nodes via contraction loses more in GINE learnability than it gains in compression.

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
