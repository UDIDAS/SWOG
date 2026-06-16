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
  NewSemirStage1 (2).ipynb  — Stage 1: band-flood coarsening + certified deletion

docs/
  SEMIR-MedicalImages.pdf   — SEMIR paper (MICCAI submission)
  SEMIR_Semantic_Minor_Ind.pdf — SEMIR supplementary

archive/                    — Previous experiments (two-stage protection, IG-JEPA, etc.)
```

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
