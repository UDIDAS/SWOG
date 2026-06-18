# SEMIR: Semantic Minor-Induced Representation Learning

Graph-minor segmentation for 3D medical images. Reproducing the SEMIR paper's binary-tensor approach for LiTS liver tumor segmentation, with the goal of replacing GT-dependent CSV pipelines in the ACM MMKG VKG system.

## Goal

Replace AuSAM (SAM2.1 + DBSCAN) with SEMIR's learned graph minors in the ACM MMKG VKG pipeline. SEMIR derives tumor phenotypes (volume, compactness, elongation, intensity) directly from graph structure, eliminating the dependency on ground-truth CSV files that don't exist in real clinical data.

## How SEMIR Works (Simple Version)

SEMIR takes a 3D CT scan and **grows regions from seed points**. Each region expands outward until it hits a tissue boundary — where intensity changes sharply. This produces ~1,000 supernodes, each covering a meaningful tissue region (a chunk of liver, a piece of tumor, etc).

A graph neural network (GINE) then classifies each supernode as tumor or background. Predictions are lifted back to voxels for the final segmentation mask.

The key is a **few-shot boundary search**: using just 5 labeled examples, the algorithm finds the exact intensity threshold where supernode boundaries align with real tissue boundaries. When they align, each supernode contains pure tumor or pure background — making classification easy.

Compare this to our earlier approach: chopping the scan into millions of individual voxels, then trying to filter out the useless ones. That gave us millions of tiny pieces with no relationship to tissue structure.

## Pipeline Architecture

```
Raw CT volume (512x512xD, ~10M voxels)
    |
    v
[1] Liver crop (bbox + margin, reduces ~50%)
    |
    v
[2] Binary tensor construction (Rust, merge_and_cut)
    — Canonical-intensity flood-fill from seed voxels
    — Expands until |seed_intensity - neighbor_intensity| > psi
    — Produces ~1K-7K boundary-aligned supernodes
    |
    v
[3] Few-shot parameter search (5 labeled examples)
    — Searches psi, alpha, beta over Cartesian grid
    — Maximizes boundary Dice: supernode edges align with GT tumor edge
    — Finds params where supernodes snap to tissue boundaries
    |
    v
[4] Voxel reassignment (distance_transform_edt)
    — Deleted voxels reassigned to nearest surviving supernode
    — Guarantees every voxel maps to exactly one supernode
    |
    v
[5] Feature extraction (7 node + 4 edge features, paper-matched)
    — Node: volume, boundary_length, compactness, elongation,
            dominant_axis, mean_intensity, intensity_std
    — Edge: log_volume_ratio, intensity_diff, centroid_distance, orientation_cosine
    |
    v
[6] GINE classifier (3-layer, hidden=128)
    — Binary: tumor vs background per supernode
    — Trained on full dataset split with weighted cross-entropy
    |
    v
[7] Voxel lifting (bijective LUT, O(N))
    — Supernode predictions mapped to voxel mask via label volume
    — Exact — no interpolation or boundary artifacts
```

## Repository Structure

```
scripts/
  v18_paper_exact.py             -- Paper-exact SEMIR (current, running)
  v17_semir.py                   -- merge_and_cut + boundary search (predecessor)
  v16_evaluate.py                -- v16b: band-flood + protection (val Dice 0.59)
  sweep_protection.py            -- Protection tightness analysis
  test_merge_and_cut.py          -- Binary tensor parameter sweep
  test_contraction.py            -- Canonical-intensity contraction test

notebooks/
  v16_luke_pipeline.ipynb        -- v16: Luke's band-flood + GINE
  v15_protected_subgraph.ipynb   -- v15: merge_and_cut + protection mask
  path_a_gine_stage1.ipynb       -- Path A: identity bands + GINE
  stage2_selective_contraction.ipynb -- v14: post-hoc contraction (failed)

fastloops/                       -- Rust crate: merge_and_cut (paper's binary tensor)
fastloops_band/                  -- Rust crate: band_build (Luke's band-flood)

docs/
  SEMIR-MedicalImages.pdf        -- SEMIR paper
  SWOG Surgery Presentation.pptx -- VKG presentation

results/                         -- Per-version results (JSON + model checkpoints)
archive/                         -- v1-v13 experiments (see archive/CHANGELOG.md)
```

## Experiment History

| Version | Approach | Nodes | Oracle | Val Dice | Key Finding |
|---------|----------|-------|--------|----------|-------------|
| v11 | merge_and_cut (no deletion) | 172K | 0.90 | 0.48 | Baseline |
| v14 | Post-hoc contraction | 52K | 0.86 | 0.13 | Contraction kills graph topology |
| v15 | merge_and_cut + protection | 164K | 0.93 | 0.50 | Protection helps oracle, not Dice |
| v16b | **band_flood + protection** | 240K | 0.98 | **0.59** | Luke's oracle boost, still too large |
| v17 | merge_and_cut + boundary search | ~1-7K | TBD | TBD | Correct algorithm, wrong search grid |
| v18 | **paper-exact reproduction** | ~1K | TBD | TBD | All 6 params searched, all 131 vols, 5 runs |

### Key Findings

1. **Graph size is the bottleneck, not oracle.** v16b has oracle 0.98 but val Dice 0.59 because 240K-node graphs are too large for 3-layer GINE message passing.

2. **merge_and_cut and band_build share the same core algorithm** — binary tensor, flood-fill, coprime iterator, moment accumulation. The ONLY difference is the merge predicate: `merge_and_cut` uses `|I_seed - I_neighbor| ≤ ψ` (continuous threshold, paper-exact), `band_build` uses `band_of[I_seed] == band_of[I_neighbor]` (discrete learned lookup). Luke's `band_build` is a generalization with non-uniform boundaries, which gave better oracle (0.978 vs 0.93) but with 196 narrow bands it fragmented into 1.6M single-voxel supernodes.

3. **merge_and_cut IS the paper's binary tensor.** It does canonical-intensity flood-fill — compares every neighbor against the seed voxel's intensity, preventing transitive drift. This naturally produces ~1K large, boundary-aligned supernodes.

4. **A hybrid approach is promising:** use `band_build` with a coarser learned dictionary (~20-30 bands instead of 196) to get both non-uniform boundaries AND actual merging. This would combine Luke's oracle advantage with the paper's compression.

4. **Post-hoc contraction breaks topology.** Merging background supernodes after construction (v14) changes grid-like graphs to hub-and-spoke, killing GINE performance.

5. **Deleted voxels must be reassigned.** Dropping them as label=-1 collapses oracle from 0.87 to 0.34. Reassignment via `distance_transform_edt` to nearest survivor preserves full tumor recall.

6. **Protection masks don't scale.** Intensity-based protection (liver_mean - Nσ) can't separate tumor from background because their HU ranges overlap. Tightening protection loses tumor voxels; loosening keeps too many nodes.

7. **Boundary alignment is the real objective.** The paper's few-shot search maximizes boundary Dice — overlap between supernode edges and GT tissue edges. This is what makes ~1K nodes meaningful: each covers a coherent tissue region.

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
