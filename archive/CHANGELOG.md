# SEMIR Experiment Changelog

Complete history of all approaches tried before pivoting to Luke's Stage 1 band-flood approach (2026-06-16).

## Timeline

### v1 — Initial notebook implementation
- **Commit**: `191c967` → `47a762d`
- **Approach**: First SEMIR reproduction attempt. Python-based edge contraction using scipy connected_components.
- **Problem**: Connected components allows transitive intensity drift — entire volume merges into one giant supernode.
- **Result**: No training possible.

### v2 — Fix attempt
- **Commit**: `47a762d` → `5b55064`
- **Files**: `semir_lits_v2_executed.ipynb`, `semir_lits_diagnostic.ipynb`
- **Results**: `semir_lits_v2/`
- **Approach**: Diagnostic investigation of the contraction problem. Identified that single-pass connected components is fundamentally wrong — need seed-based contraction.
- **Result**: Diagnostics only, no training Dice.

### v3 — Rust crate integration
- **Commit**: `5b55064` → `a732d44`
- **Files**: `rust_crate.ipynb`
- **Approach**: Integrated Luke's Rust `fastloops` crate with seed-based BFS contraction. Canonical voxel comparison prevents transitive drift.
- **Problem**: Produces ~1.7M supernodes (paper targets ~1K). BFS visits in coprime-step order, producing many single-voxel supernodes.
- **Result**: Oracle Dice 0.94 but too many nodes for training.

### v4 — Iterative consolidation
- **Commit**: `760d2f4` → `f1c94cc`
- **Approach**: Post-contraction consolidation — merge adjacent supernodes with similar canonical intensities. Aimed to reduce 1.7M → manageable count.
- **Problem**: Consolidation either doesn't merge enough (still 1M+) or over-merges (oracle drops to 0.47).
- **Result**: Best oracle 0.47 on consolidated graphs.

### v5 — Kruskal canonical-anchored contraction
- **Commit**: `f4690c4`
- **Files**: `semir_lits_full.py`
- **Results**: `semir_lits_v5/`
- **Params**: HU [0,200], ψ=3, α=15, no deletion
- **Approach**: Kruskal-like contraction sorting edges by intensity distance, merging smallest-first with canonical intensity checks. HU window [0,200] for better liver-tumor contrast.
- **Result**: Oracle 0.47, **val Dice 0.112** — first nonzero training result. But oracle ceiling too low.
- **Key insight**: HU [0,200] doubles contrast but [−50,250] became the default going forward.

### v6 — Parameter search at scale
- **Commit**: `53bd029`
- **Results**: `semir_lits_v6/`
- **Approach**: Systematic oracle-first parameter search across ψ and α values. Found ψ=2, α=5 gives oracle 0.94 but 2.2M supernodes. Stopped at oracle phase — no viable training candidate.
- **Key insight**: Oracle Dice is misleading at 2M nodes. High oracle just means "the tumor is somewhere in these 2M nodes" but the GNN can't find it.

### v7 — Clean restart with Luke's crate
- **Commit**: `926cdc4`
- **Files**: `semir_lits_v7.py`
- **Results**: `semir_lits_v7/`
- **Params**: HU [−50,250], ψ=2, α=5, no deletion, faces connectivity
- **Approach**: Clean reimplementation using Luke's exact Rust crate code. Added feature discrimination checks (Cohen's d), balanced logistic regression baseline, stopping criteria before GINE training.
- **Result**: Oracle 0.94, ~2.2M nodes. Stopped at oracle criteria — Cohen's d < 0.54, features degenerate at single-voxel resolution. No GINE training attempted.
- **Key insight**: Don't stack Rust crate changes. Cut edges stay in graph (cut_frac feature). Cohen's d is misleading at 2M nodes.

### v8 — SEMIR-JEPA (self-supervised)
- **Commit**: `7636098`
- **Files**: `run_semir_jepa.py`
- **Results**: `semir_jepa/`
- **Approach**: DINO ViT features extracted per CT slice, mapped to 3D supernodes. IG-JEPA self-supervised pre-training with topology-aware masking + VICReg. MLP probe for tumor classification.
- **Params**: ψ=3, α=15, 2-layer GNN encoder, hidden 128, 41-dim input (DINO features)
- **Result**: **val Dice 0.009**. DINO features improve discrimination (d=1.04) but don't solve the 2M-node compression problem. Class imbalance 209:1 overwhelms the probe.
- **Key insight**: Better features don't help when the graph is too large. Compression is the root cause, not feature quality.

### v9 — Pancreas exploration
- **Files**: `semir_pancreas_detections.ipynb`
- **Approach**: Explored MSD Task07 Pancreas dataset (1:1 organ-to-tumor mapping) as a simpler test case. Downloaded 281 volumes via rclone.
- **Result**: Diagnostics only. Same contraction problems as LiTS.

### v10 — Two-stage protected coarsening (v1)
- **Commit**: `d99261c`
- **Files**: `semir_lits_twostage.ipynb`
- **Results**: `semir_lits_twostage/`
- **Approach**: 
  1. Crop CT to liver ROI (bounding box + margin 32)
  2. Intensity-based protection mask (hypodense < mean − 0.5σ, dilated by 2)
  3. New `fastloops.merge_and_cut_protected()` Rust function — blocks merging across protected boundary, preserves protected supernodes during deletion
  4. GINE training on protected graphs
- **Key results**:
  - Without protection: default deletion kills 77% of tumor voxels, oracle 0.28
  - With protection: 0% tumor deletion, oracle 0.89, ~241K nodes
  - **val Dice 0.321**, test Dice 0.233
- **Key insight**: Protection works — tumor is preserved and oracle is high. But 241K nodes still too many for GINE. Training unstable.

### v11 — Two-stage v2 (tighter protection)
- **Files**: `semir_lits_twostage_v2.ipynb`
- **Results**: `semir_lits_twostage_v2/`
- **Approach**: Sweep 10 protection strategies varying threshold (0.5–1.5σ), dilation (0–2), and CC filter (0–50). Selected S2 (1.5σ, dilate=2, no CC filter).
- **Key results**:
  - S2: 146K nodes (vs 241K), recall 95%, oracle 0.90
  - **val Dice 0.476**, test Dice 0.325
  - 1.4x node reduction from v1
- **Key insight**: Tightening protection gives diminishing returns. Strategies that hit target range (10K-50K) kill too much tumor (recall < 83%). Fundamental tension between node count and tumor preservation.

### v12 — Compression grid search
- **Not committed** (exploratory)
- **Approach**: Tested grid of (ψ, delete_small) with S2 protection. ψ from 2→15, delete_small from 0→2000.
- **Key results**:
  - ψ=2, del=100: 198K nodes, oracle 0.94
  - ψ=10, del=100: 81K nodes, oracle 0.80
  - ψ=15, del=100: 51K nodes, oracle 0.73
  - delete_small has minimal effect — protected nodes dominate count
- **Key insight**: ψ is the real compression lever, not deletion. But higher ψ merges tumor with liver.

### v13 — IG-JEPA v2 (on protected graphs)
- **Files**: `semir_lits_jepa_v2.ipynb` (created, not executed)
- **Approach**: Planned IG-JEPA pre-training on Mode C protected graphs (241K nodes). GINE-based encoder with edge features, topology-aware masking, VICReg.
- **Result**: Deprioritized before execution. Decided features/SSL won't help if graph fidelity is the bottleneck.

---

## Pivot (2026-06-16)

Luke shared `NewSemirStage1 (2).ipynb` — a fundamentally different approach:
- **Learned intensity bands** via oracle-guided greedy sweep (not fixed ψ/α)
- **Band-flood contraction** using custom `fastloops_band` crate (not edge-distance based)
- **Certified deletion** with oracle budget (not threshold-based)
- **Oracle IoU 0.978** (vs our best 0.90)
- **Richer features**: moment blocks (additive, float64), log-ratio edge features (antisymmetric), shape eigendecomposition (linearity/planarity/sphericity)

Our two-stage protection approach capped at val Dice 0.48. The bottleneck was always graph construction fidelity, not the classifier or features. All work archived; fresh start building on Luke's Stage 1.

## Key Lessons Learned

1. **Connected components causes transitive drift** — seed-based contraction is mandatory
2. **Fixed intensity thresholds can't match learned bands** — oracle-guided sweep is the right approach
3. **Oracle Dice is misleading at 2M nodes** — high oracle just means tumor exists somewhere in the graph
4. **Cohen's d is misleading at extreme imbalance** — d=1.67 doesn't mean separable when positives are 1.3%
5. **Protection masks help but have a ceiling** — 0% tumor deletion but nodes stay too large
6. **Compression is the root cause** — better features, SSL, class weighting are all secondary to getting the graph right
7. **Deleted voxels must be reassigned** — without reassignment, Oracle Dice drops from 0.87 to 0.34
8. **HU window [−50, 250] is the default** — [0,200] doubles contrast but misses structures
9. **Cut edges stay in the graph** — they're marked with cut_frac feature, not removed
10. **Don't stack Rust crate changes** — test each change independently
11. **Post-hoc contraction hurts GINE** — selectively merging background supernodes (172K→52K) changes graph topology from grid-like to hub-and-spoke, GINE val Dice drops from 0.48 to 0.13
12. **SMBO works for parameter search** — Optuna TPE found good params in 40 trials (~1hr), better than manual grid search
13. **Luke's features ≈ our features** — linearity/planarity/sphericity vs elongation, +3 log-ratio edges. ~90% overlap. Features aren't the bottleneck.
14. **Luke's band-flood gives 0.978 oracle** — vs our 0.90. The gap is in graph construction quality, not compression or features.

### v14 — Stage 2 selective contraction + SMBO (2026-06-17)
- **Files**: `notebooks/stage2_selective_contraction.ipynb`
- **Results**: `results/stage2_selective/`
- **Approach**: SMBO (Optuna, 40 trials) optimizes 7 params (psi, alpha, std_mult, dilate_iter, bg_threshold, prot_threshold, n_rounds) against oracle Dice. Selective contraction merges background aggressively, protects tumor candidates, never crosses boundary.
- **SMBO found**: psi=9, alpha=83, std_mult=1.86, bg_threshold=5, prot_threshold=9, n_rounds=1 → 52K nodes, oracle 0.856
- **Result**: val Dice 0.13, test Dice 0.05 — much worse than v11 (val 0.48)
- **Key insight**: contraction changes topology in ways GINE can't handle. The 172K→52K compression breaks message passing patterns. GINE trained on original protected graphs (v11) significantly outperforms GINE on contracted graphs.
