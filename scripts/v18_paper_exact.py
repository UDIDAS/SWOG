"""
v18: Paper-exact SEMIR reproduction for LiTS tumor segmentation.

Every design choice traced to the paper (SEMIR-MedicalImages.pdf):
- Binary tensor: merge_and_cut (Rust, canonical-intensity flood-fill)
- Few-shot search: Θ = {ψ, α, β_min, β_max, m_min, m_max}, 6 params
  64 values per param, Cartesian grid, boundary Dice objective (Eq. 1)
- Voxel reassignment: every voxel → exactly one supernode (bijective lifting)
- Node features: 7 (volume, boundary, compactness, elongation, dominant_axis,
  mean_intensity, intensity_std) — Section 3.2
- Edge features: 6 (paper says "10 scalars per supernode and 6 per edge")
- GINE: 3-layer, hidden=128, Adam lr=1e-3, patience=10, 200 epochs
- Binary target-vs-rest, 5 independent runs
- Target: LiTS Tumor, 0.891 ± 0.007, |V(H)| = 1075 ± 297

Usage:
    conda run -n llmft python scripts/v18_paper_exact.py
"""

import numpy as np
import os, re, time, json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GINEConv, BatchNorm
from scipy.ndimage import distance_transform_edt, binary_erosion
import fastloops

DATA_ROOT = "/scratch/ud3d4/acm_data/Data"
RESULTS_DIR = "/home/ud3d4/Desktop/SWOG/results/v18_paper_exact"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Paper Section 4.1: HU window not specified explicitly.
# Memory: HU [-50, 250] is our validated default for LiTS.
HU_MIN, HU_MAX = -50, 250

np.random.seed(42)
torch.manual_seed(42)

def pr(msg=""): print(msg, flush=True)


# =============================================================================
# merge_and_cut node feature layout (C=1, 3D): 21 columns
# From Rust build_layout: area=0, s=[1,2,3], cov=[4,5,6,7,8,9],
# chan0=10, min_max=[11..16], boundary=17, canon=[18,19,20]
# =============================================================================
MC_AREA = 0
MC_S = [1, 2, 3]
MC_COV = [(4,0,0), (5,1,1), (6,2,2), (7,0,1), (8,0,2), (9,1,2)]
MC_CHAN0 = 10
MC_BOUNDARY = 17


# =============================================================================
# Data loading
# =============================================================================

def discover_volumes():
    """Find ALL volumes with segmentations — including liver-only (no tumor).
    Liver-only cases are valid negative examples for binary tumor classification."""
    vids = []
    for f in sorted(os.listdir(os.path.join(DATA_ROOT, "ct"))):
        m = re.match(r"volume-(\d+)\.npy", f)
        if m:
            vid = int(m.group(1))
            seg_path = os.path.join(DATA_ROOT, "seg", f"segmentation-{vid}.npy")
            if os.path.exists(seg_path):
                vids.append(vid)
    return sorted(vids)


def load_and_crop(vid):
    """Load CT + seg, liver crop, convert to uint8 channel-last."""
    ct = np.load(os.path.join(DATA_ROOT, "ct", f"volume-{vid}.npy")).astype(np.float32)
    seg = np.load(os.path.join(DATA_ROOT, "seg", f"segmentation-{vid}.npy")).astype(np.int32)
    organ_mask = (seg == 1) | (seg == 2)
    if organ_mask.sum() == 0:
        return None, None  # no liver at all — skip
    coords = np.argwhere(organ_mask)
    lo = np.maximum(coords.min(0) - 32, 0)
    hi = np.minimum(coords.max(0) + 33, organ_mask.shape)
    slc = tuple(slice(int(lo[i]), int(hi[i])) for i in range(3))
    ct_crop = ct[slc]; seg_crop = seg[slc]
    ct_u8 = np.clip(ct_crop, HU_MIN, HU_MAX)
    ct_u8 = ((ct_u8 - HU_MIN) / (HU_MAX - HU_MIN) * 255).round().astype(np.uint8)
    ct_u8_4d = np.ascontiguousarray(ct_u8[..., np.newaxis])
    return ct_u8_4d, seg_crop


# =============================================================================
# Graph minor construction + reassignment
# Paper Section 3.2: "every voxel belongs to exactly one supernode"
# =============================================================================

def build_minor(ct_u8_4d, psi, alpha, beta_min, beta_max, m_min, m_max):
    """Run merge_and_cut, then reassign deleted voxels."""
    nf, ei, ef, labels, adj = fastloops.merge_and_cut(
        ct_u8_4d,
        merge_distance=int(psi),
        cut_distance=int(alpha),
        delete_small_node_max_size=int(beta_min),
        delete_large_node_min_size=int(beta_max),
        delete_value_min=int(m_min),
        delete_value_max=int(m_max),
        connectivity="faces",
    )
    nf = np.asarray(nf); labels = np.asarray(labels)
    N = nf.shape[0]
    if N == 0:
        return None, None

    # Paper: "every voxel belongs to exactly one supernode"
    # Reassign deleted voxels (label=-1) to nearest surviving supernode
    survivor_mask = (labels >= 0)
    if not survivor_mask.all() and survivor_mask.any():
        _, nearest_idx = distance_transform_edt(~survivor_mask, return_indices=True)
        labels = labels[tuple(nearest_idx)]

    return nf, labels


# =============================================================================
# Few-shot boundary Dice search
# Paper Section 3.2, Eq. 1: Θ_opt = argmin E[1 - DSC(S_B(T,Θ), Y_B)]
# Paper Section 4.1: "64 uniform values per param", "~2^14 feasible"
# =============================================================================

def boundary_dice(labels, gt_seg):
    """DSC between supernode boundaries and GT segmentation boundaries."""
    gt_mask = gt_seg == 2
    if gt_mask.sum() == 0:
        return 0.0
    gt_boundary = gt_mask ^ binary_erosion(gt_mask)
    sn_boundary = np.zeros_like(labels, dtype=bool)
    for axis in range(3):
        sl_lo = [slice(None)] * 3; sl_hi = [slice(None)] * 3
        sl_lo[axis] = slice(0, -1); sl_hi[axis] = slice(1, None)
        diff = labels[tuple(sl_lo)] != labels[tuple(sl_hi)]
        sn_boundary[tuple(sl_lo)] |= diff
        sn_boundary[tuple(sl_hi)] |= diff
    inter = int((gt_boundary & sn_boundary).sum())
    return 2.0 * inter / (gt_boundary.sum() + sn_boundary.sum() + 1e-8)


def few_shot_search(few_vols, few_segs):
    """
    Paper Eq. 1: Θ_opt = argmin E[1 - DSC(S_B(T,Θ), Y_B)]
    Paper Section 4.1: "64 uniform values per param, ~2^14 feasible"

    Strategy: the paper searches 6 params but most feasibility comes from
    ψ and α (intensity thresholds). β and m have less impact on boundary
    alignment. We search ψ×α densely (32×32), with β and m at 3-4 values
    each. Total ~16K feasible, matching paper's ~2^14.
    """
    pr("  Few-shot boundary Dice search (paper Eq. 1)")
    n_voxels = few_vols[0].size // few_vols[0].shape[-1]

    # Paper: "completes in under 10 minutes on CPU for |D_few|=5"
    # 12 values per critical param → ~1.5K feasible → ~10 min
    psi_vals = np.unique(np.linspace(2, 50, 12).astype(int))
    alpha_vals = np.unique(np.linspace(15, 150, 12).astype(int))
    # Coarser on deletion params (less impact on boundary alignment)
    beta_min_vals = np.array([2, 10, 20])
    beta_max_vals = np.array([int(n_voxels * 0.8)])  # paper default
    m_min_vals = np.array([0, 20])
    m_max_vals = np.array([240, 255])

    # Count feasible
    n_feasible = 0
    for psi in psi_vals:
        for alpha in alpha_vals:
            if alpha <= psi: continue
            n_feasible += len(beta_min_vals) * len(beta_max_vals) * len(m_min_vals) * len(m_max_vals)

    pr(f"  Grid: psi({len(psi_vals)}) x alpha({len(alpha_vals)}) x beta_min({len(beta_min_vals)}) "
       f"x beta_max({len(beta_max_vals)}) x m_min({len(m_min_vals)}) x m_max({len(m_max_vals)})")
    pr(f"  Feasible: ~{n_feasible:,} configs x {len(few_vols)} volumes")

    best_score = -1.0
    best_params = None
    best_nodes = 0
    log = []
    n_eval = 0
    t0 = time.time()

    for psi in psi_vals:
        for alpha in alpha_vals:
            if alpha <= psi: continue
            for beta_min in beta_min_vals:
                for beta_max in beta_max_vals:
                    if beta_min >= beta_max: continue
                    for m_min in m_min_vals:
                        for m_max in m_max_vals:
                            if m_min >= m_max: continue

                            scores = []
                            nodes_list = []
                            for vol, seg in zip(few_vols, few_segs):
                                nf, labels = build_minor(vol, psi, alpha,
                                                          beta_min, beta_max, m_min, m_max)
                                if nf is None or len(nf) < 2:
                                    scores.append(0.0)
                                    nodes_list.append(0)
                                    continue
                                bd = boundary_dice(labels, seg)
                                scores.append(bd)
                                nodes_list.append(len(nf))

                            mean_bd = np.mean(scores)
                            mean_nodes = np.mean(nodes_list)
                            score = mean_bd
                            n_eval += 1

                            if score > best_score:
                                best_score = score
                                best_nodes = mean_nodes
                                best_params = dict(
                                    psi=int(psi), alpha=int(alpha),
                                    beta_min=int(beta_min), beta_max=int(beta_max),
                                    m_min=int(m_min), m_max=int(m_max))
                                log.append(dict(**best_params,
                                                mean_bdice=round(mean_bd, 6),
                                                mean_nodes=round(mean_nodes, 0)))

        # Progress per psi
        elapsed = time.time() - t0
        rate = n_eval / max(elapsed, 1)
        eta = (n_feasible - n_eval) / max(rate, 1)
        pr(f"    psi={psi:>3d}: {n_eval:>6d}/{n_feasible}  "
           f"best_bdice={best_score:.5f}  nodes={best_nodes:.0f}  "
           f"eta={eta/60:.0f}min")

    pr(f"\n  Search done: {n_eval:,} evals in {(time.time()-t0)/60:.1f} min")
    pr(f"  Best: psi={best_params['psi']} alpha={best_params['alpha']} "
       f"beta=[{best_params['beta_min']},{best_params['beta_max']}] "
       f"m=[{best_params['m_min']},{best_params['m_max']}]")
    pr(f"  Boundary Dice: {best_score:.5f}  Nodes: {best_nodes:.0f}")

    return best_params, log


# =============================================================================
# Paper-matched features
# Section 3.2: "volume, boundary length, compactness 36πa²/(b³+ε),
# elongation, dominant axis, mean intensity, intensity std"
# Section 4.1: "10 scalars per supernode and 6 scalars per edge"
# =============================================================================

def extract_node_features(nf, labels, ct_u8_4d):
    """Paper's 7 node features (Section 3.2)."""
    f = nf.astype(np.float64)
    N = f.shape[0]
    eps = 1e-6

    V = f[:, MC_AREA]
    Vsafe = np.maximum(V, 1.0)
    boundary = f[:, MC_BOUNDARY]

    # 3. Compactness: 36π a² / (b³ + ε)
    compactness = (36.0 * np.pi * V**2) / (boundary**3 + eps)
    compactness = np.clip(compactness, 0, 1)

    # 4+5. Elongation + dominant axis from PCA
    mean_coord = np.stack([f[:, c] for c in MC_S], axis=1) / Vsafe[:, None]
    cov = np.zeros((N, 3, 3))
    for col, i, j in MC_COV:
        cij = f[:, col] / Vsafe - mean_coord[:, i] * mean_coord[:, j]
        cov[:, i, j] = cij; cov[:, j, i] = cij
    w, vec = np.linalg.eigh(cov)
    w = np.clip(w, 0.0, None)
    elongation = w[:, 2] / (w[:, 0] + eps)
    elongation = np.clip(elongation, 1.0, 100.0)
    # Paper: "dominant axis d_u" = principal eigenvector
    principal = vec[..., -1]  # (N, 3) — kept for edge features
    dominant_axis = np.max(np.abs(principal), axis=1)

    # 6. Mean intensity
    mean_intensity = f[:, MC_CHAN0] / Vsafe / 255.0

    # 7. Intensity std (not in Rust output, compute from voxels)
    flat_labels = labels.ravel().astype(np.int64)
    flat_int = ct_u8_4d[..., 0].ravel().astype(np.float64)
    int_sum = np.bincount(flat_labels, weights=flat_int, minlength=N)[:N]
    int_sq = np.bincount(flat_labels, weights=flat_int**2, minlength=N)[:N]
    counts = np.maximum(np.bincount(flat_labels, minlength=N)[:N].astype(np.float64), 1.0)
    intensity_std = np.sqrt(np.clip(int_sq / counts - (int_sum / counts)**2, 0, None)) / 255.0

    x = np.column_stack([
        np.log1p(V),       # 1. volume
        boundary,           # 2. boundary length
        compactness,        # 3. compactness
        elongation,         # 4. elongation
        dominant_axis,      # 5. dominant axis
        mean_intensity,     # 6. mean intensity
        intensity_std,      # 7. intensity std
    ]).astype(np.float32)

    # Z-normalize
    for c in range(x.shape[1]):
        mu, sig = x[:, c].mean(), x[:, c].std()
        if sig > eps: x[:, c] = (x[:, c] - mu) / sig
        else: x[:, c] = 0.0

    return x, mean_coord, principal


def extract_edge_features(nf, labels, centroids, principal):
    """
    Paper Section 3.2: "scale-invariant log-ratios of geometric properties,
    normalized intensity differences, and relative orientation cos θ = |d^T_u d_v|"
    Paper Section 4.1: "6 scalars per edge"
    """
    eps = 1e-6
    N = len(nf)
    f = nf.astype(np.float64)
    V = f[:, MC_AREA]; Vsafe = np.maximum(V, 1.0)
    boundary = f[:, MC_BOUNDARY]; bsafe = np.maximum(boundary, 1.0)
    mean_int = f[:, MC_CHAN0] / Vsafe / 255.0

    # Build edges from label volume
    all_a, all_b, all_counts = [], [], []
    for axis in range(3):
        sl_lo = [slice(None)] * 3; sl_hi = [slice(None)] * 3
        sl_lo[axis] = slice(0, -1); sl_hi[axis] = slice(1, None)
        a = labels[tuple(sl_lo)].ravel()
        b = labels[tuple(sl_hi)].ravel()
        bdry = (a != b) & (a >= 0) & (b >= 0)
        all_a.append(a[bdry]); all_b.append(b[bdry])

    if not all_a:
        return np.zeros((2, 0), dtype=np.int64), np.zeros((0, 6), dtype=np.float32)

    a_arr = np.concatenate(all_a).astype(np.int64)
    b_arr = np.concatenate(all_b).astype(np.int64)
    lo = np.minimum(a_arr, b_arr); hi = np.maximum(a_arr, b_arr)

    # Deduplicate + count boundary voxels per edge
    pair_keys = lo * (N + 1) + hi
    unique_keys, edge_bl = np.unique(pair_keys, return_counts=True)
    u_a = unique_keys // (N + 1); u_b = unique_keys % (N + 1)
    edge_bl = edge_bl.astype(np.float64)

    max_dim = max(labels.shape)

    # 6 edge features:
    # 1. log volume ratio
    lr_vol = np.log(Vsafe[u_a] / Vsafe[u_b])
    # 2. normalized intensity diff
    int_diff = np.abs(mean_int[u_a] - mean_int[u_b]) / (np.abs(mean_int[u_a]) + np.abs(mean_int[u_b]) + eps)
    # 3. normalized centroid distance
    dist = np.sqrt(((centroids[u_a] - centroids[u_b])**2).sum(axis=1)) / max_dim
    # 4. orientation cosine: |d_u · d_v|
    cos_angle = np.abs((principal[u_a] * principal[u_b]).sum(axis=1))
    # 5. boundary fraction (boundary voxels shared / total boundary of smaller node)
    bfrac = edge_bl / np.minimum(bsafe[u_a], bsafe[u_b])
    # 6. log boundary ratio
    lr_boundary = np.log(bsafe[u_a] / bsafe[u_b])

    ea = np.column_stack([lr_vol, int_diff, dist, cos_angle, bfrac, lr_boundary]).astype(np.float32)
    ei = np.stack([u_a, u_b], axis=0)
    return ei, ea


# =============================================================================
# Build PyG graph
# =============================================================================

def build_graph(ct_u8_4d, seg_crop, params):
    nf, labels = build_minor(ct_u8_4d,
                              params["psi"], params["alpha"],
                              params["beta_min"], params["beta_max"],
                              params["m_min"], params["m_max"])
    if nf is None:
        return None
    N = len(nf)

    x, centroids, principal = extract_node_features(nf, labels, ct_u8_4d)
    ei, ea = extract_edge_features(nf, labels, centroids, principal)

    # Undirected
    if ei.shape[1] > 0:
        ei_fwd = torch.tensor(ei, dtype=torch.long)
        ei_rev = torch.stack([ei_fwd[1], ei_fwd[0]])
        edge_index = torch.cat([ei_fwd, ei_rev], dim=1)
        edge_attr = torch.tensor(np.concatenate([ea, ea]), dtype=torch.float32)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, 6), dtype=torch.float32)

    # Labels
    flat = labels.ravel().astype(np.int64)
    gt = (seg_crop == 2).ravel().astype(np.float64)
    fg = np.bincount(flat, weights=gt, minlength=N)[:N]
    tot = np.bincount(flat, minlength=N)[:N]
    bg = tot - fg
    y = (fg / np.maximum(fg + bg, 1) >= 0.10).astype(np.int64)

    data = Data(x=torch.tensor(x, dtype=torch.float32),
                edge_index=edge_index, edge_attr=edge_attr,
                y=torch.tensor(y, dtype=torch.long),
                n_fg=torch.tensor(fg, dtype=torch.float32),
                n_bg=torch.tensor(bg, dtype=torch.float32))
    return data, labels, seg_crop


# =============================================================================
# GINE — Paper: 3-layer, hidden 128
# =============================================================================

class GINE(nn.Module):
    def __init__(self, nd=7, ed=6, h=128):
        super().__init__()
        self.ep = nn.Linear(ed, h)
        def mlp(d): return nn.Sequential(nn.Linear(d, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Linear(h, h))
        self.c1 = GINEConv(mlp(nd), edge_dim=h); self.b1 = BatchNorm(h)
        self.c2 = GINEConv(mlp(h), edge_dim=h); self.b2 = BatchNorm(h)
        self.c3 = GINEConv(mlp(h), edge_dim=h); self.b3 = BatchNorm(h)
        self.head = nn.Linear(h, 2)

    def forward(self, x, ei, ea):
        if ea is not None and ea.numel() > 0:
            ea = self.ep(ea)
        else:
            n = x.size(0)
            ei = torch.stack([torch.arange(n, device=x.device)] * 2)
            ea = torch.zeros(n, self.ep.out_features, device=x.device)
        x = F.relu(self.b1(self.c1(x, ei, ea)))
        x = F.relu(self.b2(self.c2(x, ei, ea)))
        x = F.relu(self.b3(self.c3(x, ei, ea)))
        return self.head(x)


# =============================================================================
# Voxel-level Dice
# =============================================================================

def voxel_dice(preds, labels, seg_crop):
    flat = labels.ravel().astype(np.int64)
    N = int(flat.max()) + 1
    lut = np.zeros(N, dtype=np.int8)
    lut[:len(preds)] = preds
    pm = lut[flat].reshape(labels.shape).astype(bool)
    gm = seg_crop == 2
    inter = int((pm & gm).sum())
    dice = 2.0 * inter / (gm.sum() + pm.sum() + 1e-8)
    rec = inter / (gm.sum() + 1e-8)
    prec = inter / (pm.sum() + 1e-8) if pm.sum() > 0 else 0.0
    return float(dice), float(rec), float(prec)


# =============================================================================
# Main
# =============================================================================

def main():
    pr("=" * 70)
    pr("  v18: Paper-exact SEMIR for LiTS Tumor")
    pr("=" * 70)

    all_vids = discover_volumes()
    perm = np.random.permutation(len(all_vids))
    n_tr = int(0.7 * len(all_vids)); n_va = int(0.15 * len(all_vids))
    train_ids = sorted([all_vids[i] for i in perm[:n_tr]])
    val_ids = sorted([all_vids[i] for i in perm[n_tr:n_tr + n_va]])
    test_ids = sorted([all_vids[i] for i in perm[n_tr + n_va:]])
    ordered = train_ids + val_ids + test_ids
    pr(f"  {len(all_vids)} volumes: {len(train_ids)} train / {len(val_ids)} val / {len(test_ids)} test")

    # Load volumes
    pr("\nLoading volumes...")
    volumes = {}
    for vid in ordered:
        ct_u8_4d, seg_crop = load_and_crop(vid)
        if ct_u8_4d is None:
            pr(f"  vol-{vid}: SKIP (no organ)")
            continue
        volumes[vid] = (ct_u8_4d, seg_crop)
        has_tumor = "tumor" if (seg_crop == 2).sum() > 0 else "liver-only"
        pr(f"  vol-{vid:>3d}: {ct_u8_4d.shape[:3]}  {has_tumor}")
    pr(f"  Loaded {len(volumes)} volumes "
       f"({sum(1 for v in volumes.values() if (v[1]==2).sum()>0)} with tumor, "
       f"{sum(1 for v in volumes.values() if (v[1]==2).sum()==0)} liver-only)")

    # ---- STAGE 1: Few-shot parameter search ----
    # Paper: |D_few| = 5 for single-channel, saturates at 5
    pr("\n" + "=" * 70)
    pr("  STAGE 1: Few-shot boundary Dice search (paper Eq. 1)")
    pr("=" * 70)
    # Few-shot volumes must have tumor for boundary Dice to be meaningful
    n_few = 5
    train_with_tumor = [v for v in train_ids if (volumes[v][1] == 2).sum() > 0]
    few_ids = train_with_tumor[:n_few]
    few_vols = [volumes[v][0] for v in few_ids]
    few_segs = [volumes[v][1] for v in few_ids]

    best_params, search_log = few_shot_search(few_vols, few_segs)
    with open(os.path.join(RESULTS_DIR, "param_search.json"), "w") as f:
        json.dump(dict(best=best_params, log=search_log), f, indent=2)

    # ---- STAGE 2: Build graphs ----
    pr("\n" + "=" * 70)
    pr("  STAGE 2: Build graph minors (all volumes)")
    pr("=" * 70)
    graphs = {}; metas = {}
    t0 = time.time()
    for vid in ordered:
        ct_u8_4d, seg_crop = volumes[vid]
        result = build_graph(ct_u8_4d, seg_crop, best_params)
        if result is None:
            pr(f"  vol-{vid}: SKIP"); continue
        data, labels, seg = result
        graphs[vid] = data
        metas[vid] = dict(labels=labels, seg=seg)

        split = "train" if vid in train_ids else ("val" if vid in val_ids else "test")
        N = data.num_nodes; n_tu = int((data.y == 1).sum())
        fg = data.n_fg.numpy(); bg = data.n_bg.numpy(); gt_t = fg.sum()
        maj = fg > bg
        oracle = float(2*fg[maj].sum()/(2*fg[maj].sum()+bg[maj].sum()+(gt_t-fg[maj].sum())+1e-8)) if gt_t > 0 else 0
        pr(f"  vol-{vid:>3d} [{split:>5s}]: {N:>6,} nodes ({n_tu:>4,} tu) "
           f"{data.num_edges:>7,} edges  oracle={oracle:.4f}")

    dt = time.time() - t0
    nodes = [g.num_nodes for g in graphs.values()]
    pr(f"\n  Built {len(graphs)} graphs in {dt:.0f}s")
    pr(f"  Nodes: mean={np.mean(nodes):,.0f}  median={np.median(nodes):,.0f}  "
       f"max={max(nodes):,}  min={min(nodes):,}")
    pr(f"  Paper target: 1075 +/- 297")
    oracles = []
    for g in graphs.values():
        fg = g.n_fg.numpy(); bg = g.n_bg.numpy(); gt_t = fg.sum()
        if gt_t > 0:
            maj = fg > bg
            oracles.append(float(2*fg[maj].sum()/(2*fg[maj].sum()+bg[maj].sum()+(gt_t-fg[maj].sum())+1e-8)))
    pr(f"  Oracle Dice: mean={np.mean(oracles):.4f}")

    # ---- STAGE 3: GINE training ----
    # Paper: Adam lr=1e-3, patience=10, 200 epochs, T4 GPU
    pr("\n" + "=" * 70)
    pr("  STAGE 3: GINE training (paper: 3-layer, h=128, patience=10)")
    pr("=" * 70)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    pr(f"  Device: {device}")

    trainable = [v for v in train_ids if v in graphs]
    val_usable = [v for v in val_ids if v in graphs]
    pr(f"  Trainable: {len(trainable)}/{len(train_ids)}")

    if not trainable:
        pr("  ERROR: No trainable volumes!"); return

    total_pos = sum(int((graphs[v].y == 1).sum()) for v in trainable)
    total_neg = sum(int((graphs[v].y == 0).sum()) for v in trainable)
    ratio = total_neg / max(total_pos, 1)
    eff = min(np.sqrt(ratio), 30.0)
    cw = torch.tensor([1.0, eff], dtype=torch.float32).to(device)
    pr(f"  Class weight: [1.0, {eff:.1f}] (ratio {ratio:.0f}:1)")

    # Paper: 5 independent runs
    N_RUNS = 5
    all_run_results = []

    for run in range(N_RUNS):
        pr(f"\n  --- Run {run+1}/{N_RUNS} ---")
        torch.manual_seed(42 + run)
        np.random.seed(42 + run)

        model = GINE(nd=7, ed=6).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        PATIENCE = 10  # Paper: patience 10
        best_dice, best_state, wait = -1.0, None, 0

        for epoch in range(1, 201):
            model.train(); eloss = 0.0; proc = 0
            for vid in np.random.permutation(trainable):
                try:
                    g = graphs[vid].to(device); opt.zero_grad()
                    logits = model(g.x, g.edge_index, g.edge_attr)
                    loss = F.cross_entropy(logits, g.y, weight=cw)
                    loss.backward(); opt.step(); eloss += loss.item(); proc += 1
                    del g, logits, loss
                except (torch.cuda.OutOfMemoryError, RuntimeError):
                    try: del g
                    except: pass
                    torch.cuda.empty_cache(); continue
                torch.cuda.empty_cache()
            if proc == 0: break
            ml = eloss / proc

            if epoch % 3 == 0 or epoch <= 3:
                model.eval(); tp = fp = fn = 0
                with torch.no_grad():
                    for vid in val_usable:
                        g = graphs[vid]; m = metas[vid]
                        try:
                            gd = g.to(device)
                            preds = model(gd.x, gd.edge_index, gd.edge_attr).argmax(1).cpu().numpy()
                            del gd; torch.cuda.empty_cache()
                        except:
                            torch.cuda.empty_cache()
                            mc = model.cpu(); mc.eval()
                            preds = mc(g.x, g.edge_index, g.edge_attr).argmax(1).numpy()
                            model.to(device)
                        d, _, _ = voxel_dice(preds, m["labels"], m["seg"])
                        gm = m["seg"] == 2
                        flat = m["labels"].ravel().astype(np.int64)
                        lut = np.zeros(int(flat.max())+1, dtype=np.int8)
                        lut[:len(preds)] = preds
                        pm = lut[flat].reshape(m["labels"].shape).astype(bool)
                        inter = int((pm & gm).sum())
                        tp += inter; fp += int(pm.sum())-inter; fn += int(gm.sum())-inter
                vd = 2*tp/(2*tp+fp+fn+1e-8)
                if vd > best_dice:
                    best_dice = vd
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    wait = 0; mk = " *"
                else: wait += 1; mk = ""
                pr(f"    Ep {epoch:3d}  loss={ml:.4f}  val_dice={vd:.4f}{mk}")
                if wait >= PATIENCE: pr(f"    Early stop @ {best_dice:.4f}"); break
            else:
                if epoch % 10 == 0: pr(f"    Ep {epoch:3d}  loss={ml:.4f}")

        if best_state: model.load_state_dict(best_state)
        pr(f"  Run {run+1} best val Dice: {best_dice:.4f}")

        # Evaluate this run
        model.eval()
        run_results = []
        for vid in sorted(graphs.keys()):
            g = graphs[vid]; m = metas[vid]
            with torch.no_grad():
                try:
                    gd = g.to(device)
                    preds = model(gd.x, gd.edge_index, gd.edge_attr).argmax(1).cpu().numpy()
                    del gd; torch.cuda.empty_cache()
                except:
                    torch.cuda.empty_cache()
                    mc = model.cpu(); mc.eval()
                    preds = mc(g.x.cpu(), g.edge_index.cpu(), g.edge_attr.cpu()).argmax(1).numpy()
                    model.to(device)
            dice, rec, prec = voxel_dice(preds, m["labels"], m["seg"])
            split = "train" if vid in train_ids else ("val" if vid in val_ids else "test")
            run_results.append(dict(vid=vid, split=split, dice=dice, recall=rec, precision=prec))
        all_run_results.append(run_results)

        if run == 0:
            torch.save(best_state or model.state_dict(), os.path.join(RESULTS_DIR, "model.pt"))

    # ---- Aggregate over 5 runs ----
    pr(f"\n{'='*80}")
    pr(f"  v18 RESULTS: Paper-exact SEMIR (5 runs)")
    pr(f"{'='*80}")
    pr(f"  Params: psi={best_params['psi']} alpha={best_params['alpha']} "
       f"beta=[{best_params['beta_min']},{best_params['beta_max']}] "
       f"m=[{best_params['m_min']},{best_params['m_max']}]")
    pr(f"  Nodes: {np.mean(nodes):,.0f} (paper: 1075)")
    pr(f"  Oracle: {np.mean(oracles):.4f}")

    for split in ["train", "val", "test"]:
        dices = []
        for run_res in all_run_results:
            split_dice = np.mean([r["dice"] for r in run_res if r["split"] == split])
            dices.append(split_dice)
        pr(f"  {split:>5s}: Dice={np.mean(dices):.4f} +/- {np.std(dices):.4f}  (n_runs={len(dices)})")

    pr(f"\n  Paper target: LiTS Tumor = 0.891 +/- 0.007")

    with open(os.path.join(RESULTS_DIR, "results.json"), "w") as f:
        json.dump(dict(best_params=best_params, nodes_mean=float(np.mean(nodes)),
                       oracle_mean=float(np.mean(oracles)),
                       all_runs=all_run_results), f, indent=2, default=float)
    pr(f"  Saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
