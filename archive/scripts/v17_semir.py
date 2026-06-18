"""
v17: Full SEMIR pipeline — merge_and_cut binary tensor + few-shot boundary search
     + voxel reassignment + paper-matched features + GINE.

Usage:
    conda run -n llmft python scripts/v17_semir.py
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
RESULTS_DIR = "/home/ud3d4/Desktop/SWOG/results/v17_semir"
os.makedirs(RESULTS_DIR, exist_ok=True)
HU_MIN, HU_MAX = -50, 250
np.random.seed(42); torch.manual_seed(42)

def pr(msg=""): print(msg, flush=True)

def bbox_from_mask(mask, margin=32):
    coords = np.argwhere(mask)
    lo = np.maximum(coords.min(0) - margin, 0)
    hi = np.minimum(coords.max(0) + 1 + margin, mask.shape)
    return tuple(slice(int(lo[i]), int(hi[i])) for i in range(3))

# merge_and_cut layout for C=1, 3D: 21 columns
MC_AREA = 0
MC_S = [1, 2, 3]
MC_COV = [(4,0,0), (5,1,1), (6,2,2), (7,0,1), (8,0,2), (9,1,2)]
MC_CHAN0 = 10
MC_MINMAX = [11, 12, 13, 14, 15, 16]  # xmin,xmax,ymin,ymax,zmin,zmax
MC_BOUNDARY = 17
MC_CANON = [18, 19, 20]  # canonical voxel coords


# =============================================================================
# Volume loading
# =============================================================================

def discover_volumes():
    vids = []
    for f in sorted(os.listdir(os.path.join(DATA_ROOT, "ct"))):
        m = re.match(r"volume-(\d+)\.npy", f)
        if m:
            vid = int(m.group(1))
            seg = np.load(os.path.join(DATA_ROOT, "seg", f"segmentation-{vid}.npy"))
            if (seg == 2).sum() > 0:
                vids.append(vid)
    return sorted(vids)


def load_and_crop(vid):
    ct = np.load(os.path.join(DATA_ROOT, "ct", f"volume-{vid}.npy")).astype(np.float32)
    seg = np.load(os.path.join(DATA_ROOT, "seg", f"segmentation-{vid}.npy")).astype(np.int32)
    organ_mask = (seg == 1) | (seg == 2)
    slc = bbox_from_mask(organ_mask)
    ct_crop = ct[slc]; seg_crop = seg[slc]
    ct_u8 = np.clip(ct_crop, HU_MIN, HU_MAX)
    ct_u8 = ((ct_u8 - HU_MIN) / (HU_MAX - HU_MIN) * 255).round().astype(np.uint8)
    ct_u8_4d = np.ascontiguousarray(ct_u8[..., np.newaxis])
    return ct_u8_4d, seg_crop


# =============================================================================
# Graph minor construction + reassignment
# =============================================================================

def build_minor(ct_u8_4d, seg_crop, psi, alpha, beta_min, beta_max):
    """
    Build graph minor with merge_and_cut (paper's binary tensor),
    then reassign deleted voxels to nearest survivor.
    Returns (nf, labels_reassigned, seg_crop).
    """
    nf, ei, ef, labels, adj = fastloops.merge_and_cut(
        ct_u8_4d,
        merge_distance=psi,
        cut_distance=alpha,
        delete_small_node_max_size=beta_min,
        delete_large_node_min_size=beta_max,
    )
    nf = np.asarray(nf); labels = np.asarray(labels)
    N = nf.shape[0]
    if N == 0:
        return None, None

    # Reassign deleted voxels to nearest surviving supernode
    survivor_mask = (labels >= 0)
    if not survivor_mask.all() and survivor_mask.any():
        _, nearest_idx = distance_transform_edt(~survivor_mask, return_indices=True)
        labels = labels[tuple(nearest_idx)]

    return nf, labels


def boundary_dice(labels, gt_seg):
    """Compute boundary Dice: overlap between supernode boundaries and GT boundaries."""
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
    return 2 * inter / (gt_boundary.sum() + sn_boundary.sum() + 1e-8)


# =============================================================================
# Few-shot boundary Dice parameter search
# =============================================================================

def few_shot_search(few_vols, few_segs, n_psi=24, n_alpha=8, n_beta=8):
    """
    Few-shot boundary Dice parameter search (paper Section 3.2, Eq. 1).

    Searches Θ = {ψ, α, β_min} via Cartesian grid over few-shot subset.
    Paper uses 64 values per param; we use 24×8×8 ≈ 1,536 configs × 5 volumes.

    Objective: maximize boundary Dice between supernode boundaries and GT
    segmentation boundaries. This finds params where supernodes snap to
    tissue boundaries — the key to getting meaningful ~1K nodes.

    Constraints (paper): α > ψ (infeasible otherwise), β_min < β_max.
    """
    pr("  Few-shot boundary search...")
    pr(f"  Grid: {n_psi} psi x {n_alpha} alpha x {n_beta} beta_min = "
       f"{n_psi * n_alpha * n_beta} configs x {len(few_vols)} volumes")

    # Parameter ranges (uint8 intensity space, [0, 255])
    psi_values = np.unique(np.linspace(2, 50, n_psi).astype(int))
    alpha_values = np.unique(np.linspace(20, 150, n_alpha).astype(int))
    beta_min_values = np.unique(np.array([1, 2, 5, 8, 12, 17, 25, 40])[:n_beta])

    n_voxels = few_vols[0].size // few_vols[0].shape[-1]
    beta_max_default = int(n_voxels ** 0.8)

    best_score = -1.0
    best_params = None
    log = []
    total = len(psi_values) * len(alpha_values) * len(beta_min_values)
    done = 0

    for psi in psi_values:
        for alpha in alpha_values:
            # Paper constraint: α must be > ψ
            if alpha <= psi:
                done += len(beta_min_values)
                continue

            for beta_min in beta_min_values:
                bdice_scores = []
                oracle_scores = []
                n_nodes_list = []

                for vol, seg in zip(few_vols, few_segs):
                    nf, labels = build_minor(vol, seg, int(psi), int(alpha),
                                              int(beta_min), beta_max_default)
                    if nf is None or len(nf) < 2:
                        bdice_scores.append(0.0)
                        oracle_scores.append(0.0)
                        n_nodes_list.append(0)
                        continue

                    bd = boundary_dice(labels, seg)
                    bdice_scores.append(bd)
                    n_nodes_list.append(len(nf))

                    # Also track oracle for diagnostics
                    N = len(nf)
                    flat = labels.ravel().astype(np.int64)
                    gt = (seg == 2).ravel().astype(np.float64)
                    fg = np.bincount(flat, weights=gt, minlength=N)[:N]
                    tot = np.bincount(flat, minlength=N)[:N]
                    bg = tot - fg; gt_t = fg.sum(); maj = fg > bg
                    oracle = float(2*fg[maj].sum()/(2*fg[maj].sum()+bg[maj].sum()+(gt_t-fg[maj].sum())+1e-8)) if gt_t > 0 else 0
                    oracle_scores.append(oracle)

                mean_bd = np.mean(bdice_scores)
                mean_oracle = np.mean(oracle_scores)
                mean_nodes = np.mean(n_nodes_list)
                compression = n_voxels / max(mean_nodes, 1)
                # Primary objective: boundary Dice
                # Small bonus for compression (paper doesn't penalize, but
                # we want to break ties toward smaller graphs)
                score = mean_bd + 0.0005 * np.log10(max(compression, 1))

                log.append(dict(psi=int(psi), alpha=int(alpha), beta_min=int(beta_min),
                                mean_bdice=round(mean_bd, 6), mean_oracle=round(mean_oracle, 4),
                                mean_nodes=round(mean_nodes, 0),
                                compression=round(compression, 1), score=round(score, 6)))

                if score > best_score:
                    best_score = score
                    best_params = dict(psi=int(psi), alpha=int(alpha),
                                       beta_min=int(beta_min), beta_max=beta_max_default)

                done += 1

        # Progress
        pct = done / total * 100
        if done % (total // 10 + 1) < len(alpha_values) * len(beta_min_values):
            pr(f"    {pct:5.1f}% ({done}/{total}) best_bdice={best_score:.5f}")

    # Print top 10
    sorted_log = sorted(log, key=lambda x: -x["score"])
    pr(f"\n  Top-10 configurations:")
    pr(f"  {'psi':>4s} {'alpha':>5s} {'beta':>5s} | {'bdice':>8s} {'oracle':>7s} {'nodes':>8s} {'compress':>10s}")
    for e in sorted_log[:10]:
        pr(f"  {e['psi']:>4d} {e['alpha']:>5d} {e['beta_min']:>5d} | "
           f"{e['mean_bdice']:>8.5f} {e['mean_oracle']:>7.4f} {e['mean_nodes']:>8.0f} "
           f"{e['compression']:>10.0f}x")

    return best_params, log


# =============================================================================
# Paper-matched feature extraction (7 node + 4 edge features)
# =============================================================================

def extract_node_features(nf, labels, ct_u8_4d):
    """
    Paper's 7 node features:
    1. volume (a_u)
    2. boundary length (b_u)
    3. compactness: 36π a² / (b³ + ε)
    4. elongation: λ_max / (λ_min + ε)
    5. dominant axis: principal eigenvector direction (3 components → 1 via norm)
    6. mean intensity
    7. intensity std
    """
    f = nf.astype(np.float64)
    N = f.shape[0]
    eps = 1e-6

    V = f[:, MC_AREA]
    Vsafe = np.maximum(V, 1.0)
    boundary = f[:, MC_BOUNDARY]

    # Compactness: 36π a² / (b³ + ε)
    compactness = (36.0 * np.pi * V**2) / (boundary**3 + eps)
    compactness = np.clip(compactness, 0, 1)

    # Covariance → eigenvalues → elongation + dominant axis
    mean_coord = np.stack([f[:, c] for c in MC_S], axis=1) / Vsafe[:, None]
    cov = np.zeros((N, 3, 3))
    for col, i, j in MC_COV:
        cij = f[:, col] / Vsafe - mean_coord[:, i] * mean_coord[:, j]
        cov[:, i, j] = cij; cov[:, j, i] = cij
    w, vec = np.linalg.eigh(cov)
    w = np.clip(w, 0.0, None)
    elongation = w[:, 2] / (w[:, 0] + eps)  # λ_max / λ_min
    elongation = np.clip(elongation, 1.0, 100.0)
    # Dominant axis: principal eigenvector (largest eigenvalue)
    principal = vec[..., -1]  # (N, 3)
    # Reduce to scalar: use the largest component magnitude
    dominant_axis = np.max(np.abs(principal), axis=1)

    # Mean intensity (normalized to [0,1])
    mean_intensity = f[:, MC_CHAN0] / Vsafe / 255.0

    # Intensity std: compute from voxel data using bincount
    flat_labels = labels.ravel().astype(np.int64)
    flat_intensity = ct_u8_4d[..., 0].ravel().astype(np.float64)
    # mean per node
    int_sum = np.bincount(flat_labels, weights=flat_intensity, minlength=N)[:N]
    int_sq_sum = np.bincount(flat_labels, weights=flat_intensity**2, minlength=N)[:N]
    counts = np.bincount(flat_labels, minlength=N)[:N]
    counts_safe = np.maximum(counts, 1).astype(np.float64)
    mean_int_bc = int_sum / counts_safe
    var_int = int_sq_sum / counts_safe - mean_int_bc**2
    intensity_std = np.sqrt(np.clip(var_int, 0, None)) / 255.0  # normalize

    # Stack 7 features
    x = np.column_stack([
        np.log1p(V),           # 1. volume (log-scaled)
        boundary,              # 2. boundary length
        compactness,           # 3. compactness
        elongation,            # 4. elongation
        dominant_axis,         # 5. dominant axis
        mean_intensity,        # 6. mean intensity
        intensity_std,         # 7. intensity std
    ]).astype(np.float32)

    # Z-normalize per feature
    for c in range(x.shape[1]):
        mu, sig = x[:, c].mean(), x[:, c].std()
        if sig > eps:
            x[:, c] = (x[:, c] - mu) / sig
        else:
            x[:, c] = 0.0

    return x, mean_coord


def extract_edge_features(nf, labels, centroids):
    """
    Paper's 4 edge features:
    1. log volume ratio: log(a_u / a_v)
    2. normalized intensity diff: |Ī_u - Ī_v| / (|Ī_u| + |Ī_v| + ε)
    3. normalized centroid distance: ||c_u - c_v|| / max_dim
    4. orientation cosine: |d_u · d_v|
    """
    eps = 1e-6
    N = len(nf)
    f = nf.astype(np.float64)
    V = f[:, MC_AREA]; Vsafe = np.maximum(V, 1.0)
    mean_int = f[:, MC_CHAN0] / Vsafe / 255.0

    # Covariance → principal eigenvector
    mean_coord = centroids
    cov_mat = np.zeros((N, 3, 3))
    for col, i, j in MC_COV:
        cij = f[:, col] / Vsafe - mean_coord[:, i] * mean_coord[:, j]
        cov_mat[:, i, j] = cij; cov_mat[:, j, i] = cij
    _, vec = np.linalg.eigh(cov_mat)
    principal = vec[..., -1]

    # Build edges from label volume
    all_a, all_b = [], []
    for axis in range(3):
        sl_lo = [slice(None)] * 3; sl_hi = [slice(None)] * 3
        sl_lo[axis] = slice(0, -1); sl_hi[axis] = slice(1, None)
        a = labels[tuple(sl_lo)].ravel()
        b = labels[tuple(sl_hi)].ravel()
        boundary = (a != b) & (a >= 0) & (b >= 0)
        all_a.append(a[boundary]); all_b.append(b[boundary])

    if not all_a:
        return np.zeros((2, 0), dtype=np.int64), np.zeros((0, 4), dtype=np.float32)

    a_arr = np.concatenate(all_a).astype(np.int64)
    b_arr = np.concatenate(all_b).astype(np.int64)

    # Deduplicate
    lo = np.minimum(a_arr, b_arr); hi = np.maximum(a_arr, b_arr)
    pair_keys = lo * (N + 1) + hi
    unique_keys = np.unique(pair_keys)
    u_a = unique_keys // (N + 1); u_b = unique_keys % (N + 1)

    max_dim = max(labels.shape)

    # 4 features
    lr_vol = np.log(Vsafe[u_a] / Vsafe[u_b])
    int_diff = np.abs(mean_int[u_a] - mean_int[u_b]) / (np.abs(mean_int[u_a]) + np.abs(mean_int[u_b]) + eps)
    dist = np.sqrt(((centroids[u_a] - centroids[u_b])**2).sum(axis=1)) / max_dim
    cos_angle = np.abs((principal[u_a] * principal[u_b]).sum(axis=1))

    ea = np.column_stack([lr_vol, int_diff, dist, cos_angle]).astype(np.float32)
    ei = np.stack([u_a, u_b], axis=0)

    return ei, ea


# =============================================================================
# Build PyG graph
# =============================================================================

def build_graph(ct_u8_4d, seg_crop, params):
    """Build graph minor + extract features + create PyG Data."""
    nf, labels = build_minor(ct_u8_4d, seg_crop,
                              params["psi"], params["alpha"],
                              params["beta_min"], params["beta_max"])
    if nf is None:
        return None

    N = len(nf)
    x, centroids = extract_node_features(nf, labels, ct_u8_4d)
    ei, ea = extract_edge_features(nf, labels, centroids)

    # Make undirected
    if ei.shape[1] > 0:
        ei_fwd = torch.tensor(ei, dtype=torch.long)
        ei_rev = torch.stack([ei_fwd[1], ei_fwd[0]])
        edge_index = torch.cat([ei_fwd, ei_rev], dim=1)
        edge_attr = torch.tensor(np.concatenate([ea, ea]), dtype=torch.float32)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, 4), dtype=torch.float32)

    # Labels from GT overlap
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
# GINE model
# =============================================================================

class GINE(nn.Module):
    def __init__(self, nd=7, ed=4, h=128):
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
# Voxel-level Dice evaluation
# =============================================================================

def voxel_dice(preds, labels, seg_crop):
    """Lift supernode predictions to voxel-level, compute Dice vs GT."""
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
    pr("  v17: SEMIR — Binary Tensor + Boundary Search + GINE")
    pr("=" * 70)

    # Discover and split
    all_vids = discover_volumes()
    perm = np.random.permutation(len(all_vids))
    n_tr = int(0.7 * len(all_vids)); n_va = int(0.15 * len(all_vids))
    train_ids = sorted([all_vids[i] for i in perm[:n_tr]])
    val_ids = sorted([all_vids[i] for i in perm[n_tr:n_tr + n_va]])
    test_ids = sorted([all_vids[i] for i in perm[n_tr + n_va:]])
    ordered = train_ids + val_ids + test_ids
    pr(f"  {len(all_vids)} volumes: {len(train_ids)} train / {len(val_ids)} val / {len(test_ids)} test")

    # Load all volumes
    pr("\nLoading volumes...")
    volumes = {}
    for vid in ordered:
        ct_u8_4d, seg_crop = load_and_crop(vid)
        volumes[vid] = (ct_u8_4d, seg_crop)
    pr(f"  Loaded {len(volumes)} volumes")

    # ---- STAGE 1: Few-shot boundary search ----
    pr("\n" + "=" * 70)
    pr("  STAGE 1: Few-shot boundary Dice parameter search")
    pr("=" * 70)
    n_few = min(5, len(train_ids))
    few_ids = train_ids[:n_few]
    few_vols = [volumes[v][0] for v in few_ids]
    few_segs = [volumes[v][1] for v in few_ids]
    best_params, search_log = few_shot_search(few_vols, few_segs)
    pr(f"\n  Best params: psi={best_params['psi']} alpha={best_params['alpha']} "
       f"beta_min={best_params['beta_min']} beta_max={best_params['beta_max']}")

    with open(os.path.join(RESULTS_DIR, "param_search.json"), "w") as f:
        json.dump(dict(best=best_params, log=search_log), f, indent=2)

    # ---- STAGE 2: Build graphs ----
    pr("\n" + "=" * 70)
    pr("  STAGE 2: Build graph minors with optimal params")
    pr("=" * 70)
    graphs = {}; metas = {}
    t0 = time.time()
    for vid in ordered:
        ct_u8_4d, seg_crop = volumes[vid]
        result = build_graph(ct_u8_4d, seg_crop, best_params)
        if result is None:
            pr(f"  vol-{vid}: SKIP")
            continue
        data, labels, seg = result
        graphs[vid] = data
        metas[vid] = dict(labels=labels, seg=seg)

        split = "train" if vid in train_ids else ("val" if vid in val_ids else "test")
        N = data.num_nodes; n_tu = int((data.y == 1).sum())
        fg = data.n_fg.numpy(); bg = data.n_bg.numpy(); gt_t = fg.sum()
        maj = fg > bg
        oracle = float(2*fg[maj].sum()/(2*fg[maj].sum()+bg[maj].sum()+(gt_t-fg[maj].sum())+1e-8)) if gt_t > 0 else 0
        bd = boundary_dice(labels, seg)
        pr(f"  vol-{vid:>3d} [{split:>5s}]: {N:>6,} nodes ({n_tu:>4,} tu) "
           f"{data.num_edges:>7,} edges  oracle={oracle:.4f}  bdice={bd:.4f}")

    dt = time.time() - t0
    nodes = [g.num_nodes for g in graphs.values()]
    pr(f"\n  Built {len(graphs)} graphs in {dt:.0f}s")
    pr(f"  Nodes: mean={np.mean(nodes):,.0f} median={np.median(nodes):,.0f} "
       f"max={max(nodes):,} min={min(nodes):,}")
    oracles = []
    for g in graphs.values():
        fg = g.n_fg.numpy(); bg = g.n_bg.numpy(); gt_t = fg.sum()
        if gt_t > 0:
            maj = fg > bg
            oracles.append(float(2*fg[maj].sum()/(2*fg[maj].sum()+bg[maj].sum()+(gt_t-fg[maj].sum())+1e-8)))
    pr(f"  Oracle Dice: mean={np.mean(oracles):.4f}")

    # ---- STAGE 3: Train GINE ----
    pr("\n" + "=" * 70)
    pr("  STAGE 3: GINE training")
    pr("=" * 70)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    pr(f"  Device: {device}")
    if torch.cuda.is_available():
        pr(f"  GPU: {torch.cuda.get_device_name(0)}")

    trainable = [v for v in train_ids if v in graphs]
    val_usable = [v for v in val_ids if v in graphs]
    pr(f"  Trainable: {len(trainable)}/{len(train_ids)}, Val: {len(val_usable)}/{len(val_ids)}")

    if not trainable:
        pr("  ERROR: No trainable volumes!")
        return

    total_pos = sum(int((graphs[v].y == 1).sum()) for v in trainable)
    total_neg = sum(int((graphs[v].y == 0).sum()) for v in trainable)
    ratio = total_neg / max(total_pos, 1)
    eff = min(np.sqrt(ratio), 30.0)
    cw = torch.tensor([1.0, eff], dtype=torch.float32).to(device)
    pr(f"  Class weight: [1.0, {eff:.1f}] (ratio {ratio:.0f}:1)")

    model = GINE(nd=7, ed=4).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    PATIENCE, EPOCHS, VAL_EVERY = 15, 200, 3
    best_dice, best_state, wait = -1.0, None, 0

    for epoch in range(1, EPOCHS + 1):
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
        if proc == 0:
            pr("  All OOM"); break
        ml = eloss / proc

        if epoch % VAL_EVERY == 0 or epoch <= 3:
            model.eval(); tp = fp = fn = 0
            with torch.no_grad():
                for vid in val_usable:
                    g = graphs[vid]; m = metas[vid]
                    try:
                        gd = g.to(device)
                        preds = model(gd.x, gd.edge_index, gd.edge_attr).argmax(1).cpu().numpy()
                        del gd; torch.cuda.empty_cache()
                    except (torch.cuda.OutOfMemoryError, RuntimeError):
                        try: del gd
                        except: pass
                        torch.cuda.empty_cache()
                        mc = model.cpu(); mc.eval()
                        preds = mc(g.x, g.edge_index, g.edge_attr).argmax(1).numpy()
                        model.to(device)
                    dice_v, _, _ = voxel_dice(preds, m["labels"], m["seg"])
                    gm = m["seg"] == 2
                    flat = m["labels"].ravel().astype(np.int64)
                    lut = np.zeros(int(flat.max()) + 1, dtype=np.int8)
                    lut[:len(preds)] = preds
                    pm = lut[flat].reshape(m["labels"].shape).astype(bool)
                    inter = int((pm & gm).sum())
                    tp += inter; fp += int(pm.sum()) - inter; fn += int(gm.sum()) - inter

            vd = 2 * tp / (2 * tp + fp + fn + 1e-8)
            if vd > best_dice:
                best_dice = vd
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                wait = 0; mk = " *"
            else:
                wait += 1; mk = ""
            pr(f"  Epoch {epoch:3d}  loss={ml:.4f}  val_dice={vd:.4f}  ({proc} vols){mk}")
            if wait >= PATIENCE:
                pr(f"  Early stop, best={best_dice:.4f}"); break
        else:
            pr(f"  Epoch {epoch:3d}  loss={ml:.4f}  ({proc} vols)")

    if best_state:
        model.load_state_dict(best_state)
    pr(f"\n  Best val Dice: {best_dice:.4f}")
    torch.save(best_state or model.state_dict(), os.path.join(RESULTS_DIR, "model.pt"))

    # ---- STAGE 4: Final evaluation ----
    pr("\n" + "=" * 70)
    pr("  STAGE 4: Final evaluation")
    pr("=" * 70)
    model.eval()
    results = []
    for vid in sorted(graphs.keys()):
        g = graphs[vid]; m = metas[vid]
        with torch.no_grad():
            try:
                gd = g.to(device)
                preds = model(gd.x, gd.edge_index, gd.edge_attr).argmax(1).cpu().numpy()
                del gd; torch.cuda.empty_cache()
            except (torch.cuda.OutOfMemoryError, RuntimeError):
                try: del gd
                except: pass
                torch.cuda.empty_cache()
                mc = model.cpu(); mc.eval()
                preds = mc(g.x.cpu(), g.edge_index.cpu(), g.edge_attr.cpu()).argmax(1).numpy()
                model.to(device)

        dice, rec, prec = voxel_dice(preds, m["labels"], m["seg"])
        fg = g.n_fg.detach().cpu().numpy(); bg = g.n_bg.detach().cpu().numpy(); gt_t = fg.sum()
        maj = fg > bg
        oracle = float(2*fg[maj].sum()/(2*fg[maj].sum()+bg[maj].sum()+(gt_t-fg[maj].sum())+1e-8)) if gt_t > 0 else 0
        split = "train" if vid in train_ids else ("val" if vid in val_ids else "test")
        results.append(dict(vid=vid, split=split, dice=dice, recall=rec,
                            precision=prec, oracle=oracle, nodes=g.num_nodes))

    pr(f"\n{'='*80}")
    pr(f"  v17 RESULTS: SEMIR Binary Tensor + Boundary Search + GINE")
    pr(f"{'='*80}")
    pr(f"  Params: psi={best_params['psi']} alpha={best_params['alpha']} "
       f"beta_min={best_params['beta_min']}")
    for split in ["train", "val", "test"]:
        s = [r for r in results if r["split"] == split]
        if s:
            pr(f"  {split:>5s}: Dice={np.mean([r['dice'] for r in s]):.4f}  "
               f"Recall={np.mean([r['recall'] for r in s]):.4f}  "
               f"Prec={np.mean([r['precision'] for r in s]):.4f}  "
               f"Oracle={np.mean([r['oracle'] for r in s]):.4f}  "
               f"Nodes={np.mean([r['nodes'] for r in s]):,.0f}  (n={len(s)})")
    pr(f"\n  Best val Dice: {best_dice:.4f}")
    pr(f"  Paper target: 0.891")

    with open(os.path.join(RESULTS_DIR, "results.json"), "w") as f:
        json.dump(dict(best_params=best_params, best_val_dice=float(best_dice),
                       results=results), f, indent=2)
    pr(f"  Saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
