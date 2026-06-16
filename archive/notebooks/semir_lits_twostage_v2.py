#!/usr/bin/env python3
"""SEMIR LiTS Two-Stage v2: Tighter protection strategies to reduce graph size.

Goal: Reduce from ~241K nodes (Mode C) toward 10K-50K while keeping tumor recall >90%.
Strategies:
  S0: baseline (mean - 0.5*std, dilate=2) — current Mode C
  S1: mean - 1.0*std, dilate=2
  S2: mean - 1.5*std, dilate=2
  S3: mean - 0.5*std, dilate=1
  S4: mean - 0.5*std, dilate=0
  S5: mean - 1.0*std, dilate=1
  S6: mean - 1.0*std, dilate=0
  S7: mean - 1.0*std, dilate=1, cc_min=50
  S8: mean - 1.5*std, dilate=1, cc_min=50
  S9: mean - 1.0*std, dilate=0, cc_min=50
"""

# %% [markdown]
# # SEMIR LiTS Two-Stage v2: Tighter Protection Strategies
#
# **Problem**: Mode C (intensity protection) gives oracle Dice 0.89 but 241K nodes — too many for GINE.
# Mode D (GT protection) achieves 27K nodes. We need to tighten intensity protection.

# %%
import numpy as np
import os, re, time, json
import fastloops
from scipy.ndimage import binary_dilation, label as ndlabel

DATA_ROOT = "/scratch/ud3d4/acm_data/Data"
RESULTS_DIR = "/home/ud3d4/Desktop/SWOG/results/semir_lits_twostage_v2"
os.makedirs(RESULTS_DIR, exist_ok=True)

HU_MIN, HU_MAX = -50, 250
PSI, ALPHA = 5, 25
np.random.seed(42)

def pr(msg=""):
    print(msg, flush=True)

pr("Imports done.")

# %%
# ## Data loading helpers

def load_and_convert(vid):
    ct = np.load(os.path.join(DATA_ROOT, "ct", f"volume-{vid}.npy")).astype(np.float32)
    seg = np.load(os.path.join(DATA_ROOT, "seg", f"segmentation-{vid}.npy")).astype(np.int32)
    ct_u8 = np.clip(ct, HU_MIN, HU_MAX)
    ct_u8 = ((ct_u8 - HU_MIN) / (HU_MAX - HU_MIN) * 255).round().astype(np.uint8)
    ct_u8 = np.ascontiguousarray(ct_u8[..., np.newaxis])
    return ct, seg, ct_u8

def discover_volumes():
    ct_dir = os.path.join(DATA_ROOT, "ct")
    vids = []
    for f in sorted(os.listdir(ct_dir)):
        m = re.match(r"volume-(\d+)\.npy", f)
        if m:
            vid = int(m.group(1))
            seg_path = os.path.join(DATA_ROOT, "seg", f"segmentation-{vid}.npy")
            if os.path.exists(seg_path):
                seg = np.load(seg_path)
                if (seg == 2).sum() > 0:
                    vids.append(vid)
    return sorted(vids)

def bbox_from_mask(mask, margin=32):
    coords = np.argwhere(mask)
    z0, y0, x0 = coords.min(axis=0)
    z1, y1, x1 = coords.max(axis=0) + 1
    z0 = max(z0 - margin, 0); y0 = max(y0 - margin, 0); x0 = max(x0 - margin, 0)
    z1 = min(z1 + margin, mask.shape[0]); y1 = min(y1 + margin, mask.shape[1]); x1 = min(x1 + margin, mask.shape[2])
    return (slice(z0, z1), slice(y0, y1), slice(x0, x1))

all_vids = discover_volumes()
pr(f"Found {len(all_vids)} LiTS volumes with tumor")
diag_vids = all_vids[:10]
pr(f"Diagnostic volumes: {diag_vids}")

# %%
# ## Protection mask strategies

def remove_small_cc(mask, min_size=50):
    """Remove connected components smaller than min_size."""
    labeled, n_cc = ndlabel(mask)
    if n_cc == 0:
        return mask
    cc_sizes = np.bincount(labeled.ravel())
    # cc_sizes[0] is background
    keep = np.zeros(n_cc + 1, dtype=bool)
    for i in range(1, n_cc + 1):
        if cc_sizes[i] >= min_size:
            keep[i] = True
    return keep[labeled].astype(np.uint8)

def make_protection(ct_u8_crop, organ_crop, std_mult=0.5, dilate_iter=2, cc_min=0):
    """Parameterized intensity-based protection.

    Args:
        ct_u8_crop: uint8 channel-last CT crop
        organ_crop: boolean organ mask (liver+tumor)
        std_mult: threshold = mean - std_mult * std
        dilate_iter: dilation iterations (0=none)
        cc_min: minimum connected component size (0=no filter)
    """
    liver_vals = ct_u8_crop[..., 0][organ_crop]
    liver_mean = liver_vals.mean()
    liver_std = liver_vals.std()

    candidate = organ_crop & (ct_u8_crop[..., 0] < liver_mean - std_mult * liver_std)

    if cc_min > 0:
        candidate = remove_small_cc(candidate.astype(np.uint8), min_size=cc_min).astype(bool)

    if dilate_iter > 0:
        candidate = binary_dilation(candidate, iterations=dilate_iter)

    return candidate.astype(np.uint8)


STRATEGIES = {
    "S0": dict(std_mult=0.5, dilate_iter=2, cc_min=0),    # baseline
    "S1": dict(std_mult=1.0, dilate_iter=2, cc_min=0),
    "S2": dict(std_mult=1.5, dilate_iter=2, cc_min=0),
    "S3": dict(std_mult=0.5, dilate_iter=1, cc_min=0),
    "S4": dict(std_mult=0.5, dilate_iter=0, cc_min=0),
    "S5": dict(std_mult=1.0, dilate_iter=1, cc_min=0),
    "S6": dict(std_mult=1.0, dilate_iter=0, cc_min=0),
    "S7": dict(std_mult=1.0, dilate_iter=1, cc_min=50),
    "S8": dict(std_mult=1.5, dilate_iter=1, cc_min=50),
    "S9": dict(std_mult=1.0, dilate_iter=0, cc_min=50),
}

pr(f"Defined {len(STRATEGIES)} strategies")

# %%
# ## Oracle Dice + coarsening helpers

def oracle_dice_multi(labels_np, seg):
    flat = labels_np.ravel(); gt = (seg.ravel() == 2).astype(np.float64)
    gt_total = int(gt.sum()); valid = flat >= 0
    result = {}
    if gt_total == 0 or not valid.any():
        for th in [0.10, 0.25, 0.50]:
            result[f"oracle_{th:.2f}"] = 0.0
        result["tumor_deleted_pct"] = 0.0
        return result
    max_id = int(flat[valid].max())
    tc = np.bincount(flat[valid], weights=gt[valid], minlength=max_id + 1)
    total_c = np.bincount(flat[valid], minlength=max_id + 1)
    overlap = tc / np.maximum(total_c, 1)
    gt_mask = seg == 2
    for th in [0.10, 0.25, 0.50]:
        tumor_sids = np.where(overlap > th)[0]
        lut = np.zeros(max_id + 1, dtype=np.int32)
        lut[tumor_sids] = 1
        pred = np.where(valid, lut[flat], 0).reshape(labels_np.shape).astype(bool)
        inter = int((pred & gt_mask).sum())
        result[f"oracle_{th:.2f}"] = 2.0 * inter / (pred.sum() + gt_mask.sum() + 1e-8)
    del_tumor = int(gt[~valid].sum())
    result["tumor_deleted_pct"] = del_tumor / max(gt_total, 1) * 100.0
    return result


def run_coarsen(ct_u8, protect_mask):
    n_vox = ct_u8[..., 0].size
    kwargs = dict(merge_distance=PSI, cut_distance=ALPHA, connectivity="faces")
    if protect_mask is not None:
        pm = np.ascontiguousarray(protect_mask)
        nf, ei, ef, labels, adj = fastloops.merge_and_cut_protected(ct_u8, pm, **kwargs)
    else:
        nf, ei, ef, labels, adj = fastloops.merge_and_cut(ct_u8, **kwargs)
    return np.asarray(labels), nf, ei, ef

pr("Helpers defined.")

# %%
# ## Phase 1: Diagnostic sweep on 10 volumes

pr("\n" + "="*100)
pr("  PHASE 1: Protection strategy sweep on diagnostic volumes")
pr("="*100)

all_diag = []

for vid in diag_vids:
    ct_raw, seg, _ = load_and_convert(vid)
    n_tumor = int((seg == 2).sum())
    organ_mask = (seg == 1) | (seg == 2)
    slc = bbox_from_mask(organ_mask, margin=32)
    ct_crop = ct_raw[slc]; seg_crop = seg[slc]; organ_crop = organ_mask[slc]

    ct_u8_crop = np.clip(ct_crop, HU_MIN, HU_MAX)
    ct_u8_crop = ((ct_u8_crop - HU_MIN) / (HU_MAX - HU_MIN) * 255).round().astype(np.uint8)
    ct_u8_crop = np.ascontiguousarray(ct_u8_crop[..., np.newaxis])

    gt_tumor = seg_crop == 2
    pr(f"\n--- vol-{vid}: crop={ct_crop.shape} tumor={n_tumor:,} ---")

    for sname, sparams in sorted(STRATEGIES.items()):
        t0 = time.time()
        protect = make_protection(ct_u8_crop, organ_crop, **sparams)

        # Protection mask quality
        if gt_tumor.sum() > 0:
            recall = float((protect.astype(bool) & gt_tumor).sum() / gt_tumor.sum())
            precision = float((protect.astype(bool) & gt_tumor).sum() / max(protect.sum(), 1))
            prot_vox = int(protect.sum())
        else:
            recall = precision = 0.0; prot_vox = int(protect.sum())

        # Coarsen
        labels, nf_arr, ei_arr, ef_arr = run_coarsen(ct_u8_crop, protect)
        dt = time.time() - t0

        od = oracle_dice_multi(labels, seg_crop)
        n_sn = nf_arr.shape[0]
        n_edges = ei_arr.shape[1]

        row = dict(vid=vid, strategy=sname, **sparams,
                   n_sn=n_sn, n_edges=n_edges,
                   protect_recall=round(recall, 4),
                   protect_precision=round(precision, 4),
                   protect_voxels=prot_vox,
                   time=round(dt, 2))
        for k, v in od.items():
            row[k] = round(v, 4)
        all_diag.append(row)

        pr(f"  {sname} (std={sparams['std_mult']:.1f} dil={sparams['dilate_iter']} cc={sparams['cc_min']:>3d}): "
           f"{n_sn:>8,} SN  recall={recall:.3f}  prec={precision:.3f}  "
           f"o@.10={od['oracle_0.10']:.3f}  o@.50={od['oracle_0.50']:.3f}  "
           f"del={od['tumor_deleted_pct']:.1f}%  {dt:.1f}s")

pr("\nPhase 1 done.")

# %%
# ## Phase 1 Summary Table

pr(f"\n{'='*120}")
pr(f"  STRATEGY COMPARISON (mean over {len(diag_vids)} diagnostic volumes)")
pr(f"{'='*120}")
pr(f"{'Strat':>5s} {'std':>4s} {'dil':>3s} {'cc':>4s} | {'SN':>9s} {'edges':>9s} | "
   f"{'recall':>7s} {'prec':>7s} | {'o@.10':>6s} {'o@.25':>6s} {'o@.50':>6s} | {'del%':>5s}")
pr("-" * 120)

strategy_means = {}
for sname in sorted(STRATEGIES.keys()):
    rows = [r for r in all_diag if r["strategy"] == sname]
    sp = STRATEGIES[sname]
    m = {
        "sn": np.mean([r["n_sn"] for r in rows]),
        "edges": np.mean([r["n_edges"] for r in rows]),
        "recall": np.mean([r["protect_recall"] for r in rows]),
        "precision": np.mean([r["protect_precision"] for r in rows]),
        "o10": np.mean([r["oracle_0.10"] for r in rows]),
        "o25": np.mean([r["oracle_0.25"] for r in rows]),
        "o50": np.mean([r["oracle_0.50"] for r in rows]),
        "del": np.mean([r["tumor_deleted_pct"] for r in rows]),
    }
    strategy_means[sname] = m
    marker = ""
    if 10000 <= m["sn"] <= 50000 and m["recall"] >= 0.90:
        marker = " <-- TARGET"
    elif 10000 <= m["sn"] <= 100000 and m["recall"] >= 0.85:
        marker = " <-- close"
    pr(f"{sname:>5s} {sp['std_mult']:>4.1f} {sp['dilate_iter']:>3d} {sp['cc_min']:>4d} | "
       f"{m['sn']:>9,.0f} {m['edges']:>9,.0f} | "
       f"{m['recall']:>7.3f} {m['precision']:>7.3f} | "
       f"{m['o10']:>6.3f} {m['o25']:>6.3f} {m['o50']:>6.3f} | "
       f"{m['del']:>5.1f}%{marker}")

# Pick best strategy
best_name = None
best_score = -1
for sname, m in strategy_means.items():
    if m["recall"] >= 0.90 and m["o50"] >= 0.80:
        # Score: prefer fewer nodes
        score = m["o50"] - m["sn"] / 1e6  # penalize large graphs
        if score > best_score:
            best_score = score
            best_name = sname

# Fallback: relax recall to 0.85
if best_name is None:
    for sname, m in strategy_means.items():
        if m["recall"] >= 0.85 and m["o50"] >= 0.75:
            score = m["o50"] - m["sn"] / 1e6
            if score > best_score:
                best_score = score
                best_name = sname

# Fallback: just pick the one with best o50 that has < 100K nodes
if best_name is None:
    for sname, m in strategy_means.items():
        if m["sn"] < 100000:
            score = m["o50"]
            if score > best_score:
                best_score = score
                best_name = sname

if best_name is None:
    best_name = "S7"  # safe default

pr(f"\n>>> BEST STRATEGY: {best_name} — {STRATEGIES[best_name]}")
pr(f"    SN={strategy_means[best_name]['sn']:,.0f}  recall={strategy_means[best_name]['recall']:.3f}  "
   f"o@.50={strategy_means[best_name]['o50']:.3f}  del={strategy_means[best_name]['del']:.1f}%")

# Save diagnostics
with open(os.path.join(RESULTS_DIR, "strategy_sweep.json"), "w") as f:
    json.dump({"strategies": {k: v for k, v in STRATEGIES.items()},
               "diag_results": all_diag,
               "means": {k: {kk: float(vv) for kk, vv in v.items()} for k, v in strategy_means.items()},
               "best": best_name}, f, indent=2)
pr(f"Saved to {RESULTS_DIR}/strategy_sweep.json")

# %%
# ## Phase 2: Build graphs for all 118 volumes with best strategy + train GINE

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GINEConv, BatchNorm

BEST_PARAMS = STRATEGIES[best_name]
pr(f"\nPhase 2: Building graphs with {best_name}: {BEST_PARAMS}")

# ---- Feature extraction (from v1 notebook) ----

def _layout(C):
    return dict(area=0, s=[1, 2, 3],
                cov=[(4, 0, 0), (5, 1, 1), (6, 2, 2), (7, 0, 1), (8, 0, 2), (9, 1, 2)],
                chan0=10, boundary=10 + C + 6, D=3)

def node_invariants(node_feats, C=1, eps=1e-6):
    f = node_feats.astype(np.float64)
    L = _layout(C); D = L["D"]; N = f.shape[0]
    V = f[:, L["area"]]; Vsafe = np.maximum(V, 1.0)
    mean_coord = np.stack([f[:, c] for c in L["s"]], axis=1) / Vsafe[:, None]
    cov = np.zeros((N, D, D))
    for col, i, j in L["cov"]:
        cij = f[:, col] / Vsafe - mean_coord[:, i] * mean_coord[:, j]
        cov[:, i, j] = cij; cov[:, j, i] = cij
    w = np.linalg.eigvalsh(cov); w = np.clip(w, 0.0, None)
    _, vec = np.linalg.eigh(cov); principal = vec[..., -1]
    trace = w.sum(axis=1); degenerate = trace < eps
    denom = w[:, 2] + eps
    shape = np.stack([(w[:, 2] - w[:, 1]) / denom,
                      (w[:, 1] - w[:, 0]) / denom,
                      w[:, 0] / denom], axis=1)
    shape[degenerate] = 0.0
    line_like = np.where(degenerate, 0.0, shape[:, 0])
    chan = f[:, L["chan0"]:L["chan0"] + C] / Vsafe[:, None] / 255.0
    compactness = f[:, L["boundary"]] / np.power(Vsafe, (D - 1.0) / D)
    elongation = np.where(w[:, 0] > eps, w[:, 2] / (w[:, 0] + eps), 1.0)
    elongation = np.clip(elongation, 1.0, 100.0)
    return dict(V=V, surface=f[:, L["boundary"]], centroid=mean_coord,
                eig=w, shape=shape, line_like=line_like, principal=principal,
                chan=chan, compactness=compactness, elongation=elongation)

def edge_invariants(node_feats, edge_index, edge_feats, C=1, eps=1e-6):
    inv = node_invariants(node_feats, C, eps)
    a = edge_index[0].astype(np.int64); b = edge_index[1].astype(np.int64)
    ef = edge_feats.astype(np.float64); blsafe = np.maximum(ef[:, 0], 1.0)
    size_contrast = np.abs(inv["V"][a] - inv["V"][b]) / (inv["V"][a] + inv["V"][b] + eps)
    bfrac_a = ef[:, 0] / (inv["surface"][a] + eps)
    bfrac_b = ef[:, 0] / (inv["surface"][b] + eps)
    mean_contrast = np.abs(inv["chan"][a] - inv["chan"][b])
    shape_dissim = np.abs(inv["shape"][a] - inv["shape"][b])
    axis_align = (np.abs(np.sum(inv["principal"][a] * inv["principal"][b], axis=1))
                  * np.minimum(inv["line_like"][a], inv["line_like"][b]))
    bcontrast = (ef[:, 1] / blsafe) / 255.0
    cut_frac = ef[:, 3] / blsafe
    cols = [size_contrast[:, None], bfrac_a[:, None], bfrac_b[:, None],
            mean_contrast if mean_contrast.ndim > 1 else mean_contrast[:, None],
            shape_dissim, axis_align[:, None], bcontrast[:, None], cut_frac[:, None]]
    return np.concatenate(cols, axis=1).astype(np.float32)

def compute_intensity_std(labels_np, ct_u8):
    flat = labels_np.ravel(); valid = flat >= 0
    if not valid.any(): return np.array([], dtype=np.float32)
    max_id = int(flat[valid].max())
    vals = ct_u8[..., 0].ravel().astype(np.float64) / 255.0
    counts = np.bincount(flat[valid], minlength=max_id + 1).astype(np.float64)
    sums = np.bincount(flat[valid], weights=vals[valid], minlength=max_id + 1)
    sq_sums = np.bincount(flat[valid], weights=vals[valid] ** 2, minlength=max_id + 1)
    mean = sums / np.maximum(counts, 1.0)
    var = sq_sums / np.maximum(counts, 1.0) - mean ** 2
    return np.sqrt(np.maximum(var, 0.0)).astype(np.float32)

def build_pyg_graph(raw_nf, raw_ei, raw_ef, labels_np, seg, ct_u8, overlap_th, C=1):
    n_sn = raw_nf.shape[0]
    inv = node_invariants(raw_nf, C)
    int_std = compute_intensity_std(labels_np, ct_u8)
    if len(int_std) < n_sn: int_std = np.pad(int_std, (0, n_sn - len(int_std)))
    int_std = int_std[:n_sn]
    principal = inv["principal"].astype(np.float32)
    if len(principal):
        max_comp = np.argmax(np.abs(principal), axis=1)
        signs = np.sign(principal[np.arange(len(principal)), max_comp])
        signs[signs == 0] = 1
        principal = principal * signs[:, None]
    x = np.column_stack([
        np.log1p(inv["V"]), np.log1p(inv["surface"]),
        inv["compactness"], inv["elongation"],
        principal[:, 0], principal[:, 1], principal[:, 2],
        inv["chan"][:, 0], int_std,
    ]).astype(np.float32)
    for col in range(x.shape[1]):
        mu, sigma = float(x[:, col].mean()), float(x[:, col].std())
        if sigma > 1e-8: x[:, col] = (x[:, col] - mu) / sigma
        else: x[:, col] = 0.0
    if raw_ei.shape[1] > 0:
        ea = edge_invariants(raw_nf, raw_ei, raw_ef, C)
        ei_fwd = torch.tensor(raw_ei, dtype=torch.long)
        ei_rev = torch.stack([ei_fwd[1], ei_fwd[0]])
        edge_index = torch.cat([ei_fwd, ei_rev], dim=1)
        edge_attr = torch.tensor(np.concatenate([ea, ea]), dtype=torch.float32)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, 10), dtype=torch.float32)
    flat = labels_np.ravel(); valid = flat >= 0
    gt = (seg.ravel() == 2).astype(np.float64)
    max_id = int(flat[valid].max()) if valid.any() else -1
    y = np.zeros(n_sn, dtype=np.int64)
    if max_id >= 0:
        tc = np.bincount(flat[valid], weights=gt[valid], minlength=max_id + 1)
        total_c = np.bincount(flat[valid], minlength=max_id + 1)
        overlap = tc / np.maximum(total_c, 1)
        y[:min(n_sn, len(overlap))] = (overlap[:n_sn] >= overlap_th).astype(np.int64)
    return Data(x=torch.tensor(x, dtype=torch.float32),
                edge_index=edge_index, edge_attr=edge_attr,
                y=torch.tensor(y, dtype=torch.long))

# ---- GINE model ----
class GINE(nn.Module):
    def __init__(self, nd, ed, h=128):
        super().__init__()
        self.ep = nn.Linear(ed, h)
        def mlp(d): return nn.Sequential(nn.Linear(d, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Linear(h, h))
        self.c1 = GINEConv(mlp(nd), edge_dim=h); self.b1 = BatchNorm(h)
        self.c2 = GINEConv(mlp(h), edge_dim=h); self.b2 = BatchNorm(h)
        self.c3 = GINEConv(mlp(h), edge_dim=h); self.b3 = BatchNorm(h)
        self.head = nn.Linear(h, 2)
    def forward(self, x, ei, ea):
        if ea is not None and ea.numel() > 0: ea = self.ep(ea)
        else:
            n = x.size(0); ei = torch.stack([torch.arange(n, device=x.device)]*2)
            ea = torch.zeros(n, self.ep.out_features, device=x.device)
        x = F.relu(self.b1(self.c1(x, ei, ea)))
        x = F.relu(self.b2(self.c2(x, ei, ea)))
        x = F.relu(self.b3(self.c3(x, ei, ea)))
        return self.head(x)

pr("Model + features defined.")

# %%
# ## Build graphs for all 118 volumes

perm = np.random.permutation(len(all_vids))
n_train = int(0.7 * len(all_vids)); n_val = int(0.15 * len(all_vids))
train_ids = sorted([all_vids[i] for i in perm[:n_train]])
val_ids = sorted([all_vids[i] for i in perm[n_train:n_train + n_val]])
test_ids = sorted([all_vids[i] for i in perm[n_train + n_val:]])
pr(f"Split: {len(train_ids)} train / {len(val_ids)} val / {len(test_ids)} test")

OVERLAP_TH = 0.10
MAX_NODES_GPU = 500_000
GRAPH_CACHE = f"/dev/shm/semir_twostage_v2_{best_name}"
os.makedirs(GRAPH_CACHE, exist_ok=True)

graphs = {}; label_maps = {}; seg_maps = {}

for vid in train_ids + val_ids + test_ids:
    cache_g = os.path.join(GRAPH_CACHE, f"graph_{vid}.pt")
    cache_l = os.path.join(GRAPH_CACHE, f"labels_{vid}.npy")
    cache_s = os.path.join(GRAPH_CACHE, f"seg_{vid}.npy")

    if os.path.exists(cache_g):
        g = torch.load(cache_g, weights_only=False)
        lnp = np.load(cache_l); s = np.load(cache_s)
    else:
        ct_raw, seg, _ = load_and_convert(vid)
        organ_mask = (seg == 1) | (seg == 2)
        slc = bbox_from_mask(organ_mask, margin=32)
        ct_crop = ct_raw[slc]; seg_crop = seg[slc]; organ_crop = organ_mask[slc]

        ct_u8_crop = np.clip(ct_crop, HU_MIN, HU_MAX)
        ct_u8_crop = ((ct_u8_crop - HU_MIN) / (HU_MAX - HU_MIN) * 255).round().astype(np.uint8)
        ct_u8_crop = np.ascontiguousarray(ct_u8_crop[..., np.newaxis])

        protect = make_protection(ct_u8_crop, organ_crop, **BEST_PARAMS)
        lnp, raw_nf, raw_ei, raw_ef = run_coarsen(ct_u8_crop, protect)
        s = seg_crop

        g = build_pyg_graph(raw_nf, raw_ei, raw_ef, lnp, s, ct_u8_crop, OVERLAP_TH)
        torch.save(g, cache_g); np.save(cache_l, lnp); np.save(cache_s, s)

    graphs[vid] = g; label_maps[vid] = lnp; seg_maps[vid] = s
    split = "train" if vid in train_ids else ("val" if vid in val_ids else "test")
    n_tu = int((g.y == 1).sum()); n_bg = int((g.y == 0).sum())
    pr(f"  vol-{vid:>3d} [{split:>5s}]: {g.num_nodes:>7,} nodes ({n_tu:>5,} tu) "
       f"{g.num_edges:>8,} edges")

mean_nodes = np.mean([g.num_nodes for g in graphs.values()])
median_nodes = np.median([g.num_nodes for g in graphs.values()])
pr(f"\nTotal: {len(graphs)} graphs, mean={mean_nodes:,.0f} nodes, median={median_nodes:,.0f} nodes")

# %%
# ## Train GINE

device = "cuda:0" if torch.cuda.is_available() else "cpu"
pr(f"Device: {device}")
if torch.cuda.is_available():
    pr(f"GPU: {torch.cuda.get_device_name(0)}")

trainable = [v for v in train_ids if v in graphs and graphs[v].num_nodes <= MAX_NODES_GPU]
val_usable = [v for v in val_ids if v in graphs]
pr(f"Trainable: {len(trainable)}/{len(train_ids)}, Val: {len(val_usable)}/{len(val_ids)}")

total_pos = sum(int((graphs[v].y == 1).sum()) for v in trainable)
total_neg = sum(int((graphs[v].y == 0).sum()) for v in trainable)
ratio = total_neg / max(total_pos, 1)
eff = min(np.sqrt(ratio), 30.0)
class_weight = torch.tensor([1.0, eff], dtype=torch.float32).to(device)
pr(f"Class weight: [1.0, {eff:.1f}] (ratio: {ratio:.0f}:1)")

nd = graphs[trainable[0]].x.shape[1]
ed = graphs[trainable[0]].edge_attr.shape[1] if graphs[trainable[0]].edge_attr.numel() > 0 else 10

model = GINE(nd, ed).to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

PATIENCE = 15; EPOCHS = 200; VAL_EVERY = 3
best_dice, best_state, wait = -1.0, None, 0
history = {"train_loss": [], "val_dice": []}

for epoch in range(1, EPOCHS + 1):
    model.train(); epoch_loss = 0.0; processed = 0
    for vid in np.random.permutation(trainable):
        try:
            g = graphs[vid].to(device)
            opt.zero_grad()
            logits = model(g.x, g.edge_index, g.edge_attr)
            loss = F.cross_entropy(logits, g.y, weight=class_weight)
            loss.backward(); opt.step()
            epoch_loss += loss.item(); processed += 1
            del g, logits, loss
        except torch.cuda.OutOfMemoryError:
            try: del g
            except: pass
            torch.cuda.empty_cache(); continue
        torch.cuda.empty_cache()

    if processed == 0:
        pr("ERROR: No graphs processed"); break
    mean_loss = epoch_loss / processed
    history["train_loss"].append(mean_loss)

    if epoch % VAL_EVERY == 0 or epoch <= 3:
        model.eval(); tp = fp = fn = 0
        with torch.no_grad():
            for vid in val_usable:
                g = graphs[vid]; lnp = label_maps[vid]; s = seg_maps[vid]
                try:
                    gd = g.to(device)
                    preds = model(gd.x, gd.edge_index, gd.edge_attr).argmax(dim=1).cpu().numpy()
                    del gd; torch.cuda.empty_cache()
                except (torch.cuda.OutOfMemoryError, RuntimeError):
                    try: del gd
                    except: pass
                    torch.cuda.empty_cache()
                    mc = model.cpu()
                    preds = mc(g.x, g.edge_index, g.edge_attr).argmax(dim=1).numpy()
                    model.to(device)
                flat = lnp.ravel(); valid = flat >= 0
                if not valid.any(): continue
                mid = int(flat[valid].max())
                lut = np.zeros(mid + 1, dtype=np.int8)
                lut[:min(len(preds), mid + 1)] = preds[:min(len(preds), mid + 1)]
                pm = np.where(valid, lut[flat], 0).reshape(lnp.shape).astype(bool)
                gm = s == 2
                inter = int((pm & gm).sum())
                tp += inter; fp += int(pm.sum()) - inter; fn += int(gm.sum()) - inter

        vd = 2 * tp / (2 * tp + fp + fn + 1e-8)
        history["val_dice"].append(vd)
        if vd > best_dice:
            best_dice = vd
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0; marker = " *"
        else:
            wait += 1; marker = ""
        pr(f"  Epoch {epoch:3d}  loss={mean_loss:.4f}  val_dice={vd:.4f}  ({processed} vols){marker}")
        if wait >= PATIENCE:
            pr(f"  Early stop epoch {epoch}, best val Dice={best_dice:.4f}"); break
    else:
        pr(f"  Epoch {epoch:3d}  loss={mean_loss:.4f}  ({processed} vols)")

if best_state: model.load_state_dict(best_state)
pr(f"\nBest val Dice: {best_dice:.4f}")
torch.save(best_state or model.state_dict(), os.path.join(RESULTS_DIR, "model.pt"))

# %%
# ## Final evaluation

model.eval()
results = []

for vid in sorted(graphs.keys()):
    g = graphs[vid]; lnp = label_maps[vid]; s = seg_maps[vid]
    with torch.no_grad():
        try:
            gd = g.to(device)
            preds = model(gd.x, gd.edge_index, gd.edge_attr).argmax(dim=1).cpu().numpy()
            del gd; torch.cuda.empty_cache()
        except:
            torch.cuda.empty_cache()
            preds = model.cpu()(g.x, g.edge_index, g.edge_attr).argmax(dim=1).numpy()
            model.to(device)

    flat = lnp.ravel(); valid = flat >= 0
    pm = np.zeros(lnp.shape, dtype=bool)
    if valid.any():
        mid = int(flat[valid].max())
        lut = np.zeros(mid + 1, dtype=np.int8)
        lut[:min(len(preds), mid + 1)] = preds[:min(len(preds), mid + 1)]
        pm = np.where(valid, lut[flat], 0).reshape(lnp.shape).astype(bool)

    gm = s == 2
    inter = int((gm & pm).sum())
    dice = 2.0 * inter / (gm.sum() + pm.sum() + 1e-8)
    rec = inter / (gm.sum() + 1e-8)
    prec = inter / (pm.sum() + 1e-8) if pm.sum() > 0 else 0.0

    split = "train" if vid in train_ids else ("val" if vid in val_ids else "test")
    results.append({"vid": vid, "split": split, "dice": float(dice),
                     "recall": float(rec), "precision": float(prec),
                     "n_nodes": g.num_nodes, "n_tumor_nodes": int((g.y == 1).sum()),
                     "gt_voxels": int(gm.sum()), "pred_voxels": int(pm.sum())})

pr(f"\n{'='*80}")
pr(f"  VOXEL-LEVEL DICE RESULTS — strategy {best_name}: {BEST_PARAMS}")
pr(f"{'='*80}")
pr(f"  {'Split':>5s}  {'Dice':>8s}  {'Recall':>8s}  {'Prec':>8s}  {'N':>4s}")
pr(f"  {'-'*40}")
for split in ["train", "val", "test"]:
    scores = [r for r in results if r["split"] == split]
    if scores:
        d = np.mean([r["dice"] for r in scores])
        r_ = np.mean([r["recall"] for r in scores])
        p = np.mean([r["precision"] for r in scores])
        pr(f"  {split:>5s}  {d:8.4f}  {r_:8.4f}  {p:8.4f}  {len(scores):4d}")

pr(f"\n  Comparison with v1 (Mode C baseline):")
pr(f"    v1: 241K nodes avg, val Dice 0.32, test Dice ~0.23")
pr(f"    v2: {mean_nodes:,.0f} nodes avg, val Dice {best_dice:.4f}")
pr(f"    Node reduction: {241000/max(mean_nodes,1):.1f}x")

with open(os.path.join(RESULTS_DIR, "final_results.json"), "w") as f:
    json.dump({"strategy": best_name, "params": BEST_PARAMS,
               "psi": PSI, "alpha": ALPHA, "hu": [HU_MIN, HU_MAX],
               "overlap_th": OVERLAP_TH,
               "mean_nodes": float(mean_nodes), "median_nodes": float(median_nodes),
               "best_val_dice": float(best_dice),
               "epochs": len(history["train_loss"]),
               "history": history, "results": results}, f, indent=2)
pr(f"\nSaved to {RESULTS_DIR}/final_results.json")
pr("\nDONE.")
