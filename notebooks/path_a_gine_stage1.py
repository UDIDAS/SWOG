#!/usr/bin/env python3
"""
Path A: GINE on Band-Flood Graphs for LiTS Tumor Segmentation

Identity band_of (each intensity is its own band, no merging, no deletion).
Train a 3-layer GINE with PyG for node-level binary classification (tumor vs background).
Lift predictions to voxel-level Dice.
"""
import numpy as np
import os, sys, time, pickle, gc, json
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINEConv, BatchNorm

import fastloops_band

# ---- Paths ----
CT_DIR  = Path('/scratch/ud3d4/acm_data/Data/ct')
SEG_DIR = Path('/scratch/ud3d4/acm_data/Data/seg')
CACHE   = Path('/dev/shm/path_a_graphs')
CACHE.mkdir(exist_ok=True)

# ---- HU window ----
LOW_CLIP, HIGH_CLIP = -50, 250

# ---- Band config: identity (each intensity its own band, no deletion) ----
band_u8 = np.arange(256, dtype=np.uint8)   # identity mapping
del_u8  = np.zeros(256, dtype=np.uint8)     # no deletion

# ---- Volume IDs ----
ALL_VIDS = sorted([int(f.stem.replace('volume-', '')) for f in CT_DIR.glob('volume-*.npy')])
print(f'{len(ALL_VIDS)} volumes: {ALL_VIDS[:5]} ... {ALL_VIDS[-5:]}')

# ---- Device ----
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {DEVICE}')
if DEVICE == 'cuda':
    print(f'GPU: {torch.cuda.get_device_name()}')

# ==============================================================================
# Data loading + featurizer utilities
# ==============================================================================

def load_volume_u8(vid):
    """Load CT volume, apply HU window [-50,250], return (D,H,W) uint8."""
    vol = np.load(CT_DIR / f'volume-{vid}.npy').astype(np.float32)
    vol = np.clip(vol, LOW_CLIP, HIGH_CLIP)
    vol = (vol - LOW_CLIP) / (HIGH_CLIP - LOW_CLIP)
    return (vol * 255).round().astype(np.uint8)

def load_seg(vid):
    """Load segmentation, return (D,H,W) int16."""
    return np.load(SEG_DIR / f'segmentation-{vid}.npy').astype(np.int16)

def tumor_patch_slices(gt, min_size=100, frac=0.20):
    """Training crop: tumor bbox + 20%/axis, floored to min_size^3, clipped to bounds."""
    idx = np.argwhere(gt > 0)
    if idx.size == 0:
        return tuple(slice(0, s) for s in gt.shape)
    lo, hi = idx.min(0), idx.max(0); ext = hi - lo + 1
    half = np.ceil(frac * ext / 2).astype(int)
    lo2, hi2 = lo - half, hi + half
    cen = (lo + hi) / 2.0
    for a in range(3):
        if hi2[a] - lo2[a] + 1 < min_size:
            lo2[a] = int(np.floor(cen[a] - min_size/2)); hi2[a] = lo2[a] + min_size - 1
    for a in range(3):
        n = gt.shape[a]
        if lo2[a] < 0:   hi2[a] -= lo2[a]; lo2[a] = 0
        if hi2[a] > n-1: lo2[a] -= (hi2[a]-(n-1)); hi2[a] = n-1
        lo2[a] = max(lo2[a], 0); hi2[a] = min(hi2[a], n-1)
    return tuple(slice(int(lo2[a]), int(hi2[a]+1)) for a in range(3))

# ---- Node feature layout (C=1, nf=23) ----
BB_AREA, BB_CHAN0, BB_BOUNDARY = 0, 10, 17
BB_COV = [(4,0,0),(5,1,1),(6,2,2),(7,0,1),(8,0,2),(9,1,2)]
BB_S   = [1, 2, 3]
BB_NFG, BB_NBG = 21, 22
BB_MOMENTS = slice(0, 10)

def node_invariants(nf, C=1, eps=1e-6):
    f = nf.astype(np.float64)
    D = 3
    N = f.shape[0]
    V = f[:, BB_AREA]
    Vsafe = np.maximum(V, 1.0)
    mean_coord = np.stack([f[:, c] for c in BB_S], axis=1) / Vsafe[:, None]
    cov = np.zeros((N, D, D))
    for col, i, j in BB_COV:
        cij = f[:, col] / Vsafe - mean_coord[:, i] * mean_coord[:, j]
        cov[:, i, j] = cij; cov[:, j, i] = cij
    w, vec = np.linalg.eigh(cov)
    w = np.clip(w, 0.0, None)
    principal = vec[..., -1]
    trace = w.sum(1)
    degenerate = trace < eps
    denom = w[:, 2] + eps
    linearity  = (w[:, 2] - w[:, 1]) / denom
    planarity  = (w[:, 1] - w[:, 0]) / denom
    sphericity =  w[:, 0] / denom
    shape = np.stack([linearity, planarity, sphericity], axis=1)
    shape[degenerate] = 0.0
    line_like = np.where(degenerate, 0.0, linearity)
    chan = f[:, BB_CHAN0:BB_CHAN0 + C] / Vsafe[:, None] / 255.0
    surface = f[:, BB_BOUNDARY]
    compactness = surface / np.power(Vsafe, (D - 1.0) / D)
    return dict(V=V, surface=surface, shape=shape, line_like=line_like,
                principal=principal, degenerate=degenerate, chan=chan,
                compactness=compactness, log_size=np.log(Vsafe),
                moments=f[:, BB_MOMENTS])

def edge_features_fn(nf, ei, ef, C=1, eps=1e-6, signed=False):
    inv = node_invariants(nf, C, eps)
    a, b = ei[0].astype(np.int64), ei[1].astype(np.int64)
    e = ef.astype(np.float64)
    bl, sumd = e[:, 0], e[:, 1]
    blsafe = np.maximum(bl, 1.0)
    Va, Vb = inv['V'][a], inv['V'][b]
    size_contrast = (Va - Vb) / (Va + Vb + eps)
    bfrac_a = bl / (inv['surface'][a] + eps)
    bfrac_b = bl / (inv['surface'][b] + eps)
    mua, mub = inv['chan'][a], inv['chan'][b]
    mean_contrast = (mua - mub) / (mua + mub + eps)
    shape_dissim = np.abs(inv['shape'][a] - inv['shape'][b])
    cosang = np.abs(np.sum(inv['principal'][a] * inv['principal'][b], axis=1))
    axis_align = cosang * np.minimum(inv['line_like'][a], inv['line_like'][b])
    bcontrast = (sumd / blsafe) / 255.0
    Vas, Vbs = np.maximum(Va, 1.0), np.maximum(Vb, 1.0)
    lr_size = np.log(Vas / Vbs)
    ca, cb = np.maximum(mua, eps), np.maximum(mub, eps)
    lr_int = np.log(ca / cb)
    cmpa = np.maximum(inv['compactness'][a], eps)
    cmpb = np.maximum(inv['compactness'][b], eps)
    lr_cmp = np.log(cmpa / cmpb)
    cols = []
    def add(col):
        col = col if col.ndim == 2 else col[:, None]
        cols.append(col)
    add(np.abs(size_contrast) if not signed else size_contrast)
    add(bfrac_a); add(bfrac_b)
    add(np.abs(mean_contrast) if not signed else mean_contrast)
    add(shape_dissim)
    add(axis_align)
    add(bcontrast)
    add(lr_size); add(lr_int); add(lr_cmp)
    return np.concatenate(cols, axis=1).astype(np.float32)

print('Featurizer ready. Node dim = 7, Edge dim = 12 (for C=1)')

# ==============================================================================
# Build a single graph
# ==============================================================================

def build_graph(vid):
    """Build band-flood graph for one volume. Returns dict or None if volume has no tumor."""
    vol_u8 = load_volume_u8(vid)
    seg = load_seg(vid)
    gt = (seg == 2).astype(np.uint8)

    if gt.sum() == 0:
        return None  # no tumor

    sl = tumor_patch_slices(gt)
    vol_crop = vol_u8[sl]
    gt_crop = gt[sl]

    data = np.ascontiguousarray(vol_crop[..., None])  # (D,H,W,1) u8
    mask = np.ascontiguousarray(gt_crop)               # (D,H,W) u8

    nf, ei, ef, lab, adj = fastloops_band.band_build(data, mask, band_u8, del_u8)
    N = nf.shape[0]
    E = ei.shape[1]

    if N == 0 or E == 0:
        return None

    # Node features (7-dim)
    ninv = node_invariants(nf, C=1)
    node_mat = np.concatenate([
        ninv['log_size'][:, None],
        ninv['chan'],                    # (N,1) mean intensity
        ninv['shape'],                   # (N,3) linearity, planarity, sphericity
        ninv['compactness'][:, None],
        ninv['degenerate'][:, None].astype(np.float32),
    ], axis=1).astype(np.float32)

    # Edge features (12-dim for C=1)
    edge_X = edge_features_fn(nf, ei, ef, C=1, signed=False)

    # Labels: tumor overlap >= 0.10
    fg = nf[:, BB_NFG].astype(np.float32)
    bg = nf[:, BB_NBG].astype(np.float32)
    total = fg + bg
    frac = np.where(total > 0, fg / total, 0.0)
    y = (frac >= 0.10).astype(np.float32)

    gt_fg_total = float(gt_crop.sum())
    fg_outside = gt_fg_total - float(fg.sum())

    return dict(
        vid=vid,
        node_mat=node_mat,
        edge_index=ei.astype(np.int64),
        edge_attr=edge_X,
        y=y,
        fg=fg, bg=bg,
        gt_fg=gt_fg_total,
        fg_outside=fg_outside,
        labels=lab,       # (D,H,W) i64 voxel->node map
        gt_crop=gt_crop,  # (D,H,W) u8
        crop_slice=sl,
        N=N, E=E,
    )

# Quick test on one volume
t0 = time.time()
g = build_graph(0)
if g is not None:
    print(f'Volume 0: N={g["N"]}, E={g["E"]}, node_dim={g["node_mat"].shape[1]}, '
          f'edge_dim={g["edge_attr"].shape[1]}, tumor_nodes={g["y"].sum():.0f}/{g["N"]}, '
          f'time={time.time()-t0:.1f}s')
else:
    print('Volume 0 has no tumor')

# ==============================================================================
# Build all graphs and cache
# ==============================================================================

def build_and_cache(vid):
    """Build graph and save to /dev/shm cache. Returns summary dict or None."""
    cache_path = CACHE / f'graph_{vid}.pkl'
    if cache_path.exists():
        with open(cache_path, 'rb') as f:
            g = pickle.load(f)
        return dict(vid=vid, N=g['N'], E=g['E'], cached=True)

    g = build_graph(vid)
    if g is None:
        return None

    with open(cache_path, 'wb') as f:
        pickle.dump(g, f, protocol=pickle.HIGHEST_PROTOCOL)

    return dict(vid=vid, N=g['N'], E=g['E'], cached=False)

t0 = time.time()
results = []
failed = []

print(f'Building graphs for {len(ALL_VIDS)} volumes...')
for i, vid in enumerate(ALL_VIDS):
    try:
        r = build_and_cache(vid)
        if r is not None:
            results.append(r)
            if (i+1) % 10 == 0 or i == 0:
                elapsed = time.time() - t0
                print(f'  [{i+1}/{len(ALL_VIDS)}] vid={vid} N={r["N"]} E={r["E"]} '
                      f'cached={r["cached"]} ({elapsed:.0f}s)')
        else:
            if (i+1) % 10 == 0:
                print(f'  [{i+1}/{len(ALL_VIDS)}] vid={vid} -- no tumor')
    except Exception as ex:
        failed.append((vid, str(ex)))
        print(f'  FAILED vid={vid}: {ex}')

elapsed = time.time() - t0
print(f'\nDone: {len(results)} graphs built, {len(failed)} failed, {elapsed:.0f}s')
print(f'Total nodes: {sum(r["N"] for r in results):,}')
print(f'Total edges: {sum(r["E"] for r in results):,}')
if failed:
    print(f'Failed volumes: {[v for v,_ in failed]}')

# ==============================================================================
# Oracle Dice (upper bound with identity band_of)
# ==============================================================================

oracle_stats = []
for r in results:
    vid = r['vid']
    with open(CACHE / f'graph_{vid}.pkl', 'rb') as f:
        g = pickle.load(f)
    fg, bg = g['fg'], g['bg']
    maj = fg > bg  # majority-fg nodes
    TP = float(fg[maj].sum())
    FP = float(bg[maj].sum())
    FN = float(fg[~maj].sum()) + float(g['fg_outside'])
    dice = 2*TP / (2*TP + FP + FN + 1e-9)
    oracle_stats.append(dict(vid=vid, oracle_dice=dice, N=g['N'], E=g['E'],
                             gt_fg=g['gt_fg'], tumor_nodes=int(maj.sum())))

dices_oracle = [s['oracle_dice'] for s in oracle_stats]
print(f'\nOracle Dice: mean={np.mean(dices_oracle):.4f}, median={np.median(dices_oracle):.4f}, '
      f'min={np.min(dices_oracle):.4f}, max={np.max(dices_oracle):.4f}')
print(f'Volumes with oracle Dice < 0.5: {sum(1 for d in dices_oracle if d < 0.5)}')

ns = [s['N'] for s in oracle_stats]
print(f'Node count: mean={np.mean(ns):.0f}, median={np.median(ns):.0f}, '
      f'max={np.max(ns)}, min={np.min(ns)}')
print(f'Volumes with N > 500K: {sum(1 for n in ns if n > 500000)}')

# ==============================================================================
# Train/Val/Test split and PyG Data loading
# ==============================================================================

np.random.seed(42)
tumor_vids = sorted([r['vid'] for r in results])
perm = np.random.permutation(len(tumor_vids))
n_train = int(0.70 * len(tumor_vids))
n_val   = int(0.15 * len(tumor_vids))

train_vids = sorted([tumor_vids[i] for i in perm[:n_train]])
val_vids   = sorted([tumor_vids[i] for i in perm[n_train:n_train+n_val]])
test_vids  = sorted([tumor_vids[i] for i in perm[n_train+n_val:]])

print(f'\nSplit: train={len(train_vids)}, val={len(val_vids)}, test={len(test_vids)}')

MAX_NODES = 500_000

def load_pyg_data(vids, max_nodes=MAX_NODES):
    """Load cached graphs as PyG Data objects."""
    data_list = []
    skipped = 0
    for vid in vids:
        with open(CACHE / f'graph_{vid}.pkl', 'rb') as f:
            g = pickle.load(f)
        if g['N'] > max_nodes:
            skipped += 1
            continue
        d = Data(
            x=torch.tensor(g['node_mat'], dtype=torch.float32),
            edge_index=torch.tensor(g['edge_index'], dtype=torch.long),
            edge_attr=torch.tensor(g['edge_attr'], dtype=torch.float32),
            y=torch.tensor(g['y'], dtype=torch.float32),
        )
        d.vid = vid
        d.fg = torch.tensor(g['fg'], dtype=torch.float32)
        d.bg = torch.tensor(g['bg'], dtype=torch.float32)
        d.gt_fg = g['gt_fg']
        d.fg_outside = g['fg_outside']
        data_list.append(d)
    if skipped:
        print(f'  Skipped {skipped} graphs with N > {max_nodes}')
    return data_list

print('Loading train...')
train_data = load_pyg_data(train_vids)
print('Loading val...')
val_data   = load_pyg_data(val_vids)
print('Loading test...')
test_data  = load_pyg_data(test_vids)

print(f'Loaded: train={len(train_data)}, val={len(val_data)}, test={len(test_data)}')

# Class balance
all_y = torch.cat([d.y for d in train_data])
n_pos = all_y.sum().item()
n_neg = len(all_y) - n_pos
print(f'Train class balance: pos={n_pos:.0f} ({100*n_pos/len(all_y):.1f}%), neg={n_neg:.0f}')
w_pos = np.sqrt(n_neg / max(n_pos, 1))
w_neg = np.sqrt(n_pos / max(n_neg, 1))
s = w_pos + w_neg
w_pos, w_neg = 2*w_pos/s, 2*w_neg/s
print(f'Sqrt class weights: pos={w_pos:.4f}, neg={w_neg:.4f}')

# ==============================================================================
# GINE model
# ==============================================================================

class GINENet(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden=128, n_layers=3, dropout=0.1):
        super().__init__()
        self.node_embed = nn.Linear(node_dim, hidden)
        self.edge_embed = nn.Linear(edge_dim, hidden)

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for _ in range(n_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
            )
            self.convs.append(GINEConv(mlp, edge_dim=hidden))
            self.bns.append(BatchNorm(hidden))

        self.head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )
        self.dropout = dropout

    def forward(self, data):
        x = self.node_embed(data.x)
        edge_attr = self.edge_embed(data.edge_attr)

        for conv, bn in zip(self.convs, self.bns):
            x_new = conv(x, data.edge_index, edge_attr)
            x_new = bn(x_new)
            x_new = F.relu(x_new)
            x_new = F.dropout(x_new, p=self.dropout, training=self.training)
            x = x + x_new  # residual

        return self.head(x).squeeze(-1)

node_dim = train_data[0].x.shape[1]
edge_dim = train_data[0].edge_attr.shape[1]
print(f'\nnode_dim={node_dim}, edge_dim={edge_dim}')

model = GINENet(node_dim, edge_dim, hidden=128, n_layers=3).to(DEVICE)
n_params = sum(p.numel() for p in model.parameters())
print(f'Model parameters: {n_params:,}')

# ==============================================================================
# Training loop
# ==============================================================================

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=7, factor=0.5)

pos_weight = torch.tensor([w_pos / w_neg], device=DEVICE)

def compute_loss(logits, y):
    return F.binary_cross_entropy_with_logits(logits, y, pos_weight=pos_weight)

def compute_dice_from_graph(pred_labels, fg, bg, gt_fg, fg_outside):
    TP = float(fg[pred_labels].sum())
    FP = float(bg[pred_labels].sum())
    FN = float(fg[~pred_labels].sum()) + float(fg_outside)
    return 2*TP / (2*TP + FP + FN + 1e-9)

def train_epoch(data_list):
    model.train()
    total_loss = 0
    indices = list(range(len(data_list)))
    np.random.shuffle(indices)
    for idx in indices:
        d = data_list[idx].to(DEVICE)
        optimizer.zero_grad()
        logits = model(d)
        loss = compute_loss(logits, d.y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(data_list)

@torch.no_grad()
def evaluate(data_list):
    model.eval()
    dices = []
    total_loss = 0
    for d in data_list:
        d = d.to(DEVICE)
        logits = model(d)
        loss = compute_loss(logits, d.y)
        total_loss += loss.item()
        pred = logits > 0
        dice = compute_dice_from_graph(
            pred.cpu().numpy(), d.fg.cpu().numpy(), d.bg.cpu().numpy(),
            d.gt_fg, d.fg_outside
        )
        dices.append(dice)
    return total_loss / max(len(data_list), 1), np.mean(dices), dices

MAX_EPOCHS = 150
PATIENCE = 15
VAL_EVERY = 3

best_val_dice = -1
best_epoch = 0
best_state = None
patience_counter = 0

print(f'\nTraining for up to {MAX_EPOCHS} epochs, patience={PATIENCE}, val every {VAL_EVERY} epochs')
print(f'Train graphs: {len(train_data)}, Val graphs: {len(val_data)}\n')

t0 = time.time()
for epoch in range(1, MAX_EPOCHS + 1):
    train_loss = train_epoch(train_data)

    if epoch % VAL_EVERY == 0 or epoch == 1:
        val_loss, val_dice, _ = evaluate(val_data)
        train_loss_eval, train_dice, _ = evaluate(train_data)
        scheduler.step(val_dice)
        lr = optimizer.param_groups[0]['lr']
        elapsed = time.time() - t0
        print(f'Epoch {epoch:3d} | train_loss={train_loss:.4f} train_dice={train_dice:.4f} | '
              f'val_loss={val_loss:.4f} val_dice={val_dice:.4f} | lr={lr:.1e} | {elapsed:.0f}s')

        if val_dice > best_val_dice:
            best_val_dice = val_dice
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += VAL_EVERY

        if patience_counter >= PATIENCE:
            print(f'\nEarly stopping at epoch {epoch}. Best val Dice={best_val_dice:.4f} at epoch {best_epoch}')
            break

print(f'\nBest val Dice: {best_val_dice:.4f} at epoch {best_epoch}')
print(f'Total training time: {time.time()-t0:.0f}s')

# ==============================================================================
# Final evaluation with best model
# ==============================================================================

model.load_state_dict(best_state)
model = model.to(DEVICE)

print('\n=== Final Evaluation (best model) ===\n')

for split_name, split_data in [('train', train_data), ('val', val_data), ('test', test_data)]:
    _, mean_dice, split_dices = evaluate(split_data)
    print(f'{split_name:5s}: mean_dice={mean_dice:.4f}, median={np.median(split_dices):.4f}, '
          f'min={np.min(split_dices):.4f}, max={np.max(split_dices):.4f}, n={len(split_dices)}')

print(f'\nTarget: 0.891')
print(f'\n--- Per-volume test Dice ---')
_, _, test_dices = evaluate(test_data)
for d_obj, dice in sorted(zip(test_data, test_dices), key=lambda x: x[1]):
    print(f'  vid={d_obj.vid:3d}  N={d_obj.x.shape[0]:7d}  dice={dice:.4f}')

# ==============================================================================
# Oracle vs GINE comparison
# ==============================================================================

print('\n=== Oracle Dice (upper bound) vs GINE Dice ===\n')
oracle_map = {s['vid']: s['oracle_dice'] for s in oracle_stats}

for split_name, split_data in [('train', train_data), ('val', val_data), ('test', test_data)]:
    _, _, split_dices = evaluate(split_data)
    oracle_dices_split = [oracle_map.get(d.vid, 0) for d in split_data]
    print(f'{split_name:5s}: GINE_dice={np.mean(split_dices):.4f}, oracle_dice={np.mean(oracle_dices_split):.4f}, '
          f'gap={np.mean(oracle_dices_split)-np.mean(split_dices):.4f}')

print('\nDone.')
