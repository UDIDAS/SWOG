"""
v16b: Luke's Band-Flood Stage 1 + Protected Subgraph + GINE

Combines:
  - Luke's learned band_of dictionary (oracle 0.978) for graph construction
  - v15's intensity-based protection mask to extract a tractable subgraph
  - 3-layer GINE for tumor classification

Usage:
    conda run -n llmft python scripts/v16_evaluate.py
"""

import numpy as np
import os, re, time, json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GINEConv, BatchNorm
from scipy.ndimage import binary_dilation
import fastloops_band

DATA_ROOT = "/scratch/ud3d4/acm_data/Data"
RESULTS_DIR = "/home/ud3d4/Desktop/SWOG/results/v16b_band_protected"
os.makedirs(RESULTS_DIR, exist_ok=True)
HU_MIN, HU_MAX = -50, 250
STD_MULT = 1.5  # protection threshold: liver_mean - STD_MULT * liver_std
DILATE = 2      # dilation iterations for protection mask
np.random.seed(42); torch.manual_seed(42)

def pr(msg=""): print(msg, flush=True)

def bbox_from_mask(mask, margin=32):
    coords = np.argwhere(mask)
    lo = np.maximum(coords.min(0) - margin, 0)
    hi = np.minimum(coords.max(0) + 1 + margin, mask.shape)
    return tuple(slice(int(lo[i]), int(hi[i])) for i in range(3))

# Luke's learned dictionaries (oracle ceiling 0.978)
band_of = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 55, 55, 57, 57, 57, 57, 57, 57, 57, 57, 65, 65, 65, 65, 65, 65, 71, 71, 71, 71, 71, 76, 76, 76, 79, 79, 79, 82, 82, 82, 85, 85, 85, 88, 89, 90, 90, 90, 93, 94, 95, 95, 97, 98, 99, 100, 101, 101, 103, 103, 105, 106, 107, 108, 108, 110, 111, 112, 113, 114, 115, 116, 117, 117, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 131, 133, 134, 134, 136, 136, 138, 139, 139, 141, 141, 141, 144, 145, 145, 145, 145, 149, 149, 149, 149, 149, 154, 154, 156, 156, 156, 156, 156, 156, 162, 162, 162, 162, 162, 167, 167, 167, 167, 167, 167, 173, 173, 173, 173, 173, 173, 179, 179, 179, 179, 179, 179, 179, 179, 187, 187, 187, 187, 187, 187, 187, 187, 187, 196, 196, 196, 196, 196, 196, 196, 196, 196, 196, 196, 196, 196, 196, 196, 196, 196, 196, 196, 196, 196, 196, 196, 196, 196, 196, 196, 196, 196, 196, 196, 196, 196, 196, 196, 196, 196, 196, 196, 196, 196, 196, 196, 196, 196, 196, 196, 196, 196, 196, 196, 196, 196, 196, 196, 196, 196, 196, 196, 196], dtype=np.uint8)
deleted = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=np.uint8)

BB_AREA = 0; BB_S = [1, 2, 3]
BB_COV = [(4, 0, 0), (5, 1, 1), (6, 2, 2), (7, 0, 1), (8, 0, 2), (9, 1, 2)]
BB_CHAN0 = 10; BB_BOUNDARY = 17; BB_NFG = 21; BB_NBG = 22


def node_invariants(nf, C=1, eps=1e-6):
    f = nf.astype(np.float64); D = 3; N = f.shape[0]
    V = f[:, BB_AREA]; Vsafe = np.maximum(V, 1.0)
    mean_coord = np.stack([f[:, c] for c in BB_S], axis=1) / Vsafe[:, None]
    cov = np.zeros((N, D, D))
    for col, i, j in BB_COV:
        cij = f[:, col] / Vsafe - mean_coord[:, i] * mean_coord[:, j]
        cov[:, i, j] = cij; cov[:, j, i] = cij
    w, vec = np.linalg.eigh(cov); w = np.clip(w, 0.0, None)
    principal = vec[..., -1]; trace = w.sum(1); degenerate = trace < eps
    denom = w[:, 2] + eps
    shape = np.stack([(w[:,2]-w[:,1])/denom, (w[:,1]-w[:,0])/denom, w[:,0]/denom], axis=1)
    shape[degenerate] = 0.0; line_like = np.where(degenerate, 0.0, shape[:, 0])
    chan = f[:, BB_CHAN0:BB_CHAN0+C] / Vsafe[:, None] / 255.0
    compactness = f[:, BB_BOUNDARY] / np.power(Vsafe, (D-1.0)/D)
    return dict(V=V, surface=f[:, BB_BOUNDARY], centroid=mean_coord,
                shape=shape, line_like=line_like, principal=principal,
                chan=chan, compactness=compactness, degenerate=degenerate)


def edge_features_fn(nf, ei, ef, C=1, eps=1e-6):
    inv = node_invariants(nf, C, eps)
    a, b = ei[0].astype(np.int64), ei[1].astype(np.int64)
    e = ef.astype(np.float64); bl = e[:, 0]; blsafe = np.maximum(bl, 1.0)
    Va, Vb = inv["V"][a], inv["V"][b]
    mua, mub = inv["chan"][a], inv["chan"][b]
    cols = [
        (np.abs(Va - Vb) / (Va + Vb + eps))[:, None],
        (bl / (inv["surface"][a] + eps))[:, None],
        (bl / (inv["surface"][b] + eps))[:, None],
        np.abs(mua - mub) / (mua + mub + eps),
        np.abs(inv["shape"][a] - inv["shape"][b]),
        (np.abs(np.sum(inv["principal"][a] * inv["principal"][b], axis=1))
         * np.minimum(inv["line_like"][a], inv["line_like"][b]))[:, None],
        ((e[:, 1] / blsafe) / 255.0)[:, None],
        np.log(np.maximum(Va, 1.0) / np.maximum(Vb, 1.0))[:, None],
        np.log(np.maximum(mua, eps) / np.maximum(mub, eps)),
        np.log(np.maximum(inv["compactness"][a], eps) / np.maximum(inv["compactness"][b], eps))[:, None],
    ]
    return np.concatenate(cols, axis=1).astype(np.float32)


def build_graph(vid):
    """Luke's band-flood Stage 1 + v15 protection mask → tractable subgraph."""
    ct = np.load(os.path.join(DATA_ROOT, "ct", f"volume-{vid}.npy")).astype(np.float32)
    seg = np.load(os.path.join(DATA_ROOT, "seg", f"segmentation-{vid}.npy")).astype(np.int32)

    # Liver crop
    organ_mask = (seg == 1) | (seg == 2)
    slc = bbox_from_mask(organ_mask)
    ct_crop = ct[slc]; seg_crop = seg[slc]; organ_crop = organ_mask[slc]

    ct_u8 = np.clip(ct_crop, HU_MIN, HU_MAX)
    ct_u8 = ((ct_u8 - HU_MIN) / (HU_MAX - HU_MIN) * 255).round().astype(np.uint8)
    ct_u8_4d = np.ascontiguousarray(ct_u8[..., np.newaxis])
    mask = (seg_crop == 2).astype(np.uint8)

    # Stage 1: Luke's band-flood
    nf, ei, ef, labels, adj = fastloops_band.band_build(ct_u8_4d, mask, band_of, deleted)
    raw_nf = np.asarray(nf); raw_ei = np.asarray(ei); raw_ef = np.asarray(ef)
    labels_np = np.asarray(labels)
    n_total = raw_nf.shape[0]
    if n_total == 0:
        return None

    # Protection mask: intensity below liver_mean - STD_MULT * liver_std
    liver_vals = ct_u8[organ_crop]
    threshold = liver_vals.mean() - STD_MULT * liver_vals.std()
    candidate = organ_crop & (ct_u8 < threshold)
    protect = binary_dilation(candidate, iterations=DILATE).astype(bool)

    # Identify which supernodes overlap with protected region
    flat_labels = labels_np.ravel()
    flat_prot = protect.ravel()
    valid = flat_labels >= 0
    node_prot = np.zeros(n_total, dtype=bool)
    if valid.any():
        max_id = int(flat_labels[valid].max())
        pc = np.bincount(flat_labels[valid & flat_prot], minlength=max_id + 1)
        node_prot[:min(n_total, len(pc))] = pc[:n_total] > 0

    # Extract protected subgraph
    prot_ids = np.where(node_prot)[0]
    if len(prot_ids) == 0:
        return None

    old2new = -np.ones(n_total, dtype=np.int64)
    old2new[prot_ids] = np.arange(len(prot_ids))
    sub_nf = raw_nf[prot_ids]

    # Filter edges: keep only edges where BOTH endpoints are protected
    if raw_ei.shape[1] > 0:
        src, dst = raw_ei[0], raw_ei[1]
        emask = node_prot[src] & node_prot[dst]
        sub_ei = np.stack([old2new[src[emask]], old2new[dst[emask]]])
        sub_ef = raw_ef[emask]
    else:
        sub_ei = np.zeros((2, 0), dtype=np.int64)
        sub_ef = np.zeros((0, 2), dtype=np.float32)

    # Node features
    inv = node_invariants(sub_nf)
    x = np.column_stack([
        np.log1p(inv["V"]), inv["chan"][:, 0],
        inv["shape"][:, 0], inv["shape"][:, 1], inv["shape"][:, 2],
        inv["compactness"], inv["degenerate"].astype(np.float32),
    ]).astype(np.float32)
    for c in range(x.shape[1]):
        mu, sig = x[:, c].mean(), x[:, c].std()
        if sig > 1e-8: x[:, c] = (x[:, c] - mu) / sig
        else: x[:, c] = 0.0

    # Edge features
    E = sub_ei.shape[1]
    if E > 0:
        ea = edge_features_fn(sub_nf, sub_ei, sub_ef)
        ei_fwd = torch.tensor(sub_ei, dtype=torch.long)
        ei_rev = torch.stack([ei_fwd[1], ei_fwd[0]])
        edge_index = torch.cat([ei_fwd, ei_rev], dim=1)
        edge_attr = torch.tensor(np.concatenate([ea, ea]), dtype=torch.float32)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, 12), dtype=torch.float32)

    # Labels from GT overlap
    n_fg = np.zeros(len(prot_ids), dtype=np.float64)
    n_bg = np.zeros(len(prot_ids), dtype=np.float64)
    gt = (seg_crop.ravel() == 2).astype(np.float64)
    if valid.any():
        max_id = int(flat_labels[valid].max())
        ofg = np.bincount(flat_labels[valid], weights=gt[valid], minlength=max_id + 1)
        otot = np.bincount(flat_labels[valid], minlength=max_id + 1)
        obg = otot - ofg
        for new_i, old_i in enumerate(prot_ids):
            if old_i < len(ofg):
                n_fg[new_i] = ofg[old_i]
                n_bg[new_i] = obg[old_i]

    y = (n_fg / np.maximum(n_fg + n_bg, 1) >= 0.10).astype(np.int64)

    data = Data(x=torch.tensor(x, dtype=torch.float32),
                edge_index=edge_index, edge_attr=edge_attr,
                y=torch.tensor(y, dtype=torch.long),
                n_fg=torch.tensor(n_fg, dtype=torch.float32),
                n_bg=torch.tensor(n_bg, dtype=torch.float32))
    return data, labels_np, prot_ids, seg_crop, n_total


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


def discover_volumes():
    vids = []
    for f in sorted(os.listdir(os.path.join(DATA_ROOT, "ct"))):
        m = re.match(r"volume-(\d+)\.npy", f)
        if m:
            vid = int(m.group(1))
            seg = np.load(os.path.join(DATA_ROOT, "seg", f"segmentation-{vid}.npy"))
            if (seg == 2).sum() > 0: vids.append(vid)
    return sorted(vids)


def main():
    pr("=" * 70)
    pr("  v16: Evaluate saved model — Luke's Band-Flood + GINE")
    pr("=" * 70)

    all_vids = discover_volumes()
    perm = np.random.permutation(len(all_vids))
    n_tr = int(0.7 * len(all_vids)); n_va = int(0.15 * len(all_vids))
    train_ids = sorted([all_vids[i] for i in perm[:n_tr]])
    val_ids = sorted([all_vids[i] for i in perm[n_tr:n_tr+n_va]])
    test_ids = sorted([all_vids[i] for i in perm[n_tr+n_va:]])
    pr(f"  {len(all_vids)} volumes: {len(train_ids)} train / {len(val_ids)} val / {len(test_ids)} test")

    # Build graphs
    pr(f"\nBuilding graphs (protection: std_mult={STD_MULT}, dilate={DILATE})...")
    graphs = {}; metas = {}
    t0 = time.time()
    for vid in train_ids + val_ids + test_ids:
        result = build_graph(vid)
        if result is None:
            pr(f"  vol-{vid}: SKIP"); continue
        data, labels_np, prot_ids, seg_crop, n_total = result
        graphs[vid] = data
        metas[vid] = dict(labels=labels_np, prot_ids=prot_ids, seg=seg_crop, n_total=n_total)
        split = "train" if vid in train_ids else ("val" if vid in val_ids else "test")
        N = data.num_nodes; n_tu = int((data.y == 1).sum())
        fg = data.n_fg.numpy(); bg = data.n_bg.numpy(); gt_t = fg.sum()
        maj = fg > bg
        oracle = float(2*fg[maj].sum()/(2*fg[maj].sum()+bg[maj].sum()+(gt_t-fg[maj].sum())+1e-8)) if gt_t > 0 else 0
        pr(f"  vol-{vid:>3d} [{split:>5s}]: {N:>8,} nodes ({n_tu:>5,} tu) {data.num_edges:>9,} edges  "
           f"oracle={oracle:.4f}  (was {n_total:,})")

    dt = time.time() - t0
    nodes = [g.num_nodes for g in graphs.values()]
    pr(f"\nBuilt {len(graphs)} graphs in {dt:.0f}s")
    pr(f"Nodes: mean={np.mean(nodes):,.0f} median={np.median(nodes):,.0f} max={max(nodes):,} min={min(nodes):,}")
    oracles = []
    for g in graphs.values():
        fg = g.n_fg.numpy(); bg = g.n_bg.numpy(); gt_t = fg.sum()
        if gt_t > 0:
            maj = fg > bg; oracles.append(float(2*fg[maj].sum()/(2*fg[maj].sum()+bg[maj].sum()+(gt_t-fg[maj].sum())+1e-8)))
    pr(f"Oracle Dice: mean={np.mean(oracles):.4f}")

    # Train GINE
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    pr(f"\nDevice: {device}")
    if torch.cuda.is_available():
        pr(f"GPU: {torch.cuda.get_device_name(0)}")

    MAX_NODES = 500_000
    trainable = [v for v in train_ids if v in graphs and graphs[v].num_nodes <= MAX_NODES]
    val_usable = [v for v in val_ids if v in graphs and graphs[v].num_nodes <= MAX_NODES]
    pr(f"Trainable: {len(trainable)}/{len(train_ids)}, Val: {len(val_usable)}/{len(val_ids)}")

    total_pos = sum(int((graphs[v].y==1).sum()) for v in trainable)
    total_neg = sum(int((graphs[v].y==0).sum()) for v in trainable)
    ratio = total_neg / max(total_pos, 1)
    eff = min(np.sqrt(ratio), 30.0)
    cw = torch.tensor([1.0, eff], dtype=torch.float32).to(device)
    pr(f"Class weight: [1.0, {eff:.1f}] (ratio {ratio:.0f}:1)")

    nd = 7; ed = 12
    model = GINE(nd, ed).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    PATIENCE, EPOCHS, VAL_EVERY = 15, 200, 3
    best_dice, best_state, wait = -1.0, None, 0

    for epoch in range(1, EPOCHS+1):
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
        if proc == 0: pr("All OOM"); break
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
                        mc = model.cpu()
                        preds = mc(g.x, g.edge_index, g.edge_attr).argmax(1).numpy()
                        model.to(device)
                    # Lift: subgraph node → original supernode → voxel
                    labels_np = m["labels"]; prot_ids = m["prot_ids"]
                    flat = labels_np.ravel(); valid = flat >= 0
                    if not valid.any(): continue
                    mid = int(flat[valid].max())
                    lut = np.zeros(mid + 1, dtype=np.int8)
                    for new_i, old_i in enumerate(prot_ids):
                        if old_i <= mid and new_i < len(preds):
                            lut[old_i] = preds[new_i]
                    pm = np.where(valid, lut[flat], 0).reshape(labels_np.shape).astype(bool)
                    gm = m["seg"] == 2; inter = int((pm & gm).sum())
                    tp += inter; fp += int(pm.sum()) - inter; fn += int(gm.sum()) - inter
            vd = 2*tp/(2*tp+fp+fn+1e-8)
            if vd > best_dice:
                best_dice = vd; best_state = {k: v.cpu().clone() for k,v in model.state_dict().items()}
                wait = 0; mk = " *"
            else: wait += 1; mk = ""
            pr(f"  Epoch {epoch:3d}  loss={ml:.4f}  val_dice={vd:.4f}  ({proc} vols){mk}")
            if wait >= PATIENCE: pr(f"  Early stop, best={best_dice:.4f}"); break
        else:
            pr(f"  Epoch {epoch:3d}  loss={ml:.4f}  ({proc} vols)")

    if best_state: model.load_state_dict(best_state)
    pr(f"\nBest val Dice: {best_dice:.4f}")
    torch.save(best_state or model.state_dict(), os.path.join(RESULTS_DIR, "model.pt"))

    # Evaluate all splits
    pr("\nEvaluating...")
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
                model_cpu = model.cpu(); model_cpu.eval()
                preds = model_cpu(g.x.cpu(), g.edge_index.cpu(),
                                  g.edge_attr.cpu()).argmax(1).numpy()
                model.to(device)

        # Lift: subgraph node → original supernode → voxel
        labels_np = m["labels"]; prot_ids = m["prot_ids"]
        flat = labels_np.ravel(); valid = flat >= 0
        pm = np.zeros(labels_np.shape, dtype=bool)
        if valid.any():
            mid = int(flat[valid].max())
            lut = np.zeros(mid + 1, dtype=np.int8)
            for new_i, old_i in enumerate(prot_ids):
                if old_i <= mid and new_i < len(preds):
                    lut[old_i] = preds[new_i]
            pm = np.where(valid, lut[flat], 0).reshape(labels_np.shape).astype(bool)
        gm = m["seg"] == 2; inter = int((gm & pm).sum())
        dice = 2.0*inter/(gm.sum()+pm.sum()+1e-8)
        rec = inter/(gm.sum()+1e-8); prec = inter/(pm.sum()+1e-8) if pm.sum()>0 else 0.0

        fg = g.n_fg.detach().cpu().numpy(); bg = g.n_bg.detach().cpu().numpy(); gt_t = fg.sum()
        maj = fg > bg
        oracle = float(2*fg[maj].sum()/(2*fg[maj].sum()+bg[maj].sum()+(gt_t-fg[maj].sum())+1e-8)) if gt_t > 0 else 0

        split = "train" if vid in train_ids else ("val" if vid in val_ids else "test")
        results.append(dict(vid=vid, split=split, dice=float(dice), recall=float(rec),
                            precision=float(prec), oracle=oracle, nodes=g.num_nodes))
        pr(f"  vol-{vid:>3d} [{split:>5s}]: Dice={dice:.4f}  Recall={rec:.4f}  Prec={prec:.4f}  Oracle={oracle:.4f}")

    # Summary
    pr(f"\n{'='*80}")
    pr(f"  v16: Luke's Band-Flood + Liver Crop + GINE — Results")
    pr(f"{'='*80}")
    for split in ["train", "val", "test"]:
        s = [r for r in results if r["split"] == split]
        if s:
            pr(f"  {split:>5s}: Dice={np.mean([r['dice'] for r in s]):.4f}  "
               f"Recall={np.mean([r['recall'] for r in s]):.4f}  "
               f"Prec={np.mean([r['precision'] for r in s]):.4f}  "
               f"Oracle={np.mean([r['oracle'] for r in s]):.4f}  "
               f"Nodes={np.mean([r['nodes'] for r in s]):,.0f}  (n={len(s)})")
    pr(f"\n  Paper target: 0.891")

    with open(os.path.join(RESULTS_DIR, "results.json"), "w") as f:
        json.dump(dict(results=results), f, indent=2)
    pr(f"  Saved to {RESULTS_DIR}/results.json")


if __name__ == "__main__":
    main()
