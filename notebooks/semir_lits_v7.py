"""
SEMIR LiTS v7 — Clean restart from Luke's original Rust crate.

Uses Luke's exact code from rust_crate.ipynb:
  - Seed-based contraction (fixed seed comparison)
  - Cut edges KEPT in graph with cut_frac feature
  - Raw voxel distance for edge features
  - HU [-50, 250]
  - No node deletion initially

Procedure:
  1. Oracle-first parameter search
  2. Feature discrimination check
  3. Train GINE only if oracle >= 0.75 AND features discriminate
"""

import numpy as np
import os, re, time, json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GINEConv, BatchNorm
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import fastloops

DATA_ROOT = "/scratch/ud3d4/acm_data/Data"
RESULTS_DIR = "/home/ud3d4/Desktop/SWOG/results/semir_lits_v7"
GRAPH_CACHE = "/dev/shm/semir_v7_graphs"
FEW_DIR = "/dev/shm/semir_v7_fewshot"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Luke's defaults
HU_MIN, HU_MAX = -50, 250
N_WORKERS = min(mp.cpu_count(), 32)
np.random.seed(42)
torch.manual_seed(42)


def pr(msg=""):
    print(msg, flush=True)


def section(title):
    pr(f"\n{'='*70}")
    pr(f"  {title}")
    pr(f"{'='*70}")


# ============================================================
# Data
# ============================================================

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


# ============================================================
# Oracle + diagnostics
# ============================================================

def oracle_dice_multi(labels_np, seg):
    flat = labels_np.ravel()
    gt = (seg.ravel() == 2).astype(np.float64)
    gt_total = int(gt.sum())
    valid = flat >= 0
    result = {}
    if gt_total == 0 or not valid.any():
        for th in [0.01, 0.05, 0.10, 0.25, 0.50]:
            result[f"oracle_{th:.2f}"] = 0.0
        result["tumor_deleted_pct"] = 0.0
        return result
    max_id = int(flat[valid].max())
    tc = np.bincount(flat[valid], weights=gt[valid], minlength=max_id + 1)
    total_c = np.bincount(flat[valid], minlength=max_id + 1)
    overlap = tc / np.maximum(total_c, 1)
    gt_mask = seg == 2
    for th in [0.01, 0.05, 0.10, 0.25, 0.50]:
        tumor_sids = np.where(overlap > th)[0]
        lut = np.zeros(max_id + 1, dtype=np.int32)
        lut[tumor_sids] = 1
        pred = np.where(valid, lut[flat], 0).reshape(labels_np.shape).astype(bool)
        inter = int((pred & gt_mask).sum())
        result[f"oracle_{th:.2f}"] = 2.0 * inter / (pred.sum() + gt_mask.sum() + 1e-8)
    del_tumor = int(gt[~valid].sum())
    result["tumor_deleted_pct"] = del_tumor / max(gt_total, 1) * 100.0
    return result


def graph_purity_stats(labels_np, seg):
    flat = labels_np.ravel(); valid = flat >= 0
    gt = (seg.ravel() == 2).astype(np.float64)
    if not valid.any() or gt.sum() == 0: return {}
    max_id = int(flat[valid].max())
    tc = np.bincount(flat[valid], weights=gt[valid], minlength=max_id + 1)
    total_c = np.bincount(flat[valid], minlength=max_id + 1)
    overlap = tc / np.maximum(total_c, 1)
    tt = tc > 0
    return {
        "num_nodes": int(max_id + 1),
        "tumor_touching": int(tt.sum()),
        "tumor_ge10": int((overlap >= 0.10).sum()),
        "tumor_ge50": int((overlap >= 0.50).sum()),
        "mean_overlap": float(overlap[tt].mean()) if tt.any() else 0.0,
    }


# ============================================================
# Oracle-first search
# ============================================================

def _eval_candidate(args):
    psi, alpha, meta_list, few_dir = args
    import fastloops as fl
    import numpy as _np

    oracles = {f"oracle_{th:.2f}": [] for th in [0.01, 0.05, 0.10, 0.25, 0.50]}
    sns, dels = [], []

    for m in meta_list:
        try:
            ct_u8 = _np.load(f"{few_dir}/ct_u8_{m['vid']}.npy")
            seg = _np.load(f"{few_dir}/seg_{m['vid']}.npy")
            n_vox = int(m["n_vox"])
            nf, ei, ef, labels, adj = fl.merge_and_cut(
                ct_u8, merge_distance=psi, cut_distance=alpha,
                delete_small_node_max_size=0,
                delete_large_node_min_size=n_vox + 1,
                delete_value_min=0, delete_value_max=255,
                connectivity="faces",
            )
            labels_np = _np.asarray(labels)
            od = oracle_dice_multi(labels_np, seg)
            for k in oracles:
                oracles[k].append(od.get(k, 0.0))
            dels.append(od["tumor_deleted_pct"])
            sns.append(nf.shape[0])
        except Exception:
            return None

    if not sns:
        return None
    row = {"psi": psi, "alpha": alpha,
           "sn": float(_np.mean(sns)),
           "tumor_deleted_pct": float(_np.mean(dels)),
           "edges": 0}
    for k in oracles:
        row[k] = float(_np.mean(oracles[k]))

    if row["oracle_0.10"] < 0.75:
        row["score"] = -1e9 + row["oracle_0.10"]
    elif row["tumor_deleted_pct"] > 5.0:
        row["score"] = -1e8 - row["tumor_deleted_pct"]
    else:
        target_sn = 5000.0
        sn_term = -abs(_np.log((row["sn"] + 1.0) / target_sn))
        row["score"] = (
            4.0 * row["oracle_0.10"]
            + 1.0 * row["oracle_0.25"]
            + 0.2 * sn_term
            - 0.05 * row["tumor_deleted_pct"]
        )
    return row


# ============================================================
# Luke's feature extraction (from rust_crate.ipynb)
# ============================================================

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
    cut_frac = ef[:, 3] / blsafe  # Luke's cut_frac feature
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


# ============================================================
# GINE
# ============================================================

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


# ============================================================
# MAIN
# ============================================================

def main():
    section("SEMIR LiTS v7 — Luke's original crate, clean start")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    pr(f"Device: {device}")
    pr(f"HU window: [{HU_MIN}, {HU_MAX}]")
    pr(f"Crate: Luke's original (seed comparison, cut edges kept)")

    all_vids = discover_volumes()
    pr(f"Found {len(all_vids)} LiTS volumes with tumor")

    np.random.seed(42)
    perm = np.random.permutation(len(all_vids))
    n_train = int(0.7 * len(all_vids)); n_val = int(0.15 * len(all_vids))
    train_ids = sorted([all_vids[i] for i in perm[:n_train]])
    val_ids = sorted([all_vids[i] for i in perm[n_train:n_train + n_val]])
    test_ids = sorted([all_vids[i] for i in perm[n_train + n_val:]])
    pr(f"Split: {len(train_ids)} train / {len(val_ids)} val / {len(test_ids)} test")

    # ================================================================
    # STEP 1: Oracle-first parameter search
    # ================================================================
    section("Step 1: Oracle-first parameter search")

    N_FEW = 10
    few_vids = train_ids[:N_FEW]
    pr(f"Few-shot volumes: {few_vids}")

    os.makedirs(FEW_DIR, exist_ok=True)
    few_meta = []
    for vid in few_vids:
        ct_raw, seg, ct_u8 = load_and_convert(vid)
        np.save(f"{FEW_DIR}/ct_u8_{vid}.npy", ct_u8)
        np.save(f"{FEW_DIR}/seg_{vid}.npy", seg)
        few_meta.append({"vid": vid, "n_vox": ct_raw.size})

    psi_values = [2, 3, 4, 5, 6, 8, 10, 12, 15]
    alpha_mult_values = [2.5, 3.5, 5.0, 7.0, 10.0]
    candidates = sorted(set(
        (psi, min(int(round(psi * am)), 180))
        for psi in psi_values for am in alpha_mult_values
    ))
    pr(f"Evaluating {len(candidates)} candidates on {N_FEW} cases...")

    work = [(psi, alpha, few_meta, FEW_DIR) for psi, alpha in candidates]
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        raw = list(pool.map(_eval_candidate, work))
    rows = [r for r in raw if r is not None]
    pr(f"Evaluated {len(rows)}/{len(candidates)} in {time.time()-t0:.1f}s")

    pr(f"\nTop 20 by oracle@0.10:")
    pr(f"{'o@.01':>7s} {'o@.10':>7s} {'o@.25':>7s} {'o@.50':>7s} {'SN':>10s} {'del%':>6s}  params")
    for r in sorted(rows, key=lambda x: x["oracle_0.10"], reverse=True)[:20]:
        pr(f"{r['oracle_0.01']:7.4f} {r['oracle_0.10']:7.4f} {r['oracle_0.25']:7.4f} "
           f"{r['oracle_0.50']:7.4f} {r['sn']:10.0f} {r['tumor_deleted_pct']:5.1f}%  "
           f"psi={r['psi']} alpha={r['alpha']}")

    viable = [r for r in rows if r["oracle_0.10"] >= 0.75 and r["tumor_deleted_pct"] <= 5.0]
    if viable:
        BEST = sorted(viable, key=lambda r: r["score"], reverse=True)[0]
        pr(f"\nSelected from {len(viable)} viable candidates")
    else:
        BEST = sorted(rows, key=lambda r: r["oracle_0.10"], reverse=True)[0]
        pr(f"\nWARNING: No candidate met oracle >= 0.75. Best-oracle fallback:")

    pr(json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in BEST.items()}, indent=2))
    PSI, ALPHA = int(BEST["psi"]), int(BEST["alpha"])

    with open(os.path.join(RESULTS_DIR, "search_results.json"), "w") as f:
        json.dump({"candidates": rows, "selected": BEST}, f, indent=2)
    import shutil; shutil.rmtree(FEW_DIR, ignore_errors=True)

    # ================================================================
    # STEP 2: Feature discrimination check
    # ================================================================
    section("Step 2: Feature discrimination check")
    pr(f"Building graph for vol-{few_vids[0]} with psi={PSI} alpha={ALPHA}...")

    ct_raw, seg, ct_u8 = load_and_convert(few_vids[0])
    nf, ei, ef, labels, adj = fastloops.merge_and_cut(
        ct_u8, merge_distance=PSI, cut_distance=ALPHA,
        delete_small_node_max_size=0, delete_large_node_min_size=ct_raw.size + 1,
        delete_value_min=0, delete_value_max=255, connectivity="faces")
    labels_np = np.asarray(labels)

    data = build_pyg_graph(nf, ei, ef, labels_np, seg, ct_u8, overlap_th=0.10)
    x = data.x.numpy(); y = data.y.numpy()
    feat_names = ['log_vol', 'log_surf', 'compact', 'elong', 'ax_x', 'ax_y', 'ax_z', 'mean_int', 'int_std']

    pr(f"\nFeature discrimination (vol-{few_vids[0]}):")
    pr(f"{'Feature':<12s} {'BG mean':>8s} {'TU mean':>8s} {'Cohen d':>8s}")
    cohens = []
    for i, name in enumerate(feat_names):
        bg = x[y==0, i]; tu = x[y==1, i]
        d = abs(tu.mean()-bg.mean()) / (np.sqrt((bg.std()**2+tu.std()**2)/2)+1e-8)
        cohens.append(d)
        pr(f"  {name:<12s} {bg.mean():8.3f} {tu.mean():8.3f} {d:8.4f}")
    max_d = max(cohens)
    pr(f"\nMax Cohen's d: {max_d:.4f}")
    pr(f"Assessment: {'GOOD (>0.8)' if max_d > 0.8 else 'MODERATE (0.5-0.8)' if max_d > 0.5 else 'WEAK (<0.5)'}")

    # Graph purity
    ps = graph_purity_stats(labels_np, seg)
    pr(f"\nGraph purity: {ps}")

    # Quick balanced classifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.utils import resample
    tumor_idx = np.where(y==1)[0]; bg_idx = np.where(y==0)[0]
    if len(tumor_idx) > 0:
        bg_sample = resample(bg_idx, n_samples=min(len(bg_idx), len(tumor_idx)), random_state=42)
        bal_idx = np.concatenate([tumor_idx, bg_sample])
        n_tr = int(0.7 * len(bal_idx))
        perm_bal = np.random.RandomState(42).permutation(len(bal_idx))
        lr = LogisticRegression(max_iter=1000)
        lr.fit(x[bal_idx[perm_bal[:n_tr]]], y[bal_idx[perm_bal[:n_tr]]])
        score = lr.score(x[bal_idx[perm_bal[n_tr:]]], y[bal_idx[perm_bal[n_tr:]]])
        pr(f"Balanced logistic regression accuracy: {score:.4f}")
    else:
        pr("No tumor supernodes — cannot test classifier")

    # ================================================================
    # STEP 3: Stopping criteria
    # ================================================================
    section("Step 3: Stopping criteria")
    oracle_pass = BEST["oracle_0.10"] >= 0.75
    del_pass = BEST["tumor_deleted_pct"] <= 5.0
    pr(f"  oracle@0.10 = {BEST['oracle_0.10']:.4f} >= 0.75: {'PASS' if oracle_pass else 'FAIL'}")
    pr(f"  tumor_del   = {BEST['tumor_deleted_pct']:.2f}% <= 5%: {'PASS' if del_pass else 'FAIL'}")
    pr(f"  supernodes  = {BEST['sn']:.0f}")
    pr(f"  max Cohen d = {max_d:.4f}")

    if not (oracle_pass and del_pass):
        pr("\nSTOPPING: Oracle criteria not met. Saving diagnostics.")
        with open(os.path.join(RESULTS_DIR, "results.json"), "w") as f:
            json.dump({"phase": "stopped_at_oracle", "selected": BEST, "max_cohens_d": max_d}, f, indent=2)
        return

    # ================================================================
    # STEP 4: Build all graphs + train GINE
    # ================================================================
    section("Step 4: Build graphs")
    os.makedirs(GRAPH_CACHE, exist_ok=True)
    OVERLAP_TH = 0.10
    MAX_NODES_GPU = 3_500_000

    graphs = {}; raw_labels_map = {}; raw_segs_map = {}; oracles = []
    for vid in train_ids + val_ids + test_ids:
        cache_path = os.path.join(GRAPH_CACHE, f"graph_{vid}.pt")
        lbl_cache = os.path.join(GRAPH_CACHE, f"labels_{vid}.npy")
        seg_cache = os.path.join(GRAPH_CACHE, f"seg_{vid}.npy")
        if os.path.exists(cache_path):
            g = torch.load(cache_path, weights_only=False)
            lnp = np.load(lbl_cache); s = np.load(seg_cache)
        else:
            ct_raw, s, ct_u8 = load_and_convert(vid)
            nf, ei, ef, lab, adj = fastloops.merge_and_cut(
                ct_u8, merge_distance=PSI, cut_distance=ALPHA,
                delete_small_node_max_size=0, delete_large_node_min_size=ct_raw.size+1,
                delete_value_min=0, delete_value_max=255, connectivity="faces")
            lnp = np.asarray(lab)
            g = build_pyg_graph(nf, ei, ef, lnp, s, ct_u8, overlap_th=OVERLAP_TH)
            torch.save(g, cache_path); np.save(lbl_cache, lnp); np.save(seg_cache, s)

        graphs[vid] = g; raw_labels_map[vid] = lnp; raw_segs_map[vid] = s
        od = oracle_dice_multi(lnp, s)
        oracles.append(od["oracle_0.10"])
        split = "train" if vid in train_ids else ("val" if vid in val_ids else "test")
        n_tu = int((g.y==1).sum()); n_bg = int((g.y==0).sum())
        pr(f"  vol-{vid} [{split}]: {g.num_nodes:,} nodes ({n_tu} tu, {n_bg:,} bg), "
           f"{g.num_edges:,} edges, oracle={od['oracle_0.10']:.4f}")

    pr(f"\nMean oracle@0.10: {np.mean(oracles):.4f}")

    trainable = [v for v in train_ids if v in graphs and graphs[v].num_nodes <= MAX_NODES_GPU]
    val_usable = [v for v in val_ids if v in graphs]
    pr(f"Trainable: {len(trainable)}/{len(train_ids)}")

    # Class weights
    total_pos = sum(int((graphs[v].y==1).sum()) for v in trainable)
    total_neg = sum(int((graphs[v].y==0).sum()) for v in trainable)
    ratio = total_neg / max(total_pos, 1)
    eff = min(np.sqrt(ratio), 30.0)
    class_weight = torch.tensor([1.0, eff], dtype=torch.float32).to(device)
    pr(f"Class weight: [1.0, {eff:.1f}] (ratio: {ratio:.0f}:1)")

    nd = graphs[trainable[0]].x.shape[1]
    ed = graphs[trainable[0]].edge_attr.shape[1] if graphs[trainable[0]].edge_attr.numel() > 0 else 10

    # ================================================================
    # STEP 5: Train GINE
    # ================================================================
    section("Step 5: Train GINE")
    model = GINE(nd, ed).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    PATIENCE = 15; EPOCHS = 200; VAL_EVERY = 3
    best_dice, best_state, wait = -1.0, None, 0
    history = {"train_loss": [], "val_dice": []}

    for epoch in range(1, EPOCHS+1):
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
            raise RuntimeError("No graphs processed — all OOM")
        mean_loss = epoch_loss / processed
        history["train_loss"].append(mean_loss)

        if epoch % VAL_EVERY == 0 or epoch <= 3:
            model.eval(); tp = fp = fn = 0; fg_rates = []
            with torch.no_grad():
                for vid in val_usable:
                    g = graphs[vid]; lnp = raw_labels_map[vid]; s = raw_segs_map[vid]
                    try:
                        gd = g.to(device)
                        logits = model(gd.x, gd.edge_index, gd.edge_attr)
                        preds = logits.argmax(dim=1)
                        fg_rates.append(float((preds==1).float().mean().cpu()))
                        preds = preds.cpu().numpy()
                        del gd, logits; torch.cuda.empty_cache()
                    except (torch.cuda.OutOfMemoryError, RuntimeError):
                        try: del gd
                        except: pass
                        torch.cuda.empty_cache()
                        mc = model.cpu(); logits = mc(g.x, g.edge_index, g.edge_attr)
                        preds = logits.argmax(dim=1).numpy()
                        fg_rates.append(float((torch.tensor(preds)==1).float().mean()))
                        del logits; model.to(device)
                    flat = lnp.ravel(); valid = flat >= 0
                    if not valid.any(): continue
                    mid = int(flat[valid].max())
                    lut = np.zeros(mid+1, dtype=np.int8)
                    lut[:min(len(preds), mid+1)] = preds[:min(len(preds), mid+1)]
                    pm = np.where(valid, lut[flat], 0).reshape(lnp.shape).astype(bool)
                    gm = s == 2; inter = int((pm & gm).sum())
                    tp += inter; fp += int(pm.sum()) - inter; fn += int(gm.sum()) - inter

            vd = 2*tp/(2*tp+fp+fn+1e-8)
            history["val_dice"].append(vd)
            mfg = np.mean(fg_rates) if fg_rates else 0.0
            if vd > best_dice:
                best_dice = vd
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                wait = 0; marker = " *"
            else:
                wait += 1; marker = ""
            pr(f"  Epoch {epoch:3d}  loss={mean_loss:.4f}  val_dice={vd:.4f}  fg={mfg:.4f}  ({processed} vols){marker}")
            if wait >= PATIENCE:
                pr(f"  Early stop epoch {epoch}, best val Dice={best_dice:.4f}"); break
        else:
            pr(f"  Epoch {epoch:3d}  loss={mean_loss:.4f}  ({processed} vols)")

    if best_state: model.load_state_dict(best_state)
    pr(f"\nBest val Dice: {best_dice:.4f}")
    torch.save(best_state or model.state_dict(), os.path.join(RESULTS_DIR, "model.pt"))

    # ================================================================
    # STEP 6: Final evaluation
    # ================================================================
    section("Step 6: Final evaluation")
    model.eval(); results = []
    for vid in sorted(graphs.keys()):
        g = graphs[vid]; lnp = raw_labels_map[vid]; s = raw_segs_map[vid]
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
            lut = np.zeros(mid+1, dtype=np.int8)
            lut[:min(len(preds), mid+1)] = preds[:min(len(preds), mid+1)]
            pm = np.where(valid, lut[flat], 0).reshape(lnp.shape).astype(bool)
        gm = s == 2; inter = int((gm & pm).sum())
        dice = 2.0*inter/(gm.sum()+pm.sum()+1e-8)
        rec = inter/(gm.sum()+1e-8)
        prec = inter/(pm.sum()+1e-8) if pm.sum() > 0 else 0.0
        split = "train" if vid in train_ids else ("val" if vid in val_ids else "test")
        results.append({"vid": vid, "split": split, "dice": float(dice), "recall": float(rec), "precision": float(prec)})

    section("SUMMARY")
    for split in ["train", "val", "test"]:
        scores = [r["dice"] for r in results if r["split"] == split]
        if scores:
            pr(f"  {split:>5s}: Dice = {np.mean(scores):.4f} +/- {np.std(scores):.4f}  (n={len(scores)})")
    pr(f"\n  Params: psi={PSI} alpha={ALPHA} HU=[{HU_MIN},{HU_MAX}]")
    pr(f"  Oracle@0.10: {np.mean(oracles):.4f}")
    pr(f"  Best val Dice: {best_dice:.4f}")
    pr(f"  Paper target: Dice 0.891 +/- 0.007")

    with open(os.path.join(RESULTS_DIR, "results.json"), "w") as f:
        json.dump({"params": {"psi": PSI, "alpha": ALPHA, "hu_min": HU_MIN, "hu_max": HU_MAX, "overlap_th": OVERLAP_TH},
                    "selected": BEST, "mean_oracle": float(np.mean(oracles)),
                    "best_val_dice": float(best_dice), "max_cohens_d": float(max_d),
                    "epochs": len(history["train_loss"]), "history": history, "results": results}, f, indent=2)
    pr(f"\n  Saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
