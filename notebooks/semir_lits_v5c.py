"""
SEMIR LiTS v5c — Full-graph GPU training (no NeighborLoader)

v5a: NeighborLoader, 2.5h/epoch, val_dice=0.000 (voxel-weighted loss killed signal)
v5b: NeighborLoader + subsampling, still slow
v5c: Full-graph on GPU. 0.3s per volume. ~30s per epoch for 101 volumes.
     Skip 17 volumes > 2.7M nodes that OOM on 49GB L40S.
"""

import numpy as np
import os, re, time, json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GINEConv, BatchNorm
import fastloops

DATA_ROOT = "/scratch/ud3d4/acm_data/Data"
RESULTS_DIR = "/home/ud3d4/Desktop/SWOG/results/semir_lits_v5"
GRAPH_CACHE = "/dev/shm/semir_v5_graphs"
os.makedirs(RESULTS_DIR, exist_ok=True)

HU_MIN, HU_MAX = 0, 200
MERGE_DIST = 3
CUT_DIST = 15         # also used as consolidation threshold in iterative merging
DELETE_SMALL = 0
DELETE_LARGE_FRAC = 1.00
VALUE_MIN, VALUE_MAX = 0, 255
OVERLAP_THRESHOLD = 0.10
MAX_NODES_GPU = 3_500_000  # skip graphs larger than this (consolidated graphs are smaller)

np.random.seed(42)
torch.manual_seed(42)


def pr(msg=""):
    print(msg, flush=True)


def section(title):
    pr(f"\n{'='*70}")
    pr(f"  {title}")
    pr(f"{'='*70}")


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


def oracle_dice(labels_np, seg, overlap_th=OVERLAP_THRESHOLD):
    flat = labels_np.ravel()
    gt = (seg.ravel() == 2).astype(np.float64)
    gt_total = int(gt.sum())
    valid = flat >= 0
    if gt_total == 0 or not valid.any():
        return 0.0
    max_id = int(flat[valid].max())
    tc = np.bincount(flat[valid], weights=gt[valid], minlength=max_id + 1)
    total_c = np.bincount(flat[valid], minlength=max_id + 1)
    overlap = tc / np.maximum(total_c, 1)
    tumor_sids = np.where(overlap > overlap_th)[0]
    lut = np.zeros(max_id + 1, dtype=np.int32)
    lut[tumor_sids] = 1
    pred = np.where(valid, lut[flat], 0).reshape(labels_np.shape).astype(bool)
    gt_mask = seg == 2
    inter = int((pred & gt_mask).sum())
    return 2.0 * inter / (pred.sum() + gt_mask.sum() + 1e-8)


def compute_intensity_std(labels_np, ct_u8):
    flat = labels_np.ravel()
    valid = flat >= 0
    if not valid.any():
        return np.array([], dtype=np.float32)
    max_id = int(flat[valid].max())
    vals = ct_u8[..., 0].ravel().astype(np.float64) / 255.0
    counts = np.bincount(flat[valid], minlength=max_id + 1).astype(np.float64)
    sums = np.bincount(flat[valid], weights=vals[valid], minlength=max_id + 1)
    sq_sums = np.bincount(flat[valid], weights=vals[valid] ** 2, minlength=max_id + 1)
    mean = sums / np.maximum(counts, 1.0)
    var = sq_sums / np.maximum(counts, 1.0) - mean ** 2
    return np.sqrt(np.maximum(var, 0.0)).astype(np.float32)


def _layout(C):
    return dict(area=0, s=[1, 2, 3],
                cov=[(4, 0, 0), (5, 1, 1), (6, 2, 2), (7, 0, 1), (8, 0, 2), (9, 1, 2)],
                chan0=10, boundary=10 + C + 6, D=3)


def node_invariants(node_feats, C=1, eps=1e-6):
    f = node_feats.astype(np.float64)
    L = _layout(C)
    D = L["D"]
    N = f.shape[0]
    V = f[:, L["area"]]
    Vsafe = np.maximum(V, 1.0)
    mean_coord = np.stack([f[:, c] for c in L["s"]], axis=1) / Vsafe[:, None]
    cov = np.zeros((N, D, D))
    for col, i, j in L["cov"]:
        cij = f[:, col] / Vsafe - mean_coord[:, i] * mean_coord[:, j]
        cov[:, i, j] = cij; cov[:, j, i] = cij
    w = np.linalg.eigvalsh(cov)
    w = np.clip(w, 0.0, None)
    _, vec = np.linalg.eigh(cov)
    principal = vec[..., -1]
    trace = w.sum(axis=1)
    degenerate = trace < eps
    denom = w[:, 2] + eps
    shape = np.stack([(w[:, 2] - w[:, 1]) / denom,
                      (w[:, 1] - w[:, 0]) / denom,
                      w[:, 0] / denom], axis=1)
    shape[degenerate] = 0.0
    chan = f[:, L["chan0"]:L["chan0"] + C] / Vsafe[:, None] / 255.0
    compactness = f[:, L["boundary"]] / np.power(Vsafe, (D - 1.0) / D)
    elongation = np.where(w[:, 0] > eps, w[:, 2] / (w[:, 0] + eps), 1.0)
    elongation = np.clip(elongation, 1.0, 100.0)
    return dict(V=V, surface=f[:, L["boundary"]], centroid=mean_coord,
                eig=w, shape=shape, principal=principal,
                chan=chan, compactness=compactness, elongation=elongation)


def edge_invariants(node_feats, edge_index, edge_feats, C=1, eps=1e-6):
    inv = node_invariants(node_feats, C, eps)
    a = edge_index[0].astype(np.int64)
    b = edge_index[1].astype(np.int64)
    ef = edge_feats.astype(np.float64)
    blsafe = np.maximum(ef[:, 0], 1.0)
    size_contrast = np.abs(inv["V"][a] - inv["V"][b]) / (inv["V"][a] + inv["V"][b] + eps)
    bfrac_a = ef[:, 0] / (inv["surface"][a] + eps)
    bfrac_b = ef[:, 0] / (inv["surface"][b] + eps)
    mean_contrast = np.abs(inv["chan"][a] - inv["chan"][b])
    shape_dissim = np.abs(inv["shape"][a] - inv["shape"][b])
    line_a = inv["shape"][a, 0] if inv["shape"].ndim > 1 else np.zeros(len(a))
    line_b = inv["shape"][b, 0] if inv["shape"].ndim > 1 else np.zeros(len(b))
    axis_align = (np.abs(np.sum(inv["principal"][a] * inv["principal"][b], axis=1))
                  * np.minimum(line_a, line_b))
    bcontrast = (ef[:, 1] / blsafe) / 255.0
    cut_frac = ef[:, 3] / blsafe
    cols = [size_contrast[:, None], bfrac_a[:, None], bfrac_b[:, None],
            mean_contrast if mean_contrast.ndim > 1 else mean_contrast[:, None],
            shape_dissim, axis_align[:, None], bcontrast[:, None], cut_frac[:, None]]
    return np.concatenate(cols, axis=1).astype(np.float32)


def build_pyg_graph(raw_nf, raw_ei, raw_ef, labels_np, seg, ct_u8, C=1):
    n_sn = raw_nf.shape[0]
    inv = node_invariants(raw_nf, C)
    int_std = compute_intensity_std(labels_np, ct_u8)
    if len(int_std) < n_sn:
        int_std = np.pad(int_std, (0, n_sn - len(int_std)))
    int_std = int_std[:n_sn]
    principal = inv["principal"].astype(np.float32)
    if len(principal):
        max_comp = np.argmax(np.abs(principal), axis=1)
        signs = np.sign(principal[np.arange(len(principal)), max_comp])
        signs[signs == 0] = 1
        principal = principal * signs[:, None]
    x = np.column_stack([
        np.log1p(inv["V"]),
        np.log1p(inv["surface"]),
        inv["compactness"],
        inv["elongation"],
        principal[:, 0],
        principal[:, 1],
        principal[:, 2],
        inv["chan"][:, 0],
        int_std,
    ]).astype(np.float32)
    for col in range(x.shape[1]):
        mu, sigma = float(x[:, col].mean()), float(x[:, col].std())
        if sigma > 1e-8:
            x[:, col] = (x[:, col] - mu) / sigma
        else:
            x[:, col] = 0.0
    if raw_ei.shape[1] > 0:
        edge_attr = edge_invariants(raw_nf, raw_ei, raw_ef, C)
        ei_fwd = torch.tensor(raw_ei, dtype=torch.long)
        ei_rev = torch.stack([ei_fwd[1], ei_fwd[0]])
        edge_index = torch.cat([ei_fwd, ei_rev], dim=1)
        edge_attr_t = torch.tensor(np.concatenate([edge_attr, edge_attr]), dtype=torch.float32)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr_t = torch.zeros((0, 10), dtype=torch.float32)
    flat = labels_np.ravel()
    valid = flat >= 0
    gt = (seg.ravel() == 2).astype(np.float64)
    max_id = int(flat[valid].max()) if valid.any() else -1
    y = np.zeros(n_sn, dtype=np.int64)
    node_voxel_count = np.zeros(n_sn, dtype=np.float32)
    if max_id >= 0:
        tumor_count = np.bincount(flat[valid], weights=gt[valid], minlength=max_id + 1)
        total_count = np.bincount(flat[valid], minlength=max_id + 1)
        overlap = tumor_count / np.maximum(total_count, 1)
        y[:min(n_sn, len(overlap))] = (overlap[:n_sn] >= OVERLAP_THRESHOLD).astype(np.int64)
        node_voxel_count[:min(n_sn, len(total_count))] = total_count[:n_sn].astype(np.float32)
    return Data(
        x=torch.tensor(x, dtype=torch.float32),
        edge_index=edge_index,
        edge_attr=edge_attr_t,
        y=torch.tensor(y, dtype=torch.long),
        node_voxel_count=torch.tensor(node_voxel_count, dtype=torch.float32),
    )


class GINE(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden=128):
        super().__init__()
        self.edge_proj = nn.Linear(edge_dim, hidden)

        def mlp(d_in):
            return nn.Sequential(
                nn.Linear(d_in, hidden), nn.BatchNorm1d(hidden), nn.ReLU(),
                nn.Linear(hidden, hidden),
            )

        self.conv1 = GINEConv(mlp(node_dim), edge_dim=hidden)
        self.bn1 = BatchNorm(hidden)
        self.conv2 = GINEConv(mlp(hidden), edge_dim=hidden)
        self.bn2 = BatchNorm(hidden)
        self.conv3 = GINEConv(mlp(hidden), edge_dim=hidden)
        self.bn3 = BatchNorm(hidden)
        self.head = nn.Linear(hidden, 2)

    def forward(self, x, edge_index, edge_attr):
        if edge_attr is not None and edge_attr.numel() > 0:
            ea = self.edge_proj(edge_attr)
        else:
            n = x.size(0)
            edge_index = torch.stack([torch.arange(n, device=x.device)] * 2)
            ea = torch.zeros(n, self.edge_proj.out_features, device=x.device)
        x = F.relu(self.bn1(self.conv1(x, edge_index, ea)))
        x = F.relu(self.bn2(self.conv2(x, edge_index, ea)))
        x = F.relu(self.bn3(self.conv3(x, edge_index, ea)))
        return self.head(x)


def lifted_voxel_dice_batch(model, vids, graphs, raw_labels, raw_segs, device):
    """Compute aggregate lifted voxel Dice across volumes."""
    model.eval()
    tp = fp = fn = 0
    with torch.no_grad():
        for vid in vids:
            g = graphs[vid]
            labels_np = raw_labels[vid]
            seg = raw_segs[vid]

            try:
                g_dev = g.to(device)
                logits = model(g_dev.x, g_dev.edge_index, g_dev.edge_attr)
                preds = logits.argmax(dim=1).cpu().numpy()
                del g_dev, logits
                torch.cuda.empty_cache()
            except (torch.cuda.OutOfMemoryError, RuntimeError):
                try:
                    del g_dev
                except NameError:
                    pass
                torch.cuda.empty_cache()
                # CPU fallback for large graphs
                model_cpu = model.cpu()
                logits = model_cpu(g.x, g.edge_index, g.edge_attr)
                preds = logits.argmax(dim=1).numpy()
                del logits
                model.to(device)

            flat = labels_np.ravel()
            valid = flat >= 0
            if not valid.any():
                continue
            max_id = int(flat[valid].max())
            lut = np.zeros(max_id + 1, dtype=np.int8)
            lut[:min(len(preds), max_id + 1)] = preds[:min(len(preds), max_id + 1)]
            pred_mask = np.where(valid, lut[flat], 0).reshape(labels_np.shape).astype(bool)
            gt_mask = seg == 2
            inter = int((pred_mask & gt_mask).sum())
            tp += inter
            fp += int(pred_mask.sum()) - inter
            fn += int(gt_mask.sum()) - inter

    return 2 * tp / (2 * tp + fp + fn + 1e-8)


def main():
    section("SEMIR LiTS v5c — Full-graph GPU training")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    pr(f"Device: {device}")
    if torch.cuda.is_available():
        pr(f"GPU: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory / 1e9:.0f}GB)")

    # ---- Discover and split ----
    all_vids = discover_volumes()
    pr(f"Found {len(all_vids)} LiTS volumes with tumor")

    np.random.seed(42)
    perm = np.random.permutation(len(all_vids))
    n_train = int(0.7 * len(all_vids))
    n_val = int(0.15 * len(all_vids))
    train_ids = sorted([all_vids[i] for i in perm[:n_train]])
    val_ids = sorted([all_vids[i] for i in perm[n_train:n_train + n_val]])
    test_ids = sorted([all_vids[i] for i in perm[n_train + n_val:]])
    pr(f"Split: {len(train_ids)} train / {len(val_ids)} val / {len(test_ids)} test")

    # ---- Build or load cached graphs ----
    section("Building / loading graphs")
    graphs = {}
    raw_labels = {}
    raw_segs = {}
    oracles = []
    skipped = []

    for vid in train_ids + val_ids + test_ids:
        cache_path = os.path.join(GRAPH_CACHE, f"graph_{vid}.pt")
        labels_cache = os.path.join(GRAPH_CACHE, f"labels_{vid}.npy")
        seg_cache = os.path.join(GRAPH_CACHE, f"seg_{vid}.npy")

        if os.path.exists(cache_path):
            data = torch.load(cache_path, weights_only=False)
            labels_np = np.load(labels_cache)
            seg = np.load(seg_cache)
        else:
            ct_raw = np.load(os.path.join(DATA_ROOT, "ct", f"volume-{vid}.npy")).astype(np.float32)
            seg = np.load(os.path.join(DATA_ROOT, "seg", f"segmentation-{vid}.npy")).astype(np.int32)
            n_vox = ct_raw.size
            ct_u8 = np.clip(ct_raw, HU_MIN, HU_MAX)
            ct_u8 = ((ct_u8 - HU_MIN) / (HU_MAX - HU_MIN) * 255).round().astype(np.uint8)
            ct_u8 = np.ascontiguousarray(ct_u8[..., np.newaxis])

            raw_nf, raw_ei, raw_ef, raw_lab, raw_adj = fastloops.merge_and_cut(
                ct_u8, merge_distance=MERGE_DIST, cut_distance=CUT_DIST,
                delete_small_node_max_size=DELETE_SMALL,
                delete_large_node_min_size=int(n_vox ** DELETE_LARGE_FRAC),
                delete_value_min=VALUE_MIN, delete_value_max=VALUE_MAX,
                connectivity="faces",
            )
            labels_np = np.asarray(raw_lab)
            data = build_pyg_graph(raw_nf, raw_ei, raw_ef, labels_np, seg, ct_u8)

            torch.save(data, cache_path)
            np.save(labels_cache, labels_np)
            np.save(seg_cache, seg)

        graphs[vid] = data
        raw_labels[vid] = labels_np
        raw_segs[vid] = seg

        od = oracle_dice(labels_np, seg)
        oracles.append(od)

    # Filter by GPU capacity
    trainable = [v for v in train_ids if v in graphs and graphs[v].num_nodes <= MAX_NODES_GPU]
    skipped_train = [v for v in train_ids if v in graphs and graphs[v].num_nodes > MAX_NODES_GPU]
    val_usable = [v for v in val_ids if v in graphs]
    test_usable = [v for v in test_ids if v in graphs]

    pr(f"\nLoaded {len(graphs)} graphs")
    pr(f"Trainable (≤{MAX_NODES_GPU:,} nodes): {len(trainable)}/{len(train_ids)}")
    if skipped_train:
        pr(f"Skipped (too large): {skipped_train}")
    pr(f"Mean oracle Dice: {np.mean(oracles):.4f}")

    # ---- Class weights ----
    total_pos = sum(int((graphs[v].y == 1).sum()) for v in trainable)
    total_neg = sum(int((graphs[v].y == 0).sum()) for v in trainable)
    raw_ratio = total_neg / max(total_pos, 1)
    eff_ratio = min(np.sqrt(raw_ratio), 30.0)
    class_weight = torch.tensor([1.0, eff_ratio], dtype=torch.float32).to(device)
    pr(f"Class weight: [1.0, {eff_ratio:.1f}] (imbalance: {raw_ratio:.0f}:1)")
    pr(f"Tumor SN: {total_pos:,}; Background: {total_neg:,}")

    # ---- Model ----
    node_dim = graphs[trainable[0]].x.shape[1]
    edge_dim = graphs[trainable[0]].edge_attr.shape[1]
    pr(f"Node dim: {node_dim}; Edge dim: {edge_dim}")

    model = GINE(node_dim=node_dim, edge_dim=edge_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    # ---- Training ----
    section("Training (full-graph on GPU)")
    PATIENCE = 15
    EPOCHS = 200
    VAL_EVERY = 3

    best_dice, best_state, wait = -1.0, None, 0
    history = {"train_loss": [], "val_dice": []}

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        processed = 0
        t0 = time.time()

        for vid in np.random.permutation(trainable):
            try:
                g = graphs[vid].to(device)
                opt.zero_grad()
                logits = model(g.x, g.edge_index, g.edge_attr)
                loss = F.cross_entropy(logits, g.y, weight=class_weight)
                loss.backward()
                opt.step()
                epoch_loss += float(loss.item())
                processed += 1
                del g, logits, loss
            except torch.cuda.OutOfMemoryError:
                del g
                torch.cuda.empty_cache()
                continue
            torch.cuda.empty_cache()

        mean_loss = epoch_loss / max(processed, 1)
        dt = time.time() - t0
        history["train_loss"].append(mean_loss)

        # Validation
        if epoch % VAL_EVERY == 0 or epoch <= 3:
            val_dice = lifted_voxel_dice_batch(model, val_usable, graphs, raw_labels, raw_segs, device)
            history["val_dice"].append(val_dice)

            if val_dice > best_dice:
                best_dice = val_dice
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                wait = 0
                marker = " *"
            else:
                wait += 1
                marker = ""

            pr(f"  Epoch {epoch:3d}  loss={mean_loss:.4f}  val_dice={val_dice:.4f}  "
               f"({processed} vols, {dt:.0f}s){marker}")

            if wait >= PATIENCE:
                pr(f"  Early stop at epoch {epoch}, best val Dice={best_dice:.4f}")
                break
        else:
            pr(f"  Epoch {epoch:3d}  loss={mean_loss:.4f}  ({processed} vols, {dt:.0f}s)")

    if best_state is not None:
        model.load_state_dict(best_state)
    pr(f"\nBest lifted voxel val Dice: {best_dice:.4f}")

    torch.save(best_state or model.state_dict(), os.path.join(RESULTS_DIR, "model_v5c.pt"))

    # ---- Final evaluation ----
    section("Final evaluation (all splits)")
    model.eval()
    results = []

    for vid in sorted(graphs.keys()):
        g = graphs[vid]
        labels_np = raw_labels[vid]
        seg = raw_segs[vid]

        with torch.no_grad():
            try:
                g_dev = g.to(device)
                preds = model(g_dev.x, g_dev.edge_index, g_dev.edge_attr).argmax(dim=1).cpu().numpy()
                del g_dev
                torch.cuda.empty_cache()
            except (torch.cuda.OutOfMemoryError, RuntimeError):
                try:
                    del g_dev
                except NameError:
                    pass
                torch.cuda.empty_cache()
                model_cpu = model.cpu()
                preds = model_cpu(g.x, g.edge_index, g.edge_attr).argmax(dim=1).numpy()
                model.to(device)

        flat = labels_np.ravel()
        valid = flat >= 0
        pred_mask = np.zeros(labels_np.shape, dtype=bool)
        if valid.any():
            max_id = int(flat[valid].max())
            lut = np.zeros(max_id + 1, dtype=np.int8)
            lut[:min(len(preds), max_id + 1)] = preds[:min(len(preds), max_id + 1)]
            pred_mask = np.where(valid, lut[flat], 0).reshape(labels_np.shape).astype(bool)

        gt_mask = seg == 2
        inter = int((gt_mask & pred_mask).sum())
        dice = 2.0 * inter / (gt_mask.sum() + pred_mask.sum() + 1e-8)
        recall = inter / (gt_mask.sum() + 1e-8)
        precision = inter / (pred_mask.sum() + 1e-8) if pred_mask.sum() > 0 else 0.0

        split = "train" if vid in train_ids else ("val" if vid in val_ids else "test")
        results.append({"vid": vid, "split": split, "dice": float(dice),
                        "recall": float(recall), "precision": float(precision)})

    # Print summary
    section("SUMMARY")
    for split in ["train", "val", "test"]:
        scores = [r["dice"] for r in results if r["split"] == split]
        if scores:
            pr(f"  {split:>5s}: Dice = {np.mean(scores):.4f} +/- {np.std(scores):.4f}  (n={len(scores)})")

    pr(f"\n  Oracle Dice:     {np.mean(oracles):.4f}")
    pr(f"  Best val Dice:   {best_dice:.4f}")
    pr(f"  Paper target:    Dice 0.891 +/- 0.007")

    summary = {
        "params": {"hu_min": HU_MIN, "hu_max": HU_MAX, "psi": MERGE_DIST,
                    "alpha": CUT_DIST, "beta_min": DELETE_SMALL,
                    "overlap_threshold": OVERLAP_THRESHOLD,
                    "max_nodes_gpu": MAX_NODES_GPU},
        "mean_oracle": float(np.mean(oracles)),
        "best_val_dice": float(best_dice),
        "epochs_trained": len(history["train_loss"]),
        "trainable_volumes": len(trainable),
        "skipped_volumes": len(skipped_train),
        "history": history,
        "results": results,
    }
    with open(os.path.join(RESULTS_DIR, "results_v5c.json"), "w") as f:
        json.dump(summary, f, indent=2)
    pr(f"\n  Saved to {RESULTS_DIR}/results_v5c.json")


if __name__ == "__main__":
    main()
