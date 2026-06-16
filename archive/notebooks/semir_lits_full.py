"""
SEMIR LiTS full pipeline — Rust crate + voxel reassignment + GINE.

Key fix: deleted voxels are reassigned to nearest surviving supernode
via distance_transform_edt, matching the paper's "exact bijective lifting."

Split: 80/20 of all 131 LiTS training volumes (including 13 no-tumor).
This is an internal split — the paper's 0.891 is on the held-out 70-vol
test server we can't access.

Usage:
    ~/.conda/envs/llmft/bin/python3 notebooks/semir_lits_full.py
"""

import numpy as np
import os, re, time, json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GINEConv, BatchNorm
from scipy.ndimage import distance_transform_edt
import fastloops

DATA_ROOT = "/scratch/ud3d4/acm_data/Data"
RESULTS_DIR = "/home/ud3d4/Desktop/SWOG/results/semir_lits_v2"
os.makedirs(RESULTS_DIR, exist_ok=True)

HU_MIN, HU_MAX = -50, 250
MERGE_DISTANCE = 12   # optimized via boundary Dice search
CUT_DISTANCE = 36     # 3x merge_distance
SEED = 42

np.random.seed(SEED)
torch.manual_seed(SEED)


def pr(msg=""):
    print(msg, flush=True)


def section(title):
    pr(f"\n{'='*70}")
    pr(f"  {title}")
    pr(f"{'='*70}")


# ==================================================================
# Data loading
# ==================================================================

def discover_all_volumes():
    """Find all 131 LiTS volumes (including no-tumor ones)."""
    ct_dir = os.path.join(DATA_ROOT, "ct")
    vids = []
    for f in sorted(os.listdir(ct_dir)):
        m = re.match(r"volume-(\d+)\.npy", f)
        if m:
            vid = int(m.group(1))
            seg_path = os.path.join(DATA_ROOT, "seg", f"segmentation-{vid}.npy")
            if os.path.exists(seg_path):
                vids.append(vid)
    return sorted(vids)


def load_volume(vid):
    ct = np.load(os.path.join(DATA_ROOT, "ct", f"volume-{vid}.npy")).astype(np.float32)
    seg = np.load(os.path.join(DATA_ROOT, "seg", f"segmentation-{vid}.npy")).astype(np.int32)
    ct_u8 = np.clip(ct, HU_MIN, HU_MAX)
    ct_u8 = ((ct_u8 - HU_MIN) / (HU_MAX - HU_MIN) * 255).round().astype(np.uint8)
    ct_u8 = np.ascontiguousarray(ct_u8[..., np.newaxis])
    return ct_u8, seg


# ==================================================================
# Graph minor + reassignment
# ==================================================================

def build_graph(vid):
    """Build graph minor with voxel reassignment."""
    ct_u8, seg = load_volume(vid)

    nf, ei, ef, labels, adj = fastloops.merge_and_cut(
        ct_u8,
        merge_distance=MERGE_DISTANCE,
        cut_distance=CUT_DISTANCE,
        delete_small_node_max_size=50,  # from boundary Dice search
        connectivity='faces',
    )

    labels_np = np.array(labels)

    # --- The critical step: reassign deleted voxels ---
    deleted_mask = labels_np == -1
    surviving_mask = labels_np >= 0

    if deleted_mask.any() and surviving_mask.any():
        _, indices = distance_transform_edt(deleted_mask, return_indices=True)
        labels_np[deleted_mask] = labels_np[tuple(indices[:, deleted_mask])]

    n_supernodes = nf.shape[0]
    n_edges = ei.shape[1]

    # Node features from Luke's crate: use node_invariants
    node_inv = _node_invariants_3d(nf, c=1)

    # Build PyG graph
    # Node features: [log_size, compactness, shape(3), chan(1)] = 6 dims
    x = np.column_stack([
        node_inv["log_size"][:, None],
        node_inv["compactness"][:, None],
        node_inv["shape"],
        node_inv["chan"],
    ]).astype(np.float32)

    # Edge features from crate: [boundary_count, sum_dist, max_dist, cut_count]
    ef_np = ef.astype(np.float64)
    blsafe = np.maximum(ef_np[:, 0], 1.0)
    edge_x = np.column_stack([
        np.abs((node_inv["V"][ei[0]] - node_inv["V"][ei[1]]) /
               (node_inv["V"][ei[0]] + node_inv["V"][ei[1]] + 1e-6)),
        ef_np[:, 0][:, None] / (node_inv["surface"][ei[0]][:, None] + 1e-6),
        ef_np[:, 0][:, None] / (node_inv["surface"][ei[1]][:, None] + 1e-6),
        np.abs(node_inv["chan"][ei[0]] - node_inv["chan"][ei[1]]),
        (ef_np[:, 1] / blsafe / 255.0)[:, None],
        (ef_np[:, 3] / blsafe)[:, None],
    ]).astype(np.float32)

    # Make edges bidirectional
    ei_bi = np.concatenate([ei, ei[::-1]], axis=1)
    ef_bi = np.concatenate([edge_x, edge_x], axis=0)
    # Swap bfrac_a/bfrac_b for reverse edges
    ef_bi[n_edges:, 1], ef_bi[n_edges:, 2] = ef_bi[n_edges:, 2].copy(), ef_bi[n_edges:, 1].copy()

    # Ground truth: supernode is tumor if >50% of its voxels are GT tumor
    flat_labels = labels_np.ravel().astype(np.int64)
    flat_gt = (seg.ravel() == 2).astype(np.float64)
    tc = np.bincount(flat_labels, minlength=n_supernodes)
    tu = np.bincount(flat_labels, weights=flat_gt, minlength=n_supernodes)
    y = (tu / np.maximum(tc, 1) > 0.5).astype(np.int64)

    data = Data(
        x=torch.from_numpy(x),
        edge_index=torch.from_numpy(ei_bi.astype(np.int64)),
        edge_attr=torch.from_numpy(ef_bi),
        y=torch.from_numpy(y),
    )

    return data, labels_np, seg


def _node_invariants_3d(node_feats, c=1, eps=1e-6):
    """Luke's node_invariants from rust_crate.ipynb, for 3D."""
    f = node_feats.astype(np.float64)
    N = f.shape[0]
    V = f[:, 0]  # area
    Vsafe = np.maximum(V, 1.0)

    # Centroid: s=[1,2,3] for 3D (x, y, z sums)
    mean_coord = np.stack([f[:, 1], f[:, 2], f[:, 3]], axis=1) / Vsafe[:, None]

    # Covariance: cov indices [4,5,6,7,8,9] -> (xx,yy,zz,xy,xz,yz)
    cov = np.zeros((N, 3, 3))
    cov_map = [(4, 0, 0), (5, 1, 1), (6, 2, 2), (7, 0, 1), (8, 0, 2), (9, 1, 2)]
    for col, i, j in cov_map:
        cij = f[:, col] / Vsafe - mean_coord[:, i] * mean_coord[:, j]
        cov[:, i, j] = cij
        cov[:, j, i] = cij

    w = np.linalg.eigvalsh(cov)
    w = np.clip(w, 0.0, None)

    denom = w[:, 2] + eps
    shape = np.stack([
        (w[:, 2] - w[:, 1]) / denom,
        (w[:, 1] - w[:, 0]) / denom,
        w[:, 0] / denom,
    ], axis=1)
    degenerate = w.sum(axis=1) < eps
    shape[degenerate] = 0.0

    # Channel mean (chan0 = index 10 for 3D with c=1)
    chan = f[:, 10:10 + c] / Vsafe[:, None] / 255.0

    # Boundary (index 10+c+6 = 17 for c=1)
    boundary_idx = 10 + c + 6
    surface = f[:, boundary_idx] if boundary_idx < f.shape[1] else np.zeros(N)
    compactness = surface / np.power(Vsafe, 2.0 / 3.0)

    return dict(
        V=V,
        surface=surface,
        centroid=mean_coord,
        shape=shape,
        chan=chan,
        compactness=compactness,
        log_size=np.log(Vsafe),
    )


# ==================================================================
# GINE model
# ==================================================================

class GINE(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden=128, n_layers=3):
        super().__init__()
        self.node_enc = nn.Linear(node_dim, hidden)
        self.edge_enc = nn.Linear(edge_dim, hidden)
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for _ in range(n_layers):
            mlp = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
            self.convs.append(GINEConv(mlp, edge_dim=hidden))
            self.bns.append(BatchNorm(hidden))
        self.head = nn.Linear(hidden, 2)

    def forward(self, data):
        x = F.relu(self.node_enc(data.x))
        edge_attr = F.relu(self.edge_enc(data.edge_attr))
        for conv, bn in zip(self.convs, self.bns):
            x = F.relu(bn(conv(x, data.edge_index, edge_attr)))
        return self.head(x)


# ==================================================================
# Training
# ==================================================================

def train_gine(train_graphs, val_graphs, device, epochs=200, lr=1e-3, patience=15):
    node_dim = train_graphs[0].x.shape[1]
    edge_dim = train_graphs[0].edge_attr.shape[1]

    model = GINE(node_dim, edge_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    best_val_dice = 0.0
    best_state = None
    wait = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for g in train_graphs:
            g = g.to(device)
            optimizer.zero_grad()
            logits = model(g)
            # Class-weighted CE
            n_pos = (g.y == 1).sum().float()
            n_neg = (g.y == 0).sum().float()
            weight = torch.tensor([1.0, max(n_neg / max(n_pos, 1), 1.0)], device=device)
            loss = F.cross_entropy(logits, g.y, weight=weight)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Validation
        model.eval()
        val_dices = []
        with torch.no_grad():
            for g in val_graphs:
                g = g.to(device)
                preds = model(g).argmax(dim=1)
                tp = ((preds == 1) & (g.y == 1)).sum().float()
                fp = ((preds == 1) & (g.y == 0)).sum().float()
                fn = ((preds == 0) & (g.y == 1)).sum().float()
                dice = (2 * tp / (2 * tp + fp + fn + 1e-8)).item()
                val_dices.append(dice)

        mean_val = np.mean(val_dices) if val_dices else 0.0
        if mean_val > best_val_dice:
            best_val_dice = mean_val
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1

        if (epoch + 1) % 10 == 0 or wait == 0:
            pr(f"  Epoch {epoch+1:3d}: loss={total_loss/len(train_graphs):.4f}  "
               f"val_dice={mean_val:.4f}  best={best_val_dice:.4f}  wait={wait}")

        if wait >= patience:
            pr(f"  Early stopping at epoch {epoch+1}")
            break

    if best_state:
        model.load_state_dict(best_state)
    return model


def evaluate_voxel_dice(model, vids, graphs, labels_map, segs_map, device):
    """Compute voxel-level Dice via LUT lifting."""
    model.eval()
    results = []
    with torch.no_grad():
        for vid in vids:
            if vid not in graphs:
                continue
            g = graphs[vid].to(device)
            preds = model(g).argmax(dim=1).cpu().numpy()

            labels_np = labels_map[vid]
            seg = segs_map[vid]

            # LUT lift: supernode prediction -> voxel prediction
            n_sn = g.num_nodes
            tumor_lut = np.zeros(n_sn, dtype=bool)
            tumor_lut[preds == 1] = True
            flat_labels = labels_np.ravel().astype(np.int64)
            pred_mask = tumor_lut[flat_labels]

            gt_mask = (seg.ravel() == 2)
            tp = int((pred_mask & gt_mask).sum())
            fp = int((pred_mask & ~gt_mask).sum())
            fn = int((~pred_mask & gt_mask).sum())
            dice = 2 * tp / (2 * tp + fp + fn + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            precision = tp / (tp + fp + 1e-8)

            results.append(dict(vid=vid, dice=dice, recall=recall, precision=precision,
                                gt_tumor=int(gt_mask.sum()), pred_tumor=int(pred_mask.sum())))
    return results


# ==================================================================
# Main
# ==================================================================

def main():
    section("SEMIR LiTS — Full Pipeline with Reassignment")

    # Discover volumes
    all_vids = discover_all_volumes()
    pr(f"Found {len(all_vids)} LiTS volumes")

    # 80/20 split of all 131 volumes
    perm = np.random.permutation(len(all_vids))
    n_train = int(0.8 * len(all_vids))
    train_ids = sorted([all_vids[perm[i]] for i in range(n_train)])
    val_ids = sorted([all_vids[perm[i]] for i in range(n_train, len(all_vids))])
    pr(f"Split: {len(train_ids)} train / {len(val_ids)} val")

    # Count tumor volumes per split
    n_tumor_train = 0
    n_tumor_val = 0
    for vid in train_ids:
        seg = np.load(os.path.join(DATA_ROOT, "seg", f"segmentation-{vid}.npy"))
        if (seg == 2).sum() > 0:
            n_tumor_train += 1
    for vid in val_ids:
        seg = np.load(os.path.join(DATA_ROOT, "seg", f"segmentation-{vid}.npy"))
        if (seg == 2).sum() > 0:
            n_tumor_val += 1
    pr(f"Tumor volumes: {n_tumor_train} train / {n_tumor_val} val")

    # Build graphs for all volumes
    section("Building graphs (Rust crate + reassignment)")
    graphs = {}
    labels_map = {}
    segs_map = {}
    for i, vid in enumerate(all_vids):
        t0 = time.time()
        data, labels_np, seg = build_graph(vid)
        dt = time.time() - t0

        graphs[vid] = data
        labels_map[vid] = labels_np
        segs_map[vid] = seg

        n_tu = int((data.y == 1).sum())
        n_bg = int((data.y == 0).sum())
        split = "train" if vid in train_ids else "val"
        pr(f"  [{i+1:3d}/{len(all_vids)}] vol-{vid} [{split}]: "
           f"{data.num_nodes:,} nodes ({n_tu} tu, {n_bg:,} bg), "
           f"{data.num_edges:,} edges  {dt:.1f}s")

    # Train GINE
    section("Training GINE")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    pr(f"Device: {device}")
    if torch.cuda.is_available():
        pr(f"GPU: {torch.cuda.get_device_name(0)}")

    train_graphs_list = [graphs[v] for v in train_ids if v in graphs]
    val_graphs_list = [graphs[v] for v in val_ids if v in graphs]

    model = train_gine(train_graphs_list, val_graphs_list, device)

    # Evaluate on train and val
    section("Evaluation — Voxel-level Dice")
    for split_name, split_ids in [("train", train_ids), ("val", val_ids)]:
        results = evaluate_voxel_dice(model, split_ids, graphs, labels_map, segs_map, device)
        dices = [r["dice"] for r in results if r["gt_tumor"] > 0]

        pr(f"\n  {split_name.upper()} ({len(results)} volumes, {len(dices)} with tumor):")
        if dices:
            pr(f"    Mean Dice:   {np.mean(dices):.4f} +/- {np.std(dices):.4f}")
            pr(f"    Median Dice: {np.median(dices):.4f}")
            pr(f"    Min/Max:     {np.min(dices):.4f} / {np.max(dices):.4f}")

        # Print per-volume
        for r in sorted(results, key=lambda x: x["dice"]):
            if r["gt_tumor"] > 0:
                pr(f"    vol-{r['vid']:3d}: Dice={r['dice']:.4f}  "
                   f"Recall={r['recall']:.4f}  Precision={r['precision']:.4f}  "
                   f"GT={r['gt_tumor']:,}")

    pr(f"\n  Paper reference: LiTS Tumor Dice = 0.891 +/- 0.007 (on held-out 70-vol test set)")
    pr(f"  Our evaluation is on internal 20% validation split of the 131 training volumes.")

    # Save
    with open(os.path.join(RESULTS_DIR, "config.json"), "w") as f:
        json.dump(dict(
            merge_distance=MERGE_DISTANCE, cut_distance=CUT_DISTANCE,
            hu_min=HU_MIN, hu_max=HU_MAX, seed=SEED,
            n_train=len(train_ids), n_val=len(val_ids),
            train_ids=train_ids, val_ids=val_ids,
        ), f, indent=2)

    torch.save(model.state_dict(), os.path.join(RESULTS_DIR, "gine_model.pt"))
    pr(f"\nResults saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
