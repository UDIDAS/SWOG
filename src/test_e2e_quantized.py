"""End-to-end test: quantized CC → features → GINE → voxel Dice."""
import numpy as np
import sys
import time
import re
import os
import torch

sys.path.insert(0, ".")
from scipy.ndimage import label as nd_label
from semir.graph_minor import _node_deletion, _edge_deletion
from semir.features import extract_node_features, extract_edge_features, build_pyg_graph
from semir.gine import train_gine

DATA_ROOT = "/scratch/ud3d4/acm_data/Data"
N_BINS = 8
BETA_MIN = 50


def quantized_cc_minor(ct_norm, n_bins=N_BINS, beta_min=BETA_MIN):
    """Build graph minor via quantized connected components."""
    HWD = int(np.prod(ct_norm.shape))
    beta_max = HWD // 3

    quantized = np.clip(np.floor(ct_norm * n_bins).astype(np.int32), 0, n_bins - 1)
    labels = np.zeros_like(quantized, dtype=np.int64)
    offset = 0
    for b in range(n_bins):
        mask = quantized == b
        if mask.sum() == 0:
            continue
        cc, n_cc = nd_label(mask)
        labels[mask] = cc[mask] + offset
        offset += n_cc

    # Node deletion
    labels = _node_deletion(labels, ct_norm, beta_min=beta_min,
                            beta_max=beta_max, m_min=0.0, m_max=1.0)

    # Edge deletion (alpha=0.20 for boundary detection, keep full adj for GNN)
    adjacency = _edge_deletion(labels, ct_norm, alpha=0.20)
    full_adjacency = _edge_deletion(labels, ct_norm, alpha=1e9)

    n_supernodes = len(np.unique(labels[labels > 0]))
    return {
        "labels": labels,
        "n_supernodes": n_supernodes,
        "adjacency": adjacency,
        "full_adjacency": full_adjacency,
        "stats": {"n_supernodes_after_deletion": n_supernodes,
                  "compression_ratio": HWD / max(n_supernodes, 1)},
    }


def main():
    # Discover volumes with tumor
    np.random.seed(42)
    all_ids = []
    for f in sorted(os.listdir(os.path.join(DATA_ROOT, "ct"))):
        m = re.match(r"volume-(\d+)\.npy", f)
        if m:
            vid = int(m.group(1))
            if os.path.exists(os.path.join(DATA_ROOT, "seg", f"segmentation-{vid}.npy")):
                all_ids.append(vid)

    # Pick small-medium volumes with tumor
    tumor_ids = []
    for vid in all_ids:
        seg = np.load(os.path.join(DATA_ROOT, "seg", f"segmentation-{vid}.npy"))
        if (seg == 2).sum() > 500 and seg.size < 15_000_000:
            tumor_ids.append(vid)
        if len(tumor_ids) >= 15:
            break

    perm = np.random.permutation(len(tumor_ids))
    train_ids = [tumor_ids[i] for i in perm[:10]]
    val_ids = [tumor_ids[i] for i in perm[10:13]]
    print(f"Train: {train_ids}", flush=True)
    print(f"Val: {val_ids}", flush=True)

    # Build graph minors + features
    print("\n=== Building quantized CC graph minors ===", flush=True)
    pyg_graphs = []
    all_nf = []
    all_gm = []

    for split, ids in [("train", train_ids), ("val", val_ids)]:
        for vid in ids:
            ct = np.load(os.path.join(DATA_ROOT, "ct", f"volume-{vid}.npy")).astype(np.float32)
            seg = np.load(os.path.join(DATA_ROOT, "seg", f"segmentation-{vid}.npy")).astype(np.int32)
            ct_norm = np.clip(ct, 0, 200).astype(np.float64) / 200.0

            t0 = time.time()
            gm = quantized_cc_minor(ct_norm)
            nf = extract_node_features(gm["labels"], ct_norm)
            adj = gm.get("full_adjacency", gm["adjacency"])
            ef = extract_edge_features(gm["labels"], ct_norm, adj, nf)
            data, _ = build_pyg_graph(nf, ef, gm["labels"], gt_seg=seg)
            dt = time.time() - t0

            pyg_graphs.append(data)
            all_nf.append(nf)
            all_gm.append(gm)

            n_tu = int((data.y == 1).sum())
            n_bg = int((data.y == 0).sum())
            n_edges = data.num_edges
            print(f"  [{split}] vol-{vid}: {data.num_nodes:,} nodes "
                  f"({n_tu} tumor, {n_bg:,} bg), {n_edges:,} edges  {dt:.1f}s",
                  flush=True)

    # Train GINE
    print("\n=== Training GINE ===", flush=True)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}", flush=True)
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

    model, history = train_gine(
        pyg_graphs[:10], pyg_graphs[10:],
        epochs=200, lr=1e-3, patience=30,
        device=device, batch_size=4,
    )

    best_dice = max(history["val_dice"]) if history["val_dice"] else 0
    print(f"\nBest val supernode Dice: {best_dice:.4f}", flush=True)
    print(f"Epochs: {len(history['train_loss'])}", flush=True)

    # Voxel-level Dice
    print("\n=== Voxel Dice ===", flush=True)
    model.eval()
    model = model.to(device)

    for i, (vid, data, nf, gm) in enumerate(
            zip(train_ids + val_ids, pyg_graphs, all_nf, all_gm)):
        seg = np.load(os.path.join(DATA_ROOT, "seg", f"segmentation-{vid}.npy")).astype(np.int32)

        with torch.no_grad():
            logits = model(data.to(device))
            preds = logits.argmax(dim=1).cpu().numpy()

        sids = sorted(nf.keys())
        max_label = int(gm["labels"].max())
        lut = np.zeros(max_label + 1, dtype=np.int32)
        for idx, sid in enumerate(sids):
            if preds[idx] == 1:
                lut[sid] = 2
        pred_mask = lut[gm["labels"]]

        gt_t = (seg == 2)
        pred_t = (pred_mask == 2)
        if gt_t.sum() == 0:
            dice = 1.0 if pred_t.sum() == 0 else 0.0
        else:
            dice = float(2 * (gt_t & pred_t).sum() / (gt_t.sum() + pred_t.sum() + 1e-8))

        recall = float((gt_t & pred_t).sum() / (gt_t.sum() + 1e-8))
        prec = float((gt_t & pred_t).sum() / (pred_t.sum() + 1e-8)) if pred_t.sum() > 0 else 0

        split = "train" if i < 10 else "val"
        print(f"  [{split}] vol-{vid}: Dice={dice:.4f}  Recall={recall:.4f}  "
              f"Prec={prec:.4f}  GT={gt_t.sum():,} Pred={pred_t.sum():,}", flush=True)

    print("\nDone!", flush=True)


if __name__ == "__main__":
    main()
