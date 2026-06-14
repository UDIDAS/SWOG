"""
SEMIR LiTS v4 — Single Rust call + Luke's GNN fixes.

Uses fastloops.merge_and_cut as designed: contraction + deletion + edge
cutting in one interleaved pass. Deleted regions get re-absorbed by
neighboring seeds (the core pooling mechanism).

Luke's GNN fixes:
  - 12 edge features (features.py)
  - sqrt class weights capped at 30 (gine.py)
  - patience 30
  - HU [0, 200] liver window

Usage:
    cd /home/ud3d4/Desktop/SWOG/src
    SEMIR_N_CASES=10 conda run -n llmft python -u -m semir.run_lits_v4
"""

import os, sys, re, json, time, pickle
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import fastloops
from semir.features import extract_node_features, extract_edge_features, build_pyg_graph
from semir.gine import SEMIRClassifier, train_gine

DATA_ROOT = "/scratch/ud3d4/acm_data/Data"
RESULTS_DIR = "/home/ud3d4/Desktop/SWOG/results/semir_lits_v4"
os.makedirs(RESULTS_DIR, exist_ok=True)

HU_MIN, HU_MAX = -50, 250


def pr(msg=""):
    print(msg, flush=True)


def hu_window(vol):
    vol = np.clip(vol, HU_MIN, HU_MAX)
    return (vol - HU_MIN) / (HU_MAX - HU_MIN)


def discover_volumes():
    ct_dir = os.path.join(DATA_ROOT, "ct")
    ids = []
    for f in sorted(os.listdir(ct_dir)):
        m = re.match(r"volume-(\d+)\.npy", f)
        if m:
            vid = int(m.group(1))
            seg_path = os.path.join(DATA_ROOT, "seg", f"segmentation-{vid}.npy")
            if os.path.exists(seg_path):
                seg = np.load(seg_path)
                if (seg == 2).sum() > 0:
                    ids.append(vid)
    return sorted(ids)


def load_case(vid):
    ct = np.load(os.path.join(DATA_ROOT, "ct", f"volume-{vid}.npy")).astype(np.float32)
    seg = np.load(os.path.join(DATA_ROOT, "seg", f"segmentation-{vid}.npy")).astype(np.int32)
    return ct, seg


def oracle_dice(labels, gt_seg):
    flat = labels.ravel()
    gt = (gt_seg.ravel() == 2).astype(np.float64)
    gt_total = int(gt.sum())
    valid = flat >= 0
    if gt_total == 0 or not valid.any():
        return 0.0, 0, 0.0

    max_id = int(flat[valid].max())
    tc = np.bincount(flat[valid], weights=gt[valid], minlength=max_id + 1)
    oracle_sids = np.where(tc > 0)[0]

    lut = np.zeros(max_id + 1, dtype=np.int32)
    lut[oracle_sids] = 1
    om = np.where(valid, lut[flat], 0).reshape(labels.shape).astype(bool)
    gm = gt_seg == 2
    inter = int((om & gm).sum())
    od = 2.0 * inter / (om.sum() + gm.sum() + 1e-8)

    del_t = int(gt[~valid].sum())
    del_pct = del_t / max(gt_total, 1) * 100.0

    return od, len(oracle_sids), del_pct


def main():
    pr("=" * 60)
    pr("  SEMIR LiTS v4 — Single Rust call + Luke's GNN fixes")
    pr("=" * 60)

    # ---- Config ----
    # Luke's Rust params (from run_lits_v2.py)
    MERGE_DIST = 8        # psi=0.03 in [-50,250] → 8 in [0,200] uint8
    CUT_DIST = 51         # alpha=0.20
    DELETE_SMALL = 5      # Luke: preserve tumor rind supernodes
    VALUE_MIN = 13        # Luke: delete air/fluid
    VALUE_MAX = 242       # Luke: delete bone

    pr(f"  merge_dist={MERGE_DIST}  cut_dist={CUT_DIST}  HU=[{HU_MIN},{HU_MAX}]")
    pr(f"  delete_small={DELETE_SMALL}  value_min={VALUE_MIN}  value_max={VALUE_MAX}")

    # ---- Discover data ----
    all_vids = discover_volumes()
    pr(f"  Found {len(all_vids)} LiTS volumes with tumor")

    max_cases = int(os.environ.get("SEMIR_N_CASES", len(all_vids)))
    if max_cases < len(all_vids):
        all_vids = all_vids[:max_cases]
        pr(f"  Limited to {max_cases} cases")

    # Split
    np.random.seed(42)
    perm = np.random.permutation(len(all_vids))
    n_train = int(0.7 * len(all_vids))
    n_val = int(0.15 * len(all_vids))
    train_ids = sorted([all_vids[i] for i in perm[:n_train]])
    val_ids = sorted([all_vids[i] for i in perm[n_train:n_train + n_val]])
    test_ids = sorted([all_vids[i] for i in perm[n_train + n_val:]])
    ordered = train_ids + val_ids + test_ids
    pr(f"  Split: {len(train_ids)} train / {len(val_ids)} val / {len(test_ids)} test")

    # ============================================================
    # STAGE 1: Single Rust call — contraction + deletion + edges
    # ============================================================
    pr(f"\n{'='*60}")
    pr(f"  STAGE 1: Rust merge_and_cut (single call, deletion enabled)")
    pr(f"{'='*60}")

    minors = {}
    oracle_results = []

    for vid in ordered:
        ct_raw, seg = load_case(vid)
        n_vox = ct_raw.size

        ct_u8 = np.clip(ct_raw, HU_MIN, HU_MAX)
        ct_u8 = ((ct_u8 - HU_MIN) / (HU_MAX - HU_MIN) * 255).round().astype(np.uint8)
        ct_u8 = np.ascontiguousarray(ct_u8[..., np.newaxis])

        t0 = time.time()
        # Single call — Rust handles contraction + deletion + edge cutting
        # with interleaved re-absorption of deleted regions
        raw_nf, raw_ei, raw_ef, raw_labels, raw_adj = fastloops.merge_and_cut(
            ct_u8,
            merge_distance=MERGE_DIST,
            cut_distance=CUT_DIST,
            delete_small_node_max_size=DELETE_SMALL,
            delete_large_node_min_size=int(n_vox ** 0.8),
            delete_value_min=VALUE_MIN,
            delete_value_max=VALUE_MAX,
            connectivity="faces",
        )
        dt = time.time() - t0

        labels_np = np.asarray(raw_labels)
        n_sn = raw_nf.shape[0]
        n_edges = raw_ei.shape[1]

        # Convert labels: -1 → 0 (deleted), 0-based → 1-based
        labels_out = labels_np.copy()
        labels_out[labels_np >= 0] += 1
        labels_out[labels_np < 0] = 0

        # Build adjacency for features.py
        full_adjacency = {}
        if n_edges > 0:
            a_ids = raw_ei[0].astype(int) + 1
            b_ids = raw_ei[1].astype(int) + 1
            ef_np = raw_ef.astype(np.float64)
            for idx in range(n_edges):
                key = (int(min(a_ids[idx], b_ids[idx])),
                       int(max(a_ids[idx], b_ids[idx])))
                mean_contrast = ef_np[idx, 1] / max(ef_np[idx, 0], 1) / 255.0
                full_adjacency[key] = float(mean_contrast)

        gm = {
            "labels": labels_out,
            "n_supernodes": n_sn,
            "adjacency": full_adjacency,
            "full_adjacency": full_adjacency,
            "stats": {
                "n_voxels": n_vox,
                "n_supernodes_after_contraction": n_sn,
                "n_supernodes_after_deletion": n_sn,
                "n_edges": n_edges,
                "compression_ratio": n_vox / max(n_sn, 1),
                "time_total_s": round(dt, 2),
            },
        }
        minors[vid] = gm

        # Oracle
        od, t_sn, del_pct = oracle_dice(labels_np, seg)
        oracle_results.append({
            "vid": vid, "oracle": round(od, 4),
            "n_sn": n_sn, "n_edges": n_edges,
            "del_pct": round(del_pct, 1),
        })
        pr(f"  vol-{vid}: {n_vox:>10,} -> {n_sn:>6,} SN  "
           f"{n_edges:,} edges  oracle={od:.4f}  del={del_pct:.1f}%  {dt:.1f}s")

    mean_oracle = np.mean([r["oracle"] for r in oracle_results])
    mean_sn = np.mean([r["n_sn"] for r in oracle_results])
    pr(f"\n  Mean oracle: {mean_oracle:.4f}  Mean SN: {mean_sn:.0f}")

    with open(os.path.join(RESULTS_DIR, "oracle_analysis.json"), "w") as f:
        json.dump(oracle_results, f, indent=2)

    # ============================================================
    # STAGE 2: Feature extraction (7 node + 12 edge)
    # ============================================================
    pr(f"\n{'='*60}")
    pr(f"  STAGE 2: Feature extraction")
    pr(f"{'='*60}")

    graphs = {}
    node_feats_dict = {}

    for vid in ordered:
        ct_raw, seg = load_case(vid)
        ct = hu_window(ct_raw)
        gm = minors[vid]

        t0 = time.time()
        nf = extract_node_features(gm["labels"], ct)
        adj = gm["full_adjacency"]
        ef = extract_edge_features(gm["labels"], ct, adj, nf)
        data, mapping = build_pyg_graph(nf, ef, gm["labels"], gt_seg=seg)
        dt = time.time() - t0

        graphs[vid] = data
        node_feats_dict[vid] = nf

        n_tu = int((data.y == 1).sum()) if hasattr(data, 'y') else 0
        n_bg = int((data.y == 0).sum()) if hasattr(data, 'y') else 0
        pr(f"  vol-{vid}: {data.num_nodes:,} nodes ({n_tu} tumor, {n_bg:,} bg)  "
           f"{data.num_edges:,} edges  {dt:.1f}s")

    # ============================================================
    # STAGE 3: GINE training (Luke's settings)
    # ============================================================
    pr(f"\n{'='*60}")
    pr(f"  STAGE 3: GINE training (sqrt weights, patience 30, 12 edge feat)")
    pr(f"{'='*60}")

    import torch
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    pr(f"  Device: {device}")
    if torch.cuda.is_available():
        pr(f"  GPU: {torch.cuda.get_device_name(0)}")

    train_graphs = [graphs[v] for v in train_ids]
    val_graphs = [graphs[v] for v in val_ids]

    model, history = train_gine(
        train_graphs, val_graphs,
        epochs=200, lr=1e-3, patience=30, device=device,
    )

    best_val = max(history["val_dice"]) if history["val_dice"] else 0
    pr(f"\n  Best val Dice (supernode): {best_val:.4f}")
    pr(f"  Epochs: {len(history['train_loss'])}")

    with open(os.path.join(RESULTS_DIR, "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)
    torch.save(model.state_dict(), os.path.join(RESULTS_DIR, "best_model.pt"))

    # ============================================================
    # STAGE 4: Voxel-level evaluation
    # ============================================================
    pr(f"\n{'='*60}")
    pr(f"  STAGE 4: Voxel-level Dice")
    pr(f"{'='*60}")

    model.eval()
    model = model.to(device)

    dice_scores = []
    for vid in ordered:
        _, seg = load_case(vid)
        gm = minors[vid]
        data = graphs[vid]
        nf = node_feats_dict[vid]

        with torch.no_grad():
            logits = model(data.to(device))
            preds = logits.argmax(dim=1).cpu().numpy()

        sids = sorted(nf.keys())
        max_label = int(gm["labels"].max())
        tumor_lut = np.zeros(max_label + 1, dtype=np.int32)
        for idx, sid in enumerate(sids):
            if preds[idx] == 1:
                tumor_lut[sid] = 2
        pred_mask = tumor_lut[gm["labels"]]

        gt_tumor = (seg == 2)
        pred_tumor = (pred_mask == 2)

        if gt_tumor.sum() == 0:
            dice = 1.0 if pred_tumor.sum() == 0 else 0.0
        else:
            inter = (gt_tumor & pred_tumor).sum()
            dice = float(2 * inter / (gt_tumor.sum() + pred_tumor.sum() + 1e-8))

        recall = float((gt_tumor & pred_tumor).sum() / (gt_tumor.sum() + 1e-8))
        prec = float((gt_tumor & pred_tumor).sum() / (pred_tumor.sum() + 1e-8)) if pred_tumor.sum() > 0 else 0

        split = "train" if vid in train_ids else ("val" if vid in val_ids else "test")
        dice_scores.append({
            "vid": vid, "dice": round(dice, 4), "recall": round(recall, 4),
            "precision": round(prec, 4), "split": split,
        })
        pr(f"  vol-{vid} [{split}]: Dice={dice:.4f}  Recall={recall:.4f}  Prec={prec:.4f}")

    with open(os.path.join(RESULTS_DIR, "dice_scores.json"), "w") as f:
        json.dump(dice_scores, f, indent=2)

    # ============================================================
    # SUMMARY
    # ============================================================
    pr(f"\n{'='*60}")
    pr(f"  SUMMARY")
    pr(f"{'='*60}")

    for split in ["train", "val", "test"]:
        scores = [d["dice"] for d in dice_scores if d["split"] == split]
        if scores:
            pr(f"  {split:>5s}: Dice = {np.mean(scores):.4f} +/- {np.std(scores):.4f}  (n={len(scores)})")

    pr(f"\n  Oracle Dice: {mean_oracle:.4f}")
    pr(f"  Mean supernodes: {mean_sn:.0f}")
    pr(f"  Paper target: 0.891 +/- 0.007")

    results = {
        "merge_distance": MERGE_DIST, "cut_distance": CUT_DIST,
        "hu_window": [HU_MIN, HU_MAX], "deletion": "rust_defaults",
        "oracle": mean_oracle, "mean_sn": mean_sn,
        "train_dice": float(np.mean([d["dice"] for d in dice_scores if d["split"] == "train"])),
        "val_dice": float(np.mean([d["dice"] for d in dice_scores if d["split"] == "val"])) if val_ids else 0,
        "test_dice": float(np.mean([d["dice"] for d in dice_scores if d["split"] == "test"])) if test_ids else 0,
    }
    with open(os.path.join(RESULTS_DIR, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    pr(f"\n  Results saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
