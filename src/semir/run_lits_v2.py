"""
SEMIR LiTS pipeline v2 — with Luke's fixes and checkpointing.

Saves after each stage so we can resume if the cluster goes down.
Checkpoints go to RESULTS_DIR. To resume, just re-run — it skips completed stages.

Changes from v1:
  - Window: [-50, 250] (Luke: wider for m_min/m_max room)
  - psi=0.03, alpha=0.20 (Luke: alpha >> psi for connected graph)
  - beta_min=5 (Luke: preserve tumor rind supernodes)
  - m_min=0.05, m_max=0.95 (Luke: delete air/bone)
  - 12 edge features (Luke: ratios + shared boundary)
  - GINE edge_dim=12

Usage:
    cd /home/ud3d4/Desktop/SWOG/src
    conda run -n llmft python -m semir.run_lits_v2
"""

import os, sys, re, json, time, pickle
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATA_ROOT = "/scratch/ud3d4/acm_data/Data"
RESULTS_DIR = "/home/ud3d4/Desktop/SWOG/results/semir_lits_v2"
os.makedirs(RESULTS_DIR, exist_ok=True)


def pr(msg=""):
    print(msg, flush=True)


def checkpoint_path(name):
    return os.path.join(RESULTS_DIR, f"ckpt_{name}.pkl")


def save_checkpoint(name, data):
    path = checkpoint_path(name)
    with open(path, "wb") as f:
        pickle.dump(data, f)
    pr(f"  [SAVED] {path} ({os.path.getsize(path) / 1e6:.1f} MB)")


def load_checkpoint(name):
    path = checkpoint_path(name)
    if os.path.exists(path):
        with open(path, "rb") as f:
            data = pickle.load(f)
        pr(f"  [LOADED] {path}")
        return data
    return None


def hu_window(vol, hu_min=-50, hu_max=250):
    vol = np.clip(vol, hu_min, hu_max)
    return (vol - hu_min) / (hu_max - hu_min)


def discover_volumes():
    ct_dir = os.path.join(DATA_ROOT, "ct")
    ids = []
    for f in sorted(os.listdir(ct_dir)):
        m = re.match(r"volume-(\d+)\.npy", f)
        if m:
            vid = int(m.group(1))
            if os.path.exists(os.path.join(DATA_ROOT, "seg", f"segmentation-{vid}.npy")):
                ids.append(vid)
    return sorted(ids)


def load_case(vid):
    ct = np.load(os.path.join(DATA_ROOT, "ct", f"volume-{vid}.npy")).astype(np.float32)
    seg = np.load(os.path.join(DATA_ROOT, "seg", f"segmentation-{vid}.npy")).astype(np.int32)
    return hu_window(ct), seg


# ============================================================
# STAGE 0: Data loading + split
# ============================================================
def stage0_data():
    ckpt = load_checkpoint("stage0_split")
    if ckpt:
        return ckpt

    all_ids = discover_volumes()
    pr(f"  Found {len(all_ids)} LiTS volumes")

    np.random.seed(42)
    perm = np.random.permutation(all_ids)
    n = len(perm)
    n_train = max(3, int(0.7 * n))
    n_val = max(1, int(0.15 * n))

    split = {
        "train": sorted(perm[:n_train].tolist()),
        "val": sorted(perm[n_train:n_train + n_val].tolist()),
        "test": sorted(perm[n_train + n_val:].tolist()),
    }
    split["all_ordered"] = split["train"] + split["val"] + split["test"]
    pr(f"  Split: {len(split['train'])} train / {len(split['val'])} val / {len(split['test'])} test")

    save_checkpoint("stage0_split", split)
    return split


# ============================================================
# STAGE 1: Graph minor construction (checkpoints per-volume)
# ============================================================
PARAMS = {
    "psi": 0.03, "alpha": 0.20, "beta_min": 5,
    "beta_max": 500000, "m_min": 0.05, "m_max": 0.95,
}


def stage1_graph_minors(split):
    from semir.graph_minor import build_graph_minor

    all_vids = split["all_ordered"]
    minors_dir = os.path.join(RESULTS_DIR, "minors")
    os.makedirs(minors_dir, exist_ok=True)

    for vid in all_vids:
        minor_path = os.path.join(minors_dir, f"minor_{vid}.pkl")
        if os.path.exists(minor_path):
            continue

        ct, seg = load_case(vid)
        t0 = time.time()
        gm = build_graph_minor(ct, **PARAMS)
        dt = time.time() - t0

        # Save minor (labels + adjacency + stats, not the full volume)
        with open(minor_path, "wb") as f:
            pickle.dump(gm, f)

        s = gm["stats"]
        n_full = len(gm["full_adjacency"])
        pr(f"  vol-{vid}: {s['n_voxels']:>10,} -> {s['n_supernodes_after_deletion']:>6,} SN  "
           f"edges={n_full:,}  E/V={n_full/max(s['n_supernodes_after_deletion'],1):.1f}  {dt:.1f}s")

    # Count completed
    done = sum(1 for v in all_vids if os.path.exists(os.path.join(minors_dir, f"minor_{v}.pkl")))
    pr(f"  Graph minors: {done}/{len(all_vids)} complete")
    return minors_dir


# ============================================================
# STAGE 2: Oracle analysis
# ============================================================
def stage2_oracle(split, minors_dir):
    ckpt = load_checkpoint("stage2_oracle")
    if ckpt:
        pr(f"  Mean oracle Dice: {np.mean([r['dice'] for r in ckpt]):.4f}")
        return ckpt

    results = []
    for vid in split["all_ordered"]:
        gm = pickle.load(open(os.path.join(minors_dir, f"minor_{vid}.pkl"), "rb"))
        _, seg = load_case(vid)

        labels = gm["labels"]
        max_label = int(labels.max())
        if max_label == 0:
            results.append({"vid": vid, "dice": 0.0, "tumor_sn": 0})
            continue

        fl = labels.ravel()
        fg = (seg.ravel() == 2).astype(np.float64)
        tc = np.bincount(fl, weights=fg, minlength=max_label + 1)
        ttc = np.bincount(fl, minlength=max_label + 1)
        safe = np.where(ttc > 0, ttc, 1)
        ovl = tc / safe
        ist = (ovl > 0.1); ist[0] = False
        pred = ist[fl].astype(np.int32)
        gt = fg.astype(np.int32)
        tp = int((pred & gt).sum())
        fp = int(pred.sum()) - tp
        fn = int(gt.sum()) - tp
        dice = 2 * tp / (2 * tp + fp + fn + 1e-8)
        n_tsn = int(ist.sum())
        del_t = int(fg[fl == 0].sum())

        results.append({"vid": vid, "dice": round(dice, 4), "tumor_sn": n_tsn,
                         "deleted_tumor": del_t, "gt_tumor": int(gt.sum())})
        pr(f"  vol-{vid}: oracle={dice:.4f}  tumor_SN={n_tsn}  del={del_t:,}/{int(gt.sum()):,}")

    mean_d = np.mean([r["dice"] for r in results])
    pr(f"\n  Mean oracle Dice: {mean_d:.4f}")
    save_checkpoint("stage2_oracle", results)
    return results


# ============================================================
# STAGE 3: Feature extraction + PyG graphs (checkpoints per-volume)
# ============================================================
def stage3_features(split, minors_dir):
    from semir.features import extract_node_features, extract_edge_features, build_pyg_graph

    graphs_dir = os.path.join(RESULTS_DIR, "pyg_graphs")
    os.makedirs(graphs_dir, exist_ok=True)

    for vid in split["all_ordered"]:
        graph_path = os.path.join(graphs_dir, f"graph_{vid}.pkl")
        if os.path.exists(graph_path):
            continue

        ct, seg = load_case(vid)
        gm = pickle.load(open(os.path.join(minors_dir, f"minor_{vid}.pkl"), "rb"))

        t0 = time.time()
        nf = extract_node_features(gm["labels"], ct)
        adj = gm.get("full_adjacency", gm["adjacency"])
        ef = extract_edge_features(gm["labels"], ct, adj, nf)
        data, mapping = build_pyg_graph(nf, ef, gm["labels"], gt_seg=seg)
        dt = time.time() - t0

        with open(graph_path, "wb") as f:
            pickle.dump({"data": data, "mapping": mapping, "nf": nf}, f)

        n_tu = int((data.y == 1).sum()) if hasattr(data, "y") else 0
        n_bg = int((data.y == 0).sum()) if hasattr(data, "y") else 0
        pr(f"  vol-{vid}: {data.num_nodes:,} nodes ({n_tu} tumor, {n_bg:,} bg)  "
           f"{data.num_edges:,} edges  {dt:.1f}s")

    done = sum(1 for v in split["all_ordered"]
               if os.path.exists(os.path.join(graphs_dir, f"graph_{v}.pkl")))
    pr(f"  PyG graphs: {done}/{len(split['all_ordered'])} complete")
    return graphs_dir


# ============================================================
# STAGE 4: GINE training with per-epoch checkpointing
# ============================================================
def stage4_train(split, graphs_dir):
    import torch
    import torch.nn as nn
    from semir.gine import SEMIRClassifier

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    pr(f"  Device: {device}")
    if torch.cuda.is_available():
        pr(f"  GPU: {torch.cuda.get_device_name(0)}")

    # Load graphs
    def load_graphs(vids):
        graphs = []
        for vid in vids:
            d = pickle.load(open(os.path.join(graphs_dir, f"graph_{vid}.pkl"), "rb"))
            graphs.append(d["data"])
        return graphs

    train_graphs = load_graphs(split["train"])
    val_graphs = load_graphs(split["val"])

    # Check for training checkpoint
    train_ckpt_path = os.path.join(RESULTS_DIR, "train_checkpoint.pt")
    model = SEMIRClassifier(edge_dim=12).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0)

    # Class weights
    total_pos = sum(int((g.y == 1).sum()) for g in train_graphs)
    total_neg = sum(int((g.y == 0).sum()) for g in train_graphs)
    raw_ratio = total_neg / max(total_pos, 1)
    eff_ratio = min(np.sqrt(raw_ratio), 30.0)
    weight = torch.tensor([1.0, eff_ratio], dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)
    pr(f"  Class weight: [1.0, {eff_ratio:.1f}] (raw: {raw_ratio:.1f})")
    pr(f"  Tumor SN: {total_pos:,}  Background: {total_neg:,}")

    EPOCHS = 200
    PATIENCE = 30
    history = {"train_loss": [], "val_loss": [], "val_dice": []}
    best_dice, best_state, wait, start_epoch = -1.0, None, 0, 1

    # Resume from checkpoint if exists
    if os.path.exists(train_ckpt_path):
        ckpt = torch.load(train_ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        history = ckpt["history"]
        best_dice = ckpt["best_dice"]
        best_state = ckpt.get("best_state")
        wait = ckpt["wait"]
        start_epoch = ckpt["epoch"] + 1
        pr(f"  Resumed from epoch {start_epoch - 1} (best_dice={best_dice:.4f})")

    for epoch in range(start_epoch, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for g in train_graphs:
            g_dev = g.to(device)
            optimizer.zero_grad()
            logits = model(g_dev)
            loss = criterion(logits, g_dev.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_train = total_loss / len(train_graphs)
        history["train_loss"].append(avg_train)

        model.eval()
        vloss_total, tp_all, fp_all, fn_all = 0.0, 0, 0, 0
        with torch.no_grad():
            for g in val_graphs:
                g_dev = g.to(device)
                logits = model(g_dev)
                vloss_total += criterion(logits, g_dev.y).item()
                preds = logits.argmax(dim=1)
                tp_all += ((preds == 1) & (g_dev.y == 1)).sum().item()
                fp_all += ((preds == 1) & (g_dev.y == 0)).sum().item()
                fn_all += ((preds == 0) & (g_dev.y == 1)).sum().item()

        vloss = vloss_total / max(len(val_graphs), 1)
        dice = float(2 * tp_all / (2 * tp_all + fp_all + fn_all + 1e-8))
        history["val_loss"].append(vloss)
        history["val_dice"].append(dice)

        if dice > best_dice:
            best_dice = dice
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1

        if epoch % 5 == 0 or epoch == 1:
            pr(f"  Epoch {epoch:3d}  loss={avg_train:.4f}  vloss={vloss:.4f}  "
               f"vdice={dice:.4f}  (tp={tp_all} fp={fp_all} fn={fn_all})")

        # Checkpoint every 10 epochs
        if epoch % 10 == 0:
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "history": history,
                "best_dice": best_dice,
                "best_state": best_state,
                "wait": wait,
            }, train_ckpt_path)
            pr(f"  [CKPT] epoch {epoch}")

        if wait >= PATIENCE:
            pr(f"  Early stopping at epoch {epoch} (best={best_dice:.4f})")
            break

    # Save final model
    if best_state:
        model.load_state_dict(best_state)
    torch.save(model.state_dict(), os.path.join(RESULTS_DIR, "best_model.pt"))
    with open(os.path.join(RESULTS_DIR, "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    pr(f"\n  Best validation Dice: {best_dice:.4f}")
    return model, history


# ============================================================
# STAGE 5: Evaluation
# ============================================================
def stage5_evaluate(split, minors_dir, graphs_dir, model):
    import torch

    device = next(model.parameters()).device
    model.eval()
    dice_scores = []
    n_train = len(split["train"])
    n_val = len(split["val"])

    for i, vid in enumerate(split["all_ordered"]):
        ct, seg = load_case(vid)
        gm = pickle.load(open(os.path.join(minors_dir, f"minor_{vid}.pkl"), "rb"))
        gdata = pickle.load(open(os.path.join(graphs_dir, f"graph_{vid}.pkl"), "rb"))
        data = gdata["data"]
        nf = gdata["nf"]

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

        sp = "train" if i < n_train else ("val" if i < n_train + n_val else "test")
        dice_scores.append({"vid": vid, "dice": round(dice, 4), "recall": round(recall, 4),
                            "precision": round(prec, 4), "split": sp,
                            "gt_vox": int(gt_tumor.sum()), "pred_vox": int(pred_tumor.sum())})

    with open(os.path.join(RESULTS_DIR, "dice_scores.json"), "w") as f:
        json.dump(dice_scores, f, indent=2)

    for sp in ["train", "val", "test"]:
        scores = [d["dice"] for d in dice_scores if d["split"] == sp]
        if scores:
            pr(f"  {sp.upper()}: Dice = {np.mean(scores):.4f} +/- {np.std(scores):.4f}  (n={len(scores)})")

    pr(f"  Paper target: 0.891 +/- 0.007")
    return dice_scores


# ============================================================
# MAIN
# ============================================================
def main():
    pr("=" * 60)
    pr("  SEMIR LiTS v2 — Luke's fixes + checkpointing")
    pr("=" * 60)
    pr(f"  Params: {PARAMS}")
    pr(f"  Results: {RESULTS_DIR}")

    pr(f"\n{'='*60}\n  STAGE 0: Data split\n{'='*60}")
    split = stage0_data()

    pr(f"\n{'='*60}\n  STAGE 1: Graph minor construction\n{'='*60}")
    minors_dir = stage1_graph_minors(split)

    pr(f"\n{'='*60}\n  STAGE 2: Oracle analysis\n{'='*60}")
    oracle = stage2_oracle(split, minors_dir)

    pr(f"\n{'='*60}\n  STAGE 3: Feature extraction + PyG graphs\n{'='*60}")
    graphs_dir = stage3_features(split, minors_dir)

    pr(f"\n{'='*60}\n  STAGE 4: GINE training\n{'='*60}")
    model, history = stage4_train(split, graphs_dir)

    pr(f"\n{'='*60}\n  STAGE 5: Evaluation\n{'='*60}")
    dice_scores = stage5_evaluate(split, minors_dir, graphs_dir, model)

    # Summary
    oracle_mean = np.mean([r["dice"] for r in oracle])
    best_val = max(history["val_dice"]) if history["val_dice"] else 0
    val_scores = [d["dice"] for d in dice_scores if d["split"] == "val"]
    test_scores = [d["dice"] for d in dice_scores if d["split"] == "test"]

    pr(f"\n{'='*60}")
    pr(f"  SUMMARY (v2 — Luke's fixes)")
    pr(f"{'='*60}")
    pr(f"  Oracle Dice:        {oracle_mean:.4f}  (was 0.10 in v1)")
    pr(f"  Best val SN Dice:   {best_val:.4f}")
    pr(f"  Val voxel Dice:     {np.mean(val_scores):.4f} +/- {np.std(val_scores):.4f}")
    pr(f"  Test voxel Dice:    {np.mean(test_scores):.4f} +/- {np.std(test_scores):.4f}")
    pr(f"  Paper target:       0.891 +/- 0.007")


if __name__ == "__main__":
    main()
