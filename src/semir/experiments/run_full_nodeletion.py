"""
Full pipeline: recursive contraction + NO node deletion + GINE + Dice evaluation.
Uses liver window [0,200], psi=0.01, no deletion (beta_min=1).
"""
import os, sys, json, re, time, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from semir.graph_minor import build_graph_minor
from semir.features import extract_node_features, extract_edge_features, build_pyg_graph
from semir.gine import SEMIRClassifier

DATA_ROOT = "/scratch/ud3d4/acm_data/Data"
RESULTS_DIR = "/home/ud3d4/Desktop/SWOG/results/semir_lits_v2"
PSI = 0.01
BETA_MIN = 2  # remove single-voxel supernodes only
DEVICE = "cuda:0"

def pr(msg=""):
    print(msg, flush=True)

def load(vid):
    ct = np.load(os.path.join(DATA_ROOT, "ct", f"volume-{vid}.npy")).astype(np.float32)
    seg = np.load(os.path.join(DATA_ROOT, "seg", f"segmentation-{vid}.npy")).astype(np.int32)
    ct = np.clip(ct, 0, 200) / 200.0
    return ct, seg

def discover():
    ct_dir = os.path.join(DATA_ROOT, "ct")
    ids = []
    for f in os.listdir(ct_dir):
        m = re.match(r"volume-(\d+)\.npy", f)
        if m:
            vid = int(m.group(1))
            if os.path.exists(os.path.join(DATA_ROOT, "seg", f"segmentation-{vid}.npy")):
                ids.append(vid)
    return sorted(ids)

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    all_ids = discover()
    pr(f"Found {len(all_ids)} volumes")

    # Split 70/15/15
    np.random.seed(42)
    perm = np.random.permutation(all_ids)
    n = len(perm)
    n_train = max(3, int(0.7 * n))
    n_val = max(1, int(0.15 * n))
    train_ids = sorted(perm[:n_train].tolist())
    val_ids = sorted(perm[n_train:n_train + n_val].tolist())
    test_ids = sorted(perm[n_train + n_val:].tolist())
    all_ordered = train_ids + val_ids + test_ids
    pr(f"Split: {len(train_ids)} train / {len(val_ids)} val / {len(test_ids)} test")
    pr(f"Params: psi={PSI}, beta_min={BETA_MIN}, liver window [0,200]")

    # Stage 1: Build graph minors + features
    pr(f"\n=== Stage 1: Graph minors + features ===")
    graphs = []
    gm_labels_list = []  # store labels for lifting
    nf_list = []
    t0 = time.time()

    for i, vid in enumerate(all_ordered):
        ct, seg = load(vid)
        gm = build_graph_minor(ct, psi=PSI, alpha=1.0, beta_min=BETA_MIN, beta_max=999999999,
                               m_min=-1e9, m_max=1e9, fast=True)
        nf = extract_node_features(gm["labels"], ct)
        adj = gm.get("full_adjacency", gm["adjacency"])
        ef = extract_edge_features(gm["labels"], ct, adj, nf)
        data, _ = build_pyg_graph(nf, ef, gm["labels"], gt_seg=seg)

        graphs.append(data)
        gm_labels_list.append(gm["labels"])
        nf_list.append(nf)

        n_tu = int((data.y == 1).sum()) if hasattr(data, "y") else 0
        n_bg = int((data.y == 0).sum()) if hasattr(data, "y") else 0
        split = "train" if i < len(train_ids) else ("val" if i < len(train_ids) + len(val_ids) else "test")
        if i % 10 == 0 or i == len(all_ordered) - 1:
            pr(f"  [{i+1}/{len(all_ordered)}] vol-{vid} [{split}]: "
               f"{data.num_nodes:,} nodes ({n_tu} tu), {data.num_edges:,} edges  "
               f"[{time.time()-t0:.0f}s total]")

    pr(f"  Done: {len(graphs)} graphs in {time.time()-t0:.0f}s")

    # Stage 2: Train GINE
    pr(f"\n=== Stage 2: GINE training ===")
    train_g = graphs[:len(train_ids)]
    val_g = graphs[len(train_ids):len(train_ids) + len(val_ids)]

    # Class weights
    tp = sum(int((g.y == 1).sum()) for g in train_g)
    tn = sum(int((g.y == 0).sum()) for g in train_g)
    pr(f"  Train: {tp:,} tumor nodes, {tn:,} bg nodes, ratio=1:{tn // max(tp, 1)}")
    weight = torch.tensor([1.0, tn / max(tp, 1)], dtype=torch.float32).to(DEVICE)
    criterion = torch.nn.CrossEntropyLoss(weight=weight)

    model = SEMIRClassifier(node_dim=7, edge_dim=4, hidden_dim=128).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    best_vd = -1
    best_state = None
    wait = 0
    history = {"train_loss": [], "val_dice": []}

    for epoch in range(1, 201):
        model.train()
        loss_sum = 0
        for g in train_g:
            g_dev = g.to(DEVICE)
            optimizer.zero_grad()
            logits = model(g_dev)
            loss = criterion(logits, g_dev.y)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
        avg_loss = loss_sum / len(train_g)
        history["train_loss"].append(avg_loss)

        model.eval()
        tp_all, fp_all, fn_all = 0, 0, 0
        with torch.no_grad():
            for g in val_g:
                g_dev = g.to(DEVICE)
                preds = model(g_dev).argmax(1)
                tp_all += ((preds == 1) & (g_dev.y == 1)).sum().item()
                fp_all += ((preds == 1) & (g_dev.y == 0)).sum().item()
                fn_all += ((preds == 0) & (g_dev.y == 1)).sum().item()
        vdice = 2 * tp_all / (2 * tp_all + fp_all + fn_all + 1e-8)
        history["val_dice"].append(vdice)

        if vdice > best_vd:
            best_vd = vdice
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= 15:
                pr(f"  Early stop at epoch {epoch}")
                break

        if epoch % 10 == 0 or epoch == 1:
            pr(f"  Epoch {epoch:3d}: loss={avg_loss:.4f}  val_sn_dice={vdice:.4f}")

    if best_state:
        model.load_state_dict(best_state)
    pr(f"  Best val supernode Dice: {best_vd:.4f}")

    # Stage 3: Voxel-level Dice on all volumes
    pr(f"\n=== Stage 3: Voxel-level Dice ===")
    model.eval()
    results = {"train": [], "val": [], "test": []}

    for i, (vid, data, gm_labels, nf) in enumerate(
            zip(all_ordered, graphs, gm_labels_list, nf_list)):
        _, seg = load(vid)
        sids = sorted(nf.keys())

        with torch.no_grad():
            preds = model(data.to(DEVICE)).argmax(1).cpu().numpy()

        # LUT lift
        max_label = int(gm_labels.max())
        lut = np.zeros(max_label + 1, dtype=np.int32)
        for idx, sid in enumerate(sids):
            if preds[idx] == 1:
                lut[sid] = 2
        pred_mask = lut[gm_labels]

        gt_t = (seg == 2)
        pr_t = (pred_mask == 2)
        if gt_t.sum() == 0:
            dice = 1.0 if pr_t.sum() == 0 else 0.0
        else:
            inter = (gt_t & pr_t).sum()
            dice = float(2 * inter / (gt_t.sum() + pr_t.sum() + 1e-8))
        recall = float((gt_t & pr_t).sum() / (gt_t.sum() + 1e-8)) if gt_t.sum() > 0 else 0
        prec = float((gt_t & pr_t).sum() / (pr_t.sum() + 1e-8)) if pr_t.sum() > 0 else 0

        split = "train" if i < len(train_ids) else ("val" if i < len(train_ids) + len(val_ids) else "test")
        results[split].append({
            "vol_id": vid, "dice": round(dice, 4),
            "recall": round(recall, 4), "precision": round(prec, 4),
            "gt_vox": int(gt_t.sum()), "pred_vox": int(pr_t.sum()),
        })

    # Summary
    pr(f"\n{'='*60}")
    pr(f"  RESULTS (psi={PSI}, no deletion, liver window)")
    pr(f"{'='*60}")
    for split in ["train", "val", "test"]:
        dices = [r["dice"] for r in results[split]]
        recalls = [r["recall"] for r in results[split]]
        precs = [r["precision"] for r in results[split]]
        pr(f"\n  {split.upper()} ({len(dices)} volumes):")
        pr(f"    Dice:      mean={np.mean(dices):.4f}  median={np.median(dices):.4f}  "
           f"std={np.std(dices):.4f}")
        pr(f"    Recall:    mean={np.mean(recalls):.4f}")
        pr(f"    Precision: mean={np.mean(precs):.4f}")

    all_dices = [r["dice"] for s in results.values() for r in s]
    pr(f"\n  OVERALL: {np.mean(all_dices):.4f} mean Dice ({len(all_dices)} volumes)")
    pr(f"  SEMIR paper: 0.891")
    pr(f"  Our AuSAM:   0.540")

    # Save
    with open(os.path.join(RESULTS_DIR, "results.json"), "w") as f:
        json.dump({"psi": PSI, "results": results, "history": history,
                   "best_val_sn_dice": best_vd}, f, indent=2)
    pr(f"\n  Saved to {RESULTS_DIR}/results.json")


if __name__ == "__main__":
    main()
