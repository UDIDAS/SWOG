"""
VKG-enhanced SEMIR experiment: 11 features (7 geometric + 4 organ-context).
Usage:
    python -m semir.run_vkg_experiment --psi 0.01 --beta_min 2 --gpu 0 --tag vkg_psi01
"""
import os, sys, json, re, time, argparse, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
from semir.graph_minor import build_graph_minor
from semir.features import extract_node_features, extract_edge_features
from semir.vkg_features import compute_vkg_features, build_enhanced_pyg_graph
from semir.gine import SEMIRClassifier

DATA_ROOT = "/scratch/ud3d4/acm_data/Data"
BASE_RESULTS = "/home/ud3d4/Desktop/SWOG/results"


class FocalLoss(nn.Module):
    """Focal loss for class imbalance: down-weights easy examples."""
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha  # class weights tensor
        self.gamma = gamma

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


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
    parser = argparse.ArgumentParser()
    parser.add_argument("--psi", type=float, default=0.01)
    parser.add_argument("--beta_min", type=int, default=2)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--tag", type=str, default="exp")
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--focal", action="store_true", help="Use focal loss")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=20)
    args = parser.parse_args()

    DEVICE = f"cuda:{args.gpu}"
    RESULTS_DIR = os.path.join(BASE_RESULTS, f"semir_{args.tag}")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    all_ids = discover()
    np.random.seed(42)
    perm = np.random.permutation(all_ids)
    n = len(perm)
    n_train = max(3, int(0.7 * n))
    n_val = max(1, int(0.15 * n))
    train_ids = sorted(perm[:n_train].tolist())
    val_ids = sorted(perm[n_train:n_train + n_val].tolist())
    test_ids = sorted(perm[n_train + n_val:].tolist())
    all_ordered = train_ids + val_ids + test_ids

    pr(f"=== Experiment: {args.tag} ===")
    pr(f"  psi={args.psi}, beta_min={args.beta_min}, gpu={args.gpu}")
    pr(f"  hidden={args.hidden}, layers={args.layers}, focal={args.focal}")
    pr(f"  {len(train_ids)} train / {len(val_ids)} val / {len(test_ids)} test")
    pr(f"  Results: {RESULTS_DIR}")

    # Stage 1: Build graphs
    pr(f"\n=== Stage 1: Graph construction ===")
    graphs = []
    gm_labels_list = []
    nf_list = []
    t0 = time.time()

    for i, vid in enumerate(all_ordered):
        ct, seg = load(vid)
        gm = build_graph_minor(ct, psi=args.psi, alpha=1.0, beta_min=args.beta_min,
                               beta_max=999999999, m_min=-1e9, m_max=1e9, fast=True)
        nf = extract_node_features(gm["labels"], ct)
        nf = compute_vkg_features(gm["labels"], ct, seg, nf)  # +4 organ features
        adj = gm.get("full_adjacency", gm["adjacency"])
        ef = extract_edge_features(gm["labels"], ct, adj, nf)
        data = build_enhanced_pyg_graph(nf, ef, gm["labels"], gt_seg=seg)  # 11 features

        graphs.append(data)
        gm_labels_list.append(gm["labels"])
        nf_list.append(nf)

        if (i + 1) % 10 == 0 or i == 0 or i == len(all_ordered) - 1:
            n_tu = int((data.y == 1).sum()) if hasattr(data, "y") else 0
            pr(f"  [{i+1}/{len(all_ordered)}] vol-{vid}: "
               f"{data.num_nodes:,} nodes ({n_tu} tu), "
               f"{data.num_edges:,} edges  [{time.time()-t0:.0f}s]")

    elapsed_build = time.time() - t0
    pr(f"  Done: {len(graphs)} graphs in {elapsed_build:.0f}s")

    # Stage 2: Train GINE
    pr(f"\n=== Stage 2: GINE training ===")
    train_g = graphs[:len(train_ids)]
    val_g = graphs[len(train_ids):len(train_ids) + len(val_ids)]

    tp = sum(int((g.y == 1).sum()) for g in train_g)
    tn = sum(int((g.y == 0).sum()) for g in train_g)
    pr(f"  Tumor nodes: {tp:,} / Background: {tn:,} / Ratio: 1:{tn // max(tp, 1)}")

    weight = torch.tensor([1.0, tn / max(tp, 1)], dtype=torch.float32).to(DEVICE)

    if args.focal:
        criterion = FocalLoss(alpha=weight, gamma=2.0)
        pr(f"  Loss: Focal (gamma=2.0)")
    else:
        criterion = nn.CrossEntropyLoss(weight=weight)
        pr(f"  Loss: Weighted CrossEntropy")

    model = SEMIRClassifier(node_dim=11, edge_dim=4, hidden_dim=args.hidden).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    best_vd = -1
    best_state = None
    wait = 0
    history = {"train_loss": [], "val_dice": []}
    t0_train = time.time()

    for epoch in range(1, args.epochs + 1):
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
            if wait >= args.patience:
                pr(f"  Early stop at epoch {epoch}")
                break

        if epoch % 10 == 0 or epoch == 1:
            pr(f"  Epoch {epoch:3d}: loss={avg_loss:.4f}  val_sn_dice={vdice:.4f}  "
               f"[{time.time()-t0_train:.0f}s]")

    if best_state:
        model.load_state_dict(best_state)
    elapsed_train = time.time() - t0_train
    pr(f"  Best val supernode Dice: {best_vd:.4f} (trained in {elapsed_train:.0f}s)")

    # Save model
    torch.save(best_state, os.path.join(RESULTS_DIR, "best_model.pt"))

    # Stage 3: Voxel-level Dice
    pr(f"\n=== Stage 3: Voxel-level Dice ===")
    model.eval()
    results = {"train": [], "val": [], "test": []}

    for i, (vid, data, gm_labels, nf) in enumerate(
            zip(all_ordered, graphs, gm_labels_list, nf_list)):
        _, seg = load(vid)
        sids = sorted(nf.keys())

        with torch.no_grad():
            preds = model(data.to(DEVICE)).argmax(1).cpu().numpy()

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

        split = ("train" if i < len(train_ids) else
                 ("val" if i < len(train_ids) + len(val_ids) else "test"))
        results[split].append({
            "vol_id": vid, "dice": round(dice, 4),
            "recall": round(recall, 4), "precision": round(prec, 4),
            "gt_vox": int(gt_t.sum()), "pred_vox": int(pr_t.sum()),
        })

    # Summary
    pr(f"\n{'='*60}")
    pr(f"  RESULTS: {args.tag}")
    pr(f"  psi={args.psi}, beta_min={args.beta_min}, focal={args.focal}")
    pr(f"{'='*60}")
    for split in ["train", "val", "test"]:
        dices = [r["dice"] for r in results[split]]
        if not dices:
            continue
        pr(f"\n  {split.upper()} ({len(dices)} volumes):")
        pr(f"    Dice:      mean={np.mean(dices):.4f}  median={np.median(dices):.4f}  "
           f"std={np.std(dices):.4f}")
        pr(f"    Recall:    mean={np.mean([r['recall'] for r in results[split]]):.4f}")
        pr(f"    Precision: mean={np.mean([r['precision'] for r in results[split]]):.4f}")

    all_dices = [r["dice"] for s in results.values() for r in s]
    test_dices = [r["dice"] for r in results["test"]]
    pr(f"\n  OVERALL mean Dice:  {np.mean(all_dices):.4f}")
    pr(f"  TEST mean Dice:     {np.mean(test_dices):.4f}")
    pr(f"  SEMIR paper:        0.891")
    pr(f"  Our AuSAM:          0.540")

    # Save
    with open(os.path.join(RESULTS_DIR, "results.json"), "w") as f:
        json.dump({"config": vars(args), "results": results, "history": history,
                   "best_val_sn_dice": best_vd,
                   "build_time_s": elapsed_build, "train_time_s": elapsed_train}, f, indent=2)
    pr(f"\n  Saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
