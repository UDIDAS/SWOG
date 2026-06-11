"""
Two-stage SEMIR: liver-first, then tumor-within-liver.

Stage A: Use graph minor at ψ=0.03 to find liver supernodes (organ_fraction > 0.5)
Stage B: Train GINE only on liver supernodes for tumor classification
This dramatically improves class balance and reduces graph size.
"""
import os, sys, json, re, time, argparse, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
from semir.graph_minor import build_graph_minor
from semir.features import extract_node_features, extract_edge_features
from semir.gine import SEMIRClassifier

DATA_ROOT = "/scratch/ud3d4/acm_data/Data"
BASE_RESULTS = "/home/ud3d4/Desktop/SWOG/results"


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha
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


def build_liver_subgraph(gm_labels, ct, seg, nf_full, psi):
    """Extract only liver-region supernodes from the graph minor."""
    flat_l = gm_labels.ravel()
    max_id = int(flat_l.max())
    if max_id == 0:
        return None, None, None

    counts = np.bincount(flat_l, minlength=max_id + 1).astype(np.float64)
    liver_c = np.bincount(flat_l, weights=(seg.ravel() >= 1).astype(np.float64),
                          minlength=max_id + 1)
    tumor_c = np.bincount(flat_l, weights=(seg.ravel() == 2).astype(np.float64),
                          minlength=max_id + 1)
    safe = np.where(counts > 0, counts, 1)
    liver_frac = liver_c / safe
    tumor_frac = tumor_c / safe

    # Keep supernodes with >30% liver content
    liver_sids = set(np.where((liver_frac > 0.3) & (np.arange(max_id + 1) > 0))[0])

    if not liver_sids:
        return None, None, None

    # Filter node features to liver-only
    nf_liver = {sid: nf_full[sid] for sid in liver_sids if sid in nf_full}

    # Add liver-specific features
    organ_int = ct[seg >= 1]
    organ_mean = organ_int.mean() if len(organ_int) > 0 else 0.5

    for sid in nf_liver:
        nf_liver[sid]["liver_fraction"] = round(float(liver_frac[sid]), 4)
        nf_liver[sid]["relative_intensity"] = round(
            float(nf_liver[sid]["mean_intensity"] - organ_mean), 4)

    # Build edges only between liver supernodes
    from semir.graph_minor import _edge_deletion
    all_adj = _edge_deletion(gm_labels, ct, alpha=1e9)
    liver_adj = {(i, j): d for (i, j), d in all_adj.items()
                 if i in liver_sids and j in liver_sids}

    # Build edge features
    ef = {}
    max_dim = max(gm_labels.shape)
    for (i, j), _ in liver_adj.items():
        fi = nf_liver.get(i)
        fj = nf_liver.get(j)
        if fi is None or fj is None:
            continue
        log_vol = float(np.log(fi["volume"] / (fj["volume"] + 1e-8) + 1e-8))
        int_range = max(abs(fi["mean_intensity"]) + abs(fj["mean_intensity"]), 1e-8)
        int_diff = abs(fi["mean_intensity"] - fj["mean_intensity"]) / int_range
        ci, cj = np.array(fi["centroid"]), np.array(fj["centroid"])
        diff = cj - ci
        dist = float(np.linalg.norm(diff))
        cos_t = float(diff[0] / (dist + 1e-8))
        ef[(i, j)] = {
            "log_volume_ratio": round(log_vol, 4),
            "intensity_diff_norm": round(float(int_diff), 4),
            "distance_norm": round(dist / max_dim, 4),
            "orientation_cos": round(cos_t, 4),
        }

    # Build PyG graph with 9 features (7 geometric + liver_fraction + relative_intensity)
    import torch as th
    from torch_geometric.data import Data

    sids = sorted(nf_liver.keys())
    sid_to_idx = {s: i for i, s in enumerate(sids)}

    feat_names = ["volume", "boundary_length", "compactness", "elongation",
                  "dominant_axis", "mean_intensity", "intensity_std",
                  "liver_fraction", "relative_intensity"]
    x = np.array([[nf_liver[s].get(f, 0.0) for f in feat_names] for s in sids],
                 dtype=np.float32)
    for col in range(x.shape[1]):
        mn, mx = x[:, col].min(), x[:, col].max()
        if mx - mn > 1e-8:
            x[:, col] = (x[:, col] - mn) / (mx - mn)

    edge_idx, edge_attr = [], []
    for (i, j), e in ef.items():
        if i in sid_to_idx and j in sid_to_idx:
            edge_idx.append([sid_to_idx[i], sid_to_idx[j]])
            edge_idx.append([sid_to_idx[j], sid_to_idx[i]])
            feat = [e["log_volume_ratio"], e["intensity_diff_norm"],
                    e["distance_norm"], e["orientation_cos"]]
            edge_attr.append(feat)
            edge_attr.append(feat)

    if edge_idx:
        ei = th.tensor(edge_idx, dtype=th.long).t().contiguous()
        ea = th.tensor(edge_attr, dtype=th.float32)
    else:
        ei = th.zeros((2, 0), dtype=th.long)
        ea = th.zeros((0, 4), dtype=th.float32)

    # Labels: tumor within liver
    y = np.array([int(tumor_frac[s] > 0.1) if s <= max_id else 0
                  for s in sids], dtype=np.int64)

    data = Data(x=th.tensor(x, dtype=th.float32),
                edge_index=ei, edge_attr=ea,
                y=th.tensor(y, dtype=th.long))
    return data, sids, liver_sids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--psi", type=float, default=0.03)
    parser.add_argument("--beta_min", type=int, default=2)
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--tag", type=str, default="twostage")
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

    pr(f"=== Two-Stage SEMIR: {args.tag} ===")
    pr(f"  psi={args.psi}, beta_min={args.beta_min}, gpu={args.gpu}")
    pr(f"  {len(train_ids)} train / {len(val_ids)} val / {len(test_ids)} test")

    # Stage A: Build graph minors + extract liver-only subgraphs
    pr(f"\n=== Stage A: Graph minor + liver subgraph extraction ===")
    graphs = []
    gm_labels_list = []
    sids_list = []
    liver_sids_list = []
    t0 = time.time()
    skipped = 0

    for i, vid in enumerate(all_ordered):
        ct, seg = load(vid)
        gm = build_graph_minor(ct, psi=args.psi, alpha=1.0, beta_min=args.beta_min,
                               beta_max=999999999, m_min=-1e9, m_max=1e9, fast=True)
        nf = extract_node_features(gm["labels"], ct)
        data, sids, liver_sids = build_liver_subgraph(gm["labels"], ct, seg, nf, args.psi)

        if data is None:
            skipped += 1
            graphs.append(None)
            gm_labels_list.append(gm["labels"])
            sids_list.append([])
            liver_sids_list.append(set())
            continue

        graphs.append(data)
        gm_labels_list.append(gm["labels"])
        sids_list.append(sids)
        liver_sids_list.append(liver_sids)

        if (i + 1) % 10 == 0 or i == 0 or i == len(all_ordered) - 1:
            n_tu = int((data.y == 1).sum())
            n_bg = int((data.y == 0).sum())
            pr(f"  [{i+1}/{len(all_ordered)}] vol-{vid}: "
               f"{data.num_nodes:,} liver SN ({n_tu} tumor, {n_bg} bg), "
               f"{data.num_edges:,} edges  [{time.time()-t0:.0f}s]")

    pr(f"  Done: {len(graphs)-skipped} graphs ({skipped} skipped) in {time.time()-t0:.0f}s")

    # Stage B: Train GINE on liver-only subgraphs
    pr(f"\n=== Stage B: GINE training on liver subgraphs ===")
    train_g = [g for g in graphs[:len(train_ids)] if g is not None]
    val_g = [g for g in graphs[len(train_ids):len(train_ids) + len(val_ids)] if g is not None]

    tp = sum(int((g.y == 1).sum()) for g in train_g)
    tn = sum(int((g.y == 0).sum()) for g in train_g)
    pr(f"  Liver-only: {tp:,} tumor SN, {tn:,} bg SN, ratio=1:{tn // max(tp, 1)}")
    pr(f"  Balance: {tp / max(tp + tn, 1) * 100:.1f}% tumor")

    weight = torch.tensor([1.0, tn / max(tp, 1)], dtype=torch.float32).to(DEVICE)
    criterion = FocalLoss(alpha=weight, gamma=2.0)

    model = SEMIRClassifier(node_dim=9, edge_dim=4, hidden_dim=128).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    best_vd = -1
    best_state = None
    wait = 0
    history = {"train_loss": [], "val_dice": []}

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
        avg_loss = loss_sum / max(len(train_g), 1)
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
            pr(f"  Epoch {epoch:3d}: loss={avg_loss:.4f}  val_sn_dice={vdice:.4f}")

    if best_state:
        model.load_state_dict(best_state)
    pr(f"  Best val supernode Dice: {best_vd:.4f}")
    torch.save(best_state, os.path.join(RESULTS_DIR, "best_model.pt"))

    # Stage C: Voxel-level Dice
    pr(f"\n=== Stage C: Voxel-level Dice ===")
    model.eval()
    results = {"train": [], "val": [], "test": []}

    for i, (vid, data, gm_labels, sids) in enumerate(
            zip(all_ordered, graphs, gm_labels_list, sids_list)):
        _, seg = load(vid)

        if data is None or not sids:
            split = ("train" if i < len(train_ids) else
                     ("val" if i < len(train_ids) + len(val_ids) else "test"))
            results[split].append({
                "vol_id": vid, "dice": 0.0, "recall": 0.0, "precision": 0.0,
                "gt_vox": int((seg == 2).sum()), "pred_vox": 0,
            })
            continue

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
    pr(f"  RESULTS: {args.tag} (Two-Stage)")
    pr(f"  psi={args.psi}, liver-only subgraphs")
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

    with open(os.path.join(RESULTS_DIR, "results.json"), "w") as f:
        json.dump({"config": vars(args), "results": results, "history": history,
                   "best_val_sn_dice": best_vd}, f, indent=2)
    pr(f"\n  Saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
