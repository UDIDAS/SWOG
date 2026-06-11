"""
Luke's diagnostic suite for LiTS.

Runs on LiTS data to answer three questions:
  1. Oracle reconstruction: what's the quality ceiling of the graph minor?
  2. Deletion analysis: how many tumor voxels does node deletion remove?
  3. Threshold sweep: does lowering the GNN threshold help?

Usage:
    conda run -n llmft python -m semir.diagnose_lits
"""

import os, sys, json, time, re
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from semir.graph_minor import build_graph_minor
from semir.features import extract_node_features, extract_edge_features, build_pyg_graph
from semir.gine import train_gine

DATA_ROOT = "/scratch/ud3d4/acm_data/Data"
RESULTS_DIR = "/home/ud3d4/Desktop/SWOG/results/diagnose_lits"


def pr(msg=""):
    print(msg, flush=True)


def section(title):
    pr(f"\n{'='*70}")
    pr(f"  {title}")
    pr(f"{'='*70}")


def load_lits(vol_id):
    ct = np.load(os.path.join(DATA_ROOT, "ct", f"volume-{vol_id}.npy")).astype(np.float32)
    seg = np.load(os.path.join(DATA_ROOT, "seg", f"segmentation-{vol_id}.npy")).astype(np.int32)
    return ct, seg


def hu_window(vol, hu_min=0, hu_max=200):
    """Liver window [0, 200] — doubles contrast vs [-150, 250]."""
    vol = np.clip(vol, hu_min, hu_max)
    return (vol - hu_min) / (hu_max - hu_min)


def discover_lits():
    ct_dir = os.path.join(DATA_ROOT, "ct")
    ids = []
    for f in os.listdir(ct_dir):
        m = re.match(r"volume-(\d+)\.npy", f)
        if m:
            vid = int(m.group(1))
            if os.path.exists(os.path.join(DATA_ROOT, "seg", f"segmentation-{vid}.npy")):
                ids.append(vid)
    return sorted(ids)


def oracle_dice(labels, gt_seg):
    """Select all supernodes with ANY GT tumor overlap, reconstruct, compute Dice."""
    flat_labels = labels.ravel()
    gt_tumor = (gt_seg == 2)
    gt_flat = gt_tumor.ravel().astype(np.float64)
    gt_total = int(gt_tumor.sum())

    if gt_total == 0:
        return 1.0, 0, 0, 0.0

    max_label = int(flat_labels.max())
    if max_label == 0:
        return 0.0, 0, gt_total, 100.0

    tumor_counts = np.bincount(flat_labels, weights=gt_flat, minlength=max_label + 1)

    # Oracle: any supernode touching GT tumor
    oracle_sids = set(np.where(tumor_counts > 0)[0])
    oracle_sids.discard(0)

    oracle_lut = np.zeros(max_label + 1, dtype=np.int32)
    for sid in oracle_sids:
        oracle_lut[sid] = 1
    oracle_mask = oracle_lut[flat_labels].reshape(labels.shape).astype(bool)

    inter = int((gt_tumor & oracle_mask).sum())
    dice = float(2 * inter / (gt_tumor.sum() + oracle_mask.sum() + 1e-8))

    # Deletion loss: GT tumor voxels with label=0
    deleted = int(tumor_counts[0])
    del_pct = deleted / max(gt_total, 1) * 100.0

    return dice, len(oracle_sids), deleted, del_pct


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    all_ids = discover_lits()
    pr(f"  Found {len(all_ids)} LiTS volumes")

    # Use a representative subset for speed
    n_cases = int(os.environ.get("DIAG_N_CASES", 20))
    np.random.seed(42)
    selected = sorted(np.random.choice(all_ids, size=min(n_cases, len(all_ids)), replace=False))
    pr(f"  Using {len(selected)} cases: {selected}")

    # ================================================================
    # DIAGNOSTIC 1 & 2: Oracle reconstruction + deletion analysis
    # ================================================================
    section("DIAGNOSTIC 1 & 2: Oracle + Deletion Analysis")
    pr(f"  Testing multiple parameter configs to find what preserves tumor...")
    pr()

    configs = [
        {"psi": 0.12, "alpha": 0.12, "beta_min": 10, "label": "current (psi=0.12, beta=10)"},
        {"psi": 0.12, "alpha": 0.12, "beta_min": 2,  "label": "lower beta (psi=0.12, beta=2)"},
        {"psi": 0.12, "alpha": 0.12, "beta_min": 1,  "label": "no deletion (psi=0.12, beta=1)"},
        {"psi": 0.05, "alpha": 0.05, "beta_min": 10, "label": "tighter psi (psi=0.05, beta=10)"},
        {"psi": 0.05, "alpha": 0.05, "beta_min": 2,  "label": "tighter psi (psi=0.05, beta=2)"},
        {"psi": 0.05, "alpha": 0.05, "beta_min": 1,  "label": "tighter psi no del (psi=0.05, beta=1)"},
    ]

    all_config_results = []

    for cfg in configs:
        oracle_dices, del_losses, sn_counts = [], [], []

        for vid in selected:
            ct, seg = load_lits(vid)
            if (seg == 2).sum() == 0:
                continue
            ct = hu_window(ct)

            gm = build_graph_minor(
                ct, psi=cfg["psi"], alpha=cfg["alpha"],
                beta_min=cfg["beta_min"], beta_max=500000,
                m_min=0.0, m_max=1.0,
            )

            od, n_tumor_sn, deleted, del_pct = oracle_dice(gm["labels"], seg)
            oracle_dices.append(od)
            del_losses.append(del_pct)
            sn_counts.append(gm["stats"]["n_supernodes_after_deletion"])

        mean_od = np.mean(oracle_dices) if oracle_dices else 0
        mean_del = np.mean(del_losses) if del_losses else 0
        mean_sn = np.mean(sn_counts) if sn_counts else 0

        pr(f"  {cfg['label']:<45s}  "
           f"oracle={mean_od:.4f}  del_loss={mean_del:.1f}%  "
           f"~{mean_sn:.0f} SN  (n={len(oracle_dices)})")

        all_config_results.append({
            "config": cfg,
            "mean_oracle_dice": round(mean_od, 4),
            "mean_deletion_loss_pct": round(mean_del, 1),
            "mean_supernodes": round(mean_sn, 0),
            "n_cases": len(oracle_dices),
        })

    with open(os.path.join(RESULTS_DIR, "oracle_deletion_analysis.json"), "w") as f:
        json.dump(all_config_results, f, indent=2)

    # Pick the best config for the rest of the diagnostics
    best_cfg_idx = np.argmax([r["mean_oracle_dice"] for r in all_config_results])
    best_cfg = configs[best_cfg_idx]
    best_oracle = all_config_results[best_cfg_idx]["mean_oracle_dice"]
    pr(f"\n  Best config: {best_cfg['label']} (oracle={best_oracle:.4f})")

    # ================================================================
    # DIAGNOSTIC 3: Full pipeline with best config + threshold sweep
    # ================================================================
    section("DIAGNOSTIC 3: GINE + Threshold Sweep (best config)")

    # Split: 70/15/15
    perm = np.random.permutation(len(selected))
    n_train = max(3, int(0.7 * len(selected)))
    n_val = max(1, int(0.15 * len(selected)))
    train_ids = [selected[i] for i in perm[:n_train]]
    val_ids = [selected[i] for i in perm[n_train:n_train + n_val]]
    test_ids = [selected[i] for i in perm[n_train + n_val:]]
    ordered = train_ids + val_ids + test_ids

    pr(f"  Split: {len(train_ids)} train / {len(val_ids)} val / {len(test_ids)} test")
    pr(f"  Building graph minors + features...")

    volumes, segs, graphs, node_feats, minors_list = [], [], [], [], []
    for vid in ordered:
        ct, seg = load_lits(vid)
        ct = hu_window(ct)
        volumes.append(ct)
        segs.append(seg)

        gm = build_graph_minor(
            ct, psi=best_cfg["psi"], alpha=best_cfg["alpha"],
            beta_min=best_cfg["beta_min"], beta_max=500000,
            m_min=0.0, m_max=1.0,
        )
        minors_list.append(gm)

        nf = extract_node_features(gm["labels"], ct)
        adj = gm.get("full_adjacency", gm["adjacency"])
        ef = extract_edge_features(gm["labels"], ct, adj, nf)
        data, _ = build_pyg_graph(nf, ef, gm["labels"], gt_seg=seg)
        graphs.append(data)
        node_feats.append(nf)

        n_tu = int((data.y == 1).sum())
        n_bg = int((data.y == 0).sum())
        pr(f"    vol-{vid}: {data.num_nodes:,} nodes ({n_tu} tumor, {n_bg:,} bg), "
           f"{data.num_edges:,} edges")

    # Train GINE
    import torch
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    pr(f"\n  Training GINE on {device}...")

    train_g = graphs[:len(train_ids)]
    val_g = graphs[len(train_ids):len(train_ids) + len(val_ids)]

    model, history = train_gine(train_g, val_g, epochs=200, lr=1e-3, patience=15, device=device)
    pr(f"  Epochs: {len(history['train_loss'])}, best val dice: {max(history['val_dice']) if history['val_dice'] else 0:.4f}")

    # Collect softmax probabilities
    model.eval()
    model = model.to(device)
    all_probs = []
    for data in graphs:
        with torch.no_grad():
            logits = model(data.clone().to(device))
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        all_probs.append(probs)

    # Threshold sweep on val set
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    pr(f"\n  Threshold sweep (val set):")
    pr(f"  {'Thresh':>8s} {'Dice':>8s} {'Recall':>8s} {'Prec':>8s}")
    pr(f"  {'-'*34}")

    best_thresh, best_val_dice = 0.5, -1.0
    sweep_results = []

    for thresh in thresholds:
        tp, fp, fn = 0, 0, 0
        for i in range(len(train_ids), len(train_ids) + len(val_ids)):
            gm, seg, nf, probs = minors_list[i], segs[i], node_feats[i], all_probs[i]
            sids = sorted(nf.keys())
            max_label = int(gm["labels"].max())
            lut = np.zeros(max_label + 1, dtype=np.int32)
            for idx, sid in enumerate(sids):
                if probs[idx] >= thresh:
                    lut[sid] = 1
            pred = lut[gm["labels"].ravel()].reshape(gm["labels"].shape).astype(bool)
            gt = (seg == 2)
            inter = int((gt & pred).sum())
            tp += inter
            fp += int(pred.sum()) - inter
            fn += int(gt.sum()) - inter

        dice = float(2 * tp / (2 * tp + fp + fn + 1e-8))
        recall = float(tp / (tp + fn + 1e-8))
        prec = float(tp / (tp + fp + 1e-8))
        pr(f"  {thresh:>8.2f} {dice:>8.4f} {recall:>8.4f} {prec:>8.4f}")
        sweep_results.append({"threshold": thresh, "dice": dice, "recall": recall, "precision": prec})

        if dice > best_val_dice:
            best_val_dice = dice
            best_thresh = thresh

    pr(f"\n  Best threshold: {best_thresh} (val Dice={best_val_dice:.4f})")

    # Final per-case results with best threshold
    section("Per-case results (best threshold)")
    final_scores = []
    for i, vid in enumerate(ordered):
        gm, seg, nf, probs = minors_list[i], segs[i], node_feats[i], all_probs[i]
        sids = sorted(nf.keys())
        max_label = int(gm["labels"].max())
        lut = np.zeros(max_label + 1, dtype=np.int32)
        for idx, sid in enumerate(sids):
            if probs[idx] >= best_thresh:
                lut[sid] = 1
        pred = lut[gm["labels"].ravel()].reshape(gm["labels"].shape).astype(bool)
        gt = (seg == 2)

        if gt.sum() == 0:
            dice = 1.0 if pred.sum() == 0 else 0.0
        else:
            inter = int((gt & pred).sum())
            dice = float(2 * inter / (gt.sum() + pred.sum() + 1e-8))

        recall = float((gt & pred).sum() / (gt.sum() + 1e-8))
        prec = float((gt & pred).sum() / (pred.sum() + 1e-8)) if pred.sum() > 0 else 0

        split = "train" if i < len(train_ids) else ("val" if i < len(train_ids) + len(val_ids) else "test")
        pr(f"  vol-{vid} [{split}]: Dice={dice:.4f}  Recall={recall:.4f}  "
           f"Precision={prec:.4f}  GT={int(gt.sum()):,}  Pred={int(pred.sum()):,}")
        final_scores.append({"vol_id": vid, "split": split, "dice": dice,
                             "recall": recall, "precision": prec})

    # Summary
    section("SUMMARY FOR LUKE")
    test_dices = [s["dice"] for s in final_scores if s["split"] == "test"]
    val_dices = [s["dice"] for s in final_scores if s["split"] == "val"]
    all_dices = [s["dice"] for s in final_scores]

    pr(f"  Best config:     {best_cfg['label']}")
    pr(f"  Oracle Dice:     {best_oracle:.4f}  (quality ceiling of graph minor)")
    pr(f"  Best threshold:  {best_thresh}")
    pr(f"  Mean Dice (all): {np.mean(all_dices):.4f}")
    pr(f"  Mean Dice (val): {np.mean(val_dices):.4f}" if val_dices else "  No val cases")
    pr(f"  Mean Dice (test):{np.mean(test_dices):.4f}" if test_dices else "  No test cases")
    pr(f"  Paper target:    0.891")
    pr()
    pr(f"  If oracle << 1.0: graph minor loses tumor (fix contraction/deletion)")
    pr(f"  If oracle ~ 1.0 but Dice << oracle: GNN is the bottleneck")

    results = {
        "best_config": best_cfg,
        "oracle_dice": best_oracle,
        "best_threshold": best_thresh,
        "config_sweep": all_config_results,
        "threshold_sweep": sweep_results,
        "per_case": final_scores,
    }
    with open(os.path.join(RESULTS_DIR, "full_diagnostics.json"), "w") as f:
        json.dump(results, f, indent=2)
    pr(f"\n  Results saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
