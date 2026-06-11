"""
SEMIR end-to-end pipeline for MSD Task07 Pancreas dataset.

Runs the full SEMIR pipeline on pancreas CT volumes (.nii.gz) to produce
tumor phenotypes and VKG — replacing the GT CSV dependency.

Pancreas has 1:1 organ-to-tumor mapping (one pancreas, one tumor per case),
making it the cleanest test case for the SEMIR → VKG pipeline.

Usage:
    conda run -n llmft python -m semir.pipeline_pancreas
"""

import os
import sys
import json
import time
import re
import numpy as np
import nibabel as nib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from semir.graph_minor import build_graph_minor
from semir.features import extract_node_features, extract_edge_features, build_pyg_graph
from semir.param_search import few_shot_search
from semir.gine import SEMIRClassifier, train_gine
from semir.vkg_builder import build_vkg_from_semir

DATA_ROOT = "/scratch/ud3d4/acm_data/Task07_Pancreas"
RESULTS_DIR = "/home/ud3d4/Desktop/SWOG/results/semir_pancreas"
CSV_PATH = "/home/ud3d4/Desktop/Projects/acm_mmkg/data/Pancreas_Tumor_Analysis.csv"  # for validation only


def pr(msg=""):
    print(msg, flush=True)


def section(title):
    pr(f"\n{'='*70}")
    pr(f"  {title}")
    pr(f"{'='*70}")


def load_pancreas_case(case_name: str):
    """Load a pancreas case from .nii.gz files. Returns (ct, seg, spacing)."""
    ct_path = os.path.join(DATA_ROOT, "imagesTr", f"{case_name}.nii.gz")
    seg_path = os.path.join(DATA_ROOT, "labelsTr", f"{case_name}.nii.gz")

    ct_nii = nib.load(ct_path)
    seg_nii = nib.load(seg_path)

    ct = ct_nii.get_fdata().astype(np.float32)
    seg = seg_nii.get_fdata().astype(np.int32)

    # Extract voxel spacing from NIfTI header (mm)
    spacing = tuple(float(s) for s in ct_nii.header.get_zooms()[:3])

    return ct, seg, spacing


def hu_window(vol, hu_min=20, hu_max=180):
    """Clip and normalize CT to [0, 1]. Narrow window [20, 180] maximizes
    pancreas-tumor contrast (0.125 normalized gap vs 0.057 for [-150, 250])."""
    vol = np.clip(vol, hu_min, hu_max)
    vol = (vol - hu_min) / (hu_max - hu_min)
    return vol


def discover_pancreas_cases():
    """Find all pancreas cases that have both CT and segmentation."""
    img_dir = os.path.join(DATA_ROOT, "imagesTr")
    seg_dir = os.path.join(DATA_ROOT, "labelsTr")
    if not os.path.isdir(img_dir) or not os.path.isdir(seg_dir):
        return []

    cases = []
    for f in sorted(os.listdir(img_dir)):
        m = re.match(r"(pancreas_\d+)\.nii\.gz", f)
        if m:
            name = m.group(1)
            if os.path.exists(os.path.join(seg_dir, f"{name}.nii.gz")):
                cases.append(name)
    return cases


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Discover available cases
    all_cases = discover_pancreas_cases()
    pr(f"  Found {len(all_cases)} pancreas cases with segmentations")
    if len(all_cases) == 0:
        pr("  ERROR: No data found. Check rclone download.")
        return

    # Optional case limit (set SEMIR_N_CASES env var)
    max_cases = int(os.environ.get("SEMIR_N_CASES", len(all_cases)))
    if max_cases < len(all_cases):
        all_cases = all_cases[:max_cases]
        pr(f"  Limited to {max_cases} cases (SEMIR_N_CASES)")

    # Split: 70% train / 15% val / 15% test
    np.random.seed(42)
    perm = np.random.permutation(len(all_cases))
    n = len(perm)
    n_train = max(3, int(0.7 * n))
    n_val = max(1, int(0.15 * n))

    train_cases = sorted([all_cases[i] for i in perm[:n_train]])
    val_cases = sorted([all_cases[i] for i in perm[n_train:n_train + n_val]])
    test_cases = sorted([all_cases[i] for i in perm[n_train + n_val:]])
    ordered_cases = train_cases + val_cases + test_cases

    pr(f"  Split: {len(train_cases)} train / {len(val_cases)} val / {len(test_cases)} test")

    # ==== STAGE 0: Load data ====
    section("STAGE 0: Loading pancreas volumes")
    volumes, segs, spacings, case_ids = [], [], [], []
    for name in ordered_cases:
        ct, seg, spacing = load_pancreas_case(name)
        ct = hu_window(ct)
        volumes.append(ct)
        segs.append(seg)
        spacings.append(spacing)
        case_ids.append(name)

        n_pancreas = int((seg == 1).sum())
        n_tumor = int((seg == 2).sum())
        pr(f"  {name}: shape={ct.shape}  spacing={spacing}  "
           f"pancreas={n_pancreas:,}  tumor={n_tumor:,}  "
           f"coverage={n_tumor/max(n_pancreas,1)*100:.2f}%")

    n_train_total = len(train_cases)
    n_val_total = len(val_cases)

    # ==== STAGE 1: Parameter search ====
    if os.environ.get("SEMIR_SEARCH"):
        n_few = min(5, n_train_total)
        section(f"STAGE 1: Few-shot parameter search ({n_few} volumes)")
        best_params, search_log = few_shot_search(
            [volumes[i] for i in range(n_few)],
            [segs[i] for i in range(n_few)],
            target_label=2,
            k_psi=16, k_alpha=8, k_beta=8,
            verbose=True,
        )
        with open(os.path.join(RESULTS_DIR, "param_search_log.json"), "w") as f:
            json.dump(search_log, f, indent=2)
        sorted_log = sorted(search_log, key=lambda x: -x["score"])
        pr(f"\n  Top-5 configurations:")
        pr(f"  {'psi':>8s} {'alpha':>8s} {'beta':>6s} {'Dice':>8s} {'Score':>8s} "
           f"{'Compress':>10s} {'#Supernodes':>12s}")
        for e in sorted_log[:5]:
            pr(f"  {e['psi']:8.4f} {e['alpha']:8.4f} {e['beta_min']:6d} "
               f"{e['mean_dice']:8.4f} {e['score']:8.4f} "
               f"{e['compression']:10.1f}x {e['mean_supernodes']:12.0f}")
    else:
        section("STAGE 1: Using pre-validated parameters")
        best_params = {
            "psi": 0.12, "alpha": 0.12, "beta_min": 10,
            "beta_max": 500000, "m_min": 0.0, "m_max": 1.0,
        }
        pr(f"  psi={best_params['psi']}  alpha={best_params['alpha']}  "
           f"beta_min={best_params['beta_min']}")

    # ==== STAGE 2: Build graph minors ====
    section("STAGE 2: Graph minor construction")
    minors = []
    for vol, seg, spacing, cid in zip(volumes, segs, spacings, case_ids):
        gm = build_graph_minor(
            vol,
            psi=best_params["psi"], alpha=best_params["alpha"],
            beta_min=best_params["beta_min"], beta_max=best_params["beta_max"],
            m_min=best_params["m_min"], m_max=best_params["m_max"],
            fast=True,
        )
        minors.append(gm)
        s = gm["stats"]
        pr(f"  {cid}: {s['n_voxels']:>10,} voxels -> {s['n_supernodes_after_deletion']:>6,} "
           f"supernodes  ({s['compression_ratio']:,.0f}x)  "
           f"edges={s['n_edges']:,}  time={s['time_total_s']:.1f}s")

    with open(os.path.join(RESULTS_DIR, "graph_minor_stats.json"), "w") as f:
        json.dump([m["stats"] for m in minors], f, indent=2)

    # ==== STAGE 2b: Oracle reconstruction + deletion analysis ====
    # Luke's diagnostic: estimates quality ceiling of the Stage 1 representation.
    section("STAGE 2b: Oracle reconstruction & deletion analysis")
    oracle_results = []
    for gm, vol, seg, cid in zip(minors, volumes, segs, case_ids):
        result = _oracle_reconstruction(gm, vol, seg, cid, best_params)
        oracle_results.append(result)
        pr(f"  {cid}: oracle_dice={result['oracle_dice']:.4f}  "
           f"tumor_SN={result['n_tumor_supernodes']}  "
           f"deleted_tumor_vox={result['tumor_voxels_deleted']:,}/"
           f"{result['gt_tumor_voxels']:,} "
           f"({result['deletion_loss_pct']:.1f}% lost in deletion)")

    with open(os.path.join(RESULTS_DIR, "oracle_analysis.json"), "w") as f:
        json.dump(oracle_results, f, indent=2)

    mean_oracle = np.mean([r["oracle_dice"] for r in oracle_results])
    mean_del_loss = np.mean([r["deletion_loss_pct"] for r in oracle_results])
    pr(f"\n  Mean oracle Dice (quality ceiling): {mean_oracle:.4f}")
    pr(f"  Mean deletion loss: {mean_del_loss:.1f}%")
    pr(f"  --> If oracle << 1.0, the graph minor loses tumor extent at Stage 1.")
    pr(f"  --> If oracle is high but GINE Dice is low, the classifier is the bottleneck.")

    # ==== STAGE 3: Extract features + build PyG graphs ====
    section("STAGE 3: Feature extraction -> PyG graphs")
    pyg_graphs = []
    mappings = []
    all_node_feats = []
    all_edge_feats = []

    for gm, vol, seg, spacing, cid in zip(minors, volumes, segs, spacings, case_ids):
        nf = extract_node_features(gm["labels"], vol, voxel_spacing=spacing)
        adj_for_gnn = gm.get("full_adjacency", gm["adjacency"])
        ef = extract_edge_features(gm["labels"], vol, adj_for_gnn, nf)
        data, mapping = build_pyg_graph(nf, ef, gm["labels"], gt_seg=seg)
        pyg_graphs.append(data)
        mappings.append(mapping)
        all_node_feats.append(nf)
        all_edge_feats.append(ef)

        n_tu = int((data.y == 1).sum()) if hasattr(data, 'y') else 0
        n_bg = int((data.y == 0).sum()) if hasattr(data, 'y') else 0
        pr(f"  {cid}: {data.num_nodes:,} nodes ({n_tu} tumor, {n_bg:,} bg), "
           f"{data.num_edges:,} edges")

    # ==== STAGE 4: Train GINE ====
    section("STAGE 4: GINE training")
    import torch
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    pr(f"  Device: {device}")
    if torch.cuda.is_available():
        pr(f"  GPU: {torch.cuda.get_device_name(0)}")

    train_graphs = pyg_graphs[:n_train_total]
    val_graphs = pyg_graphs[n_train_total:n_train_total + n_val_total]

    model, history = train_gine(
        train_graphs, val_graphs,
        epochs=200, lr=1e-3, patience=15, device=device,
    )

    with open(os.path.join(RESULTS_DIR, "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    final_dice = history["val_dice"][-1] if history["val_dice"] else 0
    pr(f"\n  Final val Dice: {final_dice:.4f}")
    pr(f"  Epochs trained: {len(history['train_loss'])}")

    # ==== STAGE 5: Predict + lift -> voxel Dice with threshold sweep ====
    section("STAGE 5: Prediction + voxel lifting -> Dice scores")
    model.eval()
    model = model.to(device)

    # Luke's suggestion: sweep thresholds instead of just argmax (p>0.5)
    thresholds = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]

    # Collect softmax probabilities for all cases first
    all_probs = []
    for data in pyg_graphs:
        with torch.no_grad():
            data_dev = data.clone().to(device)
            logits = model(data_dev)
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()  # P(tumor)
        all_probs.append(probs)

    # Sweep thresholds and find best
    pr(f"  Threshold sweep (val set):")
    pr(f"  {'Thresh':>8s} {'Val Dice':>10s} {'Val Recall':>12s} {'Val Prec':>10s}")
    pr(f"  {'-'*42}")
    best_thresh, best_val_dice = 0.5, -1.0

    for thresh in thresholds:
        val_dice_sum, val_count = 0.0, 0
        val_tp, val_fp, val_fn = 0, 0, 0
        for i in range(n_train_total, n_train_total + n_val_total):
            gm, seg, nf, probs = minors[i], segs[i], all_node_feats[i], all_probs[i]
            sids = sorted(nf.keys())
            max_label = int(gm["labels"].max())
            tumor_lut = np.zeros(max_label + 1, dtype=np.int32)
            for idx, sid in enumerate(sids):
                if probs[idx] >= thresh:
                    tumor_lut[sid] = 2
            pred_mask = tumor_lut[gm["labels"]]
            gt_tumor = (seg == 2)
            pred_tumor = (pred_mask == 2)
            inter = int((gt_tumor & pred_tumor).sum())
            val_tp += inter
            val_fp += int(pred_tumor.sum()) - inter
            val_fn += int(gt_tumor.sum()) - inter

        dice = float(2 * val_tp / (2 * val_tp + val_fp + val_fn + 1e-8))
        recall = float(val_tp / (val_tp + val_fn + 1e-8))
        precision = float(val_tp / (val_tp + val_fp + 1e-8))
        pr(f"  {thresh:>8.2f} {dice:>10.4f} {recall:>12.4f} {precision:>10.4f}")

        if dice > best_val_dice:
            best_val_dice = dice
            best_thresh = thresh

    pr(f"\n  Best threshold: {best_thresh} (val Dice={best_val_dice:.4f})")

    # Apply best threshold to all cases
    dice_scores = []
    for i, (gm, vol, seg, cid, nf, probs) in enumerate(
            zip(minors, volumes, segs, case_ids, all_node_feats, all_probs)):
        sids = sorted(nf.keys())
        max_label = int(gm["labels"].max())
        tumor_lut = np.zeros(max_label + 1, dtype=np.int32)
        for idx, sid in enumerate(sids):
            if probs[idx] >= best_thresh:
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
        precision = float((gt_tumor & pred_tumor).sum() / (pred_tumor.sum() + 1e-8)) if pred_tumor.sum() > 0 else 0

        split = "train" if i < n_train_total else ("val" if i < n_train_total + n_val_total else "test")
        dice_scores.append({
            "case_id": cid, "dice": round(dice, 4),
            "recall": round(recall, 4), "precision": round(precision, 4),
            "gt_voxels": int(gt_tumor.sum()), "pred_voxels": int(pred_tumor.sum()),
            "split": split, "threshold": best_thresh,
        })
        pr(f"  {cid} [{split}]: Dice={dice:.4f}  Recall={recall:.4f}  "
           f"Precision={precision:.4f}  (GT={gt_tumor.sum():,} -> Pred={pred_tumor.sum():,})")

    with open(os.path.join(RESULTS_DIR, "dice_scores.json"), "w") as f:
        json.dump(dice_scores, f, indent=2)

    # ==== STAGE 6a: GT-based phenotype extraction (for CSV validation only) ====
    # Uses GT segmentation to identify tumor supernodes. This validates that
    # SEMIR's graph minor produces correct phenotypes — independent of GINE accuracy.
    section("STAGE 6a: GT-labeled SEMIR phenotypes (validation only)")
    gt_phenotypes = []
    for gm, vol, seg, spacing, cid, nf, ef in zip(
            minors, volumes, segs, spacings, case_ids, all_node_feats, all_edge_feats):
        sids = sorted(nf.keys())
        flat_labels = gm["labels"].ravel()
        flat_gt = (seg.ravel() == 2).astype(np.float64)
        max_label = int(gm["labels"].max())
        if max_label > 0:
            tc = np.bincount(flat_labels, weights=flat_gt, minlength=max_label + 1)
            tot = np.bincount(flat_labels, minlength=max_label + 1)
            safe = np.where(tot > 0, tot, 1)
            ov = tc / safe
            gt_tumor_sids = {int(s) for s in sids if s <= max_label and ov[s] > 0.1}
        else:
            gt_tumor_sids = set()

        vkg = build_vkg_from_semir(
            case_id=cid, dataset="MSD_Task07_Pancreas",
            labels_vol=gm["labels"], volume=vol,
            node_features=nf, edge_features=ef,
            gt_seg=seg, predictions=gt_tumor_sids,
            voxel_spacing=spacing,
        )
        gt_phenotypes.extend(vkg["phenotypes"])
        pr(f"  {cid}: {len(gt_tumor_sids)} tumor supernodes -> {len(vkg['phenotypes'])} phenotypes")

    with open(os.path.join(RESULTS_DIR, "gt_phenotypes.json"), "w") as f:
        json.dump(gt_phenotypes, f, indent=2, default=str)
    pr(f"\n  Total GT-labeled phenotypes: {len(gt_phenotypes)}")

    # ==== STAGE 6b: Build VKG from GINE predictions ====
    section(f"STAGE 6b: Build VKG from GINE predictions (threshold={best_thresh})")
    all_vkg_nodes, all_vkg_edges, all_phenotypes = [], [], []

    for i, (gm, vol, seg, spacing, cid, nf, ef, probs) in enumerate(
            zip(minors, volumes, segs, spacings, case_ids,
                all_node_feats, all_edge_feats, all_probs)):

        sids = sorted(nf.keys())
        predicted_tumor_sids = {sids[idx] for idx in range(len(sids)) if probs[idx] >= best_thresh}

        vkg = build_vkg_from_semir(
            case_id=cid, dataset="MSD_Task07_Pancreas",
            labels_vol=gm["labels"], volume=vol,
            node_features=nf, edge_features=ef,
            gt_seg=seg,
            predictions=predicted_tumor_sids,
            voxel_spacing=spacing,
        )
        all_vkg_nodes.extend(vkg["nodes"])
        all_vkg_edges.extend(vkg["edges"])
        all_phenotypes.extend(vkg["phenotypes"])
        pr(f"  {cid}: {len(vkg['nodes'])} nodes, {len(vkg['edges'])} edges, "
           f"{len(vkg['phenotypes'])} tumor phenotypes")

    # Deduplicate
    seen = set()
    deduped = [n for n in all_vkg_nodes if not (n["id"] in seen or seen.add(n["id"]))]

    vkg_data = {
        "nodes": deduped, "edges": all_vkg_edges, "phenotypes": all_phenotypes,
        "stats": {
            "n_nodes": len(deduped), "n_edges": len(all_vkg_edges),
            "n_tumors": len(all_phenotypes), "source": "semir_graph_minor",
            "schema_version": "2.0",
        },
    }
    with open(os.path.join(RESULTS_DIR, "vkg_semir.json"), "w") as f:
        json.dump(vkg_data, f, indent=2, default=str)

    pr(f"\n  SEMIR VKG totals: {len(deduped)} nodes, {len(all_vkg_edges)} edges, "
       f"{len(all_phenotypes)} tumors")

    # ==== STAGE 7: Compare GT-labeled SEMIR phenotypes vs GT masks ====
    # Uses gt_phenotypes (from Stage 6a), NOT GINE predictions.
    # This validates SEMIR's feature extraction quality, independent of classifier.
    section("STAGE 7: Compare SEMIR phenotypes (GT-labeled) vs GT-mask-derived phenotypes")
    _compare_with_gt_masks(gt_phenotypes, segs, spacings, case_ids)

    # ==== SUMMARY ====
    section("SUMMARY")
    mean_dice = np.mean([d["dice"] for d in dice_scores])
    val_dice_scores = [d["dice"] for d in dice_scores if d["split"] == "val"]
    test_dice_scores = [d["dice"] for d in dice_scores if d["split"] == "test"]
    val_dice = np.mean(val_dice_scores) if val_dice_scores else 0
    test_dice = np.mean(test_dice_scores) if test_dice_scores else 0
    mean_compress = np.mean([m["stats"]["compression_ratio"] for m in minors])

    pr(f"""
  SEMIR Pipeline Results (Pancreas, {len(ordered_cases)} volumes):
  +-----------------------------------------------------------+
  | Oracle Dice (quality ceiling):   {mean_oracle:.4f}                  |
  | Mean deletion loss:              {mean_del_loss:.1f}%                    |
  | Best GNN threshold:              {best_thresh}                      |
  | Mean voxel-level Dice:           {mean_dice:.4f}                  |
  | Validation Dice:                 {val_dice:.4f}                  |
  | Test Dice:                       {test_dice:.4f}                  |
  | Mean compression ratio:          {mean_compress:,.0f}x                     |
  | VKG nodes generated:             {len(deduped):,}                       |
  | VKG edges generated:             {len(all_vkg_edges):,}                       |
  | Tumor phenotypes extracted:      {len(all_phenotypes)}                         |
  +-----------------------------------------------------------+

  Diagnostics:
    - Oracle Dice tells us the quality ceiling of Stage 1 (graph minor).
      If oracle << 1.0, we're losing tumor extent in contraction/deletion.
    - Deletion loss tells us how many tumor voxels get label=0 (removed).
    - Threshold sweep found optimal p={best_thresh} (vs default 0.5).
    - Compactness uses 3D isoperimetric ratio: 36*pi*V^2/A^3.
    - Elongation PCA uses physical mm coordinates (spacing-corrected).
""")

    all_results = {
        "best_params": best_params,
        "dice_scores": dice_scores,
        "vkg_stats": vkg_data["stats"],
        "mean_dice": mean_dice,
        "val_dice": val_dice,
        "test_dice": test_dice,
        "mean_compression": mean_compress,
    }
    with open(os.path.join(RESULTS_DIR, "full_results.json"), "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    pr(f"  All results saved to: {RESULTS_DIR}/")


def _oracle_reconstruction(gm, volume, seg, case_id, params):
    """
    Luke's sanity check: select ALL supernodes with nonzero GT overlap as
    foreground, reconstruct the voxel mask, and compute DSC.

    This estimates the quality ceiling imposed by the graph minor
    representation BEFORE evaluating the GNN.

    Also tracks how many tumor voxels are lost during node deletion.
    """
    from semir.graph_minor import _contraction_adaptive, _node_deletion

    labels = gm["labels"]
    flat_labels = labels.ravel()
    gt_tumor = (seg == 2)
    gt_flat = gt_tumor.ravel().astype(np.float64)
    gt_tumor_voxels = int(gt_tumor.sum())

    if gt_tumor_voxels == 0:
        return {
            "case_id": case_id, "oracle_dice": 1.0,
            "n_tumor_supernodes": 0, "gt_tumor_voxels": 0,
            "tumor_voxels_deleted": 0, "deletion_loss_pct": 0.0,
        }

    max_label = int(flat_labels.max())
    if max_label == 0:
        return {
            "case_id": case_id, "oracle_dice": 0.0,
            "n_tumor_supernodes": 0, "gt_tumor_voxels": gt_tumor_voxels,
            "tumor_voxels_deleted": gt_tumor_voxels, "deletion_loss_pct": 100.0,
        }

    # Count GT tumor voxels per supernode
    tumor_counts = np.bincount(flat_labels, weights=gt_flat, minlength=max_label + 1)
    total_counts = np.bincount(flat_labels, minlength=max_label + 1)

    # Oracle: select any supernode with >0 GT tumor voxels
    oracle_sids = set(np.where(tumor_counts > 0)[0])
    oracle_sids.discard(0)  # exclude deleted voxels (label 0)

    # Reconstruct oracle mask via LUT
    oracle_lut = np.zeros(max_label + 1, dtype=np.int32)
    for sid in oracle_sids:
        oracle_lut[sid] = 1
    oracle_mask = oracle_lut[flat_labels].reshape(labels.shape).astype(bool)

    # Compute oracle Dice
    inter = int((gt_tumor & oracle_mask).sum())
    oracle_dice = float(2 * inter / (gt_tumor.sum() + oracle_mask.sum() + 1e-8))

    # Deletion analysis: how many GT tumor voxels have label=0 (deleted)?
    tumor_voxels_in_deleted = int(tumor_counts[0])
    deletion_loss_pct = tumor_voxels_in_deleted / max(gt_tumor_voxels, 1) * 100.0

    return {
        "case_id": case_id,
        "oracle_dice": round(oracle_dice, 4),
        "n_tumor_supernodes": len(oracle_sids),
        "gt_tumor_voxels": gt_tumor_voxels,
        "oracle_pred_voxels": int(oracle_mask.sum()),
        "tumor_voxels_deleted": tumor_voxels_in_deleted,
        "deletion_loss_pct": round(deletion_loss_pct, 1),
    }


def _compute_gt_phenotypes(seg, spacing):
    """Compute ground truth phenotypes directly from the segmentation mask."""
    from scipy.ndimage import distance_transform_edt, center_of_mass

    voxel_vol_mm3 = spacing[0] * spacing[1] * spacing[2]

    tumor_mask = seg == 2
    pancreas_mask = seg == 1
    tumor_voxels = int(tumor_mask.sum())
    pancreas_voxels = int(pancreas_mask.sum())

    if tumor_voxels == 0:
        return None

    tumor_vol_cm3 = tumor_voxels * voxel_vol_mm3 / 1000.0
    pancreas_vol_cm3 = pancreas_voxels * voxel_vol_mm3 / 1000.0
    coverage_pct = tumor_voxels / max(pancreas_voxels, 1) * 100.0
    diameter_mm = 2.0 * (3.0 * tumor_vol_cm3 * 1000.0 / (4.0 * np.pi)) ** (1.0 / 3.0)

    # Centroid in mm
    tumor_centroid = center_of_mass(tumor_mask)
    tumor_centroid_mm = tuple(c * s for c, s in zip(tumor_centroid, spacing))
    pancreas_centroid = center_of_mass(pancreas_mask) if pancreas_voxels > 0 else (0, 0, 0)
    pancreas_centroid_mm = tuple(c * s for c, s in zip(pancreas_centroid, spacing))

    # Distance to organ boundary
    if pancreas_mask.any():
        dist_map = distance_transform_edt(~pancreas_mask, sampling=spacing)
        min_dist = float(dist_map[tumor_mask].min())
    else:
        min_dist = 0.0

    return {
        "tumor_vol_cm3": round(tumor_vol_cm3, 4),
        "pancreas_vol_cm3": round(pancreas_vol_cm3, 4),
        "coverage_pct": round(coverage_pct, 4),
        "diameter_mm": round(diameter_mm, 2),
        "distance_to_organ_mm": round(min_dist, 2),
        "tumor_centroid_mm": tumor_centroid_mm,
        "pancreas_centroid_mm": pancreas_centroid_mm,
        "tumor_voxels": tumor_voxels,
    }


def _compare_with_gt_masks(semir_phenotypes, segs, spacings, case_ids):
    """Compare SEMIR-derived phenotypes against GT-mask-derived phenotypes."""
    from collections import Counter

    pr(f"\n  Computing ground truth phenotypes from segmentation masks...")

    gt_phenos = {}
    for seg, spacing, cid in zip(segs, spacings, case_ids):
        gt = _compute_gt_phenotypes(seg, spacing)
        if gt is not None:
            gt_phenos[cid] = gt

    pr(f"  GT phenotypes computed for {len(gt_phenos)} cases")

    # Group SEMIR phenotypes by case (sum volumes for multi-supernode tumors)
    semir_by_case = {}
    for p in semir_phenotypes:
        cid = p["case_id"]
        if cid not in semir_by_case:
            semir_by_case[cid] = []
        semir_by_case[cid].append(p)

    # Compare
    vol_errors, cov_errors, dist_errors, diam_errors = [], [], [], []
    matched = 0

    pr(f"\n  {'Case':<20s} {'GT Vol(cm3)':>12s} {'SEMIR Vol':>12s} "
       f"{'GT Diam(mm)':>12s} {'SEMIR Diam':>12s} {'GT Cov%':>10s} {'SEMIR Cov%':>10s}")
    pr(f"  {'-'*88}")

    for cid in sorted(gt_phenos.keys()):
        gt = gt_phenos[cid]
        if cid not in semir_by_case:
            continue
        matched += 1

        # Aggregate SEMIR: sum volumes across tumor supernodes
        semir_list = semir_by_case[cid]
        semir_vol = sum(p["volume_cm3"] for p in semir_list)
        semir_diam = 2.0 * (3.0 * semir_vol * 1000.0 / (4.0 * np.pi)) ** (1.0 / 3.0)
        semir_cov = sum(p["coverage_pct"] for p in semir_list)
        semir_dist = min(p["distance_to_organ_mm"] for p in semir_list)

        pr(f"  {cid:<20s} {gt['tumor_vol_cm3']:>12.2f} {semir_vol:>12.2f} "
           f"{gt['diameter_mm']:>12.2f} {semir_diam:>12.2f} "
           f"{gt['coverage_pct']:>10.2f} {semir_cov:>10.2f}")

        if gt["tumor_vol_cm3"] > 0:
            vol_errors.append(abs(semir_vol - gt["tumor_vol_cm3"]) / gt["tumor_vol_cm3"])
        if gt["diameter_mm"] > 0:
            diam_errors.append(abs(semir_diam - gt["diameter_mm"]) / gt["diameter_mm"])
        if gt["coverage_pct"] > 0:
            cov_errors.append(abs(semir_cov - gt["coverage_pct"]) / gt["coverage_pct"])
        if gt["distance_to_organ_mm"] > 0:
            dist_errors.append(abs(semir_dist - gt["distance_to_organ_mm"]) / gt["distance_to_organ_mm"])

    pr(f"\n  Matched {matched} cases")

    if matched > 0:
        pr(f"\n  Mean relative errors (SEMIR vs GT masks):")
        pr(f"  {'Feature':<25s} {'Mean Rel Error':>15s} {'Matched':>10s}")
        pr(f"  {'-'*50}")
        if vol_errors:
            pr(f"  {'Volume (cm3)':<25s} {np.mean(vol_errors):>15.2%} {len(vol_errors):>10d}")
        if diam_errors:
            pr(f"  {'Diameter (mm)':<25s} {np.mean(diam_errors):>15.2%} {len(diam_errors):>10d}")
        if cov_errors:
            pr(f"  {'Coverage (%)':<25s} {np.mean(cov_errors):>15.2%} {len(cov_errors):>10d}")
        if dist_errors:
            pr(f"  {'Distance (mm)':<25s} {np.mean(dist_errors):>15.2%} {len(dist_errors):>10d}")

    # SEMIR-exclusive features
    if semir_phenotypes:
        pr(f"\n  SEMIR-exclusive features (no GT equivalent):")
        sem_comp = np.mean([p["compactness"] for p in semir_phenotypes])
        sem_elong = np.mean([p["elongation"] for p in semir_phenotypes])
        pr(f"    Mean compactness: {sem_comp:.4f}")
        pr(f"    Mean elongation:  {sem_elong:.4f}")

        morphs = Counter(p["morphology"] for p in semir_phenotypes)
        stages = Counter(p["t_stage"] for p in semir_phenotypes)
        subs = Counter(p["subregion"] for p in semir_phenotypes)
        pr(f"    Morphology dist:  {dict(morphs)}")
        pr(f"    T-stage dist:     {dict(stages)}")
        pr(f"    Subregion dist:   {dict(subs)}")


if __name__ == "__main__":
    main()
