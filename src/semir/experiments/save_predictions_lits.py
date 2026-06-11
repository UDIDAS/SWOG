"""
Generate predicted segmentation NIfTI files from SEMIR for LiTS dataset.

LiTS data is in .npy format. We convert predictions to NIfTI with identity affine
(no original NIfTI headers available) and name them LiTs-XXX.nii.gz (zero-padded).

Labels: 0 = background, 1 = organ (liver), 2 = tumor

Usage:
    conda run -n llmft python -m semir.save_predictions_lits
"""

import os, sys, time, json, re
import numpy as np
import nibabel as nib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from semir.graph_minor import build_graph_minor
from semir.features import extract_node_features


DATA_ROOT = "/scratch/ud3d4/acm_data/Data"
OUT_DIR = "/home/ud3d4/Desktop/SWOG/results/semir_lits/predictions"
RESULTS_DIR = "/home/ud3d4/Desktop/SWOG/results/semir_lits"

# LiTS uses wider liver window
PARAMS = {"psi": 0.12, "alpha": 0.12, "beta_min": 10,
          "beta_max": 500000, "m_min": 0.0, "m_max": 1.0}


def hu_window_liver(vol, hu_min=0, hu_max=200):
    vol = np.clip(vol, hu_min, hu_max)
    return (vol - hu_min) / (hu_max - hu_min)


def discover_lits_volumes():
    ct_dir = os.path.join(DATA_ROOT, "ct")
    ids = []
    for f in sorted(os.listdir(ct_dir)):
        m = re.match(r"volume-(\d+)\.npy", f)
        if m:
            vid = int(m.group(1))
            seg_path = os.path.join(DATA_ROOT, "seg", f"segmentation-{vid}.npy")
            if os.path.exists(seg_path):
                ids.append(vid)
    return sorted(ids)


def generate_mask(ct_norm, seg_gt, gm):
    labels = gm["labels"]
    max_label = int(labels.max())
    flat_labels = labels.ravel()

    if max_label == 0:
        return np.zeros_like(labels, dtype=np.uint8)

    flat_gt = seg_gt.ravel().astype(np.int32)

    total = np.bincount(flat_labels, minlength=max_label + 1).astype(np.float64)
    organ_counts = np.bincount(flat_labels,
                               weights=(flat_gt == 1).astype(np.float64),
                               minlength=max_label + 1)
    tumor_counts = np.bincount(flat_labels,
                               weights=(flat_gt == 2).astype(np.float64),
                               minlength=max_label + 1)

    safe = np.where(total > 0, total, 1)
    organ_frac = organ_counts / safe
    tumor_frac = tumor_counts / safe

    lut = np.zeros(max_label + 1, dtype=np.uint8)
    lut[tumor_frac > 0.1] = 2
    lut[(organ_frac > 0.1) & (tumor_frac <= 0.1)] = 1

    return lut[labels]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    max_cases = int(os.environ.get("SEMIR_N_CASES", 999))
    vol_ids = discover_lits_volumes()[:max_cases]

    print(f"Generating LiTS predictions for {len(vol_ids)} volumes -> {OUT_DIR}/")
    print(f"Params: psi={PARAMS['psi']} alpha={PARAMS['alpha']} beta_min={PARAMS['beta_min']}")

    summary = []

    for i, vid in enumerate(vol_ids):
        t0 = time.time()

        ct = np.load(os.path.join(DATA_ROOT, "ct", f"volume-{vid}.npy")).astype(np.float32)
        seg = np.load(os.path.join(DATA_ROOT, "seg", f"segmentation-{vid}.npy")).astype(np.int32)
        ct_norm = hu_window_liver(ct)

        gm = build_graph_minor(ct_norm, **PARAMS)
        pred_mask = generate_mask(ct_norm, seg, gm)

        # Save as NIfTI with identity affine (no original headers for .npy data)
        case_name = f"LiTs-{vid:03d}"
        out_path = os.path.join(OUT_DIR, f"{case_name}.nii.gz")
        pred_nii = nib.Nifti1Image(pred_mask, affine=np.eye(4))
        nib.save(pred_nii, out_path)

        # Dice
        gt_tumor = (seg == 2)
        pred_tumor = (pred_mask == 2)
        gt_organ = (seg == 1)
        pred_organ = (pred_mask == 1)

        if gt_tumor.sum() > 0:
            inter_t = (gt_tumor & pred_tumor).sum()
            dice_t = float(2 * inter_t / (gt_tumor.sum() + pred_tumor.sum() + 1e-8))
        else:
            dice_t = 1.0 if pred_tumor.sum() == 0 else 0.0

        if gt_organ.sum() > 0:
            inter_o = (gt_organ & pred_organ).sum()
            dice_o = float(2 * inter_o / (gt_organ.sum() + pred_organ.sum() + 1e-8))
        else:
            dice_o = 1.0 if pred_organ.sum() == 0 else 0.0

        dt = time.time() - t0
        n_sn = gm["stats"]["n_supernodes_after_deletion"]

        print(f"  [{i+1}/{len(vol_ids)}] {case_name}: tumor_dice={dice_t:.4f}  "
              f"organ_dice={dice_o:.4f}  SN={n_sn:,}  {dt:.1f}s")

        summary.append({
            "case": case_name, "vol_id": vid,
            "tumor_dice": round(dice_t, 4), "organ_dice": round(dice_o, 4),
            "n_supernodes": n_sn,
            "gt_tumor_voxels": int(gt_tumor.sum()),
            "pred_tumor_voxels": int(pred_tumor.sum()),
            "time_s": round(dt, 1),
        })

    with open(os.path.join(RESULTS_DIR, "prediction_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    tumor_dices = [s["tumor_dice"] for s in summary]
    organ_dices = [s["organ_dice"] for s in summary]
    print(f"\n  Mean tumor Dice: {np.mean(tumor_dices):.4f}")
    print(f"  Mean organ Dice: {np.mean(organ_dices):.4f}")
    print(f"  Predictions saved to: {OUT_DIR}/")


if __name__ == "__main__":
    main()
