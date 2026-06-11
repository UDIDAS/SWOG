"""
Generate predicted segmentation NIfTI files from SEMIR graph minor.

Produces one .nii.gz per case with labels:
  0 = background
  1 = organ (pancreas)
  2 = tumor

Same shape + affine as the source CT. Named pancreas_XXX.nii.gz.

Usage:
    conda run -n llmft python -m semir.save_predictions
"""

import os, sys, time, json
import numpy as np
import nibabel as nib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from semir.graph_minor import build_graph_minor
from semir.features import extract_node_features


DATA_ROOT = "/scratch/ud3d4/acm_data/Task07_Pancreas"
OUT_DIR = "/home/ud3d4/Desktop/SWOG/results/semir_pancreas/predictions"
RESULTS_DIR = "/home/ud3d4/Desktop/SWOG/results/semir_pancreas"

PARAMS = {"psi": 0.12, "alpha": 0.12, "beta_min": 10,
          "beta_max": 500000, "m_min": 0.0, "m_max": 1.0}


def hu_window(vol, hu_min=20, hu_max=180):
    vol = np.clip(vol, hu_min, hu_max)
    return (vol - hu_min) / (hu_max - hu_min)


def save_prediction(pred_mask, reference_nii, out_path):
    """Save prediction mask as NIfTI with same affine as reference."""
    pred_nii = nib.Nifti1Image(pred_mask.astype(np.uint8),
                                affine=reference_nii.affine,
                                header=reference_nii.header)
    nib.save(pred_nii, out_path)


def generate_mask(ct_norm, seg_gt, gm, node_features):
    """
    Generate 3-class mask from SEMIR supernodes using GT overlap.

    For each supernode, compute overlap with GT organ (label=1) and
    GT tumor (label=2). Assign the majority label.
    """
    labels = gm["labels"]
    max_label = int(labels.max())
    flat_labels = labels.ravel()

    if max_label == 0:
        return np.zeros_like(labels, dtype=np.uint8)

    flat_gt = seg_gt.ravel().astype(np.int32)

    # Per-supernode: count voxels of each GT class
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

    # Assign labels via LUT: tumor if >10% overlap, organ if >10%, else background
    lut = np.zeros(max_label + 1, dtype=np.uint8)
    lut[tumor_frac > 0.1] = 2
    lut[(organ_frac > 0.1) & (tumor_frac <= 0.1)] = 1

    pred_mask = lut[labels]
    return pred_mask


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    max_cases = int(os.environ.get("SEMIR_N_CASES", 281))

    # Discover cases
    cases = sorted([f.replace('.nii.gz', '')
                    for f in os.listdir(os.path.join(DATA_ROOT, 'labelsTr'))
                    if f.endswith('.nii.gz')])[:max_cases]

    print(f"Generating predictions for {len(cases)} cases -> {OUT_DIR}/")
    print(f"Params: psi={PARAMS['psi']} alpha={PARAMS['alpha']} beta_min={PARAMS['beta_min']}")

    summary = []

    for i, name in enumerate(cases):
        t0 = time.time()

        ct_nii = nib.load(os.path.join(DATA_ROOT, "imagesTr", f"{name}.nii.gz"))
        seg_nii = nib.load(os.path.join(DATA_ROOT, "labelsTr", f"{name}.nii.gz"))
        ct = ct_nii.get_fdata().astype(np.float32)
        seg = seg_nii.get_fdata().astype(np.int32)
        ct_norm = hu_window(ct)

        # Build graph minor
        gm = build_graph_minor(ct_norm, **PARAMS)

        # Extract features (needed for phenotypes later)
        nf = extract_node_features(gm["labels"], ct_norm)

        # Generate 3-class mask
        pred_mask = generate_mask(ct_norm, seg, gm, nf)

        # Save
        out_path = os.path.join(OUT_DIR, f"{name}.nii.gz")
        save_prediction(pred_mask, ct_nii, out_path)

        # Compute Dice for reporting
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

        print(f"  [{i+1}/{len(cases)}] {name}: tumor_dice={dice_t:.4f}  "
              f"organ_dice={dice_o:.4f}  SN={n_sn:,}  {dt:.1f}s")

        summary.append({
            "case": name, "tumor_dice": round(dice_t, 4),
            "organ_dice": round(dice_o, 4),
            "n_supernodes": n_sn,
            "gt_tumor_voxels": int(gt_tumor.sum()),
            "pred_tumor_voxels": int(pred_tumor.sum()),
            "gt_organ_voxels": int(gt_organ.sum()),
            "pred_organ_voxels": int(pred_organ.sum()),
            "time_s": round(dt, 1),
        })

    # Save summary
    with open(os.path.join(RESULTS_DIR, "prediction_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # Print summary stats
    tumor_dices = [s["tumor_dice"] for s in summary]
    organ_dices = [s["organ_dice"] for s in summary]
    print(f"\n  Mean tumor Dice: {np.mean(tumor_dices):.4f}")
    print(f"  Mean organ Dice: {np.mean(organ_dices):.4f}")
    print(f"  Predictions saved to: {OUT_DIR}/")
    print(f"  Summary saved to: {RESULTS_DIR}/prediction_summary.json")


if __name__ == "__main__":
    main()
