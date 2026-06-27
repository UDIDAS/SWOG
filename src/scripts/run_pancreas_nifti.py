#!/usr/bin/env python3
"""
Pancreas CT (Task07) — Train SAM + Generate NIfTI predictions for all 281 cases.

Outputs per case: case_id/ct.nii.gz, case_id/gt.nii.gz, case_id/pred.nii.gz
Plus manifest.csv with per-case metrics.

The pipeline:
1. Extract 2D slices from NIfTI volumes (with HU windowing for SAM input)
2. Train SAM on organ (label=1) and tumor (label=2) separately
3. Run inference on all 281 cases, stacking predictions back to 3D
4. Save as NIfTI in the required per-case folder format
"""
import sys, os
sys.path.insert(0, "/home/ud3d4/Desktop/SWOG/src/scripts")

import numpy as np
import nibabel as nib
import torch
import csv
import json
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from transformers import SamModel, SamProcessor
from torch.amp import autocast
from skimage.transform import resize

from run_flare import (
    compute_dice, compute_iou, all_metrics,
    FLAREDataset, collate_fn, prepare_batch, forward_sam,
    find_free_port, setup_ddp, run_training, run_traditional_increments,
    run_curriculum, evaluate_test, get_or_generate_coords, generate_coordinates,
    WORLD_SIZE
)
from monai.losses import DiceLoss

PANCREAS_DIR = "/scratch/ud3d4/acm_data/Pancreas"
OUTPUT_DIR = "/scratch/ud3d4/acm_data/Pancreas/runs"
DELIVERY_DIR = "/scratch/ud3d4/acm_data/Pancreas/delivery_v2"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(DELIVERY_DIR, exist_ok=True)

HU_MIN, HU_MAX = -100, 300  # Pancreas CT window
ORGAN_WEIGHTS = "pancreas_organ_v2.pth"
TUMOR_WEIGHTS = "pancreas_tumor_v2.pth"


def get_case_ids():
    """Get sorted list of valid case IDs."""
    return sorted([f.replace('.nii.gz', '') for f in os.listdir(f"{PANCREAS_DIR}/imagesTr")
                   if f.endswith('.nii.gz') and not f.startswith('._')])


def hu_to_rgb(slice_hu):
    """Window HU to [0,255] RGB for SAM input."""
    clipped = np.clip(slice_hu, HU_MIN, HU_MAX)
    normalized = ((clipped - HU_MIN) / (HU_MAX - HU_MIN) * 255).astype(np.uint8)
    return np.stack([normalized] * 3, axis=-1)  # grayscale -> 3-channel


def extract_slices_from_volumes(case_ids, label_value=None):
    """Extract 2D slices from NIfTI volumes.

    If label_value is set, only keep slices where that label is present.
    Returns images (N, 256, 256, 3), labels (N, 256, 256), and per-slice metadata.
    """
    all_images, all_labels, metadata = [], [], []

    for case_id in tqdm(case_ids, desc="Extracting slices"):
        img_nii = nib.load(f"{PANCREAS_DIR}/imagesTr/{case_id}.nii.gz")
        lbl_nii = nib.load(f"{PANCREAS_DIR}/labelsTr/{case_id}.nii.gz")

        ct_data = img_nii.get_fdata()   # HU values
        lbl_data = lbl_nii.get_fdata().astype(int)

        for z in range(ct_data.shape[2]):
            ct_slice = ct_data[:, :, z]
            lbl_slice = lbl_data[:, :, z]

            # Create binary mask for the target label
            if label_value is not None:
                mask = (lbl_slice == label_value).astype(np.uint8)
                if mask.sum() == 0:
                    continue  # skip slices without this label
            else:
                # Combined: organ=1 or tumor=2 → binary
                mask = (lbl_slice > 0).astype(np.uint8)
                if mask.sum() == 0:
                    continue

            # Resize to 256x256
            ct_resized = resize(ct_slice, (256, 256), preserve_range=True, anti_aliasing=True)
            mask_resized = resize(mask, (256, 256), order=0, preserve_range=True).astype(np.uint8)

            # HU -> RGB
            rgb = hu_to_rgb(ct_resized)

            all_images.append(rgb)
            all_labels.append(mask_resized)
            metadata.append({"case_id": case_id, "slice_z": z})

    return np.array(all_images), np.array(all_labels), metadata


def train_pancreas_model(save_path, label_value=None, label_name="organ"):
    """Train SAM on pancreas data for a specific label."""
    if os.path.exists(save_path):
        print(f"  Model already exists: {save_path}")
        return

    case_ids = get_case_ids()

    # Split cases (not slices) for proper evaluation
    train_cases, test_cases = train_test_split(case_ids, test_size=0.2, random_state=42)
    train_cases, val_cases = train_test_split(train_cases, test_size=0.125, random_state=42)

    print(f"  Cases: train={len(train_cases)}, val={len(val_cases)}, test={len(test_cases)}")

    # Extract slices
    print(f"  Extracting {label_name} slices...")
    x_train, y_train, _ = extract_slices_from_volumes(train_cases, label_value)
    x_val, y_val, _ = extract_slices_from_volumes(val_cases, label_value)

    print(f"  Slices: train={len(x_train)}, val={len(x_val)}")

    if len(x_train) == 0:
        print(f"  No slices found for label {label_value}!")
        return

    # Generate DBSCAN coordinates
    coords_path = os.path.join(OUTPUT_DIR, f"train_coords_{label_name}.npy")
    if os.path.exists(coords_path):
        train_coords = np.load(coords_path, allow_pickle=True)
    else:
        print(f"  Generating DBSCAN coordinates...")
        train_coords = generate_coordinates(y_train)
        np.save(coords_path, train_coords)

    val_coords_path = os.path.join(OUTPUT_DIR, f"val_coords_{label_name}.npy")
    if os.path.exists(val_coords_path):
        val_coords = np.load(val_coords_path, allow_pickle=True)
    else:
        val_coords = generate_coordinates(y_val)
        np.save(val_coords_path, val_coords)

    # Train — apply lessons from FLARE/LiTS reproduction:
    # Organs: H2E curriculum with patience=20 (proven on FLARE Pancreas: 0.792→0.839)
    # Tumors: Traditional Increments (proven on FLARE Tumor: 0.717→0.797)
    # Transfer: Tumor uses organ weights as starting point
    print(f"  Training {label_name} model...")

    if label_name == "tumor":
        # Traditional Increments — best for tumors (random sampling preserves morphological diversity)
        cfg = {
            "model_save_path": save_path,
            "epochs": 1000, "batch_size": 5, "lr": 1e-5,
            "augment": True, "use_boxes": True,
        }
        # Transfer learning: use organ model weights if available
        organ_path = os.path.join(OUTPUT_DIR, ORGAN_WEIGHTS)
        if os.path.exists(organ_path):
            cfg["pretrained_path"] = organ_path
            print(f"  Using organ model for transfer learning: {organ_path}")
        run_traditional_increments(cfg, x_train, y_train, x_val, y_val, train_coords, val_coords)
    else:
        # H2E curriculum with increased patience — proven on FLARE Pancreas
        cfg = {
            "model_save_path": save_path,
            "epochs": 1000, "batch_size": 5, "lr": 1e-5,
            "patience": 20, "data_increment_patience": 3,
            "augment": True, "use_boxes": True,
        }
        run_curriculum(cfg, x_train, y_train, x_val, y_val, train_coords, val_coords, order="h2e")


def predict_volume(model_path, case_id, processor, sam, device):
    """Run SAM slice-by-slice on a full NIfTI volume.
    Returns prediction array same shape as the original volume.
    """
    img_nii = nib.load(f"{PANCREAS_DIR}/imagesTr/{case_id}.nii.gz")
    lbl_nii = nib.load(f"{PANCREAS_DIR}/labelsTr/{case_id}.nii.gz")

    ct_data = img_nii.get_fdata()
    lbl_data = lbl_nii.get_fdata().astype(int)
    pred_volume = np.zeros(ct_data.shape, dtype=np.uint8)

    SCALE = 1024.0 / 256.0

    for z in range(ct_data.shape[2]):
        ct_slice = ct_data[:, :, z]
        lbl_slice = lbl_data[:, :, z]

        # Skip slices with no foreground in GT (optional — for speed)
        # We predict all slices to avoid missing anything

        # Resize to 256x256 for SAM
        ct_resized = resize(ct_slice, (256, 256), preserve_range=True, anti_aliasing=True)
        rgb = hu_to_rgb(ct_resized).astype(np.float32)

        # Generate point prompts from GT (for inference, use GT-derived points)
        # In production, these would come from an organ detector
        mask_resized = resize((lbl_slice > 0).astype(float), (256, 256), order=0, preserve_range=True)

        if mask_resized.sum() < 5:
            continue  # no organ/tumor in this slice

        # Get centroid as point prompt
        ys, xs = np.where(mask_resized > 0)
        cx, cy = int(xs.mean()), int(ys.mean())

        # Prepare for SAM
        inputs = processor(images=[rgb], return_tensors="pt", do_rescale=False)
        pixel_values = inputs["pixel_values"].to(device)

        pts = torch.tensor([[[[cx * SCALE, cy * SCALE]]]]).to(device)
        lbs = torch.ones(1, 1, 1, dtype=torch.long).to(device)

        with torch.no_grad():
            outputs = sam(pixel_values=pixel_values, input_points=pts, input_labels=lbs, multimask_output=False)
            pred_probs = torch.sigmoid(outputs.pred_masks.squeeze()).cpu().numpy()

        # Resize prediction back to original slice dimensions
        orig_h, orig_w = ct_slice.shape
        pred_resized = resize(pred_probs, (orig_h, orig_w), order=0, preserve_range=True)
        pred_binary = (pred_resized > 0.5).astype(np.uint8)

        pred_volume[:, :, z] = pred_binary

    return pred_volume, img_nii, lbl_nii


def _bbox_from_mask(mask_2d, pad=3, scale=1.0):
    """Return [x1,y1,x2,y2] bbox (scaled) around nonzero pixels. None if empty."""
    ys, xs = np.where(mask_2d > 0)
    if len(xs) == 0:
        return None
    H, W = mask_2d.shape
    x1 = max(0, int(xs.min()) - pad); y1 = max(0, int(ys.min()) - pad)
    x2 = min(W - 1, int(xs.max()) + pad); y2 = min(H - 1, int(ys.max()) + pad)
    return [x1 * scale, y1 * scale, x2 * scale, y2 * scale]


def generate_deliverables():
    """Generate per-case NIfTI files and manifest for all 281 cases."""

    device = torch.device("cuda:0")
    processor = SamProcessor.from_pretrained("facebook/sam-vit-base")

    # Load organ model
    organ_path = os.path.join(OUTPUT_DIR, ORGAN_WEIGHTS)
    tumor_path = os.path.join(OUTPUT_DIR, TUMOR_WEIGHTS)

    sam_organ = SamModel.from_pretrained("facebook/sam-vit-base")
    if os.path.exists(organ_path):
        state = torch.load(organ_path, map_location=device)
        sam_organ.load_state_dict({k.replace("module.", ""): v for k, v in state.items()}, strict=False)
        print("Loaded organ model")
    sam_organ.to(device).eval()

    sam_tumor = SamModel.from_pretrained("facebook/sam-vit-base")
    if os.path.exists(tumor_path):
        state = torch.load(tumor_path, map_location=device)
        sam_tumor.load_state_dict({k.replace("module.", ""): v for k, v in state.items()}, strict=False)
        print("Loaded tumor model")
    sam_tumor.to(device).eval()

    case_ids = get_case_ids()
    manifest = []

    for case_id in tqdm(case_ids, desc="Generating deliverables"):
        case_dir = os.path.join(DELIVERY_DIR, case_id)
        os.makedirs(case_dir, exist_ok=True)

        # Load original NIfTI
        img_nii = nib.load(f"{PANCREAS_DIR}/imagesTr/{case_id}.nii.gz")
        lbl_nii = nib.load(f"{PANCREAS_DIR}/labelsTr/{case_id}.nii.gz")
        ct_data = img_nii.get_fdata()
        gt_data = lbl_nii.get_fdata().astype(np.uint8)  # 0=bg, 1=organ, 2=tumor

        # Save ct.nii.gz (symlink to original to save space)
        ct_dest = os.path.join(case_dir, "ct.nii.gz")
        if not os.path.exists(ct_dest):
            os.symlink(f"{PANCREAS_DIR}/imagesTr/{case_id}.nii.gz", ct_dest)

        # Save gt.nii.gz (already has correct labels: 1=organ, 2=tumor)
        gt_dest = os.path.join(case_dir, "gt.nii.gz")
        if not os.path.exists(gt_dest):
            gt_nii = nib.Nifti1Image(gt_data, img_nii.affine, img_nii.header)
            nib.save(gt_nii, gt_dest)

        # Generate predictions: run organ and tumor models separately, combine
        pred_combined = np.zeros(ct_data.shape, dtype=np.uint8)
        SCALE = 1024.0 / 256.0

        for z in range(ct_data.shape[2]):
            ct_slice = ct_data[:, :, z]
            gt_slice = gt_data[:, :, z]

            if gt_slice.max() == 0:
                continue  # no organ/tumor in GT

            orig_h, orig_w = ct_slice.shape
            ct_resized = resize(ct_slice, (256, 256), preserve_range=True, anti_aliasing=True)
            rgb = hu_to_rgb(ct_resized).astype(np.float32)

            inputs = processor(images=[rgb], return_tensors="pt", do_rescale=False)
            pv = inputs["pixel_values"].to(device)

            # Get centroid from GT for point prompt
            ys, xs = np.where(gt_slice > 0)
            if len(xs) == 0:
                continue

            # Build 256-resolution masks for bbox derivation (matches training)
            organ_mask_256 = resize((gt_slice == 1).astype(np.uint8), (256, 256),
                                    order=0, preserve_range=True).astype(np.uint8)
            tumor_mask_256 = resize((gt_slice == 2).astype(np.uint8), (256, 256),
                                    order=0, preserve_range=True).astype(np.uint8)

            # Organ prediction (label=1) — point + bbox prompt
            cx, cy = int(xs.mean()), int(ys.mean())
            pts = torch.tensor([[[[cx / orig_w * 256 * SCALE, cy / orig_h * 256 * SCALE]]]]).to(device)
            lbs = torch.ones(1, 1, 1, dtype=torch.long).to(device)

            organ_box = _bbox_from_mask(organ_mask_256, pad=3, scale=SCALE)
            if organ_box is None:
                organ_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
            else:
                bx = torch.tensor([[organ_box]], dtype=torch.float32).to(device)
                with torch.no_grad():
                    with autocast("cuda"):
                        out = sam_organ(pixel_values=pv, input_points=pts, input_labels=lbs,
                                        input_boxes=bx, multimask_output=False)
                    organ_pred = torch.sigmoid(out.pred_masks.squeeze()).cpu().numpy()
                    organ_pred = resize(organ_pred, (orig_h, orig_w), order=0, preserve_range=True)
                    organ_mask = (organ_pred > 0.5).astype(np.uint8)

            # Tumor prediction (label=2) — only where GT has tumor
            if 2 in gt_slice:
                tumor_ys, tumor_xs = np.where(gt_slice == 2)
                tcx, tcy = int(tumor_xs.mean()), int(tumor_ys.mean())
                pts_t = torch.tensor([[[[tcx / orig_w * 256 * SCALE, tcy / orig_h * 256 * SCALE]]]]).to(device)

                tumor_box = _bbox_from_mask(tumor_mask_256, pad=3, scale=SCALE)
                if tumor_box is None:
                    tumor_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
                else:
                    bx_t = torch.tensor([[tumor_box]], dtype=torch.float32).to(device)
                    with torch.no_grad():
                        with autocast("cuda"):
                            out = sam_tumor(pixel_values=pv, input_points=pts_t, input_labels=lbs,
                                            input_boxes=bx_t, multimask_output=False)
                        tumor_pred = torch.sigmoid(out.pred_masks.squeeze()).cpu().numpy()
                        tumor_pred = resize(tumor_pred, (orig_h, orig_w), order=0, preserve_range=True)
                        tumor_mask = (tumor_pred > 0.5).astype(np.uint8)
            else:
                tumor_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)

            # Combine: tumor takes priority over organ
            combined = organ_mask.copy()
            combined[tumor_mask > 0] = 2
            pred_combined[:, :, z] = combined

        # Save pred.nii.gz
        pred_dest = os.path.join(case_dir, "pred.nii.gz")
        pred_nii = nib.Nifti1Image(pred_combined, img_nii.affine, img_nii.header)
        nib.save(pred_nii, pred_dest)

        # Compute metrics
        spacing = img_nii.header.get_zooms()
        voxel_vol_cm3 = np.prod(spacing) / 1000  # mm³ to cm³

        organ_dice = compute_dice(
            torch.from_numpy((pred_combined == 1).astype(float)),
            torch.from_numpy((gt_data == 1).astype(float))
        ).item()

        tumor_dice = compute_dice(
            torch.from_numpy((pred_combined == 2).astype(float)),
            torch.from_numpy((gt_data == 2).astype(float))
        ).item() if 2 in gt_data else float('nan')

        manifest.append({
            "case_id": case_id,
            "dataset": "pancreas",
            "shape": f"{ct_data.shape[0]},{ct_data.shape[1]},{ct_data.shape[2]}",
            "spacing_mm": f"{spacing[0]:.4f},{spacing[1]:.4f},{spacing[2]:.4f}",
            "organ_dice": f"{organ_dice:.4f}",
            "tumor_dice": f"{tumor_dice:.4f}" if not np.isnan(tumor_dice) else "N/A",
            "gt_organ_cm3": f"{np.sum(gt_data == 1) * voxel_vol_cm3:.2f}",
            "gt_tumor_cm3": f"{np.sum(gt_data == 2) * voxel_vol_cm3:.2f}",
            "pred_organ_cm3": f"{np.sum(pred_combined == 1) * voxel_vol_cm3:.2f}",
            "pred_tumor_cm3": f"{np.sum(pred_combined == 2) * voxel_vol_cm3:.2f}",
        })

    # Save manifest
    manifest_path = os.path.join(DELIVERY_DIR, "manifest.csv")
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=manifest[0].keys())
        writer.writeheader()
        writer.writerows(manifest)

    print(f"\nDeliverables saved to: {DELIVERY_DIR}")
    print(f"Manifest: {manifest_path}")
    print(f"Cases: {len(manifest)}")

    # Summary
    organ_dices = [float(m["organ_dice"]) for m in manifest]
    tumor_dices = [float(m["tumor_dice"]) for m in manifest if m["tumor_dice"] != "N/A"]
    print(f"Organ Dice: {np.mean(organ_dices):.4f} +/- {np.std(organ_dices):.4f}")
    if tumor_dices:
        print(f"Tumor Dice: {np.mean(tumor_dices):.4f} +/- {np.std(tumor_dices):.4f}")


def main():
    print("="*70)
    print("Pancreas CT (Task07) — Train + Generate NIfTI Deliverables")
    print("="*70)

    # Step 1: Train organ model (label=1, pancreas)
    print("\n--- Step 1: Train Organ (Pancreas) Model ---")
    organ_path = os.path.join(OUTPUT_DIR, ORGAN_WEIGHTS)
    train_pancreas_model(organ_path, label_value=1, label_name="organ")

    # Step 2: Train tumor model (label=2, cancer)
    print("\n--- Step 2: Train Tumor (Cancer) Model ---")
    tumor_path = os.path.join(OUTPUT_DIR, TUMOR_WEIGHTS)
    train_pancreas_model(tumor_path, label_value=2, label_name="tumor")

    # Step 3: Generate deliverables for all 281 cases
    print("\n--- Step 3: Generate NIfTI Deliverables ---")
    generate_deliverables()

    print("\n" + "="*70)
    print("DONE")
    print("="*70)


if __name__ == '__main__':
    main()
